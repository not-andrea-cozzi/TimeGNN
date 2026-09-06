"""
GamesBuilder.py

Builder unificato per il TRAIN set "games" (posizioni mate 1-10 da partite
reali, con clock reale quando disponibile), sorgente multipla:

    - Lichess:  PGN grezzo compresso (.pgn.zst)         -> tag "lichess"
    - FICS:     PGN grezzo compresso (.pgn.bz2)          -> tag "fics"
    - Club:     CSV con colonna pgn (es. chess.com dump) -> tag "club"

SCHEMA COMPRESSO (vedi DatasetCreator/GraphBuilder.py): ogni sample salvato
nel .pt contiene SOLO i campi utili al modello (board_packed, global_flags,
clock_norm, edge_index/edge_attr uint8, mate_n, best_move_idx, value_target,
rating, game_id, ply), tutti in tipi minimi (uint8/float16/int64/int32). Le
stringhe (fen, best_move_uci, problem_id, clock_source, source) NON entrano
piu' nel tensor dataset: vengono scritte a parte in un .jsonl via
GraphBuilder.write_debug_jsonl, nello stesso ordine dei sample salvati, cosi'
si puo' sempre risalire da un sample del dataset .pt al suo FEN/mossa/game_id
originali per debug o audit senza appesantire il file usato in training.

GAME_ID: ora è un identificatore univoco universale (UUID troncato a 63 bit),
generato dal processo principale per ogni partita accettata. Non c'è più
alcuna dipendenza da SequenceIdAllocator o da contatori locali: la
generazione è indipendente da riavvii o ordine di completamento dei worker.
La coda e la persistenza sono affidate a PositionQueueRegistry, che gestisce
lo spool su disco e il resume in caso di interruzione.

NOTA IMPORTANTE (fix ancdata / hang su multiprocessing.Pool):
Il worker NON ritorna oggetti torch_geometric.data.Data (che contengono
tensori) attraverso il Pool. I tensori PyTorch condivisi tra processi
tramite file descriptor (torch.multiprocessing sharing strategy
"file_system") possono rompersi con errori tipo:

    RuntimeError: received 0 items of ancdata

che uccide silenziosamente il thread interno _handle_results del Pool e
blocca run() indefinitamente in pool.join(). Per evitarlo, _worker()
serializza ogni Data in un dict di tipi Python nativi (liste, int, float,
str) via _data_to_plain_dict(), e run() lo ricostruisce in Data nel
processo principale via _plain_dict_to_data(). Nessun tensore attraversa
mai la pipe del Pool.
"""
from __future__ import annotations

import atexit
import bz2
import io
import os
import re
import random
import signal
import threading
import time
import zipfile
import multiprocessing as mp
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import zstandard as zstd
from DatasetPipeline.Model.PositionGraphSchema import PositionGraphSchema
from DatasetPipeline.PositionQueue import (
    PositionQueueRegistry,
    PositionQueueError,
    DEFAULT_STATE_FILENAME,
)
from DatasetPipeline.Utils.position_compression import compress_position_data

# ============================================================================
# STATO GLOBALE PER WORKER (Stockfish + Syzygy + watchdog)
# ============================================================================

_engine: Optional[chess.engine.SimpleEngine] = None
_engine_pid: Optional[int] = None
_tablebase: Optional["chess.syzygy.Tablebase"] = None

_watchdog_lock = threading.Lock()
_watchdog_deadline: Optional[float] = None
_watchdog_stop = threading.Event()
_watchdog_thread: Optional[threading.Thread] = None
_WATCHDOG_POLL_SECONDS = 1.0
_WATCHDOG_MARGIN_SECONDS = 3.0

# %clk: tempo RIMANENTE sull'orologio (Lichess, chess.com).
_CLK_RE = re.compile(r"\[\s*%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]")
# %emt: tempo SPESO sulla mossa, gia' una durata (FICS Games Database).
_EMT_RE = re.compile(r"\[\s*%emt\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]")

_SOURCE_TAG_BY_KIND = {
    "lichess": "lichess",
    "fics": "fics",
    "club": "club",
}


def _watchdog_arm(time_limit: float, margin: Optional[float] = None) -> None:
    global _watchdog_deadline
    eff_margin = _WATCHDOG_MARGIN_SECONDS if margin is None else margin
    with _watchdog_lock:
        _watchdog_deadline = time.monotonic() + time_limit + eff_margin


def _watchdog_disarm() -> None:
    global _watchdog_deadline
    with _watchdog_lock:
        _watchdog_deadline = None


def _watchdog_loop() -> None:
    global _engine, _engine_pid, _watchdog_deadline
    while not _watchdog_stop.is_set():
        with _watchdog_lock:
            deadline = _watchdog_deadline
        if deadline is not None and time.monotonic() > deadline:
            pid = _engine_pid
            if pid is not None:
                try:
                    import psutil
                    psutil.Process(pid).kill()
                except Exception:
                    try:
                        os.kill(pid, 9)
                    except Exception:
                        pass
            _engine = None
            _engine_pid = None
            with _watchdog_lock:
                _watchdog_deadline = None
        _watchdog_stop.wait(_WATCHDOG_POLL_SECONDS)


def _close_engine() -> None:
    global _engine, _engine_pid, _tablebase
    _watchdog_stop.set()
    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            pass
        finally:
            _engine = None
            _engine_pid = None
    if _tablebase is not None:
        try:
            _tablebase.close()
        except Exception:
            pass
        finally:
            _tablebase = None


def _worker_sigterm_handler(signum, frame) -> None:
    """SIGTERM (pool.terminate()) non esegue atexit: chiudiamo Stockfish
    esplicitamente prima di uscire, altrimenti resta orfano."""
    _close_engine()
    os._exit(0)


# ============================================================================
# SOURCE SPEC
# ============================================================================

@dataclass
class SourceSpec:
    """Descrive una sorgente di partite da processare.

    kind: "lichess" (.pgn.zst), "fics" (.pgn.bz2 o .pgn), "club" (CSV/CSV.zip
          con colonna pgn_col).
    path: percorso del file sorgente.
    pgn_col: solo per kind="club", nome colonna contenente il PGN.
    skip_games: partite iniziali da saltare.
    max_games: limite superiore di partite lette da questa sorgente.
    tag: etichetta leggibile; se None viene derivata da kind.
    """
    kind: str
    path: str
    pgn_col: str = "pgn"
    skip_games: int = 0
    max_games: Optional[int] = None
    tag: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in ("lichess", "fics", "club"):
            raise ValueError(f"SourceSpec.kind non valido: {self.kind}")
        if self.tag is None:
            self.tag = self.kind


class _ClosingStream:
    """File-like testuale che chiude anche il raw file sottostante (utile
    per .pgn.zst dove TextIOWrapper non chiude il file binario aperto a
    mano)."""

    def __init__(self, text_stream, raw_file):
        self._text_stream = text_stream
        self._raw_file = raw_file

    def __enter__(self):
        return self._text_stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._text_stream.close()
        finally:
            if self._raw_file is not None:
                self._raw_file.close()
        return False


# ============================================================================
# GAMES BUILDER
# ============================================================================

class GamesBuilder:
    """
    Pipeline unificata multi-sorgente per il train set "games" che utilizza
    PositionQueueRegistry per la coda e la persistenza, e UUID per i game_id.

    Il flusso è il seguente:
        1. I worker analizzano le partite e restituiscono dict serializzati.
        2. Il processo principale raggruppa i sample per partita (provisional_id)
           e assegna un UUID unico a tutti i sample della partita.
        3. Ogni sample viene accodato nel registry tramite enqueue().
        4. Al termine, i dati vengono estratti dal registry (drain_all),
           suddivisi in train/val/test in modo deterministico (basato sull'UUID)
           e salvati in file .pt e .jsonl.
    """

    _PIECE_VALUES: Dict[int, int] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    def __init__(
        self,
        sources: List[SourceSpec],
        stockfish_path: str,
        output_pt: str,
        mate_range: Tuple[int, int] = (1, 10),
        search_depth: int = 8,
        analysis_time: Optional[float] = 0.2,
        workers: Optional[int] = None,
        threads: int = 1,
        hash_mb: int = 128,
        multipv: int = 1,
        seed: int = 42,
        default_move_seconds: float = 15.0,
        avg_time_by_rating: Optional[Dict[int, float]] = None,
        require_clock: bool = False,
        min_ply: int = 8,
        ply_sample_step: int = 3,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        min_game_plies: int = 20,
        max_positions_per_game: Optional[int] = 20,
        candidate_min_legal_moves: int = 1,
        candidate_max_legal_moves: Optional[int] = None,
        skip_if_in_check: bool = False,
        max_piece_count: Optional[int] = 18,
        only_decisive_games: bool = True,
        skip_time_forfeit: bool = True,
        min_material_for_mate_attempt: int = 4,
        drop_zero_clock: bool = True,
        pool_join_timeout: Optional[float] = 20.0,
        syzygy_path: Optional[str] = None,
        checkpoint_every: int = 5000,
        config_error_cls: type = ValueError,
        min_rating: Optional[int] = 1200,
        max_rating: Optional[int] = None,
        min_material_diff_for_mate_attempt: int = 3,
        require_heavy_piece: bool = True,
        skip_forced_moves: bool = False,
        dedupe_positions: bool = True,
        skip_trivial_endgame: bool = True,
        registry_state_path: Optional[str] = None,
    ):
        """
        Args:
            registry_state_path: percorso del file di stato per PositionQueueRegistry.
                Se None, viene usato DEFAULT_STATE_FILENAME nella directory di output.
        """
        self.sources = sources
        self.stockfish_path = stockfish_path
        self.output_pt = output_pt
        self.mate_range = mate_range
        self.search_depth = search_depth
        self.analysis_time = analysis_time
        self.threads = threads
        self.hash_mb = hash_mb
        self.multipv = multipv
        self.seed = seed
        self.default_move_seconds = default_move_seconds
        self.avg_time_by_rating = avg_time_by_rating or {}
        self.require_clock = require_clock
        self.min_ply = max(0, min_ply)
        self.ply_sample_step = max(1, ply_sample_step)
        self.split_ratios = split_ratios
        self.min_game_plies = min_game_plies
        self.max_positions_per_game = max_positions_per_game
        self.candidate_min_legal_moves = candidate_min_legal_moves
        self.candidate_max_legal_moves = candidate_max_legal_moves
        self.skip_if_in_check = skip_if_in_check
        self.max_piece_count = max_piece_count
        self.only_decisive_games = only_decisive_games
        self.skip_time_forfeit = skip_time_forfeit
        self.min_material_for_mate_attempt = min_material_for_mate_attempt
        self.drop_zero_clock = drop_zero_clock
        self.pool_join_timeout = pool_join_timeout
        self.syzygy_path = syzygy_path
        self.checkpoint_every = checkpoint_every
        self._config_error_cls = config_error_cls

        self.min_rating = min_rating
        self.max_rating = max_rating
        self.min_material_diff_for_mate_attempt = min_material_diff_for_mate_attempt
        self.require_heavy_piece = require_heavy_piece
        self.skip_forced_moves = skip_forced_moves
        self.dedupe_positions = dedupe_positions
        self.skip_trivial_endgame = skip_trivial_endgame

        cpu_count = os.cpu_count() or 2
        self.workers = workers or max(1, cpu_count - 1)

        # Percorso del registry (se non specificato, nella stessa cartella di output)
        if registry_state_path is None:
            out_dir = os.path.dirname(os.path.abspath(output_pt)) or "."
            registry_state_path = os.path.join(out_dir, DEFAULT_STATE_FILENAME)
        self.registry_state_path = registry_state_path

        self._validate_parameters()

    # ================================================================
    # VALIDATION
    # ================================================================

    def _validate_parameters(self) -> None:
        lo, hi = self.mate_range
        if lo < 1:
            raise self._config_error_cls("mate_range deve iniziare da almeno 1.")
        if hi < lo:
            raise self._config_error_cls("mate_range non valido.")
        if hi > 255:
            raise self._config_error_cls(
                "mate_range superiore a 255 non supportato dallo schema compresso (mate_n e' uint8)."
            )
        if not self.sources:
            raise self._config_error_cls("Serve almeno una SourceSpec in 'sources'.")
        if self.workers < 1:
            raise self._config_error_cls("workers deve essere >= 1.")
        if self.threads < 1:
            raise self._config_error_cls("threads deve essere >= 1.")
        if self.search_depth < 1:
            raise self._config_error_cls("search_depth deve essere >= 1.")
        if self.analysis_time is not None and self.analysis_time <= 0:
            raise self._config_error_cls("analysis_time deve essere > 0.")
        if len(self.split_ratios) != 3 or abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise self._config_error_cls("split_ratios deve sommare a 1.0.")
        if self.max_piece_count is not None and self.max_piece_count < 2:
            raise self._config_error_cls("max_piece_count deve essere >= 2.")
        if self.min_material_for_mate_attempt < 0:
            raise self._config_error_cls("min_material_for_mate_attempt deve essere >= 0.")
        if self.min_material_diff_for_mate_attempt < 0:
            raise self._config_error_cls("min_material_diff_for_mate_attempt deve essere >= 0.")
        if (
            self.min_rating is not None
            and self.max_rating is not None
            and self.min_rating > self.max_rating
        ):
            raise self._config_error_cls("min_rating deve essere <= max_rating.")
        for src in self.sources:
            if not os.path.exists(src.path):
                raise self._config_error_cls(f"Sorgente non trovata: {src.path} (kind={src.kind}).")
        if not (os.path.exists(self.stockfish_path) and os.access(self.stockfish_path, os.X_OK)):
            raise self._config_error_cls(f"Stockfish non trovato/eseguibile: {self.stockfish_path}.")
        if self.syzygy_path is not None and not os.path.isdir(self.syzygy_path):
            raise self._config_error_cls(f"syzygy_path non e' una cartella valida: {self.syzygy_path}.")

    # ================================================================
    # SERIALIZZAZIONE Data <-> dict di tipi nativi (schema compresso)
    # ================================================================

    @staticmethod
    def _data_to_plain_dict(data: Data) -> Dict[str, Any]:
        """Converte gli attributi TENSORE del sample compresso in tipi
        Python nativi (liste/int/float), sicuro da passare attraverso
        multiprocessing.Pool."""
        return {
            "edge_index_u8": data.edge_index_u8.tolist(),
            "edge_attr": data.edge_attr.tolist(),
            "board_packed": data.board_packed.tolist(),
            "global_flags": int(data.global_flags.item()),
            "clock_norm": float(data.clock_norm.item()),
            "mate_n": int(data.mate_n.item()),
            "best_move_idx": int(data.best_move_idx.item()),
            "value_target": float(data.value_target.item()),
            "rating": float(data.rating.item()),
            "game_id": int(data.game_id.item()),   # provvisorio (verrà sovrascritto)
            "ply": int(data.ply.item()),
        }

    @staticmethod
    def _plain_dict_to_data(d: Dict[str, Any], game_id_override: Optional[int] = None) -> Data:
        """Ricostruisce un torch_geometric.data.Data compresso da un dict
        di tipi nativi prodotto da _data_to_plain_dict, nel processo
        principale.

        Args:
            game_id_override: se fornito, sovrascrive d["game_id"] con
                questo valore (tipicamente un UUID troncato).
        """
        edge_index_u8 = torch.tensor(d["edge_index_u8"], dtype=torch.uint8)
        data = Data(edge_index=edge_index_u8.long())
        data.edge_index_u8 = edge_index_u8
        data.edge_attr = torch.tensor(d["edge_attr"], dtype=torch.uint8)
        data.board_packed = torch.tensor(d["board_packed"], dtype=torch.uint8)
        data.global_flags = torch.tensor([d["global_flags"]], dtype=torch.uint8)
        data.clock_norm = torch.tensor([d["clock_norm"]], dtype=torch.float16)
        data.mate_n = torch.tensor([d["mate_n"]], dtype=torch.uint8)
        data.best_move_idx = torch.tensor([d["best_move_idx"]], dtype=torch.uint8)
        data.value_target = torch.tensor([d["value_target"]], dtype=torch.float16)
        data.rating = torch.tensor([d["rating"]], dtype=torch.float16)
        final_game_id = game_id_override if game_id_override is not None else d.get("game_id", -1)
        data.game_id = torch.tensor([int(final_game_id)], dtype=torch.int64)
        data.ply = torch.tensor([int(d.get("ply", -1))], dtype=torch.int32)
        return data

    # ================================================================
    # WORKER INIT (Stockfish + Syzygy + watchdog + signal handling)
    # ================================================================

    @staticmethod
    def _init_worker(stockfish_path: str, threads: int, hash_mb: int, syzygy_path: Optional[str]) -> None:
        global _engine, _engine_pid, _tablebase, _watchdog_thread

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, _worker_sigterm_handler)

        try:
            os.setpgrp()   # rende il worker leader del suo gruppo processi
        except AttributeError:
            pass

        try:
            _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            _engine.configure({"Threads": threads, "Hash": hash_mb})
            try:
                _engine_pid = _engine.transport.get_pid()
            except Exception:
                _engine_pid = None
        except Exception as e:
            _engine = None
            _engine_pid = None
            raise RuntimeError(f"Impossibile avviare Stockfish: {e}")

        if syzygy_path:
            try:
                _tablebase = chess.syzygy.open_tablebase(syzygy_path)
            except Exception:
                _tablebase = None

        atexit.register(_close_engine)

        _watchdog_stop.clear()
        _watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
        _watchdog_thread.start()

    # ================================================================
    # CLOCK PARSING
    # ================================================================

    @staticmethod
    def _parse_clk(comment: str) -> Optional[float]:
        if not comment:
            return None
        m = _CLK_RE.search(comment)
        if not m:
            return None
        h, mi, s = m.groups()
        return int(h) * 3600 + int(mi) * 60 + float(s)

    @staticmethod
    def _parse_emt(comment: str) -> Optional[float]:
        if not comment:
            return None
        m = _EMT_RE.search(comment)
        if not m:
            return None
        h, mi, s = m.groups()
        return int(h) * 3600 + int(mi) * 60 + float(s)

    @staticmethod
    def _parse_time_control(time_control: str) -> Tuple[float, float]:
        if not time_control or time_control == "-":
            return 0.0, 0.0
        m = re.match(r"^(\d+)\+(\d+)$", time_control)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.match(r"^(\d+)$", time_control)
        if m:
            return float(m.group(1)), 0.0
        return 0.0, 0.0

    @staticmethod
    def _compute_move_duration(previous_clock: Optional[float], current_clock: Optional[float], increment: float) -> Optional[float]:
        if previous_clock is None or current_clock is None:
            return None
        return max(0.0, previous_clock - current_clock + increment)

    @staticmethod
    def _parse_rating(raw: str) -> Optional[int]:
        if not raw:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            digits = "".join(ch for ch in raw if ch.isdigit())
            return int(digits) if digits else None

    def _closest_bucket_time(self, rating: Optional[int]) -> Optional[float]:
        if rating is None or not self.avg_time_by_rating:
            return None
        closest = min(self.avg_time_by_rating.keys(), key=lambda b: abs(b - rating))
        return self.avg_time_by_rating[closest]

    # ================================================================
    # ECONOMIC FILTERS
    # ================================================================

    def _game_is_eligible(self, game: "chess.pgn.Game") -> bool:
        if game is None:
            return False
        try:
            ply_count = game.end().ply()
        except Exception:
            return False
        if ply_count < self.min_game_plies:
            return False
        if self.only_decisive_games:
            result = game.headers.get("Result", "")
            if result not in ("1-0", "0-1"):
                return False
        if self.skip_time_forfeit:
            termination = game.headers.get("Termination", "")
            if "Time forfeit" in termination:
                return False

        if self.min_rating is not None or self.max_rating is not None:
            white_elo = self._parse_rating(game.headers.get("WhiteElo", ""))
            black_elo = self._parse_rating(game.headers.get("BlackElo", ""))
            ratings = [r for r in (white_elo, black_elo) if r is not None]
            if ratings:
                best_rating = max(ratings)
                if self.min_rating is not None and best_rating < self.min_rating:
                    return False
                if self.max_rating is not None and min(ratings) > self.max_rating:
                    return False

        return True

    def _get_candidate_legal_moves(self, board: "chess.Board") -> Optional[List["chess.Move"]]:
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            return None
        if self.max_piece_count is not None and len(board.piece_map()) > self.max_piece_count:
            return None

        legal_moves = list(board.legal_moves)
        if len(legal_moves) < self.candidate_min_legal_moves:
            return None
        if self.candidate_max_legal_moves is not None and len(legal_moves) > self.candidate_max_legal_moves:
            return None
        if self.skip_if_in_check and board.is_check():
            return None
        if len(legal_moves) > 255:
            return None
        return legal_moves

    def _material_by_color(self, board: "chess.Board") -> Tuple[int, int]:
        white_mat = 0
        black_mat = 0
        for p in board.piece_map().values():
            val = self._PIECE_VALUES.get(p.piece_type, 0)
            if p.color == chess.WHITE:
                white_mat += val
            else:
                black_mat += val
        return white_mat, black_mat

    def _has_mating_material(self, board: "chess.Board") -> bool:
        mover = board.turn
        white_mat, black_mat = self._material_by_color(board)
        mover_mat = white_mat if mover == chess.WHITE else black_mat
        opp_mat = black_mat if mover == chess.WHITE else white_mat

        if mover_mat < self.min_material_for_mate_attempt:
            return False
        if (mover_mat - opp_mat) < self.min_material_diff_for_mate_attempt:
            return False
        return True

    def _mover_has_heavy_piece(self, board: "chess.Board") -> bool:
        mover = board.turn
        for piece_type in (chess.QUEEN, chess.ROOK):
            if board.pieces(piece_type, mover):
                return True
        return False

    def _is_trivially_drawn_endgame(self, board: "chess.Board") -> bool:
        piece_map = board.piece_map()
        has_heavy_or_pawn = any(
            p.piece_type in (chess.QUEEN, chess.ROOK, chess.PAWN)
            for p in piece_map.values()
        )
        if has_heavy_or_pawn:
            return False

        white_minors = sum(1 for p in piece_map.values() if p.color == chess.WHITE and p.piece_type in (chess.BISHOP, chess.KNIGHT))
        black_minors = sum(1 for p in piece_map.values() if p.color == chess.BLACK and p.piece_type in (chess.BISHOP, chess.KNIGHT))

        return white_minors <= 1 and black_minors <= 1

    def _syzygy_says_no_mate(self, board: "chess.Board") -> bool:
        global _tablebase
        if _tablebase is None:
            return False
        if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
            return False
        try:
            wdl = _tablebase.probe_wdl(board)
        except (KeyError, chess.syzygy.MissingTableError):
            return False
        except Exception:
            return False
        return wdl is not None and wdl <= 0

    # ================================================================
    # STOCKFISH ANALYSIS
    # ================================================================

    def _analyse_position(self, board: "chess.Board"):
        global _engine
        if _engine is None:
            return None
        try:
            if self.analysis_time is not None:
                limit = chess.engine.Limit(time=self.analysis_time, mate=self.mate_range[1])
                _watchdog_arm(self.analysis_time)
            else:
                limit = chess.engine.Limit(depth=self.search_depth, mate=self.mate_range[1])
                _watchdog_arm(5.0)
            return _engine.analyse(board, limit, multipv=self.multipv)
        except (
            chess.engine.EngineTerminatedError,
            chess.engine.EngineError,
            BrokenPipeError,
            ConnectionResetError,
            OSError,
        ):
            return None
        except Exception:
            return None
        finally:
            _watchdog_disarm()

    # ================================================================
    # WORKER: analizza una singola partita, source-agnostic
    # ================================================================

    def _worker(self, args: Tuple[int, str, str]) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """args = (provisional_game_id, pgn_text, source_tag_str).

        Ritorna (provisional_game_id, tensor_list, debug_list) — MAI
        oggetti Data con tensori (vedi nota sulla serializzazione).
        tensor_list e debug_list sono paralleli e descrivono lo stesso sample.
        Il campo "game_id" nei tensor_list è il provisional_game_id (usato
        per raggruppare i sample della stessa partita nel master).
        """
        global _engine
        game_id, pgn_text, source_tag = args

        if _engine is None:
            return game_id, [], []

        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text))
        except Exception:
            return game_id, [], []

        if game is None:
            return game_id, [], []

        if game.headers.get("Variant", "Standard").lower() not in ("standard", "normal"):
            return game_id, [], []

        if not self._game_is_eligible(game):
            return game_id, [], []

        time_control = game.headers.get("TimeControl", "")
        base_time, increment = self._parse_time_control(time_control)

        white_elo = self._parse_rating(game.headers.get("WhiteElo", ""))
        black_elo = self._parse_rating(game.headers.get("BlackElo", ""))
        mover_rating = {chess.WHITE: white_elo, chess.BLACK: black_elo}

        previous_clock = {
            chess.WHITE: base_time if base_time > 0 else None,
            chess.BLACK: base_time if base_time > 0 else None,
        }

        tensor_list: List[Dict[str, Any]] = []
        debug_list: List[Dict[str, Any]] = []
        node = game
        mate_lo, mate_hi = self.mate_range
        positions_analysed = 0

        seen_positions: set = set()

        try:
            while node.variations:
                next_node = node.variation(0)
                board = node.board()
                comment = next_node.comment or ""
                mover_color = board.turn

                if self.dedupe_positions:
                    position_key = " ".join(board.fen().split(" ")[:4])
                    if position_key in seen_positions:
                        node = next_node
                        continue
                    seen_positions.add(position_key)

                emt_seconds = self._parse_emt(comment)
                current_clock = self._parse_clk(comment)

                if emt_seconds is not None:
                    move_duration = emt_seconds
                    duration_is_real = True
                    clock_source = "real_emt"
                else:
                    move_duration = self._compute_move_duration(
                        previous_clock[mover_color], current_clock, increment
                    )
                    duration_is_real = move_duration is not None
                    clock_source = "real_clk" if duration_is_real else None

                if current_clock is not None:
                    previous_clock[mover_color] = current_clock

                if node.ply() < self.min_ply:
                    node = next_node
                    continue
                if (node.ply() - self.min_ply) % self.ply_sample_step != 0:
                    node = next_node
                    continue
                if self.require_clock and not duration_is_real:
                    node = next_node
                    continue
                if self.max_positions_per_game is not None and positions_analysed >= self.max_positions_per_game:
                    break

                legal_moves = self._get_candidate_legal_moves(board)
                if legal_moves is None:
                    node = next_node
                    continue

                if self.skip_forced_moves and len(legal_moves) == 1:
                    node = next_node
                    continue

                if self.require_heavy_piece and not self._mover_has_heavy_piece(board):
                    node = next_node
                    continue

                if not self._has_mating_material(board):
                    node = next_node
                    continue

                if self.skip_trivial_endgame and self._is_trivially_drawn_endgame(board):
                    node = next_node
                    continue

                if self.min_rating is not None or self.max_rating is not None:
                    mover_rating_val = mover_rating[mover_color]
                    if mover_rating_val is not None:
                        if self.min_rating is not None and mover_rating_val < self.min_rating:
                            node = next_node
                            continue
                        if self.max_rating is not None and mover_rating_val > self.max_rating:
                            node = next_node
                            continue

                if self._syzygy_says_no_mate(board):
                    node = next_node
                    continue

                info = self._analyse_position(board)
                positions_analysed += 1
                if not info:
                    node = next_node
                    continue

                best_info = info[0]
                score = best_info.get("score")
                if score is None:
                    node = next_node
                    continue
                relative_score = score.relative
                if not relative_score.is_mate():
                    node = next_node
                    continue
                mate_n = relative_score.mate()
                if mate_n is None or not (mate_n > 0 and mate_lo <= mate_n <= mate_hi):
                    node = next_node
                    continue

                pv = best_info.get("pv")
                if not pv:
                    node = next_node
                    continue
                best_move = pv[0]
                try:
                    best_move_idx = legal_moves.index(best_move)
                except ValueError:
                    node = next_node
                    continue

                if duration_is_real:
                    clock_seconds = move_duration
                else:
                    bucket_time = self._closest_bucket_time(mover_rating[mover_color])
                    if bucket_time is not None:
                        clock_seconds = bucket_time
                        clock_source = "rating_bucket"
                    else:
                        clock_seconds = self.default_move_seconds
                        clock_source = "default_constant"

                if self.drop_zero_clock and clock_seconds == 0.0 and not duration_is_real:
                    node = next_node
                    continue

                rating_val = mover_rating[mover_color]

                label = {"mate_n": int(mate_n), "best_move_idx": int(best_move_idx)}
                try:
                    # game_id provvisorio (provisional_game_id) – verrà sovrascritto dal master con UUID
                    data = GraphBuilder.board_to_pyg_data(
                        board,
                        clock_seconds=clock_seconds,
                        label=label,
                        legal_moves=legal_moves,
                        rating=float(rating_val) if rating_val is not None else None,
                        game_id=game_id,
                        ply=node.ply(),
                    )
                except Exception:
                    node = next_node
                    continue

                problem_id = f"{source_tag}_{game_id}_{node.ply()}"

                tensor_list.append(self._data_to_plain_dict(data))
                debug_list.append({
                    "problem_id": problem_id,
                    "fen": board.fen(),
                    "best_move_uci": best_move.uci(),
                    "mate_n": int(mate_n),
                    "ply": int(node.ply()),
                    "source": source_tag,
                    "clock_source": clock_source or "unknown",
                    "clock_seconds": float(clock_seconds),
                    "clock_is_real": bool(duration_is_real),
                    "rating": rating_val,
                })
                node = next_node

        except Exception:
            pass

        return game_id, tensor_list, debug_list

    # ================================================================
    # STREAMING SORGENTI (source-agnostic)
    # ================================================================

    def _open_pgn_text_stream(self, path: str, kind: str):
        if kind == "lichess":
            if zstd is None:
                raise self._config_error_cls(
                    "Il pacchetto 'zstandard' e' richiesto per file .pgn.zst. Installa con: pip install zstandard"
                )
            raw_file = open(path, "rb")
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(raw_file)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            return _ClosingStream(text_stream, raw_file)

        if kind == "fics":
            if path.lower().endswith(".bz2"):
                raw_file = bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
                return _ClosingStream(raw_file, None)
            raw_file = open(path, "r", encoding="utf-8", errors="replace")
            return _ClosingStream(raw_file, None)

        raise self._config_error_cls(f"_open_pgn_text_stream non applicabile a kind={kind}")

    def _iter_pgn_texts(self, text_stream, skip_games: int, max_games: Optional[int]) -> Generator[Tuple[int, str], None, None]:
        local_id = 0
        yielded = 0
        current_game: List[str] = []

        for line in text_stream:
            if line.startswith("[Event ") and current_game:
                local_id += 1
                if local_id > skip_games:
                    yield (local_id, "".join(current_game))
                    yielded += 1
                    if max_games is not None and yielded >= max_games:
                        return
                current_game = [line]
            else:
                current_game.append(line)

        if current_game:
            local_id += 1
            if local_id > skip_games:
                if max_games is None or yielded < max_games:
                    yield (local_id, "".join(current_game))

    def _iter_club_csv(self, src: SourceSpec) -> Generator[Tuple[int, str], None, None]:
        if src.path.endswith(".zip"):
            with zipfile.ZipFile(src.path) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    raise self._config_error_cls(f"Nessun CSV trovato dentro {src.path}.")
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
        else:
            df = pd.read_csv(src.path)

        if src.pgn_col not in df.columns:
            raise self._config_error_cls(
                f"Colonna '{src.pgn_col}' assente in {src.path}. Colonne disponibili: {list(df.columns)}"
            )

        series = df[src.pgn_col].dropna().iloc[src.skip_games:]
        if src.max_games is not None:
            series = series.iloc[: src.max_games]
        for local_id, pgn_text in series.items():
            yield (int(local_id) + 1, pgn_text)

    def _iter_source(self, src: SourceSpec) -> Generator[Tuple[int, str], None, None]:
        if src.kind == "club":
            yield from self._iter_club_csv(src)
            return

        with self._open_pgn_text_stream(src.path, src.kind) as text_stream:
            yield from self._iter_pgn_texts(text_stream, src.skip_games, src.max_games)

    def _iter_all_tasks(self) -> Generator[Tuple[int, str, str], None, None]:
        """Genera (provisional_game_id, pgn_text, source_tag)."""
        global_id = 0
        for src in self.sources:
            for _local_id, pgn_text in self._iter_source(src):
                global_id += 1
                yield (global_id, pgn_text, src.tag)

    def _count_tasks_estimate(self) -> Optional[int]:
        total = 0
        any_unknown = False
        for src in self.sources:
            if src.max_games is not None:
                total += src.max_games
            else:
                any_unknown = True
        return None if any_unknown else total

    # ================================================================
    # SPLIT DETERMINISTICO PER PARTITA (basato su UUID)
    # ================================================================

    def _assign_game_split(self, game_id: int, seed: int) -> str:
        """Assegna split in modo deterministico usando hash di game_id e seed."""
        rng = random.Random(seed + hash(game_id))
        value = rng.random()
        train_ratio, val_ratio, _ = self.split_ratios
        if value < train_ratio:
            return "train"
        if value < train_ratio + val_ratio:
            return "val"
        return "test"

    # ================================================================
    # RUN
    # ================================================================

    def run(self) -> Tuple[Dict[str, List[Data]], Dict[str, str]]:
        output_directory = os.path.dirname(self.output_pt)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        # Istanza del registry (singleton)
        registry = PositionQueueRegistry.instance(
            state_path=self.registry_state_path,
            shard_size=self.checkpoint_every if self.checkpoint_every > 0 else 10000,
        )

        split_data: Dict[str, List[Data]] = {"train": [], "val": [], "test": []}
        split_debug: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        source_counts: Dict[str, int] = defaultdict(int)
        clock_source_counts: Dict[str, int] = defaultdict(int)
        mate_n_counts: Dict[int, int] = defaultdict(int)

        pool = mp.Pool(
            processes=self.workers,
            initializer=self._init_worker,
            initargs=(self.stockfish_path, self.threads, self.hash_mb, self.syzygy_path),
        )

        processed_games = 0
        accepted_games = 0
        generated_positions = 0
        estimate = self._count_tasks_estimate()

        # Dizionario per associare i debug ai local_ref restituiti da enqueue
        debug_map: Dict[int, Dict[str, Any]] = {}

        def cleanup_after_failure() -> None:
            old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                pool.terminate()
                timeout = self.pool_join_timeout if self.pool_join_timeout is not None else 15.0
                deadline = time.monotonic() + timeout
                for proc in pool._pool:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    proc.join(timeout=remaining)

                still_alive = [p for p in pool._pool if p.is_alive()]
                if still_alive:
                    print(
                        f"\n[WARNING] {len(still_alive)} worker non terminati entro "
                        f"{timeout}s. Invio SIGKILL al gruppo processi..."
                    )
                    for proc in still_alive:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except (OSError, AttributeError):
                            try:
                                os.kill(proc.pid, signal.SIGKILL)
                            except OSError:
                                pass
                    time.sleep(0.5)
                    for proc in still_alive:
                        try:
                            proc.join(timeout=1.0)
                        except Exception:
                            pass
            finally:
                signal.signal(signal.SIGINT, old_sigint)

        try:
            task_stream = self._iter_all_tasks()
            results = pool.imap_unordered(self._worker, task_stream, chunksize=1)

            pbar = tqdm(results, desc="Analisi Partite (multi-sorgente)", total=estimate, dynamic_ncols=True)
            for provisional_game_id, raw_tensor_list, raw_debug_list in pbar:
                processed_games += 1

                if not raw_tensor_list:
                    continue

                accepted_games += 1

                # Genera un UUID (troncato a 63 bit) per questa partita
                game_uuid = uuid.uuid4().int & ((1 << 63) - 1)

                # Accoda tutti i sample di questa partita con lo stesso UUID
                for tensor_dict, debug_entry in zip(raw_tensor_list, raw_debug_list):
                    # Ricostruisci il Data con il game_id UUID
                    data = self._plain_dict_to_data(tensor_dict, game_id_override=game_uuid)

                    # group_key per il registry = mate_n
                    group_key = debug_entry["mate_n"]

                    # Accoda nel registry (ottieni local_ref)
                    local_ref = registry.enqueue(
                        source_tag=debug_entry["source"],
                        data=data,
                        group_key=group_key,
                    )

                    # Salva il debug associato al local_ref
                    debug_map[local_ref] = debug_entry

                generated_positions += len(raw_tensor_list)

                # Aggiorna statistiche (usando i debug)
                for dbg in raw_debug_list:
                    source_counts[dbg["source"]] += 1
                    clock_source_counts[dbg["clock_source"]] += 1
                    mate_n_counts[dbg["mate_n"]] += 1

                # Checkpoint: flush periodico? Il registry scrive automaticamente
                # quando raggiunge shard_size; possiamo chiamare flush ogni tanto
                # ma non necessario.

            pbar.close()
        except KeyboardInterrupt:
            print("\n[WARNING] Interruzione richiesta: arresto pulito dei worker in corso...")
            cleanup_after_failure()
            raise
        except Exception:
            print("\n[WARNING] Errore durante l'analisi: arresto pulito dei worker in corso...")
            cleanup_after_failure()
            raise
        finally:
            pool.close()

            if self.pool_join_timeout is not None:
                deadline = time.monotonic() + self.pool_join_timeout
                for proc in pool._pool:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    proc.join(timeout=remaining)

                still_alive = [p for p in pool._pool if p.is_alive()]
                if still_alive:
                    stale_pids = [p.pid for p in still_alive]
                    print(
                        f"\n[WARNING] {len(still_alive)} worker non terminati entro "
                        f"{self.pool_join_timeout}s. Forzo pool.terminate()."
                    )
                    pool.terminate()
                    try:
                        import psutil
                        for pid in stale_pids:
                            try:
                                parent = psutil.Process(pid)
                                for child in parent.children(recursive=True):
                                    child.terminate()
                            except psutil.NoSuchProcess:
                                continue
                    except ImportError:
                        print("[WARNING] 'psutil' non disponibile: eventuali processi Stockfish orfani vanno terminati manualmente.")

            pool.join()

        # Flush finale del registry (assicura che tutti i dati siano su disco)
        registry.flush()

        # Estrai tutti i dati dalla coda (ordine FIFO)
        queue_items = registry._drain_all()  # lista di _QueuedPosition

        # Ricostruisci le liste allineate: Data e debug
        all_data: List[Data] = []
        all_debug: List[Dict[str, Any]] = []
        for item in queue_items:
            all_data.append(item.data)
            all_debug.append(debug_map[item.local_ref])

        # Ora applica lo split deterministico per partita (UUID)
        # Raggruppa per game_id
        windows: Dict[int, List[int]] = defaultdict(list)  # game_id -> indici in all_data
        for idx, data in enumerate(all_data):
            gid = data.game_id.item()
            windows[gid].append(idx)

        for gid, indices in windows.items():
            split_name = self._assign_game_split(gid, self.seed)
            for idx in indices:
                split_data[split_name].append(all_data[idx])
                split_debug[split_name].append(all_debug[idx])

        # Mescola ogni split (per evitare bias d'ordine)
        for split_name in ("train", "val", "test"):
            if split_data[split_name]:
                # Usa un generatore deterministico
                gen = torch.Generator().manual_seed(self.seed + hash(split_name))
                perm = torch.randperm(len(split_data[split_name]), generator=gen)
                split_data[split_name] = [split_data[split_name][i] for i in perm.tolist()]
                split_debug[split_name] = [split_debug[split_name][i] for i in perm.tolist()]

        # Salva i file .pt e .jsonl
        paths = self._save_splits(split_data, split_debug, source_counts, clock_source_counts, mate_n_counts, processed_games, accepted_games)

        # Pulisci lo spool del registry (i file shard non sono più necessari)
        # Il registry ha già rimosso gli shard in _clear_spool() dopo drain_all?
        # In realtà _drain_all non rimuove gli shard; dobbiamo chiamare _clear_spool()
        registry._clear_spool()

        return split_data, paths

    def _save_splits(
        self,
        split_data: Dict[str, List[Data]],
        split_debug: Dict[str, List[Dict[str, Any]]],
        source_counts: Dict[str, int],
        clock_source_counts: Dict[str, int],
        mate_n_counts: Dict[int, int],
        processed_games: int,
        accepted_games: int,
    ) -> Dict[str, str]:
        base_path, extension = os.path.splitext(self.output_pt)
        if not extension:
            extension = ".pt"

        paths: Dict[str, str] = {}
        for split_name, data_list in split_data.items():
            output_path = f"{base_path}_{split_name}{extension}"
            tmp_path = output_path + ".tmp"
            torch.save(data_list, tmp_path)
            os.replace(tmp_path, output_path)
            paths[split_name] = output_path

            # Debug .jsonl
            debug_path = f"{base_path}_{split_name}.jsonl"
            GraphBuilder.write_debug_jsonl(split_debug[split_name], debug_path)

        total_positions = sum(len(v) for v in split_data.values())

        print("\n" + "=" * 60)
        print("GAMES BUILDER — SUMMARY (schema compresso, UUID game_id)")
        print("=" * 60)
        print(f"Partite processate: {processed_games:,}")
        print(f"Partite con posizioni mate: {accepted_games:,}")
        print(f"Train: {len(split_data['train']):,} | Val: {len(split_data['val']):,} | Test: {len(split_data['test']):,}")
        print(f"Totale posizioni: {total_positions:,}")

        if total_positions > 0:
            print("\nPer sorgente:")
            for tag, c in sorted(source_counts.items()):
                print(f"  {tag}: {c:,} ({100 * c / total_positions:.1f}%)")

            print("\nPer origine clock:")
            for cs, c in sorted(clock_source_counts.items()):
                print(f"  {cs}: {c:,} ({100 * c / total_positions:.1f}%)")

            print("\nPer profondita' mate (n):")
            for n in sorted(mate_n_counts.keys()):
                print(f"  n={n}: {mate_n_counts[n]:,}")

        print("=" * 60)
        return paths