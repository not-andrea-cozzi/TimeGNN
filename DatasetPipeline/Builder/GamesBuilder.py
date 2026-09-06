"""
GamesBuilder.py

Builder unificato per il TRAIN set "games" (posizioni tratte da finestre di
matto forzato in partite reali, con clock reale quando disponibile),
sorgente multipla:

    - Lichess:  PGN grezzo compresso (.pgn.zst)         -> tag "lichess"
    - FICS:     PGN grezzo compresso (.pgn.bz2)          -> tag "fics"
    - Club:     CSV con colonna pgn (es. chess.com dump) -> tag "club"

SCHEMA DATI: BOARD-LEVEL PER POSIZIONE (vedi PositionGraphSchema.py)
=====================================================================
Ogni CAMPIONE prodotto e' una SINGOLA POSIZIONE (grafo spaziale a 64 nodi/
caselle), con la mossa REALMENTE GIOCATA in quella posizione come target.
Non e' piu' un grafo-sequenza (superato, vedi git history / conversazione
di progetto): il modello di destinazione (DualGATModel / DualGATTimeAwareModel,
vedi PositionGraphSchema.py per la motivazione completa della scelta) opera
su grafi spaziali per-nodo, non su sequenze di eventi.

RICERCA DELLA FINESTRA DI MATTO FORZATO (invariata nella logica, cambia
cosa viene fatto con ogni ply trovato)
=====================================================================
Si cerca, con Stockfish, il PRIMO ply della partita con una posizione
"mate in N" (N mosse intere, dentro mate_range). Si segue poi il replay dei
ply REALMENTE GIOCATI nel PGN (non il principal variation di Stockfish) per
esattamente 2N-1 semi-mosse (formula standard: N mosse intere del mover =
2N-1 ply totali fino al matto incluso). Si accetta la finestra SOLO se il
replay arriva davvero a board.is_checkmate() sull'ultimo ply (verifica
stretta): se la partita reale devia dal piano di matto forzato, la finestra
viene scartata.

DIFFERENZA CHIAVE rispetto alla versione a grafo-sequenza: qui OGNI PLY
della finestra accettata diventa un CAMPIONE INDIPENDENTE (una posizione =
un Data), non nodi di un unico grafo. Tutti i campioni della stessa
finestra condividono lo stesso game_id (assegnato una volta per finestra
nel processo principale, vedi sotto), cosi' PositionQueueRegistry puo'
tenerli split-safe (nessuna finestra spezzata tra train/val/test, vedi
PositionQueue.py).

CONTRATTO WORKER -> PROCESSO PRINCIPALE
=====================================================================
Il worker (processo figlio) NON costruisce alcun torch_geometric.data.Data
e NON assegna alcun game_id. Ritorna esclusivamente dati NATIVI Python (fen
della posizione PRIMA di ogni mossa, mossa in UCI, secondi impiegati, ply):
questo evita ogni rischio "ancdata" (RuntimeError: received 0 items of
ancdata, causato da tensori PyTorch passati attraverso multiprocessing.Pool
con sharing strategy "file_system") per costruzione, non per convenzione
manuale da rispettare -- semplicemente non ci sono tensori da serializzare.

Il processo principale (run()), per ogni finestra accettata:
    1. alloca un nuovo game_id (contatore locale progressivo, run() e'
       l'unico consumer sequenziale della pipeline, nessun rischio di
       collisione: stessa garanzia gia' verificata per l'architettura
       esistente, vedi note di conversazione);
    2. ricostruisce la board ply-per-ply dal FEN iniziale della finestra
       (un solo replay leggero, nessuna nuova chiamata Stockfish: la
       verifica di matto e la classificazione sono gia' state fatte nel
       worker, qui serve solo la board PRIMA di ogni mossa per costruire
       il grafo spaziale);
    3. chiama PositionGraphSchema.build_position_data per ogni ply;
    4. chiama PositionQueueRegistry.enqueue per ogni posizione risultante,
       con group_key=mate_n (comune a tutta la finestra).

FIX DI ROBUSTEZZA APPLICATI (ereditati dall'analisi precedente, invariati
nella sostanza):

    (1) Nessun `except Exception: pass` silenzioso: ogni eccezione nel
        worker viene contata (Counter per tipo) e ritornata, run() logga
        un riepilogo aggregato a fine pipeline.
    (2) _validate_kings: controllo esplicito di presenza di entrambi i Re
        prima di calcolare materiale/mate-eligibility.
    (3) Parsing PGN/clock/rating condiviso in chess_replay_utils.py (non
        piu' duplicato).
    (4) _analyse_position ritenta (stockfish_retry_attempts, default 2) su
        errori plausibilmente transitori prima di scartare la posizione
        candidata.
"""
from __future__ import annotations

import atexit
import bz2
import io
import logging
import os
import signal
import threading
import time
import zipfile
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import pandas as pd
import zstandard as zstd
from tqdm import tqdm

from PositionGraphSchema import build_position_data
from PositionQueue import PositionQueueRegistry
from chess_replay_utils import (
    closest_bucket_time,
    compute_move_duration,
    parse_clk,
    parse_emt,
    parse_rating,
    parse_time_control,
)

logger = logging.getLogger("games_builder")

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
    max_games: limite superiore di partite lette da questa sorgente (None =
          nessun limite).
    tag: etichetta leggibile passata a PositionQueueRegistry.enqueue come
          source_tag (solo metadato/statistiche).
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


@dataclass
class _RawPlyRecord:
    """Un ply grezzo (tipi Python nativi, nessun tensore) ritornato dal
    worker per una finestra accettata. Il processo principale lo trasforma
    in un torch_geometric.data.Data via PositionGraphSchema.build_position_data.
    """
    fen_before_move: str    # FEN della posizione PRIMA di questa mossa
    move_uci: str            # mossa REALMENTE giocata in questa posizione
    clock_seconds: float     # tempo (s) impiegato per questa mossa
    ply: int                 # indice del ply all'interno della finestra (0-based)


@dataclass
class _WindowResult:
    """Risultato del worker per una partita: una finestra di matto forzato
    (se trovata) come lista di _RawPlyRecord, oppure None."""
    plies: Optional[List[_RawPlyRecord]]
    mate_n: Optional[int]
    source_tag: str
    error_counts: Counter = field(default_factory=Counter)


# ============================================================================
# GAMES BUILDER
# ============================================================================

class GamesBuilder:
    """
    Pipeline unificata multi-sorgente per il train set "games":

        PGN (Lichess .zst / FICS .bz2 / Club CSV)
          |
          v
        streaming per-partita, source-agnostic
          |
          v
        filtri economici partita
          |
          v
        ricerca del PRIMO candidato "mate in N" (Stockfish, mate_range)
          |
          v
        replay dei ply REALMENTE GIOCATI per 2N-1 semi-mosse (nel worker,
        SENZA costruire tensori: solo fen/uci/clock/ply nativi Python)
          |
          v
        verifica stretta: is_checkmate() sull'ultimo ply?
          | si                                    | no
          v                                        v
        List[_RawPlyRecord] ritornata        finestra scartata,
        al processo principale                partita successiva
          |
          v
        (PROCESSO PRINCIPALE) alloca un game_id per la finestra, ricostruisce
        la board per ogni ply, chiama build_position_data + enqueue su
        PositionQueueRegistry con group_key=mate_n
          |
          v
        (a fine raccolta) PositionQueueRegistry.build_splits(...)
          |
          v
        {"train": [Data...], "val": [...], "test": [...]}  (Data a grana
        di SINGOLA POSIZIONE, uno per ply di ogni finestra accettata)
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
        mate_range: Tuple[int, int] = (1, 10),
        search_depth: int = 8,
        analysis_time: Optional[float] = 0.2,
        workers: Optional[int] = None,
        threads: int = 1,
        hash_mb: int = 128,
        multipv: int = 1,
        default_move_seconds: float = 15.0,
        avg_time_by_rating: Optional[Dict[int, float]] = None,
        require_clock: bool = False,
        min_ply: int = 8,
        min_game_plies: int = 20,
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
        checkpoint_log_every: int = 5000,
        config_error_cls: type = ValueError,
        min_rating: Optional[int] = 1200,
        max_rating: Optional[int] = None,
        min_material_diff_for_mate_attempt: int = 3,
        require_heavy_piece: bool = True,
        skip_trivial_endgame: bool = True,
        stockfish_retry_attempts: int = 2,
        stockfish_retry_backoff_seconds: float = 0.5,
        queue_state_path: Optional[str] = None,
    ):
        """
        Args (differenze rispetto alla versione a grafo-sequenza):
            Rimossi: seed (non serve piu' qui: lo split deterministico e'
                interamente responsabilita' di PositionQueueRegistry.build_splits,
                che accetta il proprio seed), dedupe_positions (la ricerca
                della finestra non deduplica piu' posizioni intermedie: la
                ricerca si ferma alla prima finestra valida, non scansiona
                oltre).
            queue_state_path: percorso del file di stato per
                PositionQueueRegistry (contatore diagnostico). Se None,
                usa il default della classe (vedi PositionQueue.py).
        """
        self.sources = sources
        self.stockfish_path = stockfish_path
        self.mate_range = mate_range
        self.search_depth = search_depth
        self.analysis_time = analysis_time
        self.threads = threads
        self.hash_mb = hash_mb
        self.multipv = multipv
        self.default_move_seconds = default_move_seconds
        self.avg_time_by_rating = avg_time_by_rating or {}
        self.require_clock = require_clock
        self.min_ply = max(0, min_ply)
        self.min_game_plies = min_game_plies
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
        self.checkpoint_log_every = checkpoint_log_every
        self._config_error_cls = config_error_cls

        self.min_rating = min_rating
        self.max_rating = max_rating
        self.min_material_diff_for_mate_attempt = min_material_diff_for_mate_attempt
        self.require_heavy_piece = require_heavy_piece
        self.skip_trivial_endgame = skip_trivial_endgame
        self.stockfish_retry_attempts = max(1, stockfish_retry_attempts)
        self.stockfish_retry_backoff_seconds = stockfish_retry_backoff_seconds

        cpu_count = os.cpu_count() or 2
        self.workers = workers or max(1, cpu_count - 1)

        self._queue_registry = PositionQueueRegistry.instance(state_path=queue_state_path)
        self._next_game_id = 0  # contatore locale, assegnato SOLO nel processo principale

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
    # WORKER INIT (Stockfish + Syzygy + watchdog + signal handling)
    # ================================================================

    @staticmethod
    def _init_worker(stockfish_path: str, threads: int, hash_mb: int, syzygy_path: Optional[str]) -> None:
        global _engine, _engine_pid, _tablebase, _watchdog_thread

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, _worker_sigterm_handler)

        try:
            os.setpgrp()
        except AttributeError:
            pass  # Windows: non applicabile, il sistema e' pensato per Linux

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
            white_elo = parse_rating(game.headers.get("WhiteElo", ""))
            black_elo = parse_rating(game.headers.get("BlackElo", ""))
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
        return legal_moves

    @staticmethod
    def _validate_kings(board: "chess.Board") -> bool:
        """FIX ROBUSTEZZA (2): verifica esplicita che entrambi i Re siano
        presenti prima di calcolare materiale/mate-eligibility."""
        return (
            board.king(chess.WHITE) is not None
            and board.king(chess.BLACK) is not None
        )

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
        if not self._validate_kings(board):
            logger.warning("Board senza uno o entrambi i Re incontrata durante il replay: scarto la posizione.")
            return False

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
    # STOCKFISH ANALYSIS (con retry, fix di robustezza #4)
    # ================================================================

    _NON_RETRYABLE_ENGINE_ERRORS = (
        chess.engine.EngineTerminatedError,
        BrokenPipeError,
        ConnectionResetError,
    )

    def _analyse_position(self, board: "chess.Board") -> Tuple[Optional[Any], Counter]:
        """Analizza una posizione con Stockfish, con retry su errori
        plausibilmente transitori. Vedi docstring di modulo, fix (4)."""
        global _engine
        error_counts: Counter = Counter()

        for attempt in range(1, self.stockfish_retry_attempts + 1):
            if _engine is None:
                error_counts["engine_unavailable"] += 1
                return None, error_counts
            try:
                if self.analysis_time is not None:
                    limit = chess.engine.Limit(time=self.analysis_time, mate=self.mate_range[1])
                    _watchdog_arm(self.analysis_time)
                else:
                    limit = chess.engine.Limit(depth=self.search_depth, mate=self.mate_range[1])
                    _watchdog_arm(5.0)
                result = _engine.analyse(board, limit, multipv=self.multipv)
                return result, error_counts
            except self._NON_RETRYABLE_ENGINE_ERRORS as e:
                error_counts[type(e).__name__] += 1
                return None, error_counts
            except chess.engine.EngineError as e:
                error_counts[type(e).__name__] += 1
                if attempt < self.stockfish_retry_attempts:
                    time.sleep(self.stockfish_retry_backoff_seconds)
                    continue
                return None, error_counts
            except Exception as e:
                error_counts[type(e).__name__] += 1
                if attempt < self.stockfish_retry_attempts:
                    time.sleep(self.stockfish_retry_backoff_seconds)
                    continue
                return None, error_counts
            finally:
                _watchdog_disarm()

        return None, error_counts

    # ================================================================
    # RICERCA FINESTRA DI MATTO FORZATO + REPLAY (worker, no tensori)
    # ================================================================

    def _find_mate_window_start(
        self, game: "chess.pgn.Game"
    ) -> Tuple[Optional["chess.pgn.GameNode"], Optional[int], Counter]:
        """Scansiona la partita ply-per-ply cercando il PRIMO candidato con
        mate_n valido nel mate_range. Vedi docstring di modulo."""
        error_counts: Counter = Counter()
        node = game

        while node.variations:
            next_node = node.variation(0)
            board = node.board()

            if node.ply() < self.min_ply:
                node = next_node
                continue

            legal_moves = self._get_candidate_legal_moves(board)
            if legal_moves is None:
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

            if self._syzygy_says_no_mate(board):
                node = next_node
                continue

            info, position_errors = self._analyse_position(board)
            error_counts.update(position_errors)
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

            mate_lo, mate_hi = self.mate_range
            if mate_n is None or not (mate_n > 0 and mate_lo <= mate_n <= mate_hi):
                node = next_node
                continue

            return node, int(mate_n), error_counts

        return None, None, error_counts

    def _replay_mate_window(
        self,
        start_node: "chess.pgn.GameNode",
        mate_n: int,
    ) -> Tuple[Optional[List[_RawPlyRecord]], Counter]:
        """Segue i ply REALMENTE GIOCATI nel PGN a partire da start_node,
        per esattamente 2*mate_n-1 semi-mosse, ritornando SOLO dati nativi
        Python (fen/uci/clock/ply) per ciascuna posizione attraversata: la
        costruzione del grafo (build_position_data) avviene nel processo
        principale, non qui (vedi docstring di modulo).

        Verifica STRETTA di accettazione: la finestra e' valida solo se,
        dopo esattamente 2*mate_n-1 ply reali, board.is_checkmate() e' vero.

        Returns:
            (raw_plies, error_counts): raw_plies e' None se la finestra non
            supera la verifica stretta.
        """
        error_counts: Counter = Counter()
        target_plies = 2 * mate_n - 1

        node = start_node
        board = node.board()

        game_root = node.game()
        tc_raw = game_root.headers.get("TimeControl", "")
        base_time, increment = parse_time_control(tc_raw)
        previous_clock: Dict[bool, Optional[float]] = {
            chess.WHITE: base_time if base_time > 0 else None,
            chess.BLACK: base_time if base_time > 0 else None,
        }

        white_elo = parse_rating(game_root.headers.get("WhiteElo", ""))
        black_elo = parse_rating(game_root.headers.get("BlackElo", ""))
        mover_rating = {chess.WHITE: white_elo, chess.BLACK: black_elo}

        raw_plies: List[_RawPlyRecord] = []

        for step in range(target_plies):
            if not node.variations:
                return None, error_counts

            next_node = node.variation(0)
            move = next_node.move
            comment = next_node.comment or ""
            mover_color = board.turn

            if move not in board.legal_moves:
                error_counts["illegal_move_in_pgn"] += 1
                return None, error_counts

            emt_seconds = parse_emt(comment)
            current_clock = parse_clk(comment)

            if emt_seconds is not None:
                move_duration = emt_seconds
                duration_is_real = True
            else:
                move_duration = compute_move_duration(previous_clock[mover_color], current_clock, increment)
                duration_is_real = move_duration is not None

            if current_clock is not None:
                previous_clock[mover_color] = current_clock

            if self.require_clock and not duration_is_real:
                return None, error_counts

            if duration_is_real:
                clock_seconds = move_duration
            else:
                bucket_time = closest_bucket_time(mover_rating[mover_color], self.avg_time_by_rating)
                clock_seconds = bucket_time if bucket_time is not None else self.default_move_seconds

            if self.drop_zero_clock and clock_seconds == 0.0 and not duration_is_real:
                return None, error_counts

            raw_plies.append(
                _RawPlyRecord(
                    fen_before_move=board.fen(),
                    move_uci=move.uci(),
                    clock_seconds=float(clock_seconds),
                    ply=step,
                )
            )

            board.push(move)
            node = next_node

        if not board.is_checkmate():
            return None, error_counts

        return raw_plies, error_counts

    def _worker(self, args: Tuple[int, str, str]) -> Tuple[int, _WindowResult]:
        """args = (task_local_id, pgn_text, source_tag).

        Ritorna (task_local_id, _WindowResult). Nessun tensore, nessun
        game_id: solo dati nativi Python (vedi docstring di modulo)."""
        global _engine
        task_local_id, pgn_text, source_tag = args
        error_counts: Counter = Counter()

        if _engine is None:
            error_counts["engine_unavailable"] += 1
            return task_local_id, _WindowResult(None, None, source_tag, error_counts)

        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text))
        except Exception as e:
            error_counts[f"pgn_parse:{type(e).__name__}"] += 1
            return task_local_id, _WindowResult(None, None, source_tag, error_counts)

        if game is None:
            return task_local_id, _WindowResult(None, None, source_tag, error_counts)

        if game.headers.get("Variant", "Standard").lower() not in ("standard", "normal"):
            return task_local_id, _WindowResult(None, None, source_tag, error_counts)

        if not self._game_is_eligible(game):
            return task_local_id, _WindowResult(None, None, source_tag, error_counts)

        try:
            start_node, mate_n, search_errors = self._find_mate_window_start(game)
            error_counts.update(search_errors)

            if start_node is None:
                return task_local_id, _WindowResult(None, None, source_tag, error_counts)

            raw_plies, replay_errors = self._replay_mate_window(start_node, mate_n)
            error_counts.update(replay_errors)

            return task_local_id, _WindowResult(raw_plies, mate_n, source_tag, error_counts)

        except Exception as e:
            error_counts[f"unexpected:{type(e).__name__}"] += 1
            logger.warning(
                f"Errore inatteso nel worker per task_local_id={task_local_id} "
                f"(source={source_tag}): {type(e).__name__}: {e}"
            )
            return task_local_id, _WindowResult(None, None, source_tag, error_counts)

    # ================================================================
    # PROCESSO PRINCIPALE: da _RawPlyRecord a Data + enqueue
    # ================================================================

    def _enqueue_window(self, window: _WindowResult) -> int:
        """Assegna un game_id alla finestra, ricostruisce ogni posizione
        (build_position_data) ed effettua l'enqueue su
        PositionQueueRegistry. Chiamato SOLO dal processo principale.

        Returns:
            Il numero di posizioni accodate con successo (puo' essere
            inferiore al numero di raw_plies se una singola posizione
            fallisce la costruzione del grafo, vedi gestione errori sotto).
        """
        game_id = self._next_game_id
        self._next_game_id += 1

        enqueued = 0
        for raw_ply in window.plies:
            try:
                board = chess.Board(raw_ply.fen_before_move)
                move = chess.Move.from_uci(raw_ply.move_uci)
                data = build_position_data(
                    board=board,
                    best_move=move,
                    clock_seconds=raw_ply.clock_seconds,
                    game_id=game_id,
                    ply=raw_ply.ply,
                )
            except ValueError as e:
                logger.warning(
                    f"Scarto una posizione della finestra game_id={game_id} "
                    f"(ply={raw_ply.ply}): {e}"
                )
                continue

            self._queue_registry.enqueue(
                source_tag=window.source_tag,
                data=data,
                group_key=window.mate_n,
            )
            enqueued += 1

        return enqueued

    # ================================================================
    # STREAMING SORGENTI (source-agnostic)
    # ================================================================

    def _open_pgn_text_stream(self, path: str, kind: str):
        if kind == "lichess":
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
    # RUN
    # ================================================================

    def run(self) -> Dict[str, Any]:
        """Esegue la pipeline: scansiona tutte le sorgenti (worker Pool),
        per ogni finestra accettata assegna un game_id, costruisce e
        accoda le sue posizioni (nel processo principale). Ritorna solo
        statistiche: gli split finali vanno ottenuti chiamando
        self._queue_registry.build_splits(...) separatamente (una sola
        volta, quando TUTTE le sorgenti/builder che condividono la stessa
        coda -- inclusi eventuali puzzle -- hanno finito di accodare).

        Returns:
            Dict con "processed_games", "accepted_windows",
            "enqueued_positions", "error_counts", "mate_n_counts".
        """
        pool = mp.Pool(
            processes=self.workers,
            initializer=self._init_worker,
            initargs=(self.stockfish_path, self.threads, self.hash_mb, self.syzygy_path),
        )

        processed_games = 0
        accepted_windows = 0
        enqueued_positions = 0
        mate_n_counts: Dict[int, int] = defaultdict(int)
        aggregated_errors: Counter = Counter()
        estimate = self._count_tasks_estimate()

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
                    logger.warning(
                        f"{len(still_alive)} worker non terminati entro {timeout}s. "
                        f"Invio SIGKILL al gruppo processi..."
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

            pbar = tqdm(results, desc="Ricerca finestre matto forzato", total=estimate, dynamic_ncols=True)
            for task_local_id, window in pbar:
                processed_games += 1
                aggregated_errors.update(window.error_counts)

                if window.plies is None:
                    continue

                n_enqueued = self._enqueue_window(window)
                if n_enqueued == 0:
                    continue

                accepted_windows += 1
                enqueued_positions += n_enqueued
                mate_n_counts[window.mate_n] += 1

                if accepted_windows % self.checkpoint_log_every == 0:
                    logger.info(
                        f"[checkpoint] {processed_games} partite processate, "
                        f"{accepted_windows} finestre accettate, "
                        f"{enqueued_positions} posizioni accodate, "
                        f"{self._queue_registry.pending_count()} in coda."
                    )
            pbar.close()
        except KeyboardInterrupt:
            logger.warning("Interruzione richiesta: arresto pulito dei worker in corso...")
            cleanup_after_failure()
            raise
        except Exception:
            logger.warning("Errore durante l'analisi: arresto pulito dei worker in corso...")
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
                    logger.warning(
                        f"{len(still_alive)} worker non terminati entro "
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
                        logger.warning("'psutil' non disponibile: eventuali processi Stockfish orfani vanno terminati manualmente.")

            pool.join()

        self._log_summary(processed_games, accepted_windows, enqueued_positions, mate_n_counts, aggregated_errors)

        return {
            "processed_games": processed_games,
            "accepted_windows": accepted_windows,
            "enqueued_positions": enqueued_positions,
            "error_counts": dict(aggregated_errors),
            "mate_n_counts": dict(mate_n_counts),
        }

    def _log_summary(
        self,
        processed_games: int,
        accepted_windows: int,
        enqueued_positions: int,
        mate_n_counts: Dict[int, int],
        aggregated_errors: Counter,
    ) -> None:
        logger.info("=" * 60)
        logger.info("GAMES BUILDER — RIEPILOGO (schema board-level per posizione)")
        logger.info("=" * 60)
        logger.info(f"Partite processate: {processed_games:,}")
        logger.info(f"Finestre di matto forzato accettate: {accepted_windows:,}")
        logger.info(f"Posizioni accodate: {enqueued_positions:,}")

        if accepted_windows > 0:
            logger.info("Per profondita' mate (N, mosse intere):")
            for n in sorted(mate_n_counts.keys()):
                logger.info(f"  n={n}: {mate_n_counts[n]:,}")

        if aggregated_errors:
            logger.info("Errori/scarti aggregati durante l'analisi (fix di robustezza #1):")
            for err_type, count in aggregated_errors.most_common():
                logger.info(f"  {err_type}: {count:,}")
        logger.info("=" * 60)