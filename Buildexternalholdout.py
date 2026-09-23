from __future__ import annotations

import atexit
import io
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import pandas as pd

from DatasetPipeline.Model.ChessConstants import PIECE_VALUES
from DatasetPipeline.Model.PositionGraphSchema import build_position_data
from DatasetPipeline.PipelineState import PipelineState
from DatasetPipeline.TimeStatBuilder import load_avg_time_by_rating
from DatasetPipeline.Utils.compatibility_filters import (
    QualityFilterConfig,
    has_mating_material,
    has_valid_ratings,
    is_trivially_drawn_endgame,
    is_window_hard_valid,
    mover_has_heavy_piece,
    parse_rating_strict,
    validate_kings,
)
from DatasetPipeline.Utils.ipc_safe_data import (
    decode_from_ipc,
    encode_for_ipc,
    harden_process_for_ipc,
)
from DatasetPipeline.Utils.time_edge_weighting import apply_edge_type_time_weighting
from TrainPipeline.CleanDataset import clean_sharded_directory

logger = logging.getLogger("build_external_holdout")

_CLK_RE = re.compile(r"\[\s*%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]")

# ----------------------------------------------------------------------
# Stato globale per-worker (stesso pattern di GamesBuilder: un engine per
# processo, avviato dall'initializer del Pool, con watchdog dedicato per
# uccidere Stockfish se una singola analisi si impianta).
# ----------------------------------------------------------------------
_engine: Optional[chess.engine.SimpleEngine] = None
_engine_pid: Optional[int] = None

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
    global _engine, _engine_pid
    _watchdog_stop.set()
    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            pass
        finally:
            _engine = None
            _engine_pid = None


def _worker_sigterm_handler(signum, frame) -> None:
    _watchdog_stop.set()
    pid = _engine_pid
    if pid is not None:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    os._exit(0)


@dataclass
class ExternalHoldoutConfig:
    csv_path: str = "RawData/ficsgamesdb_201001_standard_movetimes_4340144.pgn"
    source_tag: str = "heldout_00_fcis"

    input_format: str = "auto"

    chunksize: int = 5_000
    max_games: Optional[int] = 100_000

    allowed_rules: Tuple[str, ...] = ("chess",)
    require_rated: bool = False

    only_decisive_games: bool = True
    skip_time_forfeit: bool = True
    min_game_plies: int = 20

    stockfish_path: str = "/usr/games/stockfish"
    threads: int = 1
    hash_mb: int = 32
    search_depth: int = 12
    analysis_time: Optional[float] = 0.2
    mate_range: Tuple[int, int] = (1, 3)

    stockfish_retry_attempts: int = 2
    stockfish_retry_backoff_seconds: float = 0.5

    skip_forced_single_move_window: bool = True
    require_heavy_piece: bool = False
    skip_trivial_endgame: bool = True
    # ALLINEATO: dataset_main.yaml -> games_pipeline.min_material_for_mate_attempt = 3
    min_material_for_mate_attempt: int = 3
    min_material_diff_for_mate_attempt: int = 3
    max_piece_count: Optional[int] = None
    candidate_min_legal_moves: int = 1
    candidate_max_legal_moves: Optional[int] = None
    skip_if_in_check: bool = False

    min_ply: int = 6
    # ALLINEATO: dataset_main.yaml -> games_pipeline.ply_sample_step = 12
    # (era 8: l'holdout campionava piu' densamente del main, cambiando la
    # distribuzione delle posizioni per partita rispetto al training).
    ply_sample_step: int = 12
    # ALLINEATO: dataset_main.yaml -> games_pipeline.max_positions_per_game = 20
    max_positions_per_game: Optional[int] = 20
    dedupe_positions: bool = True

    # NUOVO: dataset_main.yaml non imposta esplicitamente min_rating in
    # games_pipeline, quindi usa il default 1200 di GamesBuilderConfig.
    # L'holdout prima non aveva alcuna soglia minima di rating (solo
    # has_valid_ratings, che verifica presenza/validita', non la soglia):
    # partite con rating < 1200 potevano quindi entrare nell'holdout ma
    # mai nel train/val, causando distribution shift. Applicato come
    # filtro soft in _headers_are_eligible.
    min_rating: Optional[int] = 1200
    max_rating: Optional[int] = None

    time_stats_json: Optional[str] = "Dataset/avg_time_by_rating.json"
    default_move_seconds: float = 15.0

    output_dir: str = "Dataset/ExternalHoldout"
    output_shard_size: int = 20_000
    target_clean_shard_size: int = 2_000
    save_debug_jsonl: bool = True

    state_file: str = "external_holdout_state.json"
    resume_state_file: str = "external_holdout_resume.json"
    force_recompute: bool = False

    # --- Multiprocessing (allineato a GamesBuilderConfig) ---
    workers: Optional[int] = None
    pool_join_timeout: Optional[float] = 20.0
    auto_resume: bool = True
    resume_checkpoint_every: int = 500
    skip_games: int = 0

    log_level: str = "INFO"
    log_file: Optional[str] = "Dataset/ExternalHoldout/build_external_holdout.log"


CONFIG = ExternalHoldoutConfig()


class ConfigError(Exception):
    pass


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def require_executable(path: str) -> None:
    if not (os.path.exists(path) and os.access(path, os.X_OK)):
        raise ConfigError(f"Eseguibile non trovato o non eseguibile: {path}")


def free_memory() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class _ShardWriter:
    def __init__(self, out_dir: str, split_name: str, shard_size: int) -> None:
        self.split_dir = os.path.join(out_dir, split_name)
        os.makedirs(self.split_dir, exist_ok=True)
        self.split_name = split_name
        self.shard_size = max(1, int(shard_size))
        self.buf: List[Any] = []
        self.shard_idx = 0
        self.files: List[str] = []
        self.total = 0

    def append(self, record: Any) -> None:
        self.buf.append(record)
        self.total += 1
        if len(self.buf) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        import torch
        path = os.path.join(self.split_dir, f"shard_{self.shard_idx:05d}.pt")
        tmp_path = path + ".tmp"
        torch.save(self.buf, tmp_path)
        os.replace(tmp_path, path)
        self.files.append(path)
        self.buf = []
        self.shard_idx += 1

    def close(self) -> List[str]:
        self._flush()
        manifest = os.path.join(self.split_dir, "manifest.json")
        tmp_manifest = manifest + ".tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(
                {"num_shards": self.shard_idx, "shard_size": self.shard_size, "total": self.total},
                f, indent=2,
            )
        os.replace(tmp_manifest, manifest)
        return self.files


class ChesscomHoldoutBuilder:
    _PIECE_VALUES: Dict[int, int] = PIECE_VALUES

    def __init__(self, config: ExternalHoldoutConfig) -> None:
        self.config = config
        self._validate_config()

        os.makedirs(config.output_dir, exist_ok=True)

        self._avg_time_by_rating: Dict[int, float] = {}
        if config.time_stats_json and os.path.exists(config.time_stats_json):
            self._avg_time_by_rating = load_avg_time_by_rating(config.time_stats_json)
            logger.info(
                f"[holdout] avg_time_by_rating caricato da '{config.time_stats_json}': "
                f"{len(self._avg_time_by_rating)} bucket."
            )
        else:
            logger.warning(
                f"[holdout] time_stats_json='{config.time_stats_json}' assente: uso "
                f"default_move_seconds={config.default_move_seconds}s costante."
            )

        self._debug_records: List[Dict[str, Any]] = []
        self._debug_jsonl_path = (
            os.path.join(config.output_dir, "holdout_debug.jsonl")
            if config.save_debug_jsonl
            else None
        )

        self._quality_cfg = QualityFilterConfig(
            skip_forced_single_move_window=config.skip_forced_single_move_window,
            require_heavy_piece=config.require_heavy_piece,
            skip_trivial_endgame=config.skip_trivial_endgame,
            min_material_for_mate_attempt=config.min_material_for_mate_attempt,
            min_material_diff_for_mate_attempt=config.min_material_diff_for_mate_attempt,
            max_piece_count=config.max_piece_count,
            candidate_min_legal_moves=config.candidate_min_legal_moves,
            candidate_max_legal_moves=config.candidate_max_legal_moves,
            skip_if_in_check=config.skip_if_in_check,
        )

        self._input_format = self._detect_input_format()
        logger.info(f"[holdout] Formato input rilevato: '{self._input_format}' (da '{self.config.csv_path}').")

        # --- Resume state (stesso schema di GamesBuilder: contatore di
        # partite gia' processate con successo per questa sorgente,
        # persistito su disco e sommato a skip_games ad ogni riavvio). ---
        self._resume_state_path = os.path.join(config.output_dir, config.resume_state_file)
        self._resume_key = f"{self._input_format}:{config.csv_path}"
        self._resume_confirmed = 0
        self._effective_skip_games = config.skip_games
        if config.auto_resume:
            saved = self._load_resume_state()
            already_done = saved.get(self._resume_key, 0)
            if already_done:
                self._effective_skip_games += already_done
                logger.info(
                    f"[holdout] Resume attivo per '{self._resume_key}': skip_games portato a "
                    f"{self._effective_skip_games} ({already_done} gia' processate in run precedenti)."
                )

    def _validate_config(self) -> None:
        cfg = self.config
        if not os.path.exists(cfg.csv_path):
            raise ConfigError(f"File di input non trovato: '{cfg.csv_path}'")
        if cfg.mate_range[0] < 1 or cfg.mate_range[1] < cfg.mate_range[0]:
            raise ConfigError(f"mate_range non valido: {cfg.mate_range}")
        if cfg.input_format not in ("auto", "csv", "pgn"):
            raise ConfigError(
                f"input_format non valido: '{cfg.input_format}'. Attesi: 'auto', 'csv', 'pgn'."
            )
        if cfg.min_rating is not None and cfg.max_rating is not None and cfg.min_rating > cfg.max_rating:
            raise ConfigError("min_rating non puo' essere maggiore di max_rating.")
        require_executable(cfg.stockfish_path)

    def _detect_input_format(self) -> str:
        """Determina se csv_path va letto come CSV (stile Kaggle
        Chess.com, colonna 'pgn' per riga) o come file .pgn testuale
        (partite concatenate, header '[Event ...]').

        Se config.input_format e' 'csv' o 'pgn', quella scelta e'
        rispettata senza ispezionare il file. Con 'auto' (default), il
        rilevamento guarda il CONTENUTO, non l'estensione.
        """
        cfg = self.config
        if cfg.input_format in ("csv", "pgn"):
            return cfg.input_format

        try:
            with open(cfg.csv_path, "r", encoding="utf-8", errors="replace") as f:
                for _ in range(50):
                    line = f.readline()
                    if not line:
                        break
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped.startswith("[Event "):
                        return "pgn"
                    break
        except OSError as e:
            raise ConfigError(f"Impossibile leggere '{cfg.csv_path}' per rilevare il formato: {e}")

        return "csv"

    # ------------------------------------------------------------------
    # Resume state (persistenza su disco, stesso pattern di GamesBuilder)
    # ------------------------------------------------------------------
    def _load_resume_state(self) -> Dict[str, int]:
        if not os.path.exists(self._resume_state_path):
            return {}
        try:
            with open(self._resume_state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {str(k): int(v) for k, v in raw.items()}
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            logger.warning(
                f"[holdout] Stato di resume in {self._resume_state_path} illeggibile ({e}), "
                f"riparto senza resume."
            )
            return {}

    def _persist_resume_state(self) -> None:
        try:
            merged = self._load_resume_state()
            merged[self._resume_key] = self.config.skip_games + self._resume_confirmed
            os.makedirs(os.path.dirname(os.path.abspath(self._resume_state_path)) or ".", exist_ok=True)
            tmp_path = self._resume_state_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._resume_state_path)
        except Exception as e:
            logger.warning(f"[holdout] Impossibile salvare lo stato di resume: {e}")

    # ------------------------------------------------------------------
    # Worker init/teardown (un processo Stockfish per worker, con watchdog)
    # ------------------------------------------------------------------
    @staticmethod
    def _init_worker(stockfish_path: str, threads: int, hash_mb: int) -> None:
        global _engine, _engine_pid, _watchdog_thread

        harden_process_for_ipc()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, _worker_sigterm_handler)

        try:
            os.setpgrp()
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

        atexit.register(_close_engine)

        _watchdog_stop.clear()
        _watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
        _watchdog_thread.start()

    # ------------------------------------------------------------------
    # Parsing header / clock (stesso comportamento di GamesBuilder)
    # ------------------------------------------------------------------
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
    def _parse_rating(raw: str) -> Optional[int]:
        if not raw:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            digits = "".join(ch for ch in raw if ch.isdigit())
            return int(digits) if digits else None

    def _headers_are_eligible(self, headers, pgn_result: Optional[str] = None) -> bool:
        """HARD/SOFT a livello di header, allineato a
        GamesBuilder._headers_are_eligible: only_decisive_games,
        skip_time_forfeit e ora anche min_rating/max_rating, applicati
        PRIMA di qualunque replay/analisi.

        ALLINEATO: il filtro di rating (min_rating/max_rating) mancava
        completamente in precedenza; GamesBuilder._headers_are_eligible
        lo applica sul MASSIMO tra i due rating (best_rating), qui viene
        replicata la stessa logica per coerenza tra le due pipeline.
        """
        cfg = self.config

        if cfg.only_decisive_games:
            result = (headers.get("Result", "") if headers is not None else pgn_result) or ""
            if result not in ("1-0", "0-1"):
                return False

        if cfg.skip_time_forfeit and headers is not None:
            termination = headers.get("Termination", "") or ""
            if "Time forfeit" in termination:
                return False

        if headers is not None and (cfg.min_rating is not None or cfg.max_rating is not None):
            white_elo = self._parse_rating(headers.get("WhiteElo", ""))
            black_elo = self._parse_rating(headers.get("BlackElo", ""))
            ratings = [r for r in (white_elo, black_elo) if r is not None]
            if ratings:
                best_rating = max(ratings)
                worst_rating = min(ratings)
                if cfg.min_rating is not None and best_rating < cfg.min_rating:
                    return False
                if cfg.max_rating is not None and worst_rating > cfg.max_rating:
                    return False

        return True

    # ------------------------------------------------------------------
    # Filtri di posizione (soft, invariati rispetto alla versione precedente)
    # ------------------------------------------------------------------
    def _simulated_clock(self, rating: Optional[float], ply_idx: int, game_id: str) -> float:
        base = self.config.default_move_seconds
        if rating is not None and self._avg_time_by_rating:
            bucket = round(rating / 100) * 100
            base = self._avg_time_by_rating.get(bucket, base)
        noise = random.Random(f"{game_id}:{ply_idx}").gauss(1.0, 0.15)
        return max(0.5, base * max(0.3, noise))

    def _position_passes_quality_filters(self, board: "chess.Board") -> bool:
        cfg = self._quality_cfg
        if cfg.max_piece_count is not None and len(board.piece_map()) > cfg.max_piece_count:
            return False
        if not has_mating_material(board, cfg):
            return False
        if cfg.require_heavy_piece and not mover_has_heavy_piece(board):
            return False
        if cfg.skip_trivial_endgame and is_trivially_drawn_endgame(board):
            return False
        return True

    def _get_candidate_legal_moves(self, board: "chess.Board") -> Optional[List["chess.Move"]]:
        cfg = self._quality_cfg
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            return None
        legal_moves = list(board.legal_moves)
        if len(legal_moves) < cfg.candidate_min_legal_moves:
            return None
        if cfg.candidate_max_legal_moves is not None and len(legal_moves) > cfg.candidate_max_legal_moves:
            return None
        if cfg.skip_if_in_check and board.is_check():
            return None
        return legal_moves

    def _analyse_position(self, board: "chess.Board"):
        """Analisi con retry e watchdog, allineata a
        GamesBuilder._analyse_position (stessi parametri di retry/backoff)."""
        global _engine
        cfg = self.config

        for attempt in range(1, cfg.stockfish_retry_attempts + 1):
            if _engine is None:
                return None
            try:
                if cfg.analysis_time is not None:
                    limit = chess.engine.Limit(time=cfg.analysis_time, mate=cfg.mate_range[1])
                    _watchdog_arm(cfg.analysis_time)
                else:
                    limit = chess.engine.Limit(depth=cfg.search_depth, mate=cfg.mate_range[1])
                    _watchdog_arm(5.0)
                return _engine.analyse(board, limit)
            except Exception:
                if attempt >= cfg.stockfish_retry_attempts:
                    return None
                if cfg.stockfish_retry_backoff_seconds > 0:
                    time.sleep(cfg.stockfish_retry_backoff_seconds * attempt)
                continue
            finally:
                _watchdog_disarm()
        return None

    # ------------------------------------------------------------------
    # Iterazione input (CSV Kaggle o .pgn concatenato), con skip_games
    # ------------------------------------------------------------------
    def _iter_rows(self, skip_games: int, max_games: Optional[int]):
        if self._input_format == "pgn":
            yield from self._iter_rows_from_pgn(skip_games, max_games)
        else:
            yield from self._iter_rows_from_csv(skip_games, max_games)

    def _iter_rows_from_csv(self, skip_games: int, max_games: Optional[int]):
        reader = pd.read_csv(self.config.csv_path, chunksize=self.config.chunksize)
        local_id = 0
        yielded = 0
        for chunk in reader:
            if "rules" in chunk.columns:
                chunk = chunk[chunk["rules"].astype(str).str.lower().isin(self.config.allowed_rules)]
            if self.config.require_rated and "rated" in chunk.columns:
                chunk = chunk[chunk["rated"].astype(str).str.lower() == "true"]
            for record in chunk.to_dict("records"):
                local_id += 1
                if local_id <= skip_games:
                    continue
                yield local_id, record
                yielded += 1
                if max_games is not None and yielded >= max_games:
                    return

    def _iter_rows_from_pgn(self, skip_games: int, max_games: Optional[int]):
        """Stessa logica di split di GamesBuilder._iter_pgn_texts: spezza
        sul marcatore '[Event ' che apre ogni nuova partita."""
        cfg = self.config
        if cfg.require_rated:
            logger.warning(
                "[holdout] require_rated=True non e' applicabile a input .pgn "
                "(nessuna colonna 'rated' negli header PGN): ignorato."
            )
        if cfg.allowed_rules and cfg.allowed_rules != ("chess",):
            logger.warning(
                f"[holdout] allowed_rules={cfg.allowed_rules} non e' applicabile a input .pgn: ignorato."
            )

        local_id = 0
        yielded = 0
        current_game_lines: List[str] = []

        def _flush_current():
            if not current_game_lines:
                return None
            return {"pgn": "".join(current_game_lines)}

        with open(cfg.csv_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("[Event ") and current_game_lines:
                    record = _flush_current()
                    if record is not None:
                        local_id += 1
                        if local_id > skip_games:
                            yield local_id, record
                            yielded += 1
                            if max_games is not None and yielded >= max_games:
                                return
                    current_game_lines = [line]
                else:
                    current_game_lines.append(line)

            if current_game_lines:
                record = _flush_current()
                if record is not None:
                    local_id += 1
                    if local_id > skip_games and (max_games is None or yielded < max_games):
                        yield local_id, record

    def _count_tasks_estimate(self) -> Optional[int]:
        return self.config.max_games

    # ------------------------------------------------------------------
    # Elaborazione di una singola partita (eseguita nel WORKER)
    # ------------------------------------------------------------------
    def _process_game(self, local_id: int, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        cfg = self.config
        pgn_text = row.get("pgn")
        if not isinstance(pgn_text, str) or not pgn_text.strip():
            return []

        try:
            pgn_io = io.StringIO(pgn_text)
            headers = chess.pgn.read_headers(pgn_io)
        except Exception as e:
            logger.warning(f"[holdout] game locale={local_id}: header PGN illeggibile ({e}), scartata.")
            return []

        if headers is None:
            return []

        # --- HARD: rating validi su entrambi i lati (allineato a
        # compatibility_filters.has_valid_ratings, richiesto esplicitamente
        # come requisito di progetto, non negoziabile). ---
        if not has_valid_ratings(headers):
            return []

        # --- SOFT (default ON, allineato a GamesBuilder): only_decisive_games,
        # skip_time_forfeit e ora min_rating/max_rating, controllati
        # sull'header PRIMA del replay. ---
        if not self._headers_are_eligible(headers):
            return []

        pgn_io.seek(0)
        try:
            game = chess.pgn.read_game(pgn_io)
        except Exception as e:
            logger.warning(f"[holdout] game locale={local_id}: PGN illeggibile ({e}), scartata.")
            return []
        if game is None:
            return []

        try:
            game_end_ply = game.end().ply()
        except Exception:
            return []
        if game_end_ply < cfg.min_game_plies:
            return []

        white_rating = parse_rating_strict(str(row.get("white_rating") or "")) or parse_rating_strict(
            game.headers.get("WhiteElo", "")
        )
        black_rating = parse_rating_strict(str(row.get("black_rating") or "")) or parse_rating_strict(
            game.headers.get("BlackElo", "")
        )
        mover_rating = {chess.WHITE: white_rating, chess.BLACK: black_rating}

        full_game_id = f"{cfg.source_tag}_{local_id}"
        collected: List[Dict[str, Any]] = []
        seen_positions: set = set()
        window_group_key: Optional[int] = None

        node = game
        while node.variations:
            next_node = node.variation(0)
            board = node.board()
            mover_color = board.turn

            if cfg.dedupe_positions:
                position_key = " ".join(board.fen().split(" ")[:4])
                if position_key in seen_positions:
                    node = next_node
                    continue
                seen_positions.add(position_key)

            if node.ply() < cfg.min_ply:
                node = next_node
                continue
            if (node.ply() - cfg.min_ply) % cfg.ply_sample_step != 0:
                node = next_node
                continue
            if cfg.max_positions_per_game is not None and len(collected) >= cfg.max_positions_per_game:
                break

            if not validate_kings(board):
                node = next_node
                continue

            legal_moves = self._get_candidate_legal_moves(board)
            if legal_moves is None:
                node = next_node
                continue

            # --- SOFT: skip_forced_moves sulla SINGOLA posizione, stessa
            # semantica di GamesBuilder.cfg.skip_forced_moves (non su
            # finestra multi-ply: qui non esiste ancora una finestra
            # finche' non si trova un matto valido). ---
            if cfg.skip_forced_single_move_window and len(legal_moves) == 1:
                node = next_node
                continue

            if not self._position_passes_quality_filters(board):
                node = next_node
                continue

            mover_rating_val = mover_rating[mover_color]
            if mover_rating_val is None:
                node = next_node
                continue

            info = self._analyse_position(board)
            if not info:
                node = next_node
                continue

            score = info.get("score")
            if score is None:
                node = next_node
                continue
            relative_score = score.relative
            if not relative_score.is_mate():
                node = next_node
                continue

            mate_n = relative_score.mate()
            pv = info.get("pv")
            if not pv:
                node = next_node
                continue
            best_move = pv[0]
            if best_move not in legal_moves:
                node = next_node
                continue

            # --- HARD: la board successiva alla sequenza PV deve essere un
            # matto REALE, con Re presenti, e mate_n nel range configurato
            # (allineato a compatibility_filters.is_window_hard_valid). Il
            # replay dell'intera PV verifica che il matto sia effettivamente
            # raggiungibile dalla posizione corrente, non solo dichiarato
            # dallo score dell'engine sulla singola mossa. ---
            pv_board = board.copy(stack=False)
            pv_valid = True
            for pv_move in pv:
                if pv_move not in pv_board.legal_moves:
                    pv_valid = False
                    break
                pv_board.push(pv_move)
            if not pv_valid or not is_window_hard_valid(pv_board, mate_n, cfg.mate_range):
                node = next_node
                continue

            clock_seconds = self._simulated_clock(mover_rating_val, node.ply(), full_game_id)

            try:
                data = build_position_data(
                    board=board,
                    best_move=best_move,
                    clock_seconds=clock_seconds,
                    rating=float(mover_rating_val),
                    game_id=full_game_id,
                    ply=node.ply(),
                    mate_n=int(mate_n),
                )
                data = apply_edge_type_time_weighting(data)
            except ValueError as e:
                logger.warning(f"[holdout] {full_game_id} ply={node.ply()}: scarto posizione ({e}).")
                node = next_node
                continue

            if window_group_key is None:
                window_group_key = int(mate_n)

            debug_entry = {
                "problem_id": f"{full_game_id}_{node.ply()}",
                "fen": board.fen(),
                "best_move_uci": best_move.uci(),
                "mate_n": int(mate_n),
                "mate_n_window": window_group_key,
                "ply": int(node.ply()),
                "source": cfg.source_tag,
                "clock_seconds": float(clock_seconds),
                "clock_is_real": False,
                "rating": mover_rating_val,
                "game_id": full_game_id,
                "time_class": row.get("time_class"),
                "split": "holdout",
            }

            collected.append({"data": data, "debug": debug_entry})
            node = next_node

        return collected

    def _worker(self, args: Tuple[int, Dict[str, Any]]) -> Tuple[int, bytes]:
        """Eseguito nel processo worker: costruisce le posizioni per una
        partita e ritorna un payload serializzato IPC-safe (bytes via
        torch.save), stesso schema di GamesBuilder._worker, cosi' nessun
        tensore attraversa il Pool con la reduction custom di torch
        (evita il crash 'received 0 items of ancdata')."""
        local_id, row = args
        empty_payload = encode_for_ipc([])

        if _engine is None:
            return local_id, empty_payload

        try:
            records = self._process_game(local_id, row)
        except Exception as e:
            logger.warning(
                f"[holdout] Worker: eccezione durante l'analisi di game locale={local_id} "
                f"({type(e).__name__}: {e}); partita scartata.",
                exc_info=True,
            )
            records = []

        try:
            payload = encode_for_ipc(records)
        except Exception as e:
            logger.error(
                f"[holdout] Worker: impossibile serializzare i risultati per game locale={local_id} "
                f"({type(e).__name__}: {e}); partita scartata ({len(records)} posizioni perse).",
                exc_info=True,
            )
            payload = empty_payload

        return local_id, payload

    def _write_debug_jsonl(self) -> Optional[str]:
        if not self._debug_jsonl_path or not self._debug_records:
            return None
        tmp_path = self._debug_jsonl_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in self._debug_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self._debug_jsonl_path)
        return self._debug_jsonl_path

    # ------------------------------------------------------------------
    # Orchestrazione multiprocessing (allineata a GamesBuilder.run())
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        cfg = self.config
        harden_process_for_ipc()

        cpu_count = os.cpu_count() or 2
        workers = cfg.workers or max(1, cpu_count - 1)

        processed_games = 0
        accepted_games = 0
        enqueued_positions = 0
        mate_n_counts: Dict[int, int] = defaultdict(int)
        skipped_on_parent_error = 0

        writer = _ShardWriter(cfg.output_dir, "holdout_raw", cfg.output_shard_size)

        pool = mp.Pool(
            processes=workers,
            initializer=self._init_worker,
            initargs=(cfg.stockfish_path, cfg.threads, cfg.hash_mb),
        )

        estimate = self._count_tasks_estimate()
        shutdown_in_progress = threading.Event()

        def _panic_kill() -> None:
            for proc in getattr(pool, "_pool", []):
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            os._exit(1)

        def _sigint_handler(signum, frame) -> None:
            if shutdown_in_progress.is_set():
                _panic_kill()
            shutdown_in_progress.set()
            raise KeyboardInterrupt

        previous_sigint = signal.signal(signal.SIGINT, _sigint_handler)

        def _shutdown_pool(graceful_first: bool) -> None:
            try:
                if graceful_first:
                    timeout = cfg.pool_join_timeout if cfg.pool_join_timeout is not None else 15.0
                    deadline = time.monotonic() + timeout
                    for proc in pool._pool:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        proc.join(timeout=remaining)

                pool.terminate()

                kill_deadline = time.monotonic() + 5.0
                for proc in pool._pool:
                    remaining = kill_deadline - time.monotonic()
                    proc.join(timeout=max(remaining, 0.1))

                for proc in pool._pool:
                    if proc.is_alive():
                        logger.warning(
                            f"[holdout] Worker pid={proc.pid} ancora vivo dopo terminate(): "
                            f"invio SIGKILL diretto."
                        )
                        try:
                            os.kill(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except Exception as e:
                            logger.warning(f"[holdout] SIGKILL su pid={proc.pid} fallito: {e}")

                for proc in pool._pool:
                    proc.join(timeout=2.0)
            except Exception:
                logger.exception(
                    "[holdout] Errore imprevisto nello shutdown del pool: forzo SIGKILL su tutti i worker."
                )
                for proc in getattr(pool, "_pool", []):
                    try:
                        os.kill(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass

        try:
            task_stream = self._iter_rows(self._effective_skip_games, cfg.max_games)
            results = pool.imap_unordered(self._worker, task_stream, chunksize=1)

            from Common.progress import wrap_iter

            for local_id, payload in wrap_iter(
                results,
                desc="[ExternalHoldout] Analisi partite Chess.com",
                unit="game",
                total=estimate,
            ):
                processed_games += 1
                self._resume_confirmed += 1

                if cfg.auto_resume and processed_games % cfg.resume_checkpoint_every == 0:
                    self._persist_resume_state()

                try:
                    records: List[Dict[str, Any]] = decode_from_ipc(payload)
                except Exception:
                    skipped_on_parent_error += 1
                    logger.error(
                        f"[holdout] decode_from_ipc fallito per game locale={local_id} "
                        f"(game processato #{processed_games}): payload scartato, run() continua.",
                        exc_info=True,
                    )
                    continue

                if not records:
                    continue

                try:
                    accepted_games += 1
                    for rec in records:
                        writer.append(rec["data"])
                        enqueued_positions += 1
                        mate_n_counts[rec["debug"]["mate_n"]] += 1
                        if cfg.save_debug_jsonl:
                            self._debug_records.append(rec["debug"])
                except Exception:
                    skipped_on_parent_error += 1
                    logger.error(
                        f"[holdout] Errore nella scrittura dei record per game locale={local_id} "
                        f"(game processato #{processed_games}): partita scartata, run() continua.",
                        exc_info=True,
                    )
                    continue

        except KeyboardInterrupt:
            print("\n[WARNING] Interruzione richiesta: arresto forzato dei worker in corso...")
            _shutdown_pool(graceful_first=False)
            self._persist_resume_state()
            signal.signal(signal.SIGINT, previous_sigint)
            raise
        except Exception:
            logger.exception(
                "[holdout] Errore FATALE durante l'analisi: arresto forzato dei worker in corso."
            )
            _shutdown_pool(graceful_first=False)
            self._persist_resume_state()
            signal.signal(signal.SIGINT, previous_sigint)
            raise
        else:
            pool.close()
            _shutdown_pool(graceful_first=True)
        finally:
            self._persist_resume_state()
            signal.signal(signal.SIGINT, previous_sigint)

        raw_files = writer.close()

        if cfg.save_debug_jsonl:
            self._write_debug_jsonl()

        if skipped_on_parent_error:
            logger.warning(
                f"[holdout] {skipped_on_parent_error} game scartati per errori lato padre "
                f"(decode/scrittura) durante questa run."
            )

        raw_dir = writer.split_dir
        clean_dir = os.path.join(cfg.output_dir, "holdout_clean")
        clean_manifest = clean_sharded_directory(
            in_dir=raw_dir,
            out_dir=clean_dir,
            target_shard_size=cfg.target_clean_shard_size,
            workers=0,
        )

        return {
            "processed_games": processed_games,
            "accepted_games": accepted_games,
            "enqueued_positions": enqueued_positions,
            "mate_n_counts": dict(mate_n_counts),
            "skipped_games_on_parent_error": skipped_on_parent_error,
            "raw_dir": raw_dir,
            "raw_files": raw_files,
            "clean_dir": clean_dir,
            "clean_manifest": clean_manifest,
        }


def _step_build_external_holdout(cfg: ExternalHoldoutConfig, state: PipelineState) -> Dict[str, Any]:
    clean_manifest_path = os.path.join(cfg.output_dir, "holdout_clean", "manifest.json")

    def _is_ready() -> bool:
        return os.path.exists(clean_manifest_path) and os.path.getsize(clean_manifest_path) > 0

    builder = ChesscomHoldoutBuilder(cfg)
    try:
        result = builder.run()
    except Exception as e:
        state.mark_failed("external_holdout", str(e))
        raise

    state.mark_done(
        "external_holdout",
        **{k: v for k, v in result.items() if isinstance(v, (int, str, float))},
    )
    logger.info(
        f"[external_holdout] Completato: processed={result.get('processed_games', 0):,}, "
        f"accepted={result.get('accepted_games', 0):,}, enqueued={result.get('enqueued_positions', 0):,}."
    )
    return result


def main() -> Dict[str, Any]:
    cfg = CONFIG
    setup_logging(cfg.log_level, cfg.log_file)

    logger.info("=" * 70)
    logger.info("BUILD EXTERNAL HOLDOUT (CSV Kaggle Chess.com o file .pgn)")
    logger.info("=" * 70)

    os.makedirs(cfg.output_dir, exist_ok=True)
    state_path = os.path.join(cfg.output_dir, cfg.state_file)

    if cfg.force_recompute and os.path.exists(state_path):
        logger.warning(f"force_recompute=True: rimuovo stato precedente '{state_path}'.")
        os.remove(state_path)

    state = PipelineState(state_path)

    result = _step_build_external_holdout(cfg, state)

    logger.info("=" * 70)
    logger.info("BUILD EXTERNAL HOLDOUT COMPLETATO")
    logger.info("=" * 70)
    return result


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        logging.getLogger("build_external_holdout").error(f"Errore di configurazione: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        logging.getLogger("build_external_holdout").warning(
            "Interrotto dall'utente. Il resume (skip_games + resume file) permette di "
            "riprendere da dove eri arrivato rilanciando lo stesso comando."
        )
        sys.exit(130)
    except Exception as e:
        logging.getLogger("build_external_holdout").error(f"Interruzione imprevista: {e}")
        sys.exit(1)