"""
GamesBuilder.py

Builder unificato per il TRAIN set "games" (posizioni tratte da finestre di
matto forzato in partite reali, con clock reale quando disponibile),
sorgente multipla:
...
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

from DatasetPipeline.Model.PositionGraphSchema import build_position_data
from DatasetPipeline.PositionQueue import PositionQueueRegistry
from DatasetPipeline.Utils.chess_replay_utils import (
    closest_bucket_time,
    compute_move_duration,
    parse_clk,
    parse_emt,
    parse_rating,
    parse_time_control,
)
from DatasetPipeline.Utils.mate_prefilters import passes_all_prefilters

logger = logging.getLogger("games_builder")

# ============================================================================
# STATO GLOBALE PER WORKER (Stockfish + Syzygy + watchdog)
# ============================================================================

_engine: Optional[chess.engine.SimpleEngine] = None
_engine_pid: Optional[int] = None
_tablebase: Optional["chess.syzygy.Tablebase"] = None

_watchdog_lock: Optional[threading.Lock] = None
_watchdog_deadline: Optional[float] = None
_watchdog_stop: Optional[threading.Event] = None
_watchdog_thread: Optional[threading.Thread] = None

_WATCHDOG_POLL_SECONDS = 1.0
_WATCHDOG_MARGIN_SECONDS = 3.0

_worker_config: Optional["WorkerConfig"] = None


# ============================================================================
# CONFIGURAZIONE WORKER (picklable)
# ============================================================================

@dataclass(frozen=True)
class WorkerConfig:
    mate_range: Tuple[int, int]
    min_ply: int
    require_heavy_piece: bool
    min_material_for_mate_attempt: int
    min_material_diff_for_mate_attempt: int
    skip_trivial_endgame: bool
    enable_extra_prefilters: bool
    prefilter_max_free_squares: int
    prefilter_king_distance_threshold: int
    candidate_min_legal_moves: int
    candidate_max_legal_moves: Optional[int]
    skip_if_in_check: bool
    max_piece_count: Optional[int]
    require_clock: bool
    default_move_seconds: float
    avg_time_by_rating: Dict[int, float]
    drop_zero_clock: bool
    stockfish_retry_attempts: int
    stockfish_retry_backoff_seconds: float
    analysis_time: Optional[float]
    search_depth: int
    multipv: int
    min_game_plies: int
    only_decisive_games: bool
    skip_time_forfeit: bool
    min_rating: Optional[int]
    max_rating: Optional[int]


# ============================================================================
# ANALIZZATORE DI MATTO (istanziato in ogni worker)
# ============================================================================

class MateAnalyzer:
    PIECE_VALUES: Dict[int, int] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    def __init__(self, config: WorkerConfig):
        self.config = config

    def _validate_kings(self, board: chess.Board) -> bool:
        return board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None

    def _material_by_color(self, board: chess.Board) -> Tuple[int, int]:
        white = 0
        black = 0
        for p in board.piece_map().values():
            val = self.PIECE_VALUES.get(p.piece_type, 0)
            if p.color == chess.WHITE:
                white += val
            else:
                black += val
        return white, black

    def _get_candidate_legal_moves(self, board: chess.Board) -> Optional[List[chess.Move]]:
        cfg = self.config
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            return None
        if cfg.max_piece_count is not None and len(board.piece_map()) > cfg.max_piece_count:
            return None
        moves = list(board.legal_moves)
        if len(moves) < cfg.candidate_min_legal_moves:
            return None
        if cfg.candidate_max_legal_moves is not None and len(moves) > cfg.candidate_max_legal_moves:
            return None
        if cfg.skip_if_in_check and board.is_check():
            return None
        return moves

    def _has_mating_material(self, board: chess.Board) -> bool:
        cfg = self.config
        if not self._validate_kings(board):
            logger.warning("Board senza uno o entrambi i Re: scarto posizione.")
            return False
        mover = board.turn
        white_mat, black_mat = self._material_by_color(board)
        mover_mat = white_mat if mover == chess.WHITE else black_mat
        opp_mat = black_mat if mover == chess.WHITE else white_mat
        if mover_mat < cfg.min_material_for_mate_attempt:
            return False
        if (mover_mat - opp_mat) < cfg.min_material_diff_for_mate_attempt:
            return False
        return True

    def _mover_has_heavy_piece(self, board: chess.Board) -> bool:
        mover = board.turn
        for pt in (chess.QUEEN, chess.ROOK):
            if board.pieces(pt, mover):
                return True
        return False

    def _is_trivially_drawn_endgame(self, board: chess.Board) -> bool:
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

    def _syzygy_says_no_mate(self, board: chess.Board) -> bool:
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

    def _passes_extra_prefilters(self, board: chess.Board) -> bool:
        cfg = self.config
        if not cfg.enable_extra_prefilters:
            return True
        return passes_all_prefilters(
            board,
            mate_n_upper_bound=cfg.mate_range[1],
            max_free_squares=cfg.prefilter_max_free_squares,
            king_distance_threshold=cfg.prefilter_king_distance_threshold,
        )

    def _game_is_eligible(self, game: chess.pgn.Game) -> bool:
        cfg = self.config
        if game is None:
            return False
        try:
            ply_count = game.end().ply()
        except Exception:
            return False
        if ply_count < cfg.min_game_plies:
            return False
        if cfg.only_decisive_games:
            result = game.headers.get("Result", "")
            if result not in ("1-0", "0-1"):
                return False
        if cfg.skip_time_forfeit:
            termination = game.headers.get("Termination", "")
            if "Time forfeit" in termination:
                return False
        if cfg.min_rating is not None or cfg.max_rating is not None:
            white_elo = parse_rating(game.headers.get("WhiteElo", ""))
            black_elo = parse_rating(game.headers.get("BlackElo", ""))
            ratings = [r for r in (white_elo, black_elo) if r is not None]
            if ratings:
                best_rating = max(ratings)
                if cfg.min_rating is not None and best_rating < cfg.min_rating:
                    return False
                if cfg.max_rating is not None and min(ratings) > cfg.max_rating:
                    return False
        return True

    # ----------------------------------------------------------------
    # Analisi Stockfish (con retry)
    # ----------------------------------------------------------------

    _NON_RETRYABLE_ENGINE_ERRORS = (
        chess.engine.EngineTerminatedError,
        BrokenPipeError,
        ConnectionResetError,
    )

    def _analyse_position(self, board: chess.Board) -> Tuple[Optional[Any], Counter]:
        global _engine
        cfg = self.config
        error_counts: Counter = Counter()

        for attempt in range(1, cfg.stockfish_retry_attempts + 1):
            if _engine is None:
                error_counts["engine_unavailable"] += 1
                return None, error_counts
            try:
                if cfg.analysis_time is not None:
                    limit = chess.engine.Limit(time=cfg.analysis_time, mate=cfg.mate_range[1])
                    _watchdog_arm(cfg.analysis_time)
                else:
                    limit = chess.engine.Limit(depth=cfg.search_depth, mate=cfg.mate_range[1])
                    _watchdog_arm(5.0)
                result = _engine.analyse(board, limit, multipv=cfg.multipv)
                return result, error_counts
            except self._NON_RETRYABLE_ENGINE_ERRORS as e:
                error_counts[type(e).__name__] += 1
                return None, error_counts
            except chess.engine.EngineError as e:
                error_counts[type(e).__name__] += 1
                if attempt < cfg.stockfish_retry_attempts:
                    time.sleep(cfg.stockfish_retry_backoff_seconds)
                    continue
                return None, error_counts
            except Exception as e:
                error_counts[type(e).__name__] += 1
                if attempt < cfg.stockfish_retry_attempts:
                    time.sleep(cfg.stockfish_retry_backoff_seconds)
                    continue
                return None, error_counts
            finally:
                _watchdog_disarm()

        return None, error_counts

    # ----------------------------------------------------------------
    # Ricerca finestra e replay
    # ----------------------------------------------------------------

    def _find_mate_window_start(
        self, game: chess.pgn.Game
    ) -> Tuple[Optional[chess.pgn.GameNode], Optional[int], Counter]:
        cfg = self.config
        error_counts: Counter = Counter()
        node = game

        while node.variations:
            next_node = node.variation(0)
            board = node.board()

            if node.ply() < cfg.min_ply:
                node = next_node
                continue

            if self._get_candidate_legal_moves(board) is None:
                node = next_node
                continue

            if cfg.require_heavy_piece and not self._mover_has_heavy_piece(board):
                node = next_node
                continue

            if not self._has_mating_material(board):
                node = next_node
                continue

            if cfg.skip_trivial_endgame and self._is_trivially_drawn_endgame(board):
                node = next_node
                continue

            if self._syzygy_says_no_mate(board):
                node = next_node
                continue

            if not self._passes_extra_prefilters(board):
                node = next_node
                continue

            info, pos_errors = self._analyse_position(board)
            error_counts.update(pos_errors)
            if not info:
                node = next_node
                continue

            best = info[0]
            score = best.get("score")
            if score is None:
                node = next_node
                continue
            rel = score.relative
            if not rel.is_mate():
                node = next_node
                continue
            mate_n = rel.mate()
            if mate_n is None or not (mate_n > 0 and cfg.mate_range[0] <= mate_n <= cfg.mate_range[1]):
                node = next_node
                continue

            return node, int(mate_n), error_counts

        return None, None, error_counts

    def _replay_mate_window(
        self,
        start_node: chess.pgn.GameNode,
        mate_n: int,
    ) -> Tuple[Optional[List["_RawPlyRecord"]], Counter]:
        cfg = self.config
        error_counts: Counter = Counter()
        target_plies = 2 * mate_n - 1

        node = start_node
        board = node.board()
        game_root = node.game()

        tc_raw = game_root.headers.get("TimeControl", "")
        base_time, increment = parse_time_control(tc_raw)
        previous_clock = {
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

            if cfg.require_clock and not duration_is_real:
                return None, error_counts

            if duration_is_real:
                clock_seconds = move_duration
            else:
                bucket = closest_bucket_time(mover_rating[mover_color], cfg.avg_time_by_rating)
                clock_seconds = bucket if bucket is not None else cfg.default_move_seconds

            if cfg.drop_zero_clock and clock_seconds == 0.0 and not duration_is_real:
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


# ============================================================================
# FUNZIONE WORKER (eseguita nei processi figli)
# ============================================================================

def _worker_main(args: Tuple[int, str, str]) -> Tuple[int, "_WindowResult"]:
    """Funzione principale del worker. Riceve una tupla (task_local_id, pgn_text, source_tag)."""
    global _engine, _worker_config
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

    analyzer = MateAnalyzer(_worker_config)
    if not analyzer._game_is_eligible(game):
        return task_local_id, _WindowResult(None, None, source_tag, error_counts)

    try:
        start_node, mate_n, search_errors = analyzer._find_mate_window_start(game)
        error_counts.update(search_errors)

        if start_node is None:
            return task_local_id, _WindowResult(None, None, source_tag, error_counts)

        raw_plies, replay_errors = analyzer._replay_mate_window(start_node, mate_n)
        error_counts.update(replay_errors)

        return task_local_id, _WindowResult(raw_plies, mate_n, source_tag, error_counts)

    except Exception as e:
        error_counts[f"unexpected:{type(e).__name__}"] += 1
        logger.warning(
            f"Errore inatteso nel worker per task_local_id={task_local_id} "
            f"(source={source_tag}): {type(e).__name__}: {e}"
        )
        return task_local_id, _WindowResult(None, None, source_tag, error_counts)


# ============================================================================
# WATCHDOG E FUNZIONI DI SUPPORTO
# ============================================================================

def _watchdog_arm(time_limit: float, margin: Optional[float] = None) -> None:
    global _watchdog_deadline
    if _watchdog_lock is None:
        return
    eff_margin = _WATCHDOG_MARGIN_SECONDS if margin is None else margin
    with _watchdog_lock:
        _watchdog_deadline = time.monotonic() + time_limit + eff_margin


def _watchdog_disarm() -> None:
    global _watchdog_deadline
    if _watchdog_lock is None:
        return
    with _watchdog_lock:
        _watchdog_deadline = None


def _watchdog_loop() -> None:
    global _engine, _engine_pid, _watchdog_deadline
    if _watchdog_stop is None:
        return
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
    global _engine, _engine_pid, _tablebase, _watchdog_stop
    if _watchdog_stop is not None:
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
    _close_engine()
    os._exit(0)


def _init_worker(
    stockfish_path: str,
    threads: int,
    hash_mb: int,
    syzygy_path: Optional[str],
    config: WorkerConfig,
) -> None:
    global _engine, _engine_pid, _tablebase, _watchdog_thread, _watchdog_lock, _watchdog_stop, _watchdog_deadline, _worker_config

    _worker_config = config
    _watchdog_lock = threading.Lock()
    _watchdog_deadline = None
    _watchdog_stop = threading.Event()
    _watchdog_thread = None

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

    if syzygy_path:
        try:
            _tablebase = chess.syzygy.open_tablebase(syzygy_path)
        except Exception:
            _tablebase = None

    atexit.register(_close_engine)

    _watchdog_stop.clear()
    _watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
    _watchdog_thread.start()


# ============================================================================
# SOURCE SPEC E RECORD
# ============================================================================

@dataclass
class SourceSpec:
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
    fen_before_move: str
    move_uci: str
    clock_seconds: float
    ply: int


@dataclass
class _WindowResult:
    plies: Optional[List[_RawPlyRecord]]
    mate_n: Optional[int]
    source_tag: str
    error_counts: Counter = field(default_factory=Counter)


# ============================================================================
# GAMES BUILDER (processo principale)
# ============================================================================

class GamesBuilder:
    """Pipeline multi-sorgente per il train set "games"."""

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
        enable_extra_prefilters: bool = True,
        prefilter_max_free_squares: int = 2,
        prefilter_king_distance_threshold: int = 3,
    ):
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
        self.enable_extra_prefilters = enable_extra_prefilters
        self.prefilter_max_free_squares = prefilter_max_free_squares
        self.prefilter_king_distance_threshold = prefilter_king_distance_threshold

        cpu_count = os.cpu_count() or 2
        self.workers = workers or max(1, cpu_count - 1)

        self._queue_registry = PositionQueueRegistry.instance(state_path=queue_state_path)
        self._next_game_id = 0

        self._validate_parameters()

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
        if self.min_rating is not None and self.max_rating is not None and self.min_rating > self.max_rating:
            raise self._config_error_cls("min_rating deve essere <= max_rating.")
        if self.prefilter_max_free_squares < 0:
            raise self._config_error_cls("prefilter_max_free_squares deve essere >= 0.")
        if self.prefilter_king_distance_threshold < 0:
            raise self._config_error_cls("prefilter_king_distance_threshold deve essere >= 0.")
        for src in self.sources:
            if not os.path.exists(src.path):
                raise self._config_error_cls(f"Sorgente non trovata: {src.path} (kind={src.kind}).")
        if not (os.path.exists(self.stockfish_path) and os.access(self.stockfish_path, os.X_OK)):
            raise self._config_error_cls(f"Stockfish non trovato/eseguibile: {self.stockfish_path}.")
        if self.syzygy_path is not None and not os.path.isdir(self.syzygy_path):
            raise self._config_error_cls(f"syzygy_path non e' una cartella valida: {self.syzygy_path}.")

    # ----------------------------------------------------------------
    # Enqueue nel processo principale
    # ----------------------------------------------------------------

    def _enqueue_window(self, window: _WindowResult) -> int:
        game_id = self._next_game_id
        self._next_game_id += 1
        enqueued = 0
        for raw in window.plies:
            try:
                board = chess.Board(raw.fen_before_move)
                move = chess.Move.from_uci(raw.move_uci)
                data = build_position_data(
                    board=board,
                    best_move=move,
                    clock_seconds=raw.clock_seconds,
                    game_id=game_id,
                    ply=raw.ply,
                )
            except ValueError as e:
                logger.warning(f"Scarto posizione game_id={game_id} ply={raw.ply}: {e}")
                continue
            self._queue_registry.enqueue(
                source_tag=window.source_tag,
                data=data,
                group_key=window.mate_n,
            )
            enqueued += 1
        return enqueued

    # ----------------------------------------------------------------
    # Streaming sorgenti
    # ----------------------------------------------------------------

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

    def _iter_pgn_texts(self, text_stream, skip_games: int, max_games: Optional[int]):
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

    def _iter_club_csv(self, src: SourceSpec):
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
                f"Colonna '{src.pgn_col}' assente in {src.path}. Colonne: {list(df.columns)}"
            )
        series = df[src.pgn_col].dropna().iloc[src.skip_games:]
        if src.max_games is not None:
            series = series.iloc[:src.max_games]
        for local_id, pgn_text in series.items():
            yield (int(local_id) + 1, pgn_text)

    def _iter_source(self, src: SourceSpec):
        if src.kind == "club":
            yield from self._iter_club_csv(src)
            return
        with self._open_pgn_text_stream(src.path, src.kind) as text_stream:
            yield from self._iter_pgn_texts(text_stream, src.skip_games, src.max_games)

    def _iter_all_tasks(self):
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

    # ----------------------------------------------------------------
    # RUN
    # ----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        config = WorkerConfig(
            mate_range=self.mate_range,
            min_ply=self.min_ply,
            require_heavy_piece=self.require_heavy_piece,
            min_material_for_mate_attempt=self.min_material_for_mate_attempt,
            min_material_diff_for_mate_attempt=self.min_material_diff_for_mate_attempt,
            skip_trivial_endgame=self.skip_trivial_endgame,
            enable_extra_prefilters=self.enable_extra_prefilters,
            prefilter_max_free_squares=self.prefilter_max_free_squares,
            prefilter_king_distance_threshold=self.prefilter_king_distance_threshold,
            candidate_min_legal_moves=self.candidate_min_legal_moves,
            candidate_max_legal_moves=self.candidate_max_legal_moves,
            skip_if_in_check=self.skip_if_in_check,
            max_piece_count=self.max_piece_count,
            require_clock=self.require_clock,
            default_move_seconds=self.default_move_seconds,
            avg_time_by_rating=self.avg_time_by_rating,
            drop_zero_clock=self.drop_zero_clock,
            stockfish_retry_attempts=self.stockfish_retry_attempts,
            stockfish_retry_backoff_seconds=self.stockfish_retry_backoff_seconds,
            analysis_time=self.analysis_time,
            search_depth=self.search_depth,
            multipv=self.multipv,
            min_game_plies=self.min_game_plies,
            only_decisive_games=self.only_decisive_games,
            skip_time_forfeit=self.skip_time_forfeit,
            min_rating=self.min_rating,
            max_rating=self.max_rating,
        )

        pool = mp.Pool(
            processes=self.workers,
            initializer=_init_worker,
            initargs=(self.stockfish_path, self.threads, self.hash_mb, self.syzygy_path, config),
        )

        processed_games = 0
        accepted_windows = 0
        enqueued_positions = 0
        mate_n_counts: Dict[int, int] = defaultdict(int)
        aggregated_errors: Counter = Counter()
        estimate = self._count_tasks_estimate()

        def cleanup_after_failure():
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
                    logger.warning(f"{len(still_alive)} worker non terminati entro {timeout}s. Invio SIGKILL...")
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
            results = pool.imap_unordered(_worker_main, task_stream, chunksize=1)

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
                        f"[checkpoint] {processed_games} partite, "
                        f"{accepted_windows} finestre, "
                        f"{enqueued_positions} posizioni, "
                        f"{self._queue_registry.pending_count()} in coda."
                    )
            pbar.close()
        except KeyboardInterrupt:
            logger.warning("Interruzione richiesta: arresto pulito dei worker...")
            cleanup_after_failure()
            raise
        except Exception:
            logger.warning("Errore durante l'analisi: arresto pulito dei worker...")
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
                    logger.warning(f"{len(still_alive)} worker non terminati entro {self.pool_join_timeout}s. Forzo terminate().")
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
                        logger.warning("'psutil' non disponibile: eventuali Stockfish orfani vanno terminati manualmente.")
            pool.join()

        self._log_summary(processed_games, accepted_windows, enqueued_positions, mate_n_counts, aggregated_errors)
        return {
            "processed_games": processed_games,
            "accepted_windows": accepted_windows,
            "enqueued_positions": enqueued_positions,
            "error_counts": dict(aggregated_errors),
            "mate_n_counts": dict(mate_n_counts),
        }

    def _log_summary(self, processed_games, accepted_windows, enqueued_positions, mate_n_counts, aggregated_errors):
        logger.info("=" * 60)
        logger.info("GAMES BUILDER — RIEPILOGO (schema board-level per posizione)")
        logger.info("=" * 60)
        logger.info(f"Partite processate: {processed_games:,}")
        logger.info(f"Finestre accettate: {accepted_windows:,}")
        logger.info(f"Posizioni accodate: {enqueued_positions:,}")
        if accepted_windows > 0:
            logger.info("Per profondita' mate (N):")
            for n in sorted(mate_n_counts.keys()):
                logger.info(f"  n={n}: {mate_n_counts[n]:,}")
        if aggregated_errors:
            logger.info("Errori/scarti:")
            for err_type, count in aggregated_errors.most_common():
                logger.info(f"  {err_type}: {count:,}")
        logger.info("=" * 60)