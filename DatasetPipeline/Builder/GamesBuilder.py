from __future__ import annotations

import atexit
import bz2
import io
import logging
import os
import signal
import threading
import time
import uuid
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
from torch_geometric.data import Data

from DatasetPipeline.Model.PositionGraphSchema import build_position_data
from DatasetPipeline.Utils.checkpoint_store import CheckpointStore
from DatasetPipeline.Utils.compatibility_filters import (
    QualityFilterConfig,
    has_valid_ratings,
    is_game_hard_eligible,
    is_window_hard_valid,
    parse_rating_strict,
    passes_all_quality_filters_for_candidate,
    passes_decisive_game_filter,
    passes_forced_move_filter,
    passes_rating_range_filter,
    passes_time_forfeit_filter,
)
from DatasetPipeline.Utils.chess_replay_utils import (
    closest_bucket_time,
    compute_move_duration,
    parse_clk,
    parse_emt,
    parse_time_control,
)

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
    quality: QualityFilterConfig

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


# ============================================================================
# ANALIZZATORE DI MATTO (istanziato in ogni worker)
# ============================================================================

class MateWindowAnalyzer:
    """Cerca, dentro una singola partita, la prima finestra di matto
    forzato che soddisfa i requisiti HARD + i filtri SOFT configurati, e
    la rigioca costruendo direttamente i Data (PositionGraphSchema) per
    ogni ply della finestra.
    """

    def __init__(self, config: WorkerConfig):
        self.config = config

    # ------------------------------------------------------------------
    # RICERCA DELLA FINESTRA (scansione della partita, ply per ply)
    # ------------------------------------------------------------------
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
            except Exception as e:
                error_counts[type(e).__name__] += 1
                if attempt < cfg.stockfish_retry_attempts:
                    time.sleep(cfg.stockfish_retry_backoff_seconds)
                    continue
                return None, error_counts
            finally:
                _watchdog_disarm()

        return None, error_counts

    def _syzygy_says_no_mate(self, board: chess.Board) -> bool:
        global _tablebase
        if _tablebase is None:
            return False
        if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
            return False
        try:
            wdl = _tablebase.probe_wdl(board)
        except Exception:
            return False
        return wdl is not None and wdl <= 0

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

            if not passes_all_quality_filters_for_candidate(board, cfg.quality):
                node = next_node
                continue

            if self._syzygy_says_no_mate(board):
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

    # ------------------------------------------------------------------
    # REPLAY DELLA FINESTRA -> COSTRUZIONE DIRETTA DEI Data
    # ------------------------------------------------------------------
    def _replay_and_build_positions(
        self,
        start_node: chess.pgn.GameNode,
        mate_n: int,
        game_id: int,
    ) -> Tuple[Optional[List[Data]], Counter]:
        """Rigioca la finestra di matto forzato (target_plies = 2*mate_n-1
        ply) e costruisce direttamente un Data per ogni ply, tramite
        build_position_data. Ritorna None se un requisito HARD non e'
        soddisfatto in un punto qualsiasi del replay (mossa illegale,
        finestra troncata, matto finale non reale, nessun arco spaziale
        per una posizione)."""
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

        white_elo = parse_rating_strict(game_root.headers.get("WhiteElo"))
        black_elo = parse_rating_strict(game_root.headers.get("BlackElo"))
        mover_rating = {chess.WHITE: white_elo, chess.BLACK: black_elo}

        positions: List[Data] = []
        window_boards: List[chess.Board] = []

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

            window_boards.append(board.copy(stack=False))

            try:
                data = build_position_data(
                    board=board,
                    best_move=move,
                    clock_seconds=float(clock_seconds),
                    game_id=game_id,
                    ply=step,
                )
            except ValueError:
                # Requisito HARD (6): nessun arco spaziale costruibile.
                error_counts["no_spatial_edges"] += 1
                return None, error_counts

            positions.append(data)

            board.push(move)
            node = next_node

        # HARD (3)+(1)+(4): matto reale, Re presenti, mate_n nel range.
        if not is_window_hard_valid(board, mate_n, cfg.mate_range):
            error_counts["window_not_hard_valid"] += 1
            return None, error_counts

        # SOFT: sequenza a mossa forzata su tutta la finestra (default ON).
        if not passes_forced_move_filter(window_boards, cfg.quality):
            error_counts["skipped_forced_move_window"] += 1
            return None, error_counts

        return positions, error_counts


# ============================================================================
# FUNZIONI DI SUPPORTO E WORKER MAIN
# ============================================================================

@dataclass
class _WindowBuildResult:
    positions: Optional[List[Data]]
    mate_n: Optional[int]
    game_id: Optional[int]
    source_tag: str
    error_counts: Counter = field(default_factory=Counter)


def _worker_main(args: Tuple[int, str, str]) -> Tuple[int, "_WindowBuildResult"]:
    global _engine, _worker_config
    task_local_id, pgn_text, source_tag = args
    error_counts: Counter = Counter()

    if _engine is None:
        error_counts["engine_unavailable"] += 1
        return task_local_id, _WindowBuildResult(None, None, None, source_tag, error_counts)

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception as e:
        error_counts[f"pgn_parse:{type(e).__name__}"] += 1
        return task_local_id, _WindowBuildResult(None, None, None, source_tag, error_counts)

    if game is None:
        return task_local_id, _WindowBuildResult(None, None, None, source_tag, error_counts)

    if game.headers.get("Variant", "Standard").lower() not in ("standard", "normal"):
        return task_local_id, _WindowBuildResult(None, None, None, source_tag, error_counts)

    cfg = _worker_config

    # HARD: partita eleggibile (non vuota/corrotta) + rating validi per
    # ENTRAMBI i giocatori (requisito esplicito, non negoziabile).
    if not is_game_hard_eligible(game, min_plies=cfg.min_game_plies):
        error_counts["missing_or_invalid_ratings_or_too_short"] += 1
        return task_local_id, _WindowBuildResult(None, None, None, source_tag, error_counts)

    # SOFT (default ON): esclude partite terminate per tempo scaduto.
    if not passes_time_forfeit_filter(game.headers, cfg.quality):
        error_counts["skipped_time_forfeit"] += 1
        return task_local_id, _WindowBuildResult(None, None, None, source_tag, error_counts)

    # SOFT (default OFF, disponibile se riattivato): solo partite decisive.
    if not passes_decisive_game_filter(game.headers, cfg.quality):
        error_counts["skipped_not_decisive"] += 1
        return task_local_id, _WindowBuildResult(None, None, None, source_tag, error_counts)

    # SOFT (default OFF, disponibile se riattivato): range di rating.
    if not passes_rating_range_filter(game.headers, cfg.quality):
        error_counts["skipped_rating_range"] += 1
        return task_local_id, _WindowBuildResult(None, None, None, source_tag, error_counts)

    analyzer = MateWindowAnalyzer(cfg)

    try:
        start_node, mate_n, search_errors = analyzer._find_mate_window_start(game)
        error_counts.update(search_errors)

        if start_node is None:
            return task_local_id, _WindowBuildResult(None, None, None, source_tag, error_counts)

        game_id = uuid.uuid4().int & ((1 << 63) - 1)

        positions, replay_errors = analyzer._replay_and_build_positions(start_node, mate_n, game_id)
        error_counts.update(replay_errors)

        if not positions:
            return task_local_id, _WindowBuildResult(None, None, None, source_tag, error_counts)

        return task_local_id, _WindowBuildResult(positions, mate_n, game_id, source_tag, error_counts)

    except Exception as e:
        error_counts[f"unexpected:{type(e).__name__}"] += 1
        logger.warning(
            f"Errore inatteso nel worker per task_local_id={task_local_id} "
            f"(source={source_tag}): {type(e).__name__}: {e}"
        )
        return task_local_id, _WindowBuildResult(None, None, None, source_tag, error_counts)


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
# DATACLASSES E STRUTTURE DATI
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


# ============================================================================
# GAMES BUILDER
# ============================================================================

class GamesBuilder:
    """Estrae finestre di matto forzato da sorgenti PGN (Lichess/FICS/Club),
    costruisce i Data (PositionGraphSchema) e li accumula in un
    CheckpointStore che mantiene sempre e solo 3 file di output:
    train_games.pt, val_games.pt, test_games.pt, riscritti stratificati
    per mate_n ad ogni checkpoint (vedi checkpoint_store.py).

    Rispetto alla precedente RelaxedGamesBuilder:
      - i filtri sono esternalizzati in compatibility_filters.py, con
        distinzione esplicita hard/soft (vedi docstring di modulo);
      - i Data vengono davvero costruiti e accodati (bug corretto);
      - non usa piu' PositionQueueRegistry/shard: la persistenza a
        checkpoint e' gestita da CheckpointStore.
    """

    def __init__(
        self,
        sources: List[SourceSpec],
        stockfish_path: str,
        output_dir: str,
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
        min_ply: int = 0,
        min_game_plies: int = 2,
        quality: Optional[QualityFilterConfig] = None,
        pool_join_timeout: Optional[float] = 20.0,
        syzygy_path: Optional[str] = None,
        checkpoint_every: int = 5000,
        config_error_cls: type = ValueError,
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        split_seed: int = 42,
        stockfish_retry_attempts: int = 2,
        stockfish_retry_backoff_seconds: float = 0.5,
    ):
        self.sources = sources
        self.stockfish_path = stockfish_path
        self.output_dir = output_dir
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
        self.quality = quality or QualityFilterConfig()
        self.pool_join_timeout = pool_join_timeout
        self.syzygy_path = syzygy_path
        self.checkpoint_every = checkpoint_every
        self._config_error_cls = config_error_cls
        self.stockfish_retry_attempts = max(1, stockfish_retry_attempts)
        self.stockfish_retry_backoff_seconds = stockfish_retry_backoff_seconds

        cpu_count = os.cpu_count() or 2
        self.workers = workers or max(1, cpu_count - 1)

        self._store = CheckpointStore(
            output_dir=output_dir,
            split_ratios=split_ratios,
            seed=split_seed,
        )

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
        for src in self.sources:
            if not os.path.exists(src.path):
                raise self._config_error_cls(f"Sorgente non trovata: {src.path}.")
        if not (os.path.exists(self.stockfish_path) and os.access(self.stockfish_path, os.X_OK)):
            raise self._config_error_cls(f"Stockfish non trovato/eseguibile: {self.stockfish_path}.")

    # ------------------------------------------------------------------
    # LETTURA SORGENTI (invariata nella logica rispetto alla versione
    # precedente)
    # ------------------------------------------------------------------
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
            raise self._config_error_cls(f"Colonna '{src.pgn_col}' assente in {src.path}.")
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

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        config = WorkerConfig(
            mate_range=self.mate_range,
            min_ply=self.min_ply,
            quality=self.quality,
            candidate_min_legal_moves=self.quality.candidate_min_legal_moves,
            candidate_max_legal_moves=self.quality.candidate_max_legal_moves,
            skip_if_in_check=self.quality.skip_if_in_check,
            max_piece_count=self.quality.max_piece_count,
            require_clock=self.require_clock,
            default_move_seconds=self.default_move_seconds,
            avg_time_by_rating=self.avg_time_by_rating,
            drop_zero_clock=False,
            stockfish_retry_attempts=self.stockfish_retry_attempts,
            stockfish_retry_backoff_seconds=self.stockfish_retry_backoff_seconds,
            analysis_time=self.analysis_time,
            search_depth=self.search_depth,
            multipv=self.multipv,
            min_game_plies=self.min_game_plies,
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

        # chunksize dinamico: riduce l'overhead IPC quando ci sono molte
        # task piccole, senza sacrificare il bilanciamento del carico tra
        # worker (chunksize troppo alto farebbe attendere i worker piu'
        # lenti). Euristica semplice: qualche centinaio di task per worker
        # al massimo, non piu' di 50 per chunk.
        dynamic_chunksize = max(1, min(50, self.workers * 4))

        try:
            task_stream = self._iter_all_tasks()
            results = pool.imap_unordered(_worker_main, task_stream, chunksize=dynamic_chunksize)

            pbar = tqdm(results, desc="[GamesBuilder] Ricerca finestre matto", dynamic_ncols=True)
            for task_local_id, window in pbar:
                processed_games += 1
                aggregated_errors.update(window.error_counts)

                if window.positions is None:
                    continue

                self._store.add_window(
                    game_id=window.game_id,
                    group_key=window.mate_n,
                    positions=window.positions,
                )

                accepted_windows += 1
                enqueued_positions += len(window.positions)
                mate_n_counts[window.mate_n] += 1

                if processed_games % self.checkpoint_every == 0:
                    self._store.checkpoint()
                    pbar.set_postfix(
                        accepted=accepted_windows,
                        positions=enqueued_positions,
                        refresh=False,
                    )

            pbar.close()
        except KeyboardInterrupt:
            pool.terminate()
            raise
        finally:
            pool.close()
            pool.join()

        # Checkpoint finale, garantisce che l'ultimo batch parziale (sotto
        # checkpoint_every) sia comunque scritto sui 3 file di output.
        window_counts = self._store.finalize()

        logger.info(
            f"Riepilogo finale: {processed_games} partite elaborate, "
            f"{accepted_windows} finestre valide accettate (mate_range={self.mate_range})."
        )
        return {
            "processed_games": processed_games,
            "accepted_windows": accepted_windows,
            "enqueued_positions": enqueued_positions,
            "error_counts": dict(aggregated_errors),
            "mate_n_counts": dict(mate_n_counts),
            "final_window_counts": window_counts,
            "output_paths": self._store.final_paths,
        }