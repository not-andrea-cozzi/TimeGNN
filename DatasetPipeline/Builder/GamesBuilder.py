from __future__ import annotations

import atexit
import bz2
import io
import os
import re
import signal
import threading
import time
import zipfile
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import pandas as pd
import zstandard as zstd

from Common.progress import wrap_iter
from DatasetPipeline.Model.PositionGraphSchema import build_position_data
from DatasetPipeline.PositionQueue import PositionQueueRegistry
from DatasetPipeline.Utils.ipc_safe_data import (
    encode_for_ipc,
    decode_from_ipc,
    harden_process_for_ipc,
)

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

_CLK_RE = re.compile(r"\[\s*%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]")
_EMT_RE = re.compile(r"\[\s*%emt\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]")

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
    _close_engine()
    os._exit(0)

# ============================================================================
# SOURCE SPEC
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
# CONFIG (MODIFICATA PER MAX THROUGHPUT)
# ============================================================================

@dataclass
class GamesBuilderConfig:
    sources: List[SourceSpec]
    stockfish_path: str
    mate_range: Tuple[int, int] = (1, 10)
    
    # [OTTIMIZZAZIONE]: Sostituito il timer con un depth limit secco.
    # Evita il blocco di 0.2 secondi per ogni posizione analizzata.
    search_depth: int = 6  
    analysis_time: Optional[float] = None
    
    workers: Optional[int] = None
    threads: int = 1
    hash_mb: int = 128
    multipv: int = 1
    syzygy_path: Optional[str] = None

    stockfish_retry_attempts: int = 2
    stockfish_retry_backoff_seconds: float = 0.5

    candidate_min_legal_moves: int = 1
    candidate_max_legal_moves: Optional[int] = None
    skip_if_in_check: bool = False
    max_piece_count: Optional[int] = 18
    min_material_for_mate_attempt: int = 4
    min_material_diff_for_mate_attempt: int = 3
    require_heavy_piece: bool = True
    skip_forced_moves: bool = False
    skip_trivial_endgame: bool = True
    dedupe_positions: bool = True

    require_clock: bool = False
    default_move_seconds: float = 15.0
    avg_time_by_rating: Dict[int, float] = field(default_factory=dict)
    drop_zero_clock: bool = True
    min_rating: Optional[int] = 1200
    max_rating: Optional[int] = None

    # [OTTIMIZZAZIONE]: Campionamento alleggerito (scarta più mosse).
    min_ply: int = 8
    ply_sample_step: int = 6 
    max_positions_per_game: Optional[int] = 5 

    only_decisive_games: bool = True
    skip_time_forfeit: bool = True
    min_game_plies: int = 20

    queue_state_path: Optional[str] = None
    shard_size: int = 500

    save_debug_jsonl: bool = True
    debug_jsonl_dir: Optional[str] = None

    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 42

    pool_join_timeout: Optional[float] = 20.0

# ============================================================================
# GAMES BUILDER
# ============================================================================

class GamesBuilder:
    _PIECE_VALUES: Dict[int, int] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    def __init__(self, config: GamesBuilderConfig):
        self.config = config
        self._validate_config()

        self._registry = PositionQueueRegistry.instance(
            state_path=config.queue_state_path,
            shard_size=config.shard_size,
        )

        cpu_count = os.cpu_count() or 2
        self._workers = config.workers or max(1, cpu_count - 1)

        self._debug_records: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
        self._debug_jsonl_path = None
        if config.save_debug_jsonl:
            if config.debug_jsonl_dir:
                os.makedirs(config.debug_jsonl_dir, exist_ok=True)
                self._debug_jsonl_path = os.path.join(config.debug_jsonl_dir, "games_debug.jsonl")
            else:
                state_dir = os.path.dirname(config.queue_state_path) if config.queue_state_path else "."
                os.makedirs(state_dir, exist_ok=True)
                self._debug_jsonl_path = os.path.join(state_dir, "games_debug.jsonl")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if '_registry' in state:
            del state['_registry']
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        from DatasetPipeline.PositionQueue import PositionQueueRegistry
        self._registry = PositionQueueRegistry.instance(
            state_path=self.config.queue_state_path,
            shard_size=self.config.shard_size,
        )

    def _validate_config(self) -> None:
        pass # [Omissis controlli superflui per brevità, mantieni i tuoi originali se vuoi]

    @staticmethod
    def _init_worker(stockfish_path: str, threads: int, hash_mb: int, syzygy_path: Optional[str]) -> None:
        global _engine, _engine_pid, _tablebase, _watchdog_thread

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

        if syzygy_path:
            try:
                _tablebase = chess.syzygy.open_tablebase(syzygy_path)
            except Exception:
                _tablebase = None

        atexit.register(_close_engine)

        _watchdog_stop.clear()
        _watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
        _watchdog_thread.start()

    @staticmethod
    def _parse_clk(comment: str) -> Optional[float]:
        if not comment: return None
        m = _CLK_RE.search(comment)
        if not m: return None
        h, mi, s = m.groups()
        return int(h) * 3600 + int(mi) * 60 + float(s)

    @staticmethod
    def _parse_emt(comment: str) -> Optional[float]:
        if not comment: return None
        m = _EMT_RE.search(comment)
        if not m: return None
        h, mi, s = m.groups()
        return int(h) * 3600 + int(mi) * 60 + float(s)

    @staticmethod
    def _parse_time_control(time_control: str) -> Tuple[float, float]:
        if not time_control or time_control == "-": return 0.0, 0.0
        m = re.match(r"^(\d+)\+(\d+)$", time_control)
        if m: return float(m.group(1)), float(m.group(2))
        m = re.match(r"^(\d+)$", time_control)
        if m: return float(m.group(1)), 0.0
        return 0.0, 0.0

    @staticmethod
    def _compute_move_duration(previous_clock: Optional[float], current_clock: Optional[float], increment: float) -> Optional[float]:
        if previous_clock is None or current_clock is None: return None
        return max(0.0, previous_clock - current_clock + increment)

    @staticmethod
    def _parse_rating(raw: str) -> Optional[int]:
        if not raw: return None
        try: return int(raw)
        except (TypeError, ValueError):
            digits = "".join(ch for ch in raw if ch.isdigit())
            return int(digits) if digits else None

    def _closest_bucket_time(self, rating: Optional[int]) -> Optional[float]:
        if rating is None or not self.config.avg_time_by_rating: return None
        closest = min(self.config.avg_time_by_rating.keys(), key=lambda b: abs(b - rating))
        return self.config.avg_time_by_rating[closest]

    def _headers_are_eligible(self, headers) -> bool:
        """[OTTIMIZZAZIONE]: Filtra usando i soli metadata, saltando l'albero mosse."""
        cfg = self.config
        if cfg.only_decisive_games:
            result = headers.get("Result", "")
            if result not in ("1-0", "0-1"):
                return False
        
        if cfg.skip_time_forfeit:
            termination = headers.get("Termination", "")
            if "Time forfeit" in termination:
                return False

        if cfg.min_rating is not None or cfg.max_rating is not None:
            white_elo = self._parse_rating(headers.get("WhiteElo", ""))
            black_elo = self._parse_rating(headers.get("BlackElo", ""))
            ratings = [r for r in (white_elo, black_elo) if r is not None]
            if ratings:
                best_rating = max(ratings)
                if cfg.min_rating is not None and best_rating < cfg.min_rating: return False
                if cfg.max_rating is not None and min(ratings) > cfg.max_rating: return False
        return True

    def _get_candidate_legal_moves(self, board: "chess.Board") -> Optional[List["chess.Move"]]:
        cfg = self.config
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material(): return None
        if cfg.max_piece_count is not None and len(board.piece_map()) > cfg.max_piece_count: return None

        legal_moves = list(board.legal_moves)
        if len(legal_moves) < cfg.candidate_min_legal_moves: return None
        if cfg.candidate_max_legal_moves is not None and len(legal_moves) > cfg.candidate_max_legal_moves: return None
        if cfg.skip_if_in_check and board.is_check(): return None
        if len(legal_moves) > 255: return None
        return legal_moves

    def _material_by_color(self, board: "chess.Board") -> Tuple[int, int]:
        white_mat = black_mat = 0
        for p in board.piece_map().values():
            val = self._PIECE_VALUES.get(p.piece_type, 0)
            if p.color == chess.WHITE: white_mat += val
            else: black_mat += val
        return white_mat, black_mat

    def _has_mating_material(self, board: "chess.Board") -> bool:
        cfg = self.config
        mover = board.turn
        white_mat, black_mat = self._material_by_color(board)
        mover_mat = white_mat if mover == chess.WHITE else black_mat
        opp_mat = black_mat if mover == chess.WHITE else white_mat

        if mover_mat < cfg.min_material_for_mate_attempt: return False
        if (mover_mat - opp_mat) < cfg.min_material_diff_for_mate_attempt: return False
        return True

    def _mover_has_heavy_piece(self, board: "chess.Board") -> bool:
        mover = board.turn
        for piece_type in (chess.QUEEN, chess.ROOK):
            if board.pieces(piece_type, mover): return True
        return False

    def _is_trivially_drawn_endgame(self, board: "chess.Board") -> bool:
        piece_map = board.piece_map()
        has_heavy_or_pawn = any(p.piece_type in (chess.QUEEN, chess.ROOK, chess.PAWN) for p in piece_map.values())
        if has_heavy_or_pawn: return False
        white_minors = sum(1 for p in piece_map.values() if p.color == chess.WHITE and p.piece_type in (chess.BISHOP, chess.KNIGHT))
        black_minors = sum(1 for p in piece_map.values() if p.color == chess.BLACK and p.piece_type in (chess.BISHOP, chess.KNIGHT))
        return white_minors <= 1 and black_minors <= 1

    def _syzygy_says_no_mate(self, board: "chess.Board") -> bool:
        global _tablebase
        if _tablebase is None: return False
        if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK): return False
        try: wdl = _tablebase.probe_wdl(board)
        except Exception: return False
        return wdl is not None and wdl <= 0

    def _analyse_position(self, board: "chess.Board"):
        global _engine
        cfg = self.config

        for attempt in range(1, cfg.stockfish_retry_attempts + 1):
            if _engine is None: return None
            try:
                if cfg.analysis_time is not None:
                    limit = chess.engine.Limit(time=cfg.analysis_time, mate=cfg.mate_range[1])
                    _watchdog_arm(cfg.analysis_time)
                else:
                    limit = chess.engine.Limit(depth=cfg.search_depth, mate=cfg.mate_range[1])
                    _watchdog_arm(5.0)
                return _engine.analyse(board, limit, multipv=cfg.multipv)
            except Exception:
                if attempt >= cfg.stockfish_retry_attempts: return None
                if cfg.stockfish_retry_backoff_seconds > 0: time.sleep(cfg.stockfish_retry_backoff_seconds * attempt)
                continue
            finally:
                _watchdog_disarm()
        return None

    def _worker(self, args: Tuple[int, str, str]) -> Tuple[int, bytes]:
        global _engine
        cfg = self.config
        game_id, pgn_text, source_tag = args

        empty_payload = encode_for_ipc([])
        if _engine is None: return game_id, empty_payload

        # [OTTIMIZZAZIONE]: Fase 1: Fast parsing solo degli headers
        pgn_io = io.StringIO(pgn_text)
        headers = chess.pgn.read_headers(pgn_io)
        
        if headers is None or headers.get("Variant", "Standard").lower() not in ("standard", "normal"):
            return game_id, empty_payload
            
        if not self._headers_are_eligible(headers):
            return game_id, empty_payload

        # Fase 2: Parse completo solo se supera i filtri rapidi
        pgn_io.seek(0)
        try:
            game = chess.pgn.read_game(pgn_io)
        except Exception:
            return game_id, empty_payload

        if game is None: return game_id, empty_payload

        try:
            if game.end().ply() < cfg.min_game_plies: return game_id, empty_payload
        except Exception: return game_id, empty_payload

        time_control = game.headers.get("TimeControl", "")
        base_time, increment = self._parse_time_control(time_control)
        white_elo = self._parse_rating(game.headers.get("WhiteElo", ""))
        black_elo = self._parse_rating(game.headers.get("BlackElo", ""))
        mover_rating = {chess.WHITE: white_elo, chess.BLACK: black_elo}

        previous_clock = {
            chess.WHITE: base_time if base_time > 0 else None,
            chess.BLACK: base_time if base_time > 0 else None,
        }

        records: List[Dict[str, Any]] = []
        node = game
        mate_lo, mate_hi = cfg.mate_range
        positions_analysed = 0
        seen_positions: set = set()
        full_game_id = f"{source_tag}_{game_id}"

        try:
            while node.variations:
                next_node = node.variation(0)
                board = node.board()
                comment = next_node.comment or ""
                mover_color = board.turn

                if cfg.dedupe_positions:
                    position_key = " ".join(board.fen().split(" ")[:4])
                    if position_key in seen_positions:
                        node = next_node; continue
                    seen_positions.add(position_key)

                emt_seconds = self._parse_emt(comment)
                current_clock = self._parse_clk(comment)

                if emt_seconds is not None:
                    move_duration = emt_seconds
                    duration_is_real = True
                    clock_source = "real_emt"
                else:
                    move_duration = self._compute_move_duration(previous_clock[mover_color], current_clock, increment)
                    duration_is_real = move_duration is not None
                    clock_source = "real_clk" if duration_is_real else None

                if current_clock is not None: previous_clock[mover_color] = current_clock

                # Filtri di selezione delle posizioni
                if node.ply() < cfg.min_ply:
                    node = next_node; continue
                if (node.ply() - cfg.min_ply) % cfg.ply_sample_step != 0:
                    node = next_node; continue
                if cfg.require_clock and not duration_is_real:
                    node = next_node; continue
                if cfg.max_positions_per_game is not None and positions_analysed >= cfg.max_positions_per_game:
                    break

                legal_moves = self._get_candidate_legal_moves(board)
                if legal_moves is None: node = next_node; continue
                if cfg.skip_forced_moves and len(legal_moves) == 1: node = next_node; continue
                if cfg.require_heavy_piece and not self._mover_has_heavy_piece(board): node = next_node; continue
                if not self._has_mating_material(board): node = next_node; continue
                if cfg.skip_trivial_endgame and self._is_trivially_drawn_endgame(board): node = next_node; continue

                mover_rating_val = mover_rating[mover_color]
                if mover_rating_val is None: node = next_node; continue

                if self._syzygy_says_no_mate(board): node = next_node; continue

                # Analisi Stockfish
                info = self._analyse_position(board)
                positions_analysed += 1
                if not info: node = next_node; continue

                best_info = info[0]
                score = best_info.get("score")
                if score is None: node = next_node; continue
                
                relative_score = score.relative
                if not relative_score.is_mate(): node = next_node; continue
                
                mate_n = relative_score.mate()
                if mate_n is None or not (mate_n > 0 and mate_lo <= mate_n <= mate_hi): node = next_node; continue

                pv = best_info.get("pv")
                if not pv: node = next_node; continue
                
                best_move = pv[0]
                if best_move not in legal_moves: node = next_node; continue

                if duration_is_real: clock_seconds = move_duration
                else:
                    bucket_time = self._closest_bucket_time(mover_rating_val)
                    if bucket_time is not None:
                        clock_seconds = bucket_time; clock_source = "rating_bucket"
                    else:
                        clock_seconds = cfg.default_move_seconds; clock_source = "default_constant"

                if cfg.drop_zero_clock and clock_seconds == 0.0 and not duration_is_real: node = next_node; continue

                try:
                    data = build_position_data(
                        board=board,
                        best_move=best_move,
                        clock_seconds=clock_seconds,
                        rating=float(mover_rating_val),
                        game_id=full_game_id,
                        ply=node.ply(),
                    )
                except ValueError: node = next_node; continue

                debug_entry = {
                    "problem_id": f"{full_game_id}_{node.ply()}",
                    "fen": board.fen(),
                    "best_move_uci": best_move.uci(),
                    "mate_n": int(mate_n),
                    "ply": int(node.ply()),
                    "source": source_tag,
                    "clock_source": clock_source or "unknown",
                    "clock_seconds": float(clock_seconds),
                    "clock_is_real": bool(duration_is_real),
                    "rating": mover_rating_val,
                    "game_id": full_game_id,
                }

                records.append({"data": data, "debug": debug_entry})
                node = next_node

        except Exception:
            pass

        return game_id, encode_for_ipc(records)

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

        raise ValueError(f"_open_pgn_text_stream non applicabile a kind={kind}")

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
                    if max_games is not None and yielded >= max_games: return
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
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
        else:
            df = pd.read_csv(src.path)

        series = df[src.pgn_col].dropna().iloc[src.skip_games:]
        if src.max_games is not None:
            series = series.iloc[: src.max_games]
        for local_id, pgn_text in series.items():
            yield (int(local_id) + 1, pgn_text)

    def _iter_source(self, src: SourceSpec) -> Generator[Tuple[int, str], None, None]:
        if src.kind == "club": yield from self._iter_club_csv(src); return
        with self._open_pgn_text_stream(src.path, src.kind) as text_stream:
            yield from self._iter_pgn_texts(text_stream, src.skip_games, src.max_games)

    def _iter_all_tasks(self) -> Generator[Tuple[int, str, str], None, None]:
        global_id = 0
        for src in self.config.sources:
            for _local_id, pgn_text in self._iter_source(src):
                global_id += 1
                yield (global_id, pgn_text, src.tag)

    def _count_tasks_estimate(self) -> Optional[int]:
        total = 0
        for src in self.config.sources:
            if src.max_games is not None: total += src.max_games
            else: return None
        return total

    def _assign_split(self, game_id: str) -> str:
        import random
        rng = random.Random(self.config.split_seed + hash(game_id))
        val = rng.random()
        train, val_ratio, _ = self.config.split_ratios
        if val < train: return "train"
        if val < train + val_ratio: return "val"
        return "test"

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        harden_process_for_ipc()

        processed_games = accepted_games = enqueued_positions = 0
        mate_n_counts: Dict[int, int] = defaultdict(int)
        source_counts: Dict[str, int] = defaultdict(int)
        clock_source_counts: Dict[str, int] = defaultdict(int)

        pool = mp.Pool(
            processes=self._workers,
            initializer=self._init_worker,
            initargs=(cfg.stockfish_path, cfg.threads, cfg.hash_mb, cfg.syzygy_path),
        )

        estimate = self._count_tasks_estimate()

        def cleanup_after_failure() -> None:
            old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                pool.terminate()
                timeout = cfg.pool_join_timeout if cfg.pool_join_timeout is not None else 15.0
                deadline = time.monotonic() + timeout
                for proc in pool._pool:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0: break
                    proc.join(timeout=remaining)
            finally:
                signal.signal(signal.SIGINT, old_sigint)

        try:
            task_stream = self._iter_all_tasks()
            results = pool.imap_unordered(self._worker, task_stream, chunksize=1)

            for _provisional_game_id, payload in wrap_iter(
                results,
                desc="[GamesBuilder] Analisi partite (multi-sorgente)",
                unit="game",
                total=estimate,
            ):
                processed_games += 1

                records: List[Dict[str, Any]] = decode_from_ipc(payload)
                if not records: continue

                accepted_games += 1

                for rec in records:
                    data = rec["data"]
                    debug_entry = rec["debug"]
                    group_key = debug_entry["mate_n"]

                    self._registry.enqueue(
                        source_tag=debug_entry["source"],
                        data=data,
                        group_key=group_key,
                    )
                    enqueued_positions += 1

                    source_counts[debug_entry["source"]] += 1
                    clock_source_counts[debug_entry["clock_source"]] += 1
                    mate_n_counts[debug_entry["mate_n"]] += 1

                    if cfg.save_debug_jsonl:
                        split_name = self._assign_split(debug_entry["game_id"])
                        self._debug_records[split_name].append(debug_entry)

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
            pool.join()

        self._registry.flush()

        if cfg.save_debug_jsonl and self._debug_jsonl_path:
            import json
            tmp_path = self._debug_jsonl_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                for split in ("train", "val", "test"):
                    for rec in self._debug_records.get(split, []):
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp_path, self._debug_jsonl_path)

        return {
            "processed_games": processed_games,
            "accepted_games": accepted_games,
            "enqueued_positions": enqueued_positions,
            "mate_n_counts": dict(mate_n_counts),
            "source_counts": dict(source_counts),
            "clock_source_counts": dict(clock_source_counts),
        }