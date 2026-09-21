from __future__ import annotations

import io
import json
import logging
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.pgn
import pandas as pd
from tqdm import tqdm

from DatasetPipeline.Model.ChessConstants import PIECE_VALUES
from DatasetPipeline.Model.PositionGraphSchema import build_position_data
from DatasetPipeline.PipelineState import PipelineState
from DatasetPipeline.TimeStatBuilder import load_avg_time_by_rating
from DatasetPipeline.Utils.compatibility_filters import (
    QualityFilterConfig,
    has_mating_material,
    is_trivially_drawn_endgame,
    mover_has_heavy_piece,
    parse_rating_strict,
    validate_kings,
    window_is_forced_single_move_throughout,
)
from DatasetPipeline.Utils.time_edge_weighting import apply_edge_type_time_weighting
from TrainPipeline.CleanDataset import clean_sharded_directory

logger = logging.getLogger("build_external_holdout")


@dataclass
class ExternalHoldoutConfig:
    # Accetta sia un CSV in stile Kaggle Chess.com (colonna 'pgn' per riga)
    # sia un file .pgn testuale con partite concatenate (es. FICS/Lichess
    # non compresso). Il formato viene rilevato dal CONTENUTO del file
    # (non dall'estensione), vedi ChesscomHoldoutBuilder._detect_input_format.
    csv_path: str = "RawData/ficsgamesdb_201001_standard_movetimes_4340144.pgn"
    source_tag: str = "chesscom_holdout"

    # Rilevamento formato: "auto" (default) ispeziona il file; "csv" o
    # "pgn" forzano esplicitamente un parser, saltando il rilevamento
    # (utile se un file .csv contenesse per errore testo PGN o viceversa).
    input_format: str = "auto"

    chunksize: int = 5_000
    max_games: Optional[int] = 20_000

    allowed_rules: Tuple[str, ...] = ("chess",)
    require_rated: bool = False

    stockfish_path: str = "stockfish"
    threads: int = 1
    hash_mb: int = 128
    search_depth: int = 12
    analysis_time: Optional[float] = 0.2
    mate_range: Tuple[int, int] = (1, 10)

    min_game_plies: int = 8

    skip_forced_single_move_window: bool = True
    require_heavy_piece: bool = False
    skip_trivial_endgame: bool = True
    min_material_for_mate_attempt: int = 2
    min_material_diff_for_mate_attempt: int = 2
    max_piece_count: Optional[int] = None
    candidate_min_legal_moves: int = 1
    candidate_max_legal_moves: Optional[int] = None
    skip_if_in_check: bool = False

    min_ply: int = 6
    ply_sample_step: int = 8
    max_positions_per_game: Optional[int] = 10
    dedupe_positions: bool = True

    time_stats_json: Optional[str] = "Dataset/avg_time_by_rating.json"
    default_move_seconds: float = 15.0

    output_dir: str = "Dataset/ExternalHoldout"
    output_shard_size: int = 20_000
    target_clean_shard_size: int = 2_000
    save_debug_jsonl: bool = True

    state_file: str = "external_holdout_state.json"
    force_recompute: bool = False

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

        self._engine = None
        self._input_format = self._detect_input_format()
        logger.info(f"[holdout] Formato input rilevato: '{self._input_format}' (da '{self.config.csv_path}').")

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
        require_executable(cfg.stockfish_path)

    def _detect_input_format(self) -> str:
        """Determina se csv_path va letto come CSV (stile Kaggle
        Chess.com, colonna 'pgn' per riga) o come file .pgn testuale
        (partite concatenate, header '[Event ...]').

        Se config.input_format e' 'csv' o 'pgn', quella scelta e'
        rispettata senza ispezionare il file. Con 'auto' (default), il
        rilevamento guarda il CONTENUTO, non l'estensione: un file .pgn
        rinominato .csv (o viceversa) viene comunque riconosciuto
        correttamente, a differenza di un dispatch basato solo su
        os.path.splitext.
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
                    # Prima riga non vuota, non PGN: trattala come header
                    # CSV e ferma l'ispezione (una riga basta per un header).
                    break
        except OSError as e:
            raise ConfigError(f"Impossibile leggere '{cfg.csv_path}' per rilevare il formato: {e}")

        return "csv"

    def _start_engine(self) -> None:
        import chess.engine
        self._engine = chess.engine.SimpleEngine.popen_uci(self.config.stockfish_path)
        self._engine.configure({"Threads": self.config.threads, "Hash": self.config.hash_mb})

    def _stop_engine(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def _analyse(self, board: "chess.Board"):
        import chess.engine
        if self._engine is None:
            return None
        try:
            if self.config.analysis_time is not None:
                limit = chess.engine.Limit(time=self.config.analysis_time, mate=self.config.mate_range[1])
            else:
                limit = chess.engine.Limit(depth=self.config.search_depth, mate=self.config.mate_range[1])
            return self._engine.analyse(board, limit)
        except Exception:
            return None

    def _iter_rows(self):
        """Dispatcher unico: produce record in un formato uniforme
        (un dict con almeno la chiave 'pgn'), indipendentemente dal fatto
        che l'input sia un CSV Kaggle o un file .pgn testuale. Tutto il
        codice a valle (_process_game) legge sempre row['pgn'],
        row.get('white_rating'), row.get('black_rating'), row.get('time_class'):
        non deve sapere da quale formato il record proviene.
        """
        if self._input_format == "pgn":
            yield from self._iter_rows_from_pgn()
        else:
            yield from self._iter_rows_from_csv()

    def _iter_rows_from_csv(self):
        reader = pd.read_csv(self.config.csv_path, chunksize=self.config.chunksize)
        yielded = 0
        for chunk in reader:
            if "rules" in chunk.columns:
                chunk = chunk[chunk["rules"].astype(str).str.lower().isin(self.config.allowed_rules)]
            if self.config.require_rated and "rated" in chunk.columns:
                chunk = chunk[chunk["rated"].astype(str).str.lower() == "true"]
            for record in chunk.to_dict("records"):
                yield record
                yielded += 1
                if self.config.max_games is not None and yielded >= self.config.max_games:
                    return

    def _iter_rows_from_pgn(self):
        """Legge un file .pgn testuale con partite concatenate, spezzando
        sul marcatore '[Event ' che apre ogni nuova partita (stessa logica
        di GamesBuilder._iter_pgn_texts). Ogni partita diventa un record
        con chiave 'pgn' contenente il testo PGN completo (header + mosse),
        cosi' _process_game puo' trattarlo esattamente come farebbe con
        row['pgn'] proveniente da un CSV.

        require_rated e allowed_rules (pensati per le colonne 'rated' e
        'rules' del CSV Kaggle) non hanno un equivalente diretto negli
        header PGN standard: vengono ignorati qui, loggando un avviso una
        sola volta se richiesti esplicitamente, invece di fallire
        silenziosamente o filtrare in modo scorretto.
        """
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
                        yield record
                        yielded += 1
                        if cfg.max_games is not None and yielded >= cfg.max_games:
                            return
                    current_game_lines = [line]
                else:
                    current_game_lines.append(line)

            if current_game_lines and (cfg.max_games is None or yielded < cfg.max_games):
                record = _flush_current()
                if record is not None:
                    yield record

    def _simulated_clock(self, rating: Optional[float], ply_idx: int) -> float:
        base = self.config.default_move_seconds
        if rating is not None and self._avg_time_by_rating:
            bucket = round(rating / 100) * 100
            base = self._avg_time_by_rating.get(bucket, base)
        noise = random.Random(f"holdout:{ply_idx}").gauss(1.0, 0.15)
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

    def _process_game(self, local_id: int, row: Dict[str, Any]) -> List[Any]:
        cfg = self.config
        pgn_text = row.get("pgn")
        if not isinstance(pgn_text, str) or not pgn_text.strip():
            return []

        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text))
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

        # row['white_rating']/row['black_rating'] esistono solo per record
        # provenienti dal CSV Kaggle; per record da .pgn (dove il dict ha
        # solo la chiave 'pgn', vedi _iter_rows_from_pgn) row.get(...)
        # ritorna None e si cade direttamente sugli header PGN sotto.
        white_rating = parse_rating_strict(str(row.get("white_rating") or "")) or parse_rating_strict(
            game.headers.get("WhiteElo", "")
        )
        black_rating = parse_rating_strict(str(row.get("black_rating") or "")) or parse_rating_strict(
            game.headers.get("BlackElo", "")
        )
        mover_rating = {chess.WHITE: white_rating, chess.BLACK: black_rating}

        full_game_id = f"{cfg.source_tag}_{local_id}"
        collected: List[Any] = []
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
            if not self._position_passes_quality_filters(board):
                node = next_node
                continue

            mover_rating_val = mover_rating[mover_color]
            if mover_rating_val is None:
                node = next_node
                continue

            info = self._analyse(board)
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
            lo, hi = cfg.mate_range
            if mate_n is None or not (mate_n > 0 and lo <= mate_n <= hi):
                node = next_node
                continue

            pv = info.get("pv")
            if not pv:
                node = next_node
                continue
            best_move = pv[0]
            if best_move not in legal_moves:
                node = next_node
                continue

            if cfg.skip_forced_single_move_window and window_is_forced_single_move_throughout([board]):
                node = next_node
                continue

            clock_seconds = self._simulated_clock(mover_rating_val, node.ply())

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

            collected.append(data)

            if cfg.save_debug_jsonl:
                self._debug_records.append({
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
                })

            node = next_node

        return collected

    def _write_debug_jsonl(self) -> Optional[str]:
        if not self._debug_jsonl_path or not self._debug_records:
            return None
        tmp_path = self._debug_jsonl_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in self._debug_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self._debug_jsonl_path)
        return self._debug_jsonl_path

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        processed_games = 0
        accepted_games = 0
        enqueued_positions = 0
        mate_n_counts: Dict[int, int] = defaultdict(int)

        writer = _ShardWriter(cfg.output_dir, "holdout_raw", cfg.output_shard_size)

        self._start_engine()
        try:
            rows = self._iter_rows()
            for local_id, row in enumerate(
                tqdm(rows, desc="[ExternalHoldout] Analisi partite Chess.com", unit="game"), start=1
            ):
                processed_games += 1
                try:
                    positions = self._process_game(local_id, row)
                except Exception:
                    logger.exception(f"[holdout] game locale={local_id}: eccezione non gestita, partita scartata.")
                    positions = []

                if positions:
                    accepted_games += 1
                    enqueued_positions += len(positions)
                    for data in positions:
                        writer.append(data)
        finally:
            self._stop_engine()

        raw_files = writer.close()

        for rec in self._debug_records:
            mate_n_counts[rec["mate_n"]] += 1

        if cfg.save_debug_jsonl:
            self._write_debug_jsonl()

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
            "raw_dir": raw_dir,
            "raw_files": raw_files,
            "clean_dir": clean_dir,
            "clean_manifest": clean_manifest,
        }


def _step_build_external_holdout(cfg: ExternalHoldoutConfig, state: PipelineState) -> Dict[str, Any]:
    clean_manifest_path = os.path.join(cfg.output_dir, "holdout_clean", "manifest.json")

    def _is_ready() -> bool:
        return os.path.exists(clean_manifest_path) and os.path.getsize(clean_manifest_path) > 0

    if state.is_done("external_holdout", skip=cfg.force_recompute) and _is_ready():
        logger.info("[external_holdout] Gia' completato: skip.")
        return state.get_meta("external_holdout")

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
        logging.getLogger("build_external_holdout").warning("Interrotto dall'utente.")
        sys.exit(130)
    except Exception as e:
        logging.getLogger("build_external_holdout").error(f"Interruzione imprevista: {e}")
        sys.exit(1)