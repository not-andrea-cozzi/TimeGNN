from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None

from DatasetPipeline.Builder.GamesBuilder import GamesBuilder, GamesBuilderConfig, SourceSpec
from DatasetPipeline.Builder.PuzzleBuilder import PuzzleBuilder, PuzzleBuilderConfig
from DatasetPipeline.TimeStatBuilder import TimeStatsBuilder, load_avg_time_by_rating
from DatasetPipeline.PositionQueue import PositionQueueRegistry
from DatasetPipeline.PipelineState import PipelineState

logger = logging.getLogger("dataset_main")


class ConfigError(Exception):
    """Errore bloccante di configurazione della pipeline."""


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


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise ConfigError(f"File di configurazione non trovato: {config_path}")
    if yaml is None:
        raise ConfigError("Modulo 'pyyaml' non installato. Esegui: pip install pyyaml")
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except Exception as e:
            raise ConfigError(f"Errore nel parsing YAML ({config_path}): {e}")
    if not isinstance(cfg, dict):
        raise ConfigError("Il file YAML deve definire un dizionario di primo livello.")
    return cfg


def require_executable(path: str) -> None:
    if not (os.path.exists(path) and os.access(path, os.X_OK)):
        raise ConfigError(f"Eseguibile non trovato o non eseguibile: {path}")


# ---------------------------------------------------------------------------
# ShardWriter per lo split finale (streaming su disco)
# ---------------------------------------------------------------------------
class _ShardWriter:
    """Accumula fino a `shard_size` record e li salva in <split>_<NNNNN>.pt.

    RAM di picco: `shard_size` oggetti `Data`. Non materializza mai l'intero
    split in memoria.
    """

    def __init__(self, out_dir: str, split_name: str, shard_size: int) -> None:
        self.out_dir = out_dir
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
        path = os.path.join(self.out_dir, f"{self.split_name}_{self.shard_idx:05d}.pt")
        tmp_path = path + ".tmp"
        torch.save(self.buf, tmp_path)
        os.replace(tmp_path, path)
        self.files.append(path)
        self.buf = []
        self.shard_idx += 1

    def close(self) -> List[str]:
        self._flush()
        manifest = os.path.join(self.out_dir, f"{self.split_name}_index.json")
        tmp_manifest = manifest + ".tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump({"files": self.files, "total": self.total}, f, indent=2)
        os.replace(tmp_manifest, manifest)
        return self.files


# ---------------------------------------------------------------------------
# STEP 1: time_stats
# ---------------------------------------------------------------------------
def _step_time_stats(cfg: Dict[str, Any], state: PipelineState, dataset_dir: str) -> Optional[str]:
    pipe_cfg = cfg.get("pipeline", {})
    ts_cfg = cfg.get("time_stats", {})
    raw_cfg = cfg.get("raw_data", {})

    out_path = os.path.join(dataset_dir, ts_cfg.get("output_filename", "avg_time_by_rating.json"))

    if state.is_done("time_stats", skip=pipe_cfg.get("force_recompute", False)) and os.path.exists(out_path):
        logger.info(f"[time_stats] Gia' completato ({out_path}): skip.")
        return out_path

    games_zst = raw_cfg.get("games_zst")
    if not games_zst:
        logger.warning("[time_stats] raw_data.games_zst non specificato: skip step (avg_time_by_rating resta vuoto).")
        return None
    if not os.path.exists(games_zst):
        raise ConfigError(f"[time_stats] raw_data.games_zst non trovato: {games_zst}")

    builder = TimeStatsBuilder(
        zst_path=games_zst,
        max_games=ts_cfg.get("max_games", 50_000),
        bucket_size=ts_cfg.get("bucket_size", 100),
    )
    try:
        stats = builder.build_and_save(out_path)
        state.mark_done("time_stats", output=out_path, buckets=len(stats))
        logger.info(f"[time_stats] Completato: {len(stats)} bucket -> {out_path}")
        return out_path
    except Exception as e:
        state.mark_failed("time_stats", str(e))
        raise


# ---------------------------------------------------------------------------
# STEP 2: games_pipeline
# ---------------------------------------------------------------------------
def _build_game_sources(raw_cfg: Dict[str, Any], games_cfg: Dict[str, Any]) -> List[SourceSpec]:
    sources: List[SourceSpec] = []

    games_zst = raw_cfg.get("games_zst")
    if games_zst:
        if not os.path.exists(games_zst):
            raise ConfigError(f"raw_data.games_zst non trovato: {games_zst}")
        sources.append(SourceSpec(
            kind="lichess",
            path=games_zst,
            skip_games=games_cfg.get("skip_games_part1", 0),
            max_games=games_cfg.get("max_games") or None,
            tag=raw_cfg.get("games_source_tag", "games_lichess"),
        ))

    fics_pgn = raw_cfg.get("fics_pgn")
    if fics_pgn:
        if not os.path.exists(fics_pgn):
            raise ConfigError(f"raw_data.fics_pgn non trovato: {fics_pgn}")
        sources.append(SourceSpec(
            kind="fics",
            path=fics_pgn,
            skip_games=games_cfg.get("fics_skip_games", 0),
            max_games=games_cfg.get("fics_max_games"),
            tag=raw_cfg.get("fics_source_tag", "fics"),
        ))

    club_csv = raw_cfg.get("club_csv")
    if club_csv:
        if not os.path.exists(club_csv):
            raise ConfigError(f"raw_data.club_csv non trovato: {club_csv}")
        sources.append(SourceSpec(
            kind="club",
            path=club_csv,
            pgn_col=games_cfg.get("club_pgn_col", "pgn"),
            skip_games=games_cfg.get("club_skip_games", 0),
            max_games=games_cfg.get("club_max_games") or None,
            tag=raw_cfg.get("club_source_tag", "club"),
        ))

    if not sources:
        raise ConfigError("games_pipeline: nessuna sorgente valida in raw_data (games_zst/fics_pgn/club_csv).")
    return sources


def _step_games_pipeline(
    cfg: Dict[str, Any], state: PipelineState, dataset_dir: str,
    queue_state_path: str, resume_state_path: str,
) -> Dict[str, Any]:
    pipe_cfg = cfg.get("pipeline", {})
    games_cfg = cfg.get("games_pipeline", {})
    engine_cfg = cfg.get("engine", {})
    raw_cfg = cfg.get("raw_data", {})

    if pipe_cfg.get("use_existing_games", False):
        logger.info("[games_pipeline] use_existing_games=true: skip (uso shard/queue gia' presenti).")
        return {}

    if state.is_done("games_pipeline", skip=pipe_cfg.get("force_recompute", False)):
        logger.info("[games_pipeline] Gia' completato: skip.")
        return state.get_meta("games_pipeline")

    stockfish_path = engine_cfg.get("stockfish_path")
    if not stockfish_path:
        raise ConfigError("engine.stockfish_path mancante nello YAML.")
    require_executable(stockfish_path)

    games_output_dir = os.path.join(dataset_dir, pipe_cfg.get("games_subfolder", "Games"))
    os.makedirs(games_output_dir, exist_ok=True)

    avg_time_by_rating: Dict[int, float] = {}
    time_stats_path = os.path.join(dataset_dir, cfg.get("time_stats", {}).get("output_filename", "avg_time_by_rating.json"))
    if os.path.exists(time_stats_path):
        avg_time_by_rating = load_avg_time_by_rating(time_stats_path)
        logger.info(f"[games_pipeline] avg_time_by_rating caricato: {len(avg_time_by_rating)} bucket.")
    else:
        logger.warning(f"[games_pipeline] {time_stats_path} non trovato: uso default_move_seconds come fallback.")

    mate_range = (games_cfg.get("mate_range_min", 1), games_cfg.get("mate_range_max", 10))
    if mate_range[0] < 1 or mate_range[1] < mate_range[0]:
        raise ConfigError(f"games_pipeline: mate_range non valido: {mate_range}")

    sources = _build_game_sources(raw_cfg, games_cfg)

    gb_config = GamesBuilderConfig(
        sources=sources,
        stockfish_path=stockfish_path,
        mate_range=mate_range,
        search_depth=games_cfg.get("search_depth", 8),
        analysis_time=games_cfg.get("analysis_time", 0.2),
        workers=games_cfg.get("workers"),
        threads=engine_cfg.get("threads", 1),
        hash_mb=engine_cfg.get("hash_mb", 128),
        multipv=1,
        syzygy_path=engine_cfg.get("syzygy_path"),
        stockfish_retry_attempts=games_cfg.get("stockfish_retry_attempts", 2),
        stockfish_retry_backoff_seconds=games_cfg.get("stockfish_retry_backoff_seconds", 0.5),
        candidate_min_legal_moves=games_cfg.get("candidate_min_legal_moves", 1),
        candidate_max_legal_moves=games_cfg.get("candidate_max_legal_moves"),
        skip_if_in_check=games_cfg.get("skip_if_in_check", False),
        max_piece_count=games_cfg.get("max_piece_count", 18),
        min_material_for_mate_attempt=games_cfg.get("min_material_for_mate_attempt", 4),
        min_material_diff_for_mate_attempt=games_cfg.get("min_material_diff_for_mate_attempt", 3),
        require_heavy_piece=games_cfg.get("require_heavy_piece", True),
        skip_forced_moves=games_cfg.get("skip_forced_moves", True),
        skip_trivial_endgame=games_cfg.get("skip_trivial_endgame", True),
        dedupe_positions=games_cfg.get("dedupe_positions", True),
        require_clock=games_cfg.get("require_clock", True),
        default_move_seconds=games_cfg.get("default_move_seconds", 15.0),
        avg_time_by_rating=avg_time_by_rating,
        drop_zero_clock=games_cfg.get("drop_zero_clock", True),
        min_rating=games_cfg.get("min_rating"),
        max_rating=games_cfg.get("max_rating"),
        min_ply=games_cfg.get("min_ply", 8),
        ply_sample_step=games_cfg.get("ply_sample_step", 3),
        max_positions_per_game=games_cfg.get("max_positions_per_game", 20),
        only_decisive_games=games_cfg.get("only_decisive_games", True),
        skip_time_forfeit=games_cfg.get("skip_time_forfeit", True),
        min_game_plies=games_cfg.get("min_game_plies", 20),
        queue_state_path=queue_state_path,
        shard_size=games_cfg.get("shard_size", 5000),
        save_debug_jsonl=games_cfg.get("save_debug_jsonl", True),
        debug_jsonl_dir=games_output_dir,
        split_ratios=(
            cfg.get("splits", {}).get("train_ratio", 0.7),
            cfg.get("splits", {}).get("val_ratio", 0.1),
            cfg.get("splits", {}).get("test_ratio", 0.2),
        ),
        split_seed=pipe_cfg.get("seed", 42),
        pool_join_timeout=games_cfg.get("pool_join_timeout", 20.0),
        auto_resume=True,
        resume_state_path=resume_state_path,
        resume_checkpoint_every=games_cfg.get("checkpoint_every", 2000),
    )

    builder = GamesBuilder(gb_config)
    try:
        result = builder.run()
    except Exception as e:
        state.mark_failed("games_pipeline", str(e))
        raise

    state.mark_done("games_pipeline", **{k: v for k, v in result.items() if isinstance(v, (int, str, float))})
    logger.info(
        f"[games_pipeline] Completato: processed={result.get('processed_games', 0):,}, "
        f"accepted={result.get('accepted_games', 0):,}, enqueued={result.get('enqueued_positions', 0):,}."
    )
    return result


# ---------------------------------------------------------------------------
# STEP 3: puzzle_pipeline
# ---------------------------------------------------------------------------
def _decompress_puzzle_csv_if_needed(raw_cfg: Dict[str, Any], puzzle_cfg: Dict[str, Any], dataset_dir: str) -> Optional[str]:
    puzzles_zst = raw_cfg.get("puzzles_zst")
    if not puzzles_zst:
        return None
    if not os.path.exists(puzzles_zst):
        raise ConfigError(f"raw_data.puzzles_zst non trovato: {puzzles_zst}")

    out_csv = os.path.join(dataset_dir, puzzle_cfg.get("decompressed_csv_filename", "lichess_puzzles.csv"))
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        logger.info(f"[puzzle_pipeline] CSV decompresso gia' presente: {out_csv} (skip decompressione).")
        return out_csv

    import zstandard as zstd
    chunk_size = puzzle_cfg.get("chunk_size_bytes", 1_048_576)
    logger.info(f"[puzzle_pipeline] Decompressione {puzzles_zst} -> {out_csv} ...")
    dctx = zstd.ZstdDecompressor()
    with open(puzzles_zst, "rb") as fin, open(out_csv, "wb") as fout:
        dctx.copy_stream(fin, fout, read_size=chunk_size, write_size=chunk_size)
    logger.info("[puzzle_pipeline] Decompressione completata.")
    return out_csv


def _step_puzzle_pipeline(
    cfg: Dict[str, Any], state: PipelineState, dataset_dir: str,
    queue_state_path: str,
) -> Dict[str, Any]:
    pipe_cfg = cfg.get("pipeline", {})
    puzzle_cfg = cfg.get("puzzle_pipeline", {})
    raw_cfg = cfg.get("raw_data", {})

    if state.is_done("puzzle_pipeline", skip=pipe_cfg.get("force_recompute", False)):
        logger.info("[puzzle_pipeline] Gia' completato: skip.")
        return state.get_meta("puzzle_pipeline")

    csv_path = _decompress_puzzle_csv_if_needed(raw_cfg, puzzle_cfg, dataset_dir)
    if csv_path is None:
        logger.warning("[puzzle_pipeline] raw_data.puzzles_zst non specificato: skip step.")
        return {}

    puzzles_output_dir = os.path.join(dataset_dir, pipe_cfg.get("puzzles_subfolder", "Puzzles"))
    os.makedirs(puzzles_output_dir, exist_ok=True)

    avg_time_by_rating: Dict[int, float] = {}
    time_stats_path = os.path.join(dataset_dir, cfg.get("time_stats", {}).get("output_filename", "avg_time_by_rating.json"))
    if os.path.exists(time_stats_path):
        avg_time_by_rating = load_avg_time_by_rating(time_stats_path)

    mate_range = (
        cfg.get("games_pipeline", {}).get("mate_range_min", 1),
        cfg.get("games_pipeline", {}).get("mate_range_max", 5),
    )

    pb_config = PuzzleBuilderConfig(
        csv_path=csv_path,
        mate_range=mate_range,
        max_puzzles=puzzle_cfg.get("max_puzzles"),
        max_puzzles_per_theme=puzzle_cfg.get("max_puzzles_per_theme"),
        avg_time_by_rating=avg_time_by_rating,
        queue_state_path=queue_state_path,
        shard_size=5000,
        save_debug_jsonl=True,
        debug_jsonl_dir=puzzles_output_dir,
        split_ratios=(
            cfg.get("splits", {}).get("train_ratio", 0.7),
            cfg.get("splits", {}).get("val_ratio", 0.1),
            cfg.get("splits", {}).get("test_ratio", 0.2),
        ),
        split_seed=pipe_cfg.get("seed", 42),
        source_tag=raw_cfg.get("puzzles_source_tag", "puzzle"),
        min_rating=puzzle_cfg.get("min_rating"),
        max_rating=puzzle_cfg.get("max_rating"),
        max_piece_count=puzzle_cfg.get("max_piece_count"),
        min_material_for_mate_attempt=puzzle_cfg.get("min_material_for_mate_attempt", 0),
        min_material_diff_for_mate_attempt=puzzle_cfg.get("min_material_diff_for_mate_attempt", 0),
        require_heavy_piece=puzzle_cfg.get("require_heavy_piece", False),
        skip_trivial_endgame=puzzle_cfg.get("skip_trivial_endgame", True),
        dedupe_positions=puzzle_cfg.get("dedupe_positions", True),
    )

    builder = PuzzleBuilder(pb_config)
    try:
        result = builder.run()
    except Exception as e:
        state.mark_failed("puzzle_pipeline", str(e))
        raise

    state.mark_done("puzzle_pipeline", **{k: v for k, v in result.items() if isinstance(v, (int, str, float))})
    logger.info(
        f"[puzzle_pipeline] Completato: processed={result.get('processed_puzzles', 0):,}, "
        f"accepted={result.get('accepted_puzzles', 0):,}, enqueued={result.get('enqueued_positions', 0):,}."
    )
    return result


# ---------------------------------------------------------------------------
# STEP 4: finalize_splits (streaming: pass 1 metadata + pass 2 scrittura a shard)
# ---------------------------------------------------------------------------
def _step_finalize_splits(
    cfg: Dict[str, Any], state: PipelineState, dataset_dir: str,
    queue_state_path: str,
) -> Dict[str, Any]:
    pipe_cfg = cfg.get("pipeline", {})
    splits_cfg = cfg.get("splits", {})

    if state.is_done("finalize_splits", skip=pipe_cfg.get("force_recompute", False)):
        logger.info("[finalize_splits] Gia' completato: skip.")
        return state.get_meta("finalize_splits")

    merged_dir = os.path.join(dataset_dir, pipe_cfg.get("merged_subfolder", "Train"))
    os.makedirs(merged_dir, exist_ok=True)

    split_ratios = (
        splits_cfg.get("train_ratio", 0.7),
        splits_cfg.get("val_ratio", 0.1),
        splits_cfg.get("test_ratio", 0.2),
    )
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ConfigError(f"splits: le percentuali devono sommare a 1.0 (attuale: {split_ratios}).")

    # quanti Data tenere in RAM per shard di output (abbassa se i Data sono grandi)
    output_shard_size = int(splits_cfg.get("output_shard_size", 20_000))
    seed = pipe_cfg.get("seed", 42)

    registry = PositionQueueRegistry.instance(state_path=queue_state_path)

    # ---- Pass 1: metadata-only, niente Data in RAM ----
    logger.info("[finalize_splits] Pass 1/2: calcolo assegnazione split (metadata-only)...")
    try:
        assignment = registry.build_split_assignment(split_ratios=split_ratios, seed=seed)
    except Exception as e:
        state.mark_failed("finalize_splits", str(e))
        raise

    # ---- Pass 2: stream decompression -> shard writer ----
    logger.info(f"[finalize_splits] Pass 2/2: scrittura a shard (output_shard_size={output_shard_size:,})...")
    writers = {
        name: _ShardWriter(merged_dir, name, output_shard_size)
        for name in ("train", "val", "test")
    }

    try:
        for split_name, data in registry.iter_shard_positions(assignment):
            writers[split_name].append(data)
            del data  # lascia andare subito il Data dopo la scrittura
    except Exception as e:
        state.mark_failed("finalize_splits", str(e))
        raise

    out_paths: Dict[str, List[str]] = {}
    meta: Dict[str, Any] = {}
    for name, w in writers.items():
        files = w.close()
        out_paths[name] = files
        meta[name] = w.total
        logger.info(
            f"[finalize_splits] {name}: {w.total:,} posizioni in {len(files)} shard -> {merged_dir}"
        )

    # ---- Debug jsonl (se presenti) ----
    split_assignment = assignment  # game_id -> split
    games_output_dir = os.path.join(dataset_dir, pipe_cfg.get("games_subfolder", "Games"))
    pending_games_debug = os.path.join(games_output_dir, "games_debug_records.pending.jsonl")
    if os.path.exists(pending_games_debug):
        GamesBuilder.write_debug_jsonl_from_pending(
            pending_games_debug,
            os.path.join(games_output_dir, "games_debug.jsonl"),
            split_assignment,
        )

    puzzles_output_dir = os.path.join(dataset_dir, pipe_cfg.get("puzzles_subfolder", "Puzzles"))
    pending_puzzle_debug = os.path.join(puzzles_output_dir, "puzzle_debug_records.pending.jsonl")
    if os.path.exists(pending_puzzle_debug):
        PuzzleBuilder.write_debug_jsonl_from_pending(
            pending_puzzle_debug,
            os.path.join(puzzles_output_dir, "puzzle_debug.jsonl"),
            split_assignment,
        )

    registry.commit_splits()

    state.mark_done(
        "finalize_splits",
        **meta,
        **{f"{k}_files": v for k, v in out_paths.items()},
    )
    logger.info(f"[finalize_splits] Completato: {meta}")
    return meta


# ---------------------------------------------------------------------------
# ORCHESTRAZIONE
# ---------------------------------------------------------------------------
def main(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)
    pipe_cfg = cfg.get("pipeline", {})

    log_level = pipe_cfg.get("log_level", "INFO")
    log_file = pipe_cfg.get("log_file")
    setup_logging(log_level, log_file)

    logger.info("=" * 70)
    logger.info("DATASET PIPELINE (time_stats -> games -> puzzles -> finalize_splits)")
    logger.info("=" * 70)

    dataset_dir = pipe_cfg.get("dataset_dir", "Dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    state_path = os.path.join(dataset_dir, pipe_cfg.get("state_file", "pipeline_state.json"))
    state = PipelineState(state_path)

    queue_state_path = os.path.join(dataset_dir, "position_queue_state.json")
    resume_state_path = os.path.join(dataset_dir, "games_builder_resume.json")

    step = pipe_cfg.get("step") or "all"
    results: Dict[str, Any] = {}

    steps_to_run = (
        ["time_stats", "games_pipeline", "puzzle_pipeline", "finalize_splits"]
        if step == "all" else [step]
    )

    for s in steps_to_run:
        if s == "time_stats":
            results["time_stats"] = _step_time_stats(cfg, state, dataset_dir)
        elif s == "games_pipeline":
            results["games_pipeline"] = _step_games_pipeline(
                cfg, state, dataset_dir, queue_state_path, resume_state_path
            )
        elif s == "puzzle_pipeline":
            results["puzzle_pipeline"] = _step_puzzle_pipeline(
                cfg, state, dataset_dir, queue_state_path
            )
        elif s == "finalize_splits":
            results["finalize_splits"] = _step_finalize_splits(
                cfg, state, dataset_dir, queue_state_path
            )
        else:
            raise ConfigError(f"pipeline.step sconosciuto: '{s}'")

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETATA")
    logger.info("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline completa: time_stats -> games_pipeline -> puzzle_pipeline -> finalize_splits."
    )
    parser.add_argument("--config", default="Yaml/dataset_main.yaml")
    args = parser.parse_args()

    try:
        main(config_path=args.config)
    except ConfigError as e:
        logging.getLogger("dataset_main").error(f"Errore di configurazione: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        logging.getLogger("dataset_main").warning("Interrotto dall'utente. Rilancia lo stesso comando per riprendere (resume/state).")
        sys.exit(130)
    except Exception as e:
        logging.getLogger("dataset_main").error(f"Interruzione imprevista: {e}")
        sys.exit(1)