from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import zstandard as zstd
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None

from DatasetPipeline.Builder.GamesBuilder import GamesBuilder, GamesBuilderConfig, SourceSpec
from DatasetPipeline.Builder.PuzzleBuilder import PuzzleBuilder, PuzzleBuilderConfig
from DatasetPipeline.PositionQueue import PositionQueueRegistry
from DatasetPipeline.TimeStatBuilder import TimeStatsBuilder, load_avg_time_by_rating
from DatasetPipeline.PipelineState import PipelineState, retry, file_ready, torch_pt_ready

logger = logging.getLogger("main")


class PipelineConfigError(Exception):
    """Errore bloccante di configurazione o validazione parametri."""


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
    logger.info(f"Logging inizializzato: livello={log_level.upper()}, file={log_file or '(nessuno, solo stdout)'}.")


def require_file(path: str, hint: str = "") -> None:
    if not os.path.exists(path):
        msg = f"File richiesto non trovato: {path}."
        if hint:
            msg += f" {hint}"
        raise PipelineConfigError(msg)


def require_executable(path: str, hint: str = "") -> None:
    if not (os.path.exists(path) and os.access(path, os.X_OK)):
        msg = f"Eseguibile non trovato o permessi di esecuzione mancanti: {path}."
        if hint:
            msg += f" {hint}"
        raise PipelineConfigError(msg)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    logger.info(f"Caricamento configurazione YAML da '{config_path}'...")
    require_file(config_path, "Specificare un file YAML valido tramite --config.")
    if yaml is None:
        raise PipelineConfigError(
            "Modulo 'pyyaml' non installato. Esegui: pip install pyyaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except Exception as e:
            raise PipelineConfigError(f"Errore nel parsing del file YAML ({config_path}): {e}")
    if not isinstance(cfg, dict):
        raise PipelineConfigError("Il file di configurazione YAML deve definire un dizionario di primo livello.")
    logger.info(f"Configurazione caricata: {len(cfg)} sezioni di primo livello ({', '.join(cfg.keys())}).")
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    logger.info("Validazione configurazione...")
    required_sections = [
        "pipeline", "engine", "raw_data", "time_stats",
        "games_pipeline", "puzzle_pipeline", "splits",
    ]
    for section in required_sections:
        if section not in cfg:
            raise PipelineConfigError(f"Sezione mancante nel file YAML: '{section}'.")

    splits = cfg.get("splits", {})
    t_ratio = splits.get("train_ratio", 0.8)
    v_ratio = splits.get("val_ratio", 0.1)
    te_ratio = splits.get("test_ratio", 0.1)
    total = round(t_ratio + v_ratio + te_ratio, 5)
    if total != 1.0:
        raise PipelineConfigError(f"La somma degli split ratio deve essere 1.0 (attuale: {total}).")

    m_train = (cfg["games_pipeline"].get("mate_range_min", 1), cfg["games_pipeline"].get("mate_range_max", 5))
    if m_train[0] > m_train[1] or m_train[0] < 1:
        raise PipelineConfigError(f"Range mate non valido per train games: {m_train}")

    logger.info(
        f"Configurazione valida: split_ratios=({t_ratio}, {v_ratio}, {te_ratio}), "
        f"mate_range={m_train}."
    )


@retry(max_attempts=3, base_delay=3.0, exceptions=(OSError, zstd.ZstdError))
def decompress_zst_csv(zst_path: str, out_csv: str, chunk_size: int = 1024 * 1024) -> str:
    require_file(zst_path, "Verifica il percorso del file compresso dei puzzle Lichess.")

    if file_ready(out_csv):
        logger.info(f"{out_csv} presente e valido, decompressione saltata.")
        return out_csv

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    tmp_out = out_csv + ".tmp"

    total_size = os.path.getsize(zst_path)
    logger.info(f"Decompressione di '{zst_path}' ({total_size / (1024 * 1024):.2f} MB) -> '{out_csv}'...")
    dctx = zstd.ZstdDecompressor()

    t0 = time.monotonic()
    try:
        with open(zst_path, "rb") as f_in, open(tmp_out, "wb") as f_out:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Decomprimo {os.path.basename(zst_path)}") as pbar:
                reader = dctx.stream_reader(f_in)
                while True:
                    chunk = reader.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.n = f_in.tell()
                    pbar.refresh()
        os.replace(tmp_out, out_csv)
    except BaseException:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        raise
    elapsed = time.monotonic() - t0
    out_size = os.path.getsize(out_csv) / (1024 * 1024)
    logger.info(f"Decompressione completata in {elapsed:.2f}s: '{out_csv}' ({out_size:.2f} MB).")
    return out_csv


def run_step(state: PipelineState, step_name: str, is_ready_fn, do_fn) -> None:
    if state.is_done(step_name) and is_ready_fn():
        logger.info(f"[SKIP] Step '{step_name}' gia' completato e verificato.")
        return
    if state.is_done(step_name) and not is_ready_fn():
        logger.warning(f"[REDO] Step '{step_name}' marcato completato ma output mancante o non valido. Rieseguo.")

    logger.info(f"[RUN] Avvio step '{step_name}'...")
    t0 = time.monotonic()
    try:
        do_fn()
    except PipelineConfigError:
        state.mark_failed(step_name, "Config Error")
        logger.error(f"[FAILED] Step '{step_name}' interrotto per errore di configurazione.")
        raise
    except Exception as e:
        state.mark_failed(step_name, str(e))
        logger.error(f"[FAILED] Step '{step_name}' interrotto: {type(e).__name__}: {e}")
        raise
    elapsed = time.monotonic() - t0
    state.mark_done(step_name)
    logger.info(f"[DONE] Step '{step_name}' terminato con successo in {elapsed:.2f}s.")


VALID_STEPS = [
    "time_stats", "games_pipeline", "decompress_puzzles",
    "build_puzzles", "finalize_splits",
]


def _build_sources(raw_cfg: Dict[str, Any], games_cfg: Dict[str, Any]) -> List[SourceSpec]:
    sources: List[SourceSpec] = []

    lichess_path = raw_cfg.get("games_zst")
    if lichess_path and os.path.exists(lichess_path):
        sources.append(
            SourceSpec(
                kind="lichess",
                path=lichess_path,
                skip_games=games_cfg.get("skip_games_part1", 0),
                max_games=games_cfg.get("max_games", 200000),
                tag="lichess",
            )
        )
        logger.info(f"Sorgente Lichess aggiunta: '{lichess_path}' (max_games={games_cfg.get('max_games', 200000)}).")
    else:
        logger.info("raw_data.games_zst non configurato o file assente: sorgente Lichess saltata.")

    fics_path = raw_cfg.get("fics_pgn")
    if fics_path and os.path.exists(fics_path):
        sources.append(
            SourceSpec(
                kind="fics",
                path=fics_path,
                skip_games=games_cfg.get("fics_skip_games", 0),
                max_games=games_cfg.get("fics_max_games"),
                tag="fics",
            )
        )
        logger.info(f"Sorgente FICS aggiunta: '{fics_path}'.")
    else:
        logger.info("raw_data.fics_pgn non configurato o file assente: sorgente FICS saltata.")

    club_path = raw_cfg.get("club_csv")
    if club_path and os.path.exists(club_path):
        sources.append(
            SourceSpec(
                kind="club",
                path=club_path,
                pgn_col=games_cfg.get("club_pgn_col", "pgn"),
                skip_games=games_cfg.get("club_skip_games", 0),
                max_games=games_cfg.get("club_max_games"),
                tag="club",
            )
        )
        logger.info(f"Sorgente Club aggiunta: '{club_path}'.")
    else:
        logger.info("raw_data.club_csv non configurato o file assente: sorgente Club saltata.")

    logger.info(f"Totale sorgenti configurate per games_pipeline: {len(sources)}.")
    return sources


def main(config_path: str = "Yaml/main.yaml") -> None:
    cfg = load_yaml_config(config_path)
    validate_config(cfg)

    pipe_cfg = cfg["pipeline"]
    engine_cfg = cfg["engine"]
    raw_cfg = cfg["raw_data"]
    stats_cfg = cfg["time_stats"]
    games_cfg = cfg["games_pipeline"]
    puzzle_cfg = cfg["puzzle_pipeline"]
    split_cfg = cfg["splits"]

    log_level = pipe_cfg.get("log_level", "INFO")
    log_file = pipe_cfg.get("log_file")
    setup_logging(log_level, log_file)

    logger.info("=" * 70)
    logger.info("AVVIO PIPELINE DATASET TIMEGNN (con PositionQueueRegistry)")
    logger.info("=" * 70)

    step = pipe_cfg.get("step")
    if step is not None and step not in VALID_STEPS:
        raise PipelineConfigError(
            f"'pipeline.step' non valido: '{step}'. Valori ammessi: {VALID_STEPS} oppure null/assente per l'intera pipeline."
        )
    logger.info(f"Modalita': {'step singolo=' + step if step else 'pipeline completa (tutti gli step)'}.")

    stockfish_path = engine_cfg["stockfish_path"]
    dataset_dir = pipe_cfg.get("dataset_dir", "Dataset")
    merged_dir = os.path.join(dataset_dir, pipe_cfg.get("merged_subfolder", "Train"))
    games_output_dir = os.path.join(dataset_dir, pipe_cfg.get("games_subfolder", "Games"))
    puzzles_dir = os.path.join(dataset_dir, pipe_cfg.get("puzzles_subfolder", "Puzzles"))

    for d in (dataset_dir, games_output_dir, puzzles_dir, merged_dir):
        os.makedirs(d, exist_ok=True)
    logger.info(
        f"Directory pipeline pronte: dataset_dir='{dataset_dir}', "
        f"games_output_dir='{games_output_dir}', puzzles_dir='{puzzles_dir}', "
        f"merged_dir='{merged_dir}'."
    )

    state_file = pipe_cfg.get("state_file", "pipeline_state.json")
    state_path = os.path.join(dataset_dir, state_file)

    queue_state_path = os.path.join(dataset_dir, pipe_cfg.get("queue_state_file", "position_queue_state.json"))

    force = pipe_cfg.get("force_recompute", False)
    if force and os.path.exists(state_path):
        logger.warning(f"Flag FORCE attivo: azzeramento stato precedente ('{state_path}').")
        os.remove(state_path)
        # Reset anche il registry
        if os.path.exists(queue_state_path):
            os.remove(queue_state_path)
        spool_dir = os.path.join(os.path.dirname(queue_state_path), "position_queue_state_spool")
        if os.path.exists(spool_dir):
            import shutil
            shutil.rmtree(spool_dir)

    state = PipelineState(state_path)
    logger.info(f"Stato pipeline caricato da '{state_path}'.")

    time_stats_path = os.path.join(dataset_dir, stats_cfg.get("output_filename", "avg_time_by_rating.json"))
    puzzle_csv_path = os.path.join(dataset_dir, puzzle_cfg.get("decompressed_csv_filename", "lichess_puzzles.csv"))

    mate_train_range = (games_cfg.get("mate_range_min", 1), games_cfg.get("mate_range_max", 5))
    split_ratios = (
        split_cfg.get("train_ratio", 0.8),
        split_cfg.get("val_ratio", 0.1),
        split_cfg.get("test_ratio", 0.1),
    )
    logger.info(f"mate_train_range={mate_train_range}, split_ratios={split_ratios}.")

    ctx: Dict[str, Any] = {}

    # --------------------------------------------------------------------
    # STEP 1: time_stats
    # --------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 1/5: time_stats")
    logger.info("-" * 70)

    games_zst_path = raw_cfg.get("games_zst")
    games_zst_available = bool(games_zst_path) and os.path.exists(games_zst_path)

    def _step_time_stats() -> None:
        require_file(raw_cfg["games_zst"], "File PGN compresso necessario per il calcolo delle durate mosse.")
        logger.info(
            f"Costruzione TimeStatsBuilder: zst_path='{raw_cfg['games_zst']}', "
            f"max_games={stats_cfg.get('max_games', 50000)}, bucket_size={stats_cfg.get('bucket_size', 100)}."
        )
        builder = TimeStatsBuilder(
            zst_path=raw_cfg["games_zst"],
            max_games=stats_cfg.get("max_games", 50000),
            bucket_size=stats_cfg.get("bucket_size", 100),
        )
        stats = builder.build_and_save(time_stats_path)
        ctx["avg_time_by_rating"] = stats
        logger.info(f"time_stats: {len(stats)} bucket di rating calcolati, salvati in '{time_stats_path}'.")

    if not games_zst_available and not file_ready(time_stats_path):
        logger.info(
            "raw_data.games_zst non configurato o file assente: step 'time_stats' "
            "saltato, avg_time_by_rating restera' vuoto (fallback su "
            "default_move_seconds/clock costante nei builder a valle)."
        )
        ctx.setdefault("avg_time_by_rating", {})
    elif games_zst_available and (step is None or step == "time_stats"):
        run_step(state, "time_stats", is_ready_fn=lambda: file_ready(time_stats_path), do_fn=_step_time_stats)

    if file_ready(time_stats_path) and "avg_time_by_rating" not in ctx:
        ctx["avg_time_by_rating"] = load_avg_time_by_rating(time_stats_path)
        logger.info(f"avg_time_by_rating ricaricato da '{time_stats_path}': {len(ctx['avg_time_by_rating'])} bucket.")

    # --------------------------------------------------------------------
    # STEP 2: games_pipeline
    # --------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 2/5: games_pipeline (GamesBuilder con PositionQueueRegistry)")
    logger.info("-" * 70)

    def _step_games_pipeline() -> None:
        require_executable(stockfish_path, "Eseguibile Stockfish mancante o non avviabile.")

        sources = _build_sources(raw_cfg, games_cfg)
        if not sources:
            logger.warning(
                "Nessuna sorgente (Lichess, FICS o Club) configurata o trovata: "
                "step 'games_pipeline' non produce alcuna posizione."
            )
            ctx["games_result"] = {"processed_games": 0, "accepted_games": 0, "enqueued_positions": 0}
            return

        gb_config = GamesBuilderConfig(
            sources=sources,
            stockfish_path=stockfish_path,
            mate_range=mate_train_range,
            search_depth=games_cfg.get("search_depth", 8),
            analysis_time=games_cfg.get("time_limit_seconds", 0.2),
            workers= 8, #games_cfg.get("workers", 8),
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

            require_clock=games_cfg.get("require_clock", False),
            default_move_seconds=games_cfg.get("default_move_seconds", 15.0),
            avg_time_by_rating=ctx.get("avg_time_by_rating", {}),
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
            shard_size=games_cfg.get("shard_size", 500),

            save_debug_jsonl=games_cfg.get("save_debug_jsonl", True),
            debug_jsonl_dir=games_output_dir,

            split_ratios=split_ratios,
            split_seed=pipe_cfg.get("seed", 42),
        )

        builder = GamesBuilder(gb_config)
        logger.info("GamesBuilder pronto, avvio run()...")
        t0 = time.monotonic()
        result = builder.run()
        elapsed = time.monotonic() - t0

        ctx["games_result"] = result
        logger.info(
            f"games_pipeline completato in {elapsed:.2f}s: "
            f"{result.get('processed_games', 0):,} partite elaborate, "
            f"{result.get('accepted_games', 0):,} partite accettate, "
            f"{result.get('enqueued_positions', 0):,} posizioni accodate."
        )
        if result.get("source_counts"):
            logger.info("Riepilogo per sorgente durante games_pipeline:")
            for tag, count in sorted(result["source_counts"].items(), key=lambda kv: -kv[1]):
                logger.info(f"    {tag}: {count:,}")

    if step is None or step == "games_pipeline":
        run_step(
            state,
            "games_pipeline",
            is_ready_fn=lambda: state.is_done("games_pipeline"),
            do_fn=_step_games_pipeline,
        )

    # --------------------------------------------------------------------
    # STEP 3: decompress_puzzles
    # --------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 3/5: decompress_puzzles")
    logger.info("-" * 70)

    def _step_decompress_puzzles() -> None:
        decompress_zst_csv(
            zst_path=raw_cfg["puzzles_zst"],
            out_csv=puzzle_csv_path,
            chunk_size=puzzle_cfg.get("chunk_size_bytes", 1024 * 1024),
        )

    puzzles_zst_path = raw_cfg.get("puzzles_zst")
    puzzles_available = bool(puzzles_zst_path) and os.path.exists(puzzles_zst_path)

    if not puzzles_available and not file_ready(puzzle_csv_path):
        logger.info(
            "raw_data.puzzles_zst non configurato o file assente: step "
            "'decompress_puzzles' e 'build_puzzles' saltati."
        )
    elif step is None or step == "decompress_puzzles":
        run_step(
            state,
            "decompress_puzzles",
            is_ready_fn=lambda: file_ready(puzzle_csv_path),
            do_fn=_step_decompress_puzzles,
        )

    # --------------------------------------------------------------------
    # STEP 4: build_puzzles
    # --------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 4/5: build_puzzles (PuzzleBuilder con PositionQueueRegistry)")
    logger.info("-" * 70)

    def _step_build_puzzles() -> None:
        if "avg_time_by_rating" not in ctx:
            ctx["avg_time_by_rating"] = (
                load_avg_time_by_rating(time_stats_path) if file_ready(time_stats_path) else {}
            )

        pb_config = PuzzleBuilderConfig(
            csv_path=puzzle_csv_path,
            mate_range=mate_train_range,
            max_puzzles=puzzle_cfg.get("max_puzzles", 100000),
            avg_time_by_rating=ctx["avg_time_by_rating"],
            chunksize=puzzle_cfg.get("chunksize", 50000),
            queue_state_path=queue_state_path,
            shard_size=puzzle_cfg.get("shard_size", 500),
            save_debug_jsonl=puzzle_cfg.get("save_debug_jsonl", True),
            debug_jsonl_dir=puzzles_dir,
            split_ratios=split_ratios,
            split_seed=pipe_cfg.get("seed", 42),
            max_positions_per_puzzle=puzzle_cfg.get("max_positions_per_puzzle"),
        )

        builder = PuzzleBuilder(pb_config)
        logger.info("PuzzleBuilder pronto, avvio run()...")
        t0 = time.monotonic()
        result = builder.run()
        elapsed = time.monotonic() - t0
        ctx["puzzle_result"] = result
        logger.info(
            f"build_puzzles completato in {elapsed:.2f}s: "
            f"{result.get('accepted_puzzles', 0):,} puzzle accettati, "
            f"{result.get('enqueued_positions', 0):,} posizioni accodate."
        )

    if file_ready(puzzle_csv_path):
        if step is None or step == "build_puzzles":
            run_step(
                state,
                "build_puzzles",
                is_ready_fn=lambda: state.is_done("build_puzzles"),
                do_fn=_step_build_puzzles,
            )
    else:
        logger.info("puzzle_csv non pronto: step 'build_puzzles' saltato.")

    # --------------------------------------------------------------------
    # STEP 5: finalize_splits
    # --------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 5/5: finalize_splits (merge games + puzzles e salvataggio)")
    logger.info("-" * 70)

    final_paths = {
        split: os.path.join(merged_dir, f"{split}.pt")
        for split in ("train", "val", "test")
    }

    def _step_finalize_splits() -> None:
        registry = PositionQueueRegistry.instance(state_path=queue_state_path)
        logger.info("Drenaggio registry e calcolo split stratificato...")
        splits = registry.build_splits(split_ratios=split_ratios, seed=pipe_cfg.get("seed", 42))

        os.makedirs(merged_dir, exist_ok=True)
        for split_name, data_list in splits.items():
            out_path = final_paths[split_name]
            tmp_path = out_path + ".tmp"
            torch.save(data_list, tmp_path)
            os.replace(tmp_path, out_path)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            logger.info(f"Salvato {split_name}: {len(data_list)} posizioni in '{out_path}' ({size_mb:.2f} MB).")

        if not splits.get("train"):
            raise PipelineConfigError("Split 'train' vuoto: nessuna posizione disponibile.")

    if step is None or step == "finalize_splits":
        run_step(
            state,
            "finalize_splits",
            is_ready_fn=lambda: all(torch_pt_ready(p) for p in final_paths.values()),
            do_fn=_step_finalize_splits,
        )

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETATA CON SUCCESSO")
    logger.info(f"Dataset finale (train/val/test) in: {merged_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Costruzione dataset TimeGNN (schema board-level con PositionQueueRegistry).")
    parser.add_argument("--config", default="Yaml/dataset_main.yaml", help="Percorso del file YAML di configurazione.")
    args = parser.parse_args()

    try:
        main(config_path=args.config)
    except PipelineConfigError as e:
        logger.error(f"Errore di configurazione pipeline: {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Interruzione imprevista della pipeline: {e}")
        logger.error("Rilanciare lo script per riprendere dall'ultimo checkpoint valido.")
        sys.exit(1)