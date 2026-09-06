"""

STEP DELLA PIPELINE (con resume via PipelineState)
=====================================================================
    1. time_stats             — statistiche tempo medio per rating da PGN
                                 (opzionale: se raw_data.games_zst manca,
                                 saltato, avg_time_by_rating resta vuoto).
    2. games_pipeline          — GamesBuilder.run(): scansiona le sorgenti
                                 partite, accoda posizioni sul registry
                                 condiviso. NON produce ancora un .pt: le
                                 posizioni restano nel registry (in-memory
                                 + shard su disco per resume, vedi
                                 PositionQueue.py) fino a finalize_splits.
    3. decompress_puzzles      — decomprime il CSV puzzle Lichess (.zst).
    4. build_puzzles           — PuzzleBuilder.run(): accoda le posizioni
                                 puzzle sullo STESSO registry condiviso,
                                 con game_id_start dal risultato di
                                 games_pipeline.
    5. finalize_splits         — UNICO punto in cui si chiama
                                 PositionQueueRegistry.build_splits():
                                 drena il registry (posizioni di games E
                                 puzzle insieme) e produce/salva
                                 train.pt/val.pt/test.pt.

"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import zstandard as zstd
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None

from DatasetPipeline.Builder.GamesBuilder import GamesBuilder, SourceSpec
from DatasetPipeline.Builder.PuzzleBuilder import PuzzleBuilder
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
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
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


@retry(max_attempts=3, base_delay=3.0, exceptions=(OSError, zstd.ZstdError))
def decompress_zst_csv(zst_path: str, out_csv: str, chunk_size: int = 1024 * 1024) -> str:
    require_file(zst_path, "Verifica il percorso del file compresso dei puzzle Lichess.")

    if file_ready(out_csv):
        logger.info(f"{out_csv} presente e valido, decompressione saltata.")
        return out_csv

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    tmp_out = out_csv + ".tmp"

    total_size = os.path.getsize(zst_path)
    dctx = zstd.ZstdDecompressor()

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
    return out_csv


def run_step(state: PipelineState, step_name: str, is_ready_fn, do_fn) -> None:
    if state.is_done(step_name) and is_ready_fn():
        logger.info(f"[SKIP] Step '{step_name}' gia' completato e verificato.")
        return
    if state.is_done(step_name) and not is_ready_fn():
        logger.warning(f"[REDO] Step '{step_name}' marcato completato ma output mancante o non valido. Rieseguo.")

    logger.info(f"[RUN] Avvio step '{step_name}'...")
    try:
        do_fn()
    except PipelineConfigError:
        state.mark_failed(step_name, "Config Error")
        raise
    except Exception as e:
        state.mark_failed(step_name, str(e))
        raise
    state.mark_done(step_name)
    logger.info(f"[DONE] Step '{step_name}' terminato con successo.")


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
    else:
        logger.info("raw_data.club_csv non configurato o file assente: sorgente Club saltata.")

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

    step = pipe_cfg.get("step")
    if step is not None and step not in VALID_STEPS:
        raise PipelineConfigError(
            f"'pipeline.step' non valido: '{step}'. Valori ammessi: {VALID_STEPS} oppure null/assente per l'intera pipeline."
        )

    stockfish_path = engine_cfg["stockfish_path"]
    dataset_dir = pipe_cfg.get("dataset_dir", "Dataset")
    puzzles_dir = os.path.join(dataset_dir, pipe_cfg.get("puzzles_subfolder", "Puzzles"))
    merged_dir = os.path.join(dataset_dir, pipe_cfg.get("merged_subfolder", "Train"))

    for d in (dataset_dir, puzzles_dir, merged_dir):
        os.makedirs(d, exist_ok=True)

    state_file = pipe_cfg.get("state_file", "pipeline_state.json")
    state_path = os.path.join(dataset_dir, state_file)

    queue_state_path = os.path.join(dataset_dir, pipe_cfg.get("queue_state_file", "position_queue_state.json"))

    force = pipe_cfg.get("force_recompute", False)
    if force and os.path.exists(state_path):
        logger.warning("Flag FORCE attivo: azzeramento stato precedente.")
        os.remove(state_path)

    state = PipelineState(state_path)

    time_stats_path = os.path.join(dataset_dir, stats_cfg.get("output_filename", "avg_time_by_rating.json"))
    puzzle_csv_path = os.path.join(dataset_dir, puzzle_cfg.get("decompressed_csv_filename", "lichess_puzzles.csv"))

    mate_train_range = (games_cfg.get("mate_range_min", 1), games_cfg.get("mate_range_max", 5))
    split_ratios = (
        split_cfg.get("train_ratio", 0.8),
        split_cfg.get("val_ratio", 0.1),
        split_cfg.get("test_ratio", 0.1),
    )

    ctx: Dict[str, Any] = {}

    queue_registry = PositionQueueRegistry.instance(state_path=queue_state_path)

    # ========================================================================
    # STEP 1: Statistiche tempo medio per rating
    # ========================================================================
    games_zst_path = raw_cfg.get("games_zst")
    games_zst_available = bool(games_zst_path) and os.path.exists(games_zst_path)

    def _step_time_stats() -> None:
        require_file(raw_cfg["games_zst"], "File PGN compresso necessario per il calcolo delle durate mosse.")
        builder = TimeStatsBuilder(
            zst_path=raw_cfg["games_zst"],
            max_games=stats_cfg.get("max_games", 50000),
            bucket_size=stats_cfg.get("bucket_size", 100),
        )
        stats = builder.build_and_save(time_stats_path)
        ctx["avg_time_by_rating"] = stats

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

    # ========================================================================
    # STEP 2: Pipeline partite -> posizioni accodate sul registry condiviso
    # ========================================================================
    def _step_games_pipeline() -> None:
        require_executable(stockfish_path, "Eseguibile Stockfish mancante o non avviabile.")

        sources = _build_sources(raw_cfg, games_cfg)
        if not sources:
            logger.info(
                "Nessuna sorgente (Lichess, FICS o Club) configurata o trovata: "
                "step 'games_pipeline' produce zero posizioni, si procede solo con i puzzle."
            )
            ctx["games_result"] = {"accepted_windows": 0, "enqueued_positions": 0}
            return

        builder = GamesBuilder(
            sources=sources,
            stockfish_path=stockfish_path,
            mate_range=mate_train_range,

            search_depth=games_cfg.get("search_depth", 8),
            analysis_time=games_cfg.get("time_limit_seconds", 0.2),

            workers=games_cfg.get("workers"),
            threads=engine_cfg.get("threads", 1),
            hash_mb=engine_cfg.get("hash_mb", 128),
            multipv=1,

            default_move_seconds=15.0,
            avg_time_by_rating=ctx.get("avg_time_by_rating", {}),
            require_clock=games_cfg.get("require_clock", False),

            min_ply=games_cfg.get("min_ply", 8),
            min_game_plies=games_cfg.get("min_game_plies", 20),

            candidate_min_legal_moves=games_cfg.get("candidate_min_legal_moves", 1),
            candidate_max_legal_moves=games_cfg.get("candidate_max_legal_moves"),
            skip_if_in_check=games_cfg.get("skip_if_in_check", False),

            max_piece_count=games_cfg.get("max_piece_count", 18),
            only_decisive_games=games_cfg.get("only_decisive_games", True),
            skip_time_forfeit=games_cfg.get("skip_time_forfeit", True),
            min_material_for_mate_attempt=games_cfg.get("min_material_for_mate_attempt", 4),
            drop_zero_clock=games_cfg.get("drop_zero_clock", True),
            min_rating=games_cfg.get("min_rating", 700),
            max_rating=games_cfg.get("max_rating"),
            min_material_diff_for_mate_attempt=games_cfg.get("min_material_diff_for_mate_attempt", 3),
            require_heavy_piece=games_cfg.get("require_heavy_piece", True),
            skip_trivial_endgame=games_cfg.get("skip_trivial_endgame", True),

            pool_join_timeout=games_cfg.get("pool_join_timeout", 20.0),
            syzygy_path=engine_cfg.get("syzygy_path"),
            checkpoint_log_every=games_cfg.get("checkpoint_every", 5000),
            config_error_cls=PipelineConfigError,
            queue_state_path=queue_state_path,
        )
        ctx["games_result"] = builder.run()

    if step is None or step == "games_pipeline":
        run_step(
            state,
            "games_pipeline",
            is_ready_fn=lambda: state.is_done("games_pipeline"),
            do_fn=_step_games_pipeline,
        )

    # ========================================================================
    # STEP 3: Decompressione puzzle Lichess
    # ========================================================================
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

    # ========================================================================
    # STEP 4: Posizioni puzzle -> accodate sul registry condiviso
    # ========================================================================
    def _step_build_puzzles() -> None:
        if "avg_time_by_rating" not in ctx:
            ctx["avg_time_by_rating"] = (
                load_avg_time_by_rating(time_stats_path) if file_ready(time_stats_path) else {}
            )

        games_result = ctx.get("games_result", {"accepted_windows": 0})
        game_id_start = games_result.get("accepted_windows", 0)

        debug_jsonl_path = os.path.join(puzzles_dir, "puzzle_debug.jsonl")

        builder = PuzzleBuilder(
            csv_path=puzzle_csv_path,
            mate_range=mate_train_range,
            max_puzzles=puzzle_cfg.get("max_puzzles", 100000),
            avg_time_by_rating=ctx["avg_time_by_rating"],
            chunksize=puzzle_cfg.get("chunksize", 50000),
            queue_state_path=queue_state_path,
            debug_jsonl_path=debug_jsonl_path,
            config_error_cls=PipelineConfigError,
        )
        ctx["puzzle_result"] = builder.run(game_id_start=game_id_start)

    if file_ready(puzzle_csv_path):
        if step is None or step == "build_puzzles":
            run_step(
                state,
                "build_puzzles",
                is_ready_fn=lambda: state.is_done("build_puzzles"),
                do_fn=_step_build_puzzles,
            )

    # ========================================================================
    # STEP 5: FINALIZE — drena il registry condiviso, produce e salva gli
    # split finali (UNICA chiamata a build_splits in tutta la pipeline).
    # ========================================================================
    final_paths = {
        split: os.path.join(merged_dir, f"{split}.pt")
        for split in ("train", "val", "test")
    }

    def _step_finalize_splits() -> None:
        splits = queue_registry.build_splits(split_ratios=split_ratios, seed=pipe_cfg.get("seed", 42))

        os.makedirs(merged_dir, exist_ok=True)
        for split_name, data_list in splits.items():
            out_path = final_paths[split_name]
            tmp_path = out_path + ".tmp"
            torch.save(data_list, tmp_path)
            os.replace(tmp_path, out_path)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            logger.info(f"Salvato {split_name}: {len(data_list)} posizioni in {out_path} ({size_mb:.2f} MB)")

        if not splits.get("train"):
            raise PipelineConfigError("Split 'train' vuoto dopo finalize_splits: nessuna posizione disponibile.")

    if step is None or step == "finalize_splits":
        run_step(
            state,
            "finalize_splits",
            is_ready_fn=lambda: all(torch_pt_ready(p) for p in final_paths.values()),
            do_fn=_step_finalize_splits,
        )

    logger.info("Pipeline completata con successo.")
    logger.info(f"Dataset train/val/test pronti in: {merged_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Costruzione dataset TimeGNN (schema board-level per posizione).")
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