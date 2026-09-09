from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None

from DatasetPipeline.Builder.GamesBuilder import GamesBuilder, GamesBuilderConfig, SourceSpec
from DatasetPipeline.TimeStatBuilder import load_avg_time_by_rating

logger = logging.getLogger("build_shards_standalone")


class ConfigError(Exception):
    """Errore bloccante di configurazione dello script standalone."""


# ============================================================================
# LOGGING
# ============================================================================

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


# ============================================================================
# CONFIG LOADING
# ============================================================================

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


def _infer_kind(path: str) -> str:
    """Deduce il kind di SourceSpec dall'estensione del file:
    - .zst -> "lichess" (stream zstd-compresso, formato Lichess standard)
    - qualsiasi altra estensione (.pgn, .bz2, testo piano) -> "fics"
      (GamesBuilder._open_pgn_text_stream gestisce sia .bz2 che testo
      piano sotto "fics", vedi GamesBuilder.py)
    "club" non e' supportato qui perche' richiede un CSV con colonna PGN,
    caso diverso da "lista di file PGN": se serve, aggiungere kind
    esplicito nella entry YAML (vedi campo opzionale `kind`).
    """
    if path.lower().endswith(".zst"):
        return "lichess"
    return "fics"


def _build_sources_from_list(sources_cfg: List[Dict[str, Any]]) -> List[SourceSpec]:
    """Costruisce i SourceSpec dalla lista dichiarata nello YAML
    (games_pipeline.sources), una entry per file:

        sources:
          - path: "RawData/batch1.pgn.zst"
            tag: "games_batch1"
          - path: "RawData/batch2.pgn"
            tag: "games_batch2"
            kind: "fics"          # opzionale, altrimenti dedotto da estensione
            skip_games: 0          # opzionale, default 0
            max_games: 50000       # opzionale, default nessun limite

    Ogni entry DEVE avere path + tag: senza un tag univoco per sorgente,
    GamesBuilder._validate_config solleva un errore esplicito (game_id
    duplicati tra file diversi altrimenti), quindi il fallimento e'
    comunque sicuro anche se lo si dimentica qui.
    """
    if not sources_cfg:
        raise ConfigError(
            "games_pipeline.sources e' vuoto o assente: specificare almeno "
            "una entry {path, tag} nello YAML."
        )

    sources: List[SourceSpec] = []
    seen_tags = set()
    for i, entry in enumerate(sources_cfg):
        path = entry.get("path")
        tag = entry.get("tag")
        if not path:
            raise ConfigError(f"sources[{i}]: campo 'path' mancante.")
        if not tag:
            raise ConfigError(f"sources[{i}] (path='{path}'): campo 'tag' mancante.")
        if tag in seen_tags:
            raise ConfigError(
                f"sources[{i}]: tag '{tag}' duplicato nella lista. Ogni "
                f"sorgente deve avere un tag univoco (vedi GamesBuilder."
                f"_validate_config)."
            )
        seen_tags.add(tag)

        if not os.path.exists(path):
            raise ConfigError(f"sources[{i}] (tag='{tag}'): file non trovato: '{path}'.")

        kind = entry.get("kind") or _infer_kind(path)

        sources.append(
            SourceSpec(
                kind=kind,
                path=path,
                pgn_col=entry.get("pgn_col", "pgn"),
                skip_games=entry.get("skip_games", 0),
                max_games=entry.get("max_games"),
                tag=tag,
            )
        )
        logger.info(
            f"Sorgente #{i}: path='{path}', tag='{tag}', kind='{kind}', "
            f"skip_games={entry.get('skip_games', 0)}, max_games={entry.get('max_games')}."
        )

    return sources


def require_executable(path: str) -> None:
    if not (os.path.exists(path) and os.access(path, os.X_OK)):
        raise ConfigError(f"Eseguibile non trovato o non eseguibile: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml_config(config_path)

    pipe_cfg = cfg.get("pipeline", {})
    engine_cfg = cfg.get("engine", {})
    games_cfg = cfg.get("games_pipeline", {})

    log_level = pipe_cfg.get("log_level", "INFO")
    log_file = pipe_cfg.get("log_file")
    setup_logging(log_level, log_file)

    logger.info("=" * 70)
    logger.info("BUILD SHARDS STANDALONE (GamesBuilder diretto, nessun merge finale)")
    logger.info("=" * 70)

    stockfish_path = engine_cfg.get("stockfish_path")
    if not stockfish_path:
        raise ConfigError("engine.stockfish_path mancante nello YAML.")
    require_executable(stockfish_path)

    dataset_dir = pipe_cfg.get("dataset_dir", "Dataset")
    games_output_dir = os.path.join(dataset_dir, pipe_cfg.get("games_subfolder", "Games"))
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(games_output_dir, exist_ok=True)

    # NOTA CONDIVISIONE SPOOL CON DatasetMain.py: questo path deve
    # risolvere ESATTAMENTE alla stessa stringa (stesso dataset_dir +
    # stesso queue_state_file) usata in Yaml/dataset_main.yaml, perche'
    # PositionQueueRegistry deriva la cartella di spool da
    # _spool_dir_for(state_path) = {dirname(state_path)}/{stem}_spool.
    # BuildShard.py e DatasetMain.py girano come processi separati: non
    # condividono il singleton in RAM, SOLO lo spool su disco. Se i due
    # YAML divergono su dataset_dir o queue_state_file, gli shard scritti
    # qui finiscono in una cartella diversa e DatasetMain non li vedra'
    # mai al momento di finalize_splits.
    queue_state_path = os.path.join(
        dataset_dir, pipe_cfg.get("queue_state_file", "position_queue_state.json")
    )
    resume_state_path = os.path.join(
        dataset_dir, pipe_cfg.get("resume_state_file", "games_builder_resume.json")
    )

    # avg_time_by_rating: opzionale, riusa un time_stats.json gia'
    # calcolato dalla pipeline principale se disponibile, altrimenti
    # ricade sul default costante/rating-bucket assente (identico
    # comportamento a DatasetMain quando lo step time_stats e' saltato).
    avg_time_by_rating: Dict[int, float] = {}
    time_stats_path = pipe_cfg.get("time_stats_json")
    if time_stats_path:
        if os.path.exists(time_stats_path):
            avg_time_by_rating = load_avg_time_by_rating(time_stats_path)
            logger.info(f"avg_time_by_rating caricato da '{time_stats_path}': {len(avg_time_by_rating)} bucket.")
        else:
            logger.warning(
                f"pipeline.time_stats_json='{time_stats_path}' specificato ma non trovato: "
                f"procedo con avg_time_by_rating vuoto (fallback su default_move_seconds)."
            )

    mate_range = (games_cfg.get("mate_range_min", 1), games_cfg.get("mate_range_max", 10))
    if mate_range[0] < 1 or mate_range[1] < mate_range[0]:
        raise ConfigError(f"games_pipeline: mate_range non valido: {mate_range}")

    split_ratios = (
        games_cfg.get("split_train_ratio", 0.7),
        games_cfg.get("split_val_ratio", 0.1),
        games_cfg.get("split_test_ratio", 0.2),
    )
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ConfigError(f"split_ratios deve sommare a 1.0 (attuale: {split_ratios}).")

    sources = _build_sources_from_list(games_cfg.get("sources", []))

    flush_every_seconds = games_cfg.get("flush_every_seconds", 1800.0)
    if flush_every_seconds is not None and flush_every_seconds <= 0:
        flush_every_seconds = None
    flush_every_seconds = 1800.0

    gb_config = GamesBuilderConfig(
        sources=sources,
        stockfish_path=stockfish_path,
        mate_range=mate_range,
        search_depth=games_cfg.get("search_depth", 8),
        analysis_time=games_cfg.get("time_limit_seconds", 0.2),
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

        require_clock=games_cfg.get("require_clock", False),
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

        split_ratios=split_ratios,
        split_seed=pipe_cfg.get("seed", 42),

        pool_join_timeout=games_cfg.get("pool_join_timeout", 20.0),

        auto_resume=games_cfg.get("auto_resume", True),
        resume_state_path=resume_state_path,
        resume_checkpoint_every=games_cfg.get("resume_checkpoint_every", 500),

        flush_every_seconds=flush_every_seconds,
    )

    logger.info(
        f"GamesBuilderConfig pronto: {len(sources)} sorgenti, mate_range={mate_range}, "
        f"queue_state_path='{queue_state_path}', resume_state_path='{resume_state_path}', "
        f"flush_every_seconds={flush_every_seconds}."
    )
    for src in sources:
        logger.info(f"    - tag='{src.tag}' kind='{src.kind}' path='{src.path}'")

    builder = GamesBuilder(gb_config)
    logger.info("GamesBuilder pronto, avvio run()...")
    result = builder.run()

    logger.info("=" * 70)
    logger.info("BUILD SHARDS STANDALONE COMPLETATO")
    logger.info(
        f"processed_games={result.get('processed_games', 0):,}, "
        f"accepted_games={result.get('accepted_games', 0):,}, "
        f"enqueued_positions={result.get('enqueued_positions', 0):,}."
    )
    if result.get("source_counts"):
        logger.info("Posizioni per sorgente (tag):")
        for tag, count in sorted(result["source_counts"].items(), key=lambda kv: -kv[1]):
            logger.info(f"    {tag}: {count:,}")
    logger.info(
        f"Shard scritti nello spool di '{queue_state_path}'. "
        f"NESSUN merge/split finale eseguito: per produrre train/val/test.pt "
        f"lanciare DatasetMain.py con step=finalize_splits (stesso "
        f"queue_state_path), oppure chiamare manualmente "
        f"registry.build_splits() + registry.commit_splits()."
    )
    logger.info("=" * 70)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Costruisce shard di posizioni da una lista di file PGN/PGN.ZST "
            "con tag associati, usando lo stesso GamesBuilder della pipeline "
            "principale. Non esegue merge/split finale."
        )
    )
    parser.add_argument(
        "--config",
        default="Yaml/build_shards.yaml",
        help="Percorso del file YAML di configurazione (lista sorgenti + parametri GamesBuilder).",
    )
    args = parser.parse_args()

    try:
        main(config_path=args.config)
    except ConfigError as e:
        logging.getLogger("build_shards_standalone").error(f"Errore di configurazione: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        logging.getLogger("build_shards_standalone").warning(
            "Interrotto dall'utente. Il resume (auto_resume=true) permette "
            "di riprendere da dove eri arrivato rilanciando lo stesso comando."
        )
        sys.exit(130)
    except Exception as e:
        logging.getLogger("build_shards_standalone").error(f"Interruzione imprevista: {e}")
        sys.exit(1)