from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from TrainPipeline.CleanDataset import clean_sharded_directory
from TrainPipeline.Steps.runner import run_step, free_memory

logger = logging.getLogger("step0_clean")


def run_clean_step(
    cfg: Dict[str, Any],
    state: Dict[str, Any],
    step_filter: Optional[str] = None,
) -> None:
    """
    STEP 0 della pipeline di training: clean + reshard shard-by-shard.

    Legge le cartelle shardate prodotte da `finalize_splits`
    (Dataset/Train/train/, val/, test/), pulisce ogni Data con `_clean_chunk`
    e riscrive in nuove cartelle dedicate (train_clean/, val_clean/, test_clean/)
    con la dimensione target `clean.target_shard_size`.

    Vincoli:
      - Non carica mai l'intero dataset in RAM: un solo shard di input per volta.
      - La scrittura di ogni shard di output è atomica (tmp + os.replace).
      - Idempotente: se i manifest di output esistono già, salta.

    Parameters
    ----------
    cfg : dict
        Config completa (deve contenere la sezione `clean`).
    state : dict
        Stato della pipeline (viene aggiornato con state["clean"] = "done" | "error").
    step_filter : str | None
        Se impostato e diverso da "clean", il passo viene saltato.
    """
    if step_filter is not None and step_filter != "clean":
        return

    clean_cfg = cfg.get("clean", {})
    if not clean_cfg.get("enabled", True):
        logger.info("clean disabilitato da config.")
        return

    # Input (output di finalize_splits)
    in_train_dir = clean_cfg["input_dir_train"]
    in_val_dir = clean_cfg["input_dir_val"]
    in_test_dir = clean_cfg.get("input_dir_test")

    # Output (nuove cartelle dedicate)
    out_train_dir = clean_cfg["output_dir_train"]
    out_val_dir = clean_cfg["output_dir_val"]
    out_test_dir = clean_cfg.get("output_dir_test")

    # Parametri di resharding
    target_shard_size = int(clean_cfg.get("target_shard_size", 8000))
    workers = int(clean_cfg.get("workers", 0))

    # --- Pre-check: i manifest di input devono esistere ---
    for d in [in_train_dir, in_val_dir] + ([in_test_dir] if in_test_dir else []):
        manifest = os.path.join(d, "manifest.json")
        if not os.path.exists(manifest):
            raise FileNotFoundError(f"Manifest di input non trovato: {manifest}")

    # --- Condizione di "già fatto" ---
    def _is_ready() -> bool:
        ready = (
            os.path.exists(os.path.join(out_train_dir, "manifest.json"))
            and os.path.exists(os.path.join(out_val_dir, "manifest.json"))
        )
        if out_test_dir:
            ready = ready and os.path.exists(
                os.path.join(out_test_dir, "manifest.json")
            )
        return ready

    # --- Lavoro effettivo, un split alla volta ---
    def _do() -> None:
        logger.info("-" * 70)
        logger.info("STEP 0: clean + reshard (shard-by-shard)")
        logger.info("-" * 70)

        logger.info(f"[train] {in_train_dir} -> {out_train_dir}")
        clean_sharded_directory(
            in_train_dir, out_train_dir, target_shard_size, workers
        )
        free_memory()

        logger.info(f"[val] {in_val_dir} -> {out_val_dir}")
        clean_sharded_directory(
            in_val_dir, out_val_dir, target_shard_size, workers
        )
        free_memory()

        if in_test_dir and out_test_dir:
            logger.info(f"[test] {in_test_dir} -> {out_test_dir}")
            clean_sharded_directory(
                in_test_dir, out_test_dir, target_shard_size, workers
            )
            free_memory()

    run_step(state, "clean", _is_ready, _do)