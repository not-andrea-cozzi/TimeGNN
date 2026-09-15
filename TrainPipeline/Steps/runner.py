from __future__ import annotations

import gc
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger("pipeline.runner")


def free_memory() -> None:
    """Forza GC e svuota la cache CUDA se disponibile."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_step(
    state: Dict[str, Any],
    step_name: str,
    is_done: Callable[[], bool],
    do: Callable[[], None],
) -> None:
    """
    Esegue `do()` solo se lo step non è già marcato 'done' in `state`
    e se `is_done()` (output già presente) è False.

    Lo stato NON viene persistito qui: la persistenza è responsabilità
    del chiamante (es. TrainMain salva `state` su disco a fine run).
    """
    if state.get(step_name) == "done":
        logger.info(f"Step '{step_name}' già completato, skip.")
        return

    try:
        if is_done():
            logger.info(f"Step '{step_name}': output già presente, skip.")
            state[step_name] = "done"
            return

        logger.info(f"Step '{step_name}': esecuzione...")
        do()
        state[step_name] = "done"
        logger.info(f"Step '{step_name}': completato.")

    except Exception as e:
        state[step_name] = "error"
        logger.exception(f"Step '{step_name}' fallito: {e}")
        raise