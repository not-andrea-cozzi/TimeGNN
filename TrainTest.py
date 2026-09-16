from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from typing import List, Optional

import torch
from torch_geometric.data import Data

import TrainMain

from DatasetPipeline.Model.ChessConstants import MOVE_VOCAB_SIZE

logger = logging.getLogger("train_main_test")

SHARD_FILENAME_TEMPLATE = "shard_{:05d}.pt"
MANIFEST_FILENAME = "manifest.json"

SUPPORTED_MODELS = ("basic", "time_aware")


# ----------------------------------------------------------------------
# Mini-state in-memory per il TuningStep
# ----------------------------------------------------------------------
class _MiniState:
    def __init__(self) -> None:
        self._done: dict = {}

    def is_done(self, step: str, skip: bool = False) -> bool:
        return (not skip) and step in self._done

    def mark_done(self, step: str, **kwargs) -> None:
        self._done[step] = kwargs

    def mark_failed(self, step: str, reason: str) -> None:
        logger.error(f"[tuning] step '{step}' fallito: {reason}")


# ----------------------------------------------------------------------
# Import robusto di run_tuning_step
# ----------------------------------------------------------------------
def _import_run_tuning_step():
    candidates = (
        "TrainPipeline.Steps.TuningStep",
        "TrainPipeline.Steps.tuning_step",
        "TrainPipeline.TuningStep",
    )
    last_err: Optional[Exception] = None
    for mod_path in candidates:
        try:
            mod = __import__(mod_path, fromlist=["run_tuning_step"])
            fn = getattr(mod, "run_tuning_step", None)
            if fn is not None:
                return fn
        except Exception as e:
            last_err = e
            continue
    raise ImportError(
        "Impossibile importare run_tuning_step da TrainPipeline.Steps.TuningStep. "
        "Verifica il percorso del modulo con: "
        "  grep -rn 'def run_tuning_step' TrainPipeline/"
        f" Ultimo errore: {last_err}"
    )


# ----------------------------------------------------------------------
# Costruzione del mini-dataset shardato
# ----------------------------------------------------------------------
def _read_manifest(shard_dir: str) -> dict:
    manifest_path = os.path.join(shard_dir, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"manifest.json non trovato in '{shard_dir}'. Verifica che la "
            f"pipeline di dataset (DatasetMain.py, step 'clean') sia gia' "
            f"stata eseguita e abbia prodotto questa cartella."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_n_items(shard_dir: str, n_items: int, label: str) -> List[Data]:
    manifest = _read_manifest(shard_dir)
    num_shards = manifest["num_shards"]
    total_available = manifest["total"]

    if total_available < n_items:
        raise ValueError(
            f"'{shard_dir}' contiene solo {total_available} campioni "
            f"({label}), ma ne sono richiesti {n_items}. Riduci --n-{label} "
            f"o usa un dataset piu' grande."
        )

    collected: List[Data] = []
    for shard_idx in range(num_shards):
        if len(collected) >= n_items:
            break
        shard_path = os.path.join(shard_dir, SHARD_FILENAME_TEMPLATE.format(shard_idx))
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard mancante: {shard_path}")
        items = torch.load(shard_path, weights_only=False)
        collected.extend(items)
        del items

    collected = collected[:n_items]
    logger.info(f"[{label}] Raccolti {len(collected)}/{n_items} campioni da '{shard_dir}'.")
    return collected


def _write_single_shard_dataset(items: List[Data], out_dir: str) -> None:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    shard_path = os.path.join(out_dir, SHARD_FILENAME_TEMPLATE.format(0))
    tmp_path = shard_path + ".tmp"
    torch.save(items, tmp_path)
    os.replace(tmp_path, shard_path)

    manifest = {"num_shards": 1, "shard_size": len(items), "total": len(items)}
    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_manifest, manifest_path)

    logger.info(
        f"Mini-shard scritto: {shard_path} ({len(items)} campioni) "
        f"-> manifest in {manifest_path}"
    )


def build_mini_dataset(
    train_dir: str,
    val_dir: str,
    out_root: str,
    n_train: int,
    n_val: int,
) -> tuple:
    train_items = _collect_n_items(train_dir, n_train, "train")
    val_items = _collect_n_items(val_dir, n_val, "val")

    train_mini_dir = os.path.join(out_root, "train_mini")
    val_mini_dir = os.path.join(out_root, "val_mini")

    _write_single_shard_dataset(train_items, train_mini_dir)
    _write_single_shard_dataset(val_items, val_mini_dir)

    del train_items, val_items
    return train_mini_dir, val_mini_dir


# ----------------------------------------------------------------------
# Config smoke test
# ----------------------------------------------------------------------
_COMMON_SMOKE_FIELDS: dict = {
    "enabled": True,
    "train_dir": None,
    "val_dir": None,
    "num_workers": 0,
    "persistent_workers": False,
    "prefetch_factor": None,
    "pin_memory": False,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "seed": 42,
    "patience": 5,
    "embedding_dims": 16,
    "gat_hidden_dim_event": 8,
    "gat_hidden_dim_embed": 16,
    "gat_hidden_dim_concat": 16,
    "num_heads": 2,
    "num_layers": 1,
    "dropout": 0.0,
    "use_batch_norm": False,
    "activation": "elu",
    "compile": False,
    "memory_cleanup_threshold_gb": 1.5,
}


def build_basic_smoke_test_cfg(checkpoint_path: str, batch_size: int, epochs: int) -> dict:
    return {
        **_COMMON_SMOKE_FIELDS,
        "checkpoint": checkpoint_path,
        "epochs": epochs,
        "batch_size": batch_size,
    }


def build_time_aware_smoke_test_cfg(
    checkpoint_path: str,
    batch_size: int,
    epochs: int,
    lambda_decay: float = 0.1,
) -> dict:
    return {
        **_COMMON_SMOKE_FIELDS,
        "checkpoint": checkpoint_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "lambda_decay": lambda_decay,
    }


def build_tuning_cfg() -> dict:
    return {"enabled": True, "move_vocab_size": MOVE_VOCAB_SIZE}


def _build_model_cfg(model_name: str, checkpoint_path: str, batch_size: int, epochs: int) -> dict:
    if model_name == "basic":
        return build_basic_smoke_test_cfg(checkpoint_path, batch_size, epochs)
    if model_name == "time_aware":
        return build_time_aware_smoke_test_cfg(checkpoint_path, batch_size, epochs)
    raise ValueError(f"Modello non supportato: {model_name!r}. Attesi: {SUPPORTED_MODELS}.")


def _cleanup_stale_checkpoints(checkpoint_dir: str, stem: str) -> None:
    for suffix in ("_last.pt", "_best.pt", "_last_scheduler.pt", "_best_scheduler.pt"):
        stale = os.path.join(checkpoint_dir, stem + suffix)
        if os.path.exists(stale):
            os.remove(stale)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test: verifica che il training giri end-to-end su un piccolo sottoinsieme."
    )
    parser.add_argument("--train-dir", default="Dataset/Train/train_clean")
    parser.add_argument("--val-dir", default="Dataset/Train/val_clean")
    parser.add_argument("--out-root", default="Dataset/_smoke_test")
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="basic",
    )
    parser.add_argument("--no-tuning", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device. 'auto' usa CUDA se disponibile e compatibile, altrimenti CPU.",
    )
    args = parser.parse_args()

    TrainMain.setup_logging("INFO")
    logger.info("=" * 70)
    logger.info(f"SMOKE TEST TRAINING [{args.model}] (solo verifica funzionamento, NON valuta qualita')")
    logger.info("=" * 70)

    # --- Selezione device ---
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise TrainMain.PipelineConfigError(
                "--device cuda richiesto ma torch.cuda.is_available() == False. "
                "Controlla driver NVIDIA e installazione PyTorch."
            )
        device = "cuda"
    else:  # auto
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Verifica che i kernel CUDA siano effettivamente eseguibili su questa GPU.
    # Evita il crash 'no kernel image is available' a meta' training.
    if device == "cuda":
        try:
            _probe = torch.zeros(1, device="cuda")
            _ = _probe + 1
            del _probe
            torch.cuda.synchronize()
        except Exception as e:
            raise TrainMain.PipelineConfigError(
                f"CUDA selezionato ma la GPU non e' utilizzabile con questa build "
                f"di PyTorch ({torch.__version__}): {e}\n"
                f"Reinstalla PyTorch con una build CUDA che supporti la tua GPU. "
                f"Per Pascal (sm_61) usa: pip install torch==2.4.1 --index-url "
                f"https://download.pytorch.org/whl/cu121"
            ) from e

        gpu_name = torch.cuda.get_device_name(0)
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        logger.info(f"Device: cuda ({gpu_name}, sm_{cc_major}{cc_minor}).")
    else:
        logger.info("Device: cpu.")

    # AMP: su CPU non ha senso, su CUDA lo abilitiamo.
    use_amp = (device == "cuda")
    logger.info(f"AMP: {use_amp}.")

    if not os.path.isdir(args.train_dir):
        raise TrainMain.PipelineConfigError(
            f"--train-dir non trovato: '{args.train_dir}'. Esegui prima la "
            f"pipeline di dataset (DatasetMain.py) fino allo step 'clean'."
        )
    if not os.path.isdir(args.val_dir):
        raise TrainMain.PipelineConfigError(f"--val-dir non trovato: '{args.val_dir}'.")

    t0 = time.monotonic()
    train_mini_dir, val_mini_dir = build_mini_dataset(
        args.train_dir, args.val_dir, args.out_root, args.n_train, args.n_val
    )
    logger.info(f"Mini-dataset pronto in {time.monotonic() - t0:.2f}s.")

    logger.info(f"[tuning] move_vocab_size = {MOVE_VOCAB_SIZE} (costante MOVE_VOCAB_SIZE, non inferita dal dataset).")

    checkpoint_dir = os.path.join(args.out_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_stem = f"smoke_{args.model}"
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_stem}.pt")

    _cleanup_stale_checkpoints(checkpoint_dir, checkpoint_stem)

    model_cfg = _build_model_cfg(
        args.model, checkpoint_path, args.batch_size, args.epochs
    )

    smoke_cfg = {
        "train_basic": (
            model_cfg if args.model == "basic" else {"enabled": False}
        ),
        "train_time_aware": (
            model_cfg if args.model == "time_aware" else {"enabled": False}
        ),
        "evaluate": {"enabled": False},
        "pipeline": {},
        "tuning": {"enabled": False} if args.no_tuning else build_tuning_cfg(),
    }

    # ------------------------------------------------------------------
    # Tuning step
    # ------------------------------------------------------------------
    tuning_meta: dict = {}
    if not args.no_tuning:
        logger.info("-" * 70)
        logger.info("Esecuzione tuning step (class_weights + warmup + norm)...")
        logger.info("-" * 70)

        try:
            run_tuning_step = _import_run_tuning_step()
        except ImportError as e:
            logger.error(f"Impossibile importare run_tuning_step: {e}")
            raise

        t_tuning_0 = time.monotonic()
        try:
            tuning_meta = run_tuning_step(
                cfg=smoke_cfg,
                state=_MiniState(),
                dataset_dir=args.out_root,
                train_dir=train_mini_dir,
                steps_per_epoch=None,
                total_planned_epochs=None,
            )
        except Exception:
            logger.error(
                "TUNING STEP FALLITO: run_tuning_step ha sollevato un'eccezione. "
                "Vedi traceback sopra/sotto per la causa.",
                exc_info=True,
            )
            raise

        logger.info(
            f"Tuning completato in {time.monotonic() - t_tuning_0:.2f}s: {tuning_meta}"
        )
    else:
        logger.info("Tuning disattivato (--no-tuning): salto run_tuning_step.")

    logger.info("-" * 70)
    logger.info(
        f"Avvio run_training('{args.model}', ...) su {args.n_train} train / {args.n_val} val, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, device={device}."
    )
    logger.info("-" * 70)

    t0 = time.monotonic()
    try:
        TrainMain.run_training(
            smoke_cfg,
            args.model,
            train_mini_dir,
            val_mini_dir,
            checkpoint_path,
            device,
            True,
            tuning_meta=tuning_meta,
        )
    except Exception:
        logger.error(
            "SMOKE TEST FALLITO: run_training ha sollevato un'eccezione. "
            "Vedi traceback sopra/sotto per la causa.",
            exc_info=True,
        )
        raise

    elapsed = time.monotonic() - t0
    logger.info("=" * 70)
    logger.info(f"SMOKE TEST OK: run_training('{args.model}') completata senza eccezioni in {elapsed:.2f}s.")
    logger.info("=" * 70)

    best_path = os.path.join(checkpoint_dir, f"{checkpoint_stem}_best.pt")
    last_path = os.path.join(checkpoint_dir, f"{checkpoint_stem}_last.pt")
    for p, label in ((last_path, "last"), (best_path, "best")):
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / 1024**2
            logger.info(f"Checkpoint {label} scritto: {p} ({size_mb:.2f} MB)")
        else:
            logger.warning(f"Checkpoint {label} atteso ma non trovato: {p}")

    if not args.keep_output:
        logger.info(
            f"Rimozione output di test in '{args.out_root}' "
            f"(usa --keep-output per conservarli)."
        )
        shutil.rmtree(args.out_root, ignore_errors=True)
    else:
        logger.info(f"Output di test conservati in '{args.out_root}' (--keep-output).")


if __name__ == "__main__":
    try:
        main()
    except TrainMain.PipelineConfigError as e:
        logging.getLogger("train_main_test").error(f"Errore di configurazione: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)