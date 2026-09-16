from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
from typing import List, Optional

import torch
from torch_geometric.data import Data

# Riusa la pipeline di training reale, non la reimplementa: se
# TrainMain.py viene modificato, questo test resta allineato.
import TrainMain

logger = logging.getLogger("train_main_test")

SHARD_FILENAME_TEMPLATE = "shard_{:05d}.pt"
MANIFEST_FILENAME = "manifest.json"


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
    """Legge shard in ordine (shard_00000.pt, shard_00001.pt, ...) finche'
    non accumula almeno n_items elementi, poi tronca esattamente a
    n_items. Non carica l'intero dataset: si ferma appena ha abbastanza.
    """
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
    """Scrive `items` come un UNICO shard + manifest.json, nel formato
    atteso da ShardedGraphDataset (stesso formato di
    TrainPipeline/Shard/Sharding.py e TrainPipeline/CleanDataset.py)."""
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

    logger.info(f"Mini-shard scritto: {shard_path} ({len(items)} campioni) -> manifest in {manifest_path}")


def build_mini_dataset(
    train_dir: str,
    val_dir: str,
    out_root: str,
    n_train: int,
    n_val: int,
) -> tuple:
    """Costruisce Dataset/_smoke_test/{train_mini,val_mini} a partire dagli
    shard reali, e ritorna (train_mini_dir, val_mini_dir)."""
    train_items = _collect_n_items(train_dir, n_train, "train")
    val_items = _collect_n_items(val_dir, n_val, "val")

    train_mini_dir = os.path.join(out_root, "train_mini")
    val_mini_dir = os.path.join(out_root, "val_mini")

    _write_single_shard_dataset(train_items, train_mini_dir)
    _write_single_shard_dataset(val_items, val_mini_dir)

    del train_items, val_items
    return train_mini_dir, val_mini_dir


# ----------------------------------------------------------------------
# Config minimale per il test (pensata per MX330, 2GB VRAM)
# ----------------------------------------------------------------------
def build_smoke_test_cfg(
    checkpoint_path: str,
    batch_size: int,
    epochs: int,
) -> dict:
    """Sezione 'train_basic' minimale. Valori scelti per essere leggeri
    su una GPU da 2GB (MX330): batch_size piccolo, hidden dims ridotte,
    niente BatchNorm (rischioso con batch piccoli/ultimo batch da 1
    elemento), niente num_workers (evita overhead di processi extra per
    un test da pochi secondi), niente compile.
    """
    return {
        "enabled": True,
        "checkpoint": checkpoint_path,
        "train_dir": None,  # sovrascritto dal chiamante di run_training
        "val_dir": None,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": None,
        "pin_memory": False,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "seed": 42,
        "patience": 1,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test: verifica che il training giri end-to-end su un piccolo sottoinsieme."
    )
    parser.add_argument("--train-dir", default="Dataset/Train/train_clean", help="Cartella shardata di train reale.")
    parser.add_argument("--val-dir", default="Dataset/Train/val_clean", help="Cartella shardata di val reale.")
    parser.add_argument("--out-root", default="Dataset/_smoke_test", help="Dove scrivere mini-dataset e checkpoint di test.")
    parser.add_argument("--n-train", type=int, default=1000, help="Numero di campioni di train da usare.")
    parser.add_argument("--n-val", type=int, default=100, help="Numero di campioni di val da usare.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (piccolo per MX330/2GB).")
    parser.add_argument("--epochs", type=int, default=1, help="Numero di epoche (default 1: solo verifica che giri).")
    parser.add_argument("--keep-output", action="store_true", help="Non cancellare --out-root a fine test.")
    args = parser.parse_args()

    TrainMain.setup_logging("INFO")
    logger.info("=" * 70)
    logger.info("SMOKE TEST TRAINING (solo verifica funzionamento, NON valuta qualita')")
    logger.info("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU rilevata: {gpu_name} ({total_mem_gb:.2f} GB VRAM totale).")
    else:
        logger.warning("Nessuna GPU CUDA rilevata: il test girera' su CPU (piu' lento ma comunque valido come smoke test).")

    # AMP disabilitato di proposito: su una GPU piccola come la MX330
    # (architettura Pascal, nessun supporto tensor-core/bf16 affidabile)
    # l'autocast puo' non dare benefici e complica il debug di un test
    # che deve solo verificare la correttezza del ciclo, non le performance.
    use_amp = False
    logger.info(f"Device: {device}, AMP: {use_amp} (disattivato di proposito per lo smoke test).")

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

    checkpoint_dir = os.path.join(args.out_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "smoke_basic.pt")

    # Rimuove eventuali checkpoint di un test precedente, cosi' questa
    # run parte sempre da zero (niente resume accidentale tra run di test
    # diverse con parametri diversi, es. batch_size cambiato).
    for pattern in ("smoke_basic_last.pt", "smoke_basic_best.pt", "smoke_basic_last_scheduler.pt", "smoke_basic_best_scheduler.pt"):
        stale = os.path.join(checkpoint_dir, pattern)
        if os.path.exists(stale):
            os.remove(stale)

    smoke_cfg = {
        "train_basic": build_smoke_test_cfg(checkpoint_path, args.batch_size, args.epochs),
        # train_time_aware presente solo perche' validate_config lo
        # richiede se in futuro questo script venisse esteso; qui non
        # viene mai eseguito (run_training e' chiamato direttamente sotto,
        # non passando da TrainMain.main()).
        "train_time_aware": {"enabled": False},
        "evaluate": {"enabled": False},
        "pipeline": {},
    }

    logger.info("-" * 70)
    logger.info(f"Avvio run_training('basic', ...) su {args.n_train} train / {args.n_val} val, "
                f"batch_size={args.batch_size}, epochs={args.epochs}, device={device}.")
    logger.info("-" * 70)

    t0 = time.monotonic()
    try:
        TrainMain.run_training(
            smoke_cfg,
            "basic",
            train_mini_dir,
            val_mini_dir,
            checkpoint_path,
            device,
            use_amp,
            tuning_meta={},
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
    logger.info(f"SMOKE TEST OK: run_training completata senza eccezioni in {elapsed:.2f}s.")
    logger.info("=" * 70)

    best_path = os.path.join(checkpoint_dir, "smoke_basic_best.pt")
    last_path = os.path.join(checkpoint_dir, "smoke_basic_last.pt")
    for p, label in ((last_path, "last"), (best_path, "best")):
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / 1024**2
            logger.info(f"Checkpoint {label} scritto: {p} ({size_mb:.2f} MB)")
        else:
            logger.warning(f"Checkpoint {label} atteso ma non trovato: {p}")

    if not args.keep_output:
        logger.info(f"Rimozione output di test in '{args.out_root}' (usa --keep-output per conservarli).")
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