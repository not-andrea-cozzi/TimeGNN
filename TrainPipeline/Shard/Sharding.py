"""
step1_shard_dataset.py

Step 1 della training pipeline: riscrive train_clean.pt / val_clean.pt
(liste monolitiche di Data, lente da caricare con torch.load per via del
pickling per-oggetto su ~300k elementi) in shard piu' piccoli su disco
(default 8000 Data/shard), cosi' lo Step 2 (IterableDataset) puo' caricare
un file alla volta invece dell'intero dataset in un colpo solo.

Layout output:
    <out_dir>/<split>/shard_00000.pt
    <out_dir>/<split>/shard_00001.pt
    ...
    <out_dir>/<split>/manifest.json   # {"num_shards": N, "shard_size": S, "total": T}

Ogni shard_NNNNN.pt e' semplicemente torch.save(List[Data]) di al piu'
shard_size elementi: nessuna compressione custom qui (i Data sono gia'
minimi dopo Step 0), la torch.load di un singolo shard da ~50MB e' rapida.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import List

import torch
from torch_geometric.data import Data

from Common.progress import wrap_iter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("step1_shard")

SHARD_FILENAME_TEMPLATE = "shard_{:05d}.pt"
MANIFEST_FILENAME = "manifest.json"


def shard_split(in_path: str, out_dir: str, shard_size: int = 8000) -> dict:
    """Legge in_path (lista di Data), scrive shard da shard_size elementi
    in out_dir, scrive manifest.json con i metadati per il loader lazy."""
    logger.info(f"[{os.path.basename(in_path)}] Caricamento...")
    t0 = time.monotonic()
    data_list: List[Data] = torch.load(in_path, weights_only=False)
    total = len(data_list)
    logger.info(f"[{os.path.basename(in_path)}] {total:,} Data caricati in {time.monotonic() - t0:.2f}s.")

    os.makedirs(out_dir, exist_ok=True)

    num_shards = 0
    t0 = time.monotonic()
    label = os.path.basename(in_path)
    offsets = list(range(0, total, shard_size))
    for start in wrap_iter(offsets, desc=f"[{label}] Scrittura shard", unit="shard", total=len(offsets)):
        chunk = data_list[start : start + shard_size]
        shard_path = os.path.join(out_dir, SHARD_FILENAME_TEMPLATE.format(num_shards))
        tmp_path = shard_path + ".tmp"
        torch.save(chunk, tmp_path)
        os.replace(tmp_path, shard_path)
        num_shards += 1

    manifest = {"num_shards": num_shards, "shard_size": shard_size, "total": total}
    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.monotonic() - t0
    logger.info(
        f"[{os.path.basename(in_path)}] {num_shards} shard scritti in {elapsed:.2f}s "
        f"-> '{out_dir}' ({total:,} Data totali, shard_size={shard_size})."
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: risharding train_clean.pt / val_clean.pt")
    parser.add_argument("--train", default="Dataset/Train/train_clean.pt")
    parser.add_argument("--val", default="Dataset/Train/val_clean.pt")
    parser.add_argument("--out-dir", default="Dataset/Train/shards")
    parser.add_argument("--shard-size", type=int, default=8000)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("STEP 1: risharding dataset (train + val)")
    logger.info("=" * 60)

    train_out = os.path.join(args.out_dir, "train")
    val_out = os.path.join(args.out_dir, "val")

    shard_split(args.train, train_out, args.shard_size)
    shard_split(args.val, val_out, args.shard_size)

    logger.info("=" * 60)
    logger.info(f"Completato: shard in '{args.out_dir}/train' e '{args.out_dir}/val'.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()