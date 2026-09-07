from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List

import torch
from torch_geometric.data import Data

from Common.progress import wrap_iter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("step0_clean")

# Campi richiesti dal forward di DualGATModel / DualGATTimeAwareModel
# (vedi timegnn/models/gat_basic.py, gat_time_decay.py) + y per la loss.
KEEP_FIELDS = ("event_ids", "x", "edge_index", "edge_attr", "time", "y", "num_nodes")


def _clean_single(data: Data) -> Data:
    """Ritorna un nuovo Data con solo i campi in KEEP_FIELDS."""
    cleaned = Data()
    for key in KEEP_FIELDS:
        if hasattr(data, key):
            val = getattr(data, key)
            if val is not None:
                cleaned[key] = val
    return cleaned


def _clean_chunk(chunk: List[Data]) -> List[Data]:
    return [_clean_single(d) for d in chunk]


def _chunkify(items: List[Data], n_chunks: int) -> List[List[Data]]:
    if n_chunks <= 1 or len(items) == 0:
        return [items]
    size = max(1, (len(items) + n_chunks - 1) // n_chunks)
    return [items[i : i + size] for i in range(0, len(items), size)]


def clean_file(in_path: str, out_path: str, workers: int = 4) -> int:
    """Carica in_path, pulisce ogni Data (in parallelo con `workers`
    processi), salva in out_path. Ritorna il numero di Data processati."""
    logger.info(f"[{os.path.basename(in_path)}] Caricamento...")
    t0 = time.monotonic()
    data_list: List[Data] = torch.load(in_path, weights_only=False)
    logger.info(f"[{os.path.basename(in_path)}] {len(data_list):,} Data caricati in {time.monotonic() - t0:.2f}s.")

    t0 = time.monotonic()
    chunks = _chunkify(data_list, workers)
    cleaned: List[Data] = []
    label = os.path.basename(in_path)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in wrap_iter(
            pool.map(_clean_chunk, chunks),
            desc=f"[{label}] Pulizia chunk",
            unit="chunk",
            total=len(chunks),
        ):
            cleaned.extend(result)
    logger.info(
        f"[{os.path.basename(in_path)}] Pulizia completata in {time.monotonic() - t0:.2f}s "
        f"({len(cleaned):,} Data, campi tenuti={KEEP_FIELDS})."
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    torch.save(cleaned, tmp_path)
    os.replace(tmp_path, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f"[{os.path.basename(in_path)}] Salvato '{out_path}' ({size_mb:.2f} MB).")

    return len(cleaned)

