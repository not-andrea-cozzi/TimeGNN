from __future__ import annotations

import argparse
import gc
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

KEEP_FIELDS = (
    "event_ids",
    "x",
    "edge_index",
    "edge_attr",
    "time",
    "y",
    "legal_move_mask",
    "position_mate_n",
    "num_nodes",
)


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


def clean_file(in_path: str, out_path: str, workers: int = 0) -> int:
    """
    Carica in_path, pulisce ogni Data e salva in out_path.
    Ritorna il numero di Data processati.

    Se workers <= 0 (default), la pulizia avviene in SINGOLO processo,
    senza ProcessPoolExecutor: questo evita la condivisione di memoria
    tra processi e il conseguente errore
    'unable to open shared memory object ... Too many open files'.

    Se workers >= 1, usa il multiprocessing (richiede un limite di
    file aperti adeguato: `ulimit -n 8192`).
    """
    basename = os.path.basename(in_path)

    # ------------------------------------------------------------------
    # 1. Caricamento
    # ------------------------------------------------------------------
    logger.info(f"[{basename}] Caricamento...")
    t0 = time.monotonic()
    data_list: List[Data] = torch.load(in_path, weights_only=False)
    logger.info(
        f"[{basename}] {len(data_list):,} Data caricati in "
        f"{time.monotonic() - t0:.2f}s."
    )

    # ------------------------------------------------------------------
    # 2. Pulizia
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    cleaned: List[Data] = []

    if workers is None or workers <= 0:
        # --- Singolo processo: nessuna shared memory, nessun fd aperto ---
        logger.info(f"[{basename}] Pulizia in singolo processo (workers={workers}).")
        chunk_size = 2000
        total = len(data_list)
        for i in wrap_iter(
            range(0, total, chunk_size),
            desc=f"[{basename}] Pulizia",
            unit="chunk",
            total=(total + chunk_size - 1) // chunk_size,
        ):
            chunk = data_list[i : i + chunk_size]
            cleaned.extend(_clean_chunk(chunk))
            del chunk
            gc.collect()
        # libera la lista originale per ridurre la RAM
        del data_list
        gc.collect()
    else:
        # --- Multiprocessing (sconsigliato su dataset grandi) ---
        logger.info(f"[{basename}] Pulizia con {workers} worker.")
        chunks = _chunkify(data_list, workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in wrap_iter(
                pool.map(_clean_chunk, chunks),
                desc=f"[{basename}] Pulizia chunk",
                unit="chunk",
                total=len(chunks),
            ):
                cleaned.extend(result)
        del data_list, chunks
        gc.collect()

    logger.info(
        f"[{basename}] Pulizia completata in {time.monotonic() - t0:.2f}s "
        f"({len(cleaned):,} Data, campi tenuti={KEEP_FIELDS})."
    )

    # ------------------------------------------------------------------
    # 3. Salvataggio atomico
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    torch.save(cleaned, tmp_path)
    os.replace(tmp_path, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f"[{basename}] Salvato '{out_path}' ({size_mb:.2f} MB).")

    # Salva la lunghezza PRIMA di liberare la lista
    n_cleaned = len(cleaned)
    del cleaned
    gc.collect()

    return n_cleaned


