from __future__ import annotations

import glob
import json
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from DatasetPipeline.Utils.position_compression import (
    compress_position_data,
    decompress_position_data,
)

logger = logging.getLogger("position_queue")

DEFAULT_STATE_FILENAME = "position_queue_state.json"
DEFAULT_SHARD_SIZE = 5000
SHARD_FILENAME_TEMPLATE = "shard_{:08d}.pt"
SHARD_GLOB_PATTERN = "shard_*.pt"


class PositionQueueError(RuntimeError):
    pass


@dataclass
class _QueuedPosition:
    local_ref: int
    source_tag: str
    group_key: int
    data: Data


def _spool_dir_for(state_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(state_path)) or "."
    stem = os.path.splitext(os.path.basename(state_path))[0]
    return os.path.join(base_dir, f"{stem}_spool")


def _extract_game_id(data: Data) -> str:
    raw = data.game_id
    if isinstance(raw, str):
        return raw
    if hasattr(raw, "item"):
        return str(raw.item())
    return str(raw)


class PositionQueueRegistry:
    """Registry disco-driven: gli shard su disco sono la source of truth.

    Nessuna coda in RAM di tutti i Data accodati: si tiene solo un buffer
    pari a `shard_size` record prima di flushare su disco. La lettura per
    lo split avviene in streaming (un shard per volta).
    """

    _instance: Optional["PositionQueueRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self, state_path: str, shard_size: int = DEFAULT_SHARD_SIZE) -> None:
        self._state_path = state_path
        self._shard_size = max(1, shard_size)
        self._spool_dir = _spool_dir_for(state_path)
        os.makedirs(self._spool_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._pending_shard: List[_QueuedPosition] = []
        self._next_local_ref = 0
        self._next_shard_index = 0
        self._enqueued_count = self._load_enqueued_count()
        self._last_split_assignment: Optional[Dict[str, str]] = None

        self._reload_existing_shards()

    # ------------------------------------------------------------------ #
    # Singleton
    # ------------------------------------------------------------------ #
    @classmethod
    def instance(cls, state_path: Optional[str] = None, shard_size: int = DEFAULT_SHARD_SIZE) -> "PositionQueueRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(state_path or DEFAULT_STATE_FILENAME, shard_size=shard_size)
            elif state_path is not None and state_path != cls._instance._state_path:
                logger.warning(
                    f"PositionQueueRegistry gia' istanziato con state_path="
                    f"'{cls._instance._state_path}'; ignorato il nuovo "
                    f"state_path='{state_path}' richiesto."
                )
            return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    # ------------------------------------------------------------------ #
    # Stato su disco
    # ------------------------------------------------------------------ #
    def _load_enqueued_count(self) -> int:
        if not os.path.exists(self._state_path):
            return 0
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return int(raw.get("total_enqueued", 0))
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            logger.warning(f"Stato coda in {self._state_path} illeggibile ({e}), riparto da 0.")
            return 0

    def _persist_enqueued_count(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._state_path)) or ".", exist_ok=True)
        tmp_path = self._state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"total_enqueued": self._enqueued_count}, f, indent=2)
        os.replace(tmp_path, self._state_path)

    def _existing_shard_paths(self) -> List[str]:
        pattern = os.path.join(self._spool_dir, SHARD_GLOB_PATTERN)
        return sorted(glob.glob(pattern))

    def _reload_existing_shards(self) -> None:
        """NON decompressa nulla: calcola solo il prossimo indice di shard.
        I dati restano su disco e verranno letti in streaming allo split.
        """
        shard_paths = self._existing_shard_paths()
        if not shard_paths:
            self._next_shard_index = 0
            return

        existing_indices: List[int] = []
        for path in shard_paths:
            name = os.path.basename(path)
            try:
                idx = int(name[len("shard_"):-len(".pt")])
                existing_indices.append(idx)
            except ValueError:
                continue
        self._next_shard_index = (max(existing_indices) + 1) if existing_indices else 0

        logger.info(
            f"[PositionQueueRegistry] Trovati {len(shard_paths)} shard residui in "
            f"{self._spool_dir}; verranno letti in streaming (nessuna decompressione in RAM)."
        )

    # ------------------------------------------------------------------ #
    # Scrittura shard
    # ------------------------------------------------------------------ #
    def _flush_pending_shard_locked(self) -> None:
        if not self._pending_shard:
            return

        records = [
            {
                "source_tag": item.source_tag,
                "group_key": item.group_key,
                "game_id": _extract_game_id(item.data),
                "data": compress_position_data(item.data, mate_n=item.group_key),
            }
            for item in self._pending_shard
        ]

        shard_path = os.path.join(
            self._spool_dir, SHARD_FILENAME_TEMPLATE.format(self._next_shard_index)
        )
        tmp_path = shard_path + ".tmp"
        torch.save(records, tmp_path)
        os.replace(tmp_path, shard_path)

        self._next_shard_index += 1
        # libera immediatamente il buffer: nessun Data resta in RAM dopo il flush
        self._pending_shard = []
        del records

    def _clear_spool(self) -> None:
        for path in self._existing_shard_paths():
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"Impossibile rimuovere lo shard consumato {path}: {e}")

    # ------------------------------------------------------------------ #
    # API pubblica
    # ------------------------------------------------------------------ #
    def enqueue(self, source_tag: str, data: Data, group_key: int) -> int:
        if not hasattr(data, "game_id") or data.game_id is None:
            raise PositionQueueError(
                f"enqueue rifiutato per source_tag='{source_tag}': il Data non ha un game_id valido."
            )

        with self._lock:
            local_ref = self._next_local_ref
            self._next_local_ref += 1

            item = _QueuedPosition(
                local_ref=local_ref,
                source_tag=source_tag,
                group_key=int(group_key),
                data=data,
            )
            self._pending_shard.append(item)
            self._enqueued_count += 1

            if len(self._pending_shard) >= self._shard_size:
                self._flush_pending_shard_locked()

        return local_ref

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending_shard)

    def flush(self) -> None:
        with self._lock:
            self._flush_pending_shard_locked()

    # ------------------------------------------------------------------ #
    # Streaming: pass 1 (metadata) e pass 2 (posizioni)
    # ------------------------------------------------------------------ #
    def iter_shard_metadata(self) -> Iterator[Tuple[str, int, str]]:
        """Pass 1: yield (game_id, group_key, source_tag) SENZA decompressione.

        Un solo shard in RAM per volta. Se lo shard e' legacy (senza game_id
        salvato), decompressione per singolo record.
        """
        self.flush()
        for path in self._existing_shard_paths():
            try:
                records = torch.load(path, weights_only=False, map_location="cpu")
            except Exception as e:
                logger.warning(f"Shard {path} illeggibile ({e}): scartato.")
                continue
            try:
                for rec in records:
                    gid = rec.get("game_id")
                    if gid is None:
                        # shard legacy: decompressione solo per questo record
                        d = decompress_position_data(rec["data"])
                        gid = _extract_game_id(d)
                        del d
                    yield gid, rec["group_key"], rec["source_tag"]
            finally:
                del records

    def iter_shard_positions(
        self, split_assignment: Dict[str, str]
    ) -> Iterator[Tuple[str, Data]]:
        """Pass 2: yield (split_name, Data) decomprimendo una posizione per volta.

        Un solo shard in RAM per volta (record compressi). Il Data viene
        decompresso on-the-fly e restituito al chiamante, che lo scrive e
        lo dimentica.
        """
        self.flush()
        for path in self._existing_shard_paths():
            try:
                records = torch.load(path, weights_only=False, map_location="cpu")
            except Exception as e:
                logger.warning(f"Shard {path} illeggibile ({e}): scartato.")
                continue
            try:
                for rec in records:
                    gid = rec.get("game_id")
                    data = decompress_position_data(rec["data"])
                    if gid is None:
                        gid = _extract_game_id(data)
                    split = split_assignment.get(gid)
                    if split is None:
                        raise PositionQueueError(
                            f"game_id={gid!r} senza split assegnato: "
                            f"la lista di shard e' cambiata tra pass 1 e pass 2?"
                        )
                    yield split, data
            finally:
                del records

    # ------------------------------------------------------------------ #
    # Assegnazione split (pass 1, leggera)
    # ------------------------------------------------------------------ #
    def build_split_assignment(
        self,
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int = 42,
    ) -> Dict[str, str]:
        """Calcola game_id -> split leggendo solo i metadati degli shard.

        RAM di picco: un dizionario {game_id: (group_key, source_tag)} e
        poi {game_id: split}. Per 1M finestre ~ 100 MB.
        """
        if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
            raise PositionQueueError("split_ratios deve contenere 3 valori che sommano a 1.0.")

        # ---- Pass 1: metadata-only, un solo shard in RAM per volta ----
        window_strata: Dict[str, Tuple[int, str]] = {}
        for gid, group_key, source_tag in self.iter_shard_metadata():
            prev = window_strata.get(gid)
            if prev is None:
                window_strata[gid] = (int(group_key), str(source_tag))
            elif prev != (int(group_key), str(source_tag)):
                raise PositionQueueError(
                    f"game_id={gid!r} ha metadati incoerenti tra shard: "
                    f"{prev} vs ({group_key}, {source_tag})."
                )

        if not window_strata:
            raise PositionQueueError("Nessuna finestra trovata negli shard.")

        groups_of_windows: Dict[Tuple[int, str], List[str]] = defaultdict(list)
        for gid, stratum in window_strata.items():
            groups_of_windows[stratum].append(gid)

        # ---- Assegnazione deterministica per strato ----
        generator = torch.Generator().manual_seed(seed)
        train_ratio, val_ratio, _test_ratio = split_ratios
        game_id_to_split: Dict[str, str] = {}

        for stratum in sorted(groups_of_windows.keys(), key=lambda s: (s[0], s[1])):
            gids = sorted(groups_of_windows[stratum])
            n = len(gids)
            n_train = min(int(train_ratio * n), n)
            n_val = min(int(val_ratio * n), n - n_train)

            perm = torch.randperm(n, generator=generator).tolist()
            shuffled = [gids[i] for i in perm]

            for gid in shuffled[:n_train]:
                game_id_to_split[gid] = "train"
            for gid in shuffled[n_train:n_train + n_val]:
                game_id_to_split[gid] = "val"
            for gid in shuffled[n_train + n_val:]:
                game_id_to_split[gid] = "test"

        with self._lock:
            self._last_split_assignment = dict(game_id_to_split)

        n_train = sum(1 for v in game_id_to_split.values() if v == "train")
        n_val = sum(1 for v in game_id_to_split.values() if v == "val")
        n_test = sum(1 for v in game_id_to_split.values() if v == "test")
        logger.info(
            f"[PositionQueueRegistry] build_split_assignment: "
            f"{len(window_strata)} finestre -> "
            f"train={n_train} val={n_val} test={n_test} "
            f"(strati distinti: {len(groups_of_windows)})."
        )
        return game_id_to_split

    # ------------------------------------------------------------------ #
    # API legacy / commit
    # ------------------------------------------------------------------ #
    def get_split_assignment(self) -> Dict[str, str]:
        with self._lock:
            if self._last_split_assignment is None:
                raise PositionQueueError(
                    "get_split_assignment chiamato prima di build_split_assignment: "
                    "nessuna assegnazione game_id -> split disponibile."
                )
            return dict(self._last_split_assignment)

    def commit_splits(self) -> None:
        with self._lock:
            self._persist_enqueued_count()
            self._clear_spool()
        logger.info(
            "[PositionQueueRegistry] commit_splits: file finali confermati "
            "su disco, spool residuo ripulito."
        )

    def build_splits(self, *args, **kwargs):  # pragma: no cover
        """DEPRECATO: materializza tutti gli split in RAM -> OOM con dataset grandi.

        Usa `build_split_assignment()` + `iter_shard_positions()`.
        """
        raise PositionQueueError(
            "build_splits e' deprecato (materializza tutto in RAM). "
            "Usa build_split_assignment() + iter_shard_positions()."
        )