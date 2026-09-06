"""
Il chiamante (GamesBuilder, PuzzleBuilder) ha gia' calcolato, per una
finestra di matto forzato accettata, la lista di posizioni che la
compongono (ognuna gia' un torch_geometric.data.Data pronto, costruito con
PositionGraphSchema.build_position_data). Chiama, PER OGNI POSIZIONE:

    local_ref = registry.enqueue(source_tag, position_data, group_key)

group_key resta la chiave di stratificazione (tipicamente mate_n della
finestra di provenienza, replicato su ogni posizione della finestra).

Il game_id e' assegnato dal CHIAMANTE (GamesBuilder._enqueue_window,
PuzzleBuilder.run) PRIMA di enqueue, come uuid.uuid4().int troncato a 63
bit -- univoco per costruzione, per qualunque run passata, presente o
futura, senza bisogno di alcuna mappa di traduzione persistita.

STORIA DI QUESTO DESIGN (perche' NON esiste piu' una _id_map)
================================================================
Una versione precedente di questa classe assegnava game_id "grezzi" come
contatori locali al processo (self._next_game_id in GamesBuilder,
next_game_id in PuzzleBuilder), poi li rimappava qui a un "safe_id"
univoco globale tramite una tabella (source_tag, orig_id) -> safe_id.

Questo design aveva un difetto strutturale: un contatore locale al
processo riparte da 0 ad ogni riavvio. Quando la pipeline veniva
interrotta e ripresa (crash, resume manuale), run diverse generavano lo
STESSO orig_id per finestre reali DIVERSE (l'ordine di completamento dei
worker paralleli, via pool.imap_unordered, non e' deterministico tra
run). La tabella di traduzione, se non persistita correttamente tra
riavvii (o se popolata in modo incoerente per via di uno stato "ctx"
perso quando uno step veniva skippato come "gia' completato"), poteva
assegnare lo STESSO safe_id a due finestre con group_key (mate_n)
diversi -- il bug osservato in produzione ("game_id=1 ha posizioni con
group_key diversi ([1, 8])").

La soluzione adottata elimina l'intera classe di bug alla radice: invece
di mitigare le collisioni con una tabella di traduzione (che a sua volta
richiede persistenza corretta, gestione di reload, e coordinamento tra
GamesBuilder e PuzzleBuilder per gli offset), il game_id e' ora generato
come identificatore univoco per costruzione (uuid4), niente affatto
soggetto a riavvii di processo o ordine di completamento dei worker.
PositionQueueRegistry non deve piu' occuparsi di tradurre nulla: il
game_id che riceve in enqueue() e' gia' l'identificatore finale, stabile.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from DatasetPipeline.Utils.position_compression import (
    compress_position_data,
    decompress_position_data,
)

logger = logging.getLogger("position_queue")

DEFAULT_STATE_FILENAME = "position_queue_state.json"
DEFAULT_SHARD_SIZE = 500
SHARD_FILENAME_TEMPLATE = "shard_{:08d}.pt"
SHARD_GLOB_PATTERN = "shard_*.pt"


class PositionQueueError(RuntimeError):
    """Errore di uso scorretto della coda (es. build_splits senza dati)."""


@dataclass
class _QueuedPosition:
    """Una posizione in coda, in attesa di essere drenata in build_splits.

    `data` e' sempre tenuta in formato ORIGINALE (non compresso) mentre e'
    in memoria (coda in-memory + buffer pendente): la compressione si
    applica esclusivamente al momento della scrittura su disco (vedi
    _flush_pending_shard_locked), cosi' il pending_count()/drain in-memory
    prima di un eventuale flush non richiede mai una decompressione.
    """
    local_ref: int
    source_tag: str
    group_key: int
    data: Data


def _spool_dir_for(state_path: str) -> str:
    """Deriva la directory di spool dal path del file di stato JSON:
    stessa cartella, sottocartella dedicata basata sul nome del file di
    stato (senza estensione) + '_spool', cosi' piu' registry con
    state_path diversi (es. in test) non condividono lo spool per errore.
    """
    base_dir = os.path.dirname(os.path.abspath(state_path)) or "."
    stem = os.path.splitext(os.path.basename(state_path))[0]
    return os.path.join(base_dir, f"{stem}_spool")


class PositionQueueRegistry:
    """Singleton: coda di posizioni (in-memory + spool su disco per
    resume) + split stratificato.

    Non assegna e non traduce piu' alcun game_id: quello e' gia' un
    identificatore univoco per costruzione (uuid troncato), assegnato dal
    chiamante prima di enqueue (vedi docstring di modulo). Questa classe
    si occupa esclusivamente di:
        1. accodare le posizioni in arrivo (FIFO, in-memory + shard su
           disco per sopravvivere a un crash, vedi docstring di modulo);
        2. ricaricare shard non ancora drenati da run precedenti,
           decomprimendoli in modo trasparente (vedi
           position_compression.py);
        3. drenare la coda e produrre gli split train/val/test,
           stratificati per group_key (tipicamente mate_n).
    """

    _instance: Optional["PositionQueueRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self, state_path: str, shard_size: int = DEFAULT_SHARD_SIZE) -> None:
        """Non chiamare direttamente: usare PositionQueueRegistry.instance()."""
        self._state_path = state_path
        self._shard_size = max(1, shard_size)
        self._spool_dir = _spool_dir_for(state_path)
        os.makedirs(self._spool_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._queue: "Queue[_QueuedPosition]" = Queue()
        self._pending_shard: List[_QueuedPosition] = []
        self._next_local_ref = 0
        self._next_shard_index = 0
        self._enqueued_count = self._load_enqueued_count()

        self._reload_existing_shards()

    # ------------------------------------------------------------------
    # SINGLETON
    # ------------------------------------------------------------------
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
        """Distrugge l'istanza Singleton. Da usare SOLO nei test."""
        with cls._instance_lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # PERSISTENZA (contatore diagnostico)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # SPOOL SU DISCO (shard batch)
    # ------------------------------------------------------------------
    def _existing_shard_paths(self) -> List[str]:
        pattern = os.path.join(self._spool_dir, SHARD_GLOB_PATTERN)
        return sorted(glob.glob(pattern))

    def _reload_existing_shards(self) -> None:
        shard_paths = self._existing_shard_paths()
        if not shard_paths:
            self._next_shard_index = 0
            return

        reloaded = 0
        for path in shard_paths:
            try:
                records = torch.load(path, weights_only=False)
            except Exception as e:
                logger.warning(f"Shard {path} illeggibile ({e}): scartato.")
                continue

            for rec in records:
                decompressed_data = decompress_position_data(rec["data"])
                item = _QueuedPosition(
                    local_ref=self._next_local_ref,
                    source_tag=rec["source_tag"],
                    group_key=rec["group_key"],
                    data=decompressed_data,
                )
                self._next_local_ref += 1
                self._queue.put(item)
                reloaded += 1

        existing_indices = []
        for path in shard_paths:
            name = os.path.basename(path)
            try:
                idx = int(name[len("shard_"):-len(".pt")])
                existing_indices.append(idx)
            except ValueError:
                continue
        self._next_shard_index = (max(existing_indices) + 1) if existing_indices else 0

        if reloaded:
            logger.info(
                f"[PositionQueueRegistry] Ricaricate {reloaded:,} posizioni da "
                f"{len(shard_paths)} shard residui in {self._spool_dir}."
            )

    def _flush_pending_shard_locked(self) -> None:
        if not self._pending_shard:
            return

        records = [
            {
                "source_tag": item.source_tag,
                "group_key": item.group_key,
                "data": compress_position_data(item.data, mate_n=item.group_key),
            }
            for item in self._pending_shard
        ]

        shard_path = os.path.join(self._spool_dir, SHARD_FILENAME_TEMPLATE.format(self._next_shard_index))
        tmp_path = shard_path + ".tmp"
        torch.save(records, tmp_path)
        os.replace(tmp_path, shard_path)

        self._next_shard_index += 1
        self._pending_shard = []

    def _clear_spool(self) -> None:
        for path in self._existing_shard_paths():
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"Impossibile rimuovere lo shard consumato {path}: {e}")

    # ------------------------------------------------------------------
    # ENQUEUE
    # ------------------------------------------------------------------
    def enqueue(self, source_tag: str, data: Data, group_key: int) -> int:
        """Accoda una posizione. Il game_id dentro `data` DEVE essere gia'
        un identificatore univoco per costruzione (uuid troncato,
        assegnato dal chiamante) -- questa classe non lo traduce, non lo
        rimappa, non lo verifica per unicita' (farlo richiederebbe
        comunque una tabella globale, esattamente il design abbandonato:
        vedi docstring di modulo). Si fida della garanzia probabilistica
        di uuid4 (collisione ~1 su 2^63, trascurabile)."""
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
            self._queue.put(item)
            self._pending_shard.append(item)
            self._enqueued_count += 1

            if len(self._pending_shard) >= self._shard_size:
                self._flush_pending_shard_locked()

        return local_ref

    def pending_count(self) -> int:
        return self._queue.qsize()

    def flush(self) -> None:
        with self._lock:
            self._flush_pending_shard_locked()

    # ------------------------------------------------------------------
    # DRAIN + SPLIT STRATIFICATO
    # ------------------------------------------------------------------
    def _drain_all(self) -> List[_QueuedPosition]:
        with self._lock:
            self._flush_pending_shard_locked()
            drained: List[_QueuedPosition] = []
            while not self._queue.empty():
                drained.append(self._queue.get())
        return drained

    def build_splits(
        self,
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int = 42,
    ) -> Dict[str, List[Data]]:
        if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
            raise PositionQueueError("split_ratios deve contenere 3 valori che sommano a 1.0.")

        drained = self._drain_all()
        if not drained:
            raise PositionQueueError("build_splits chiamato con la coda vuota.")

        windows: Dict[int, List[_QueuedPosition]] = defaultdict(list)
        for item in drained:
            game_id = int(item.data.game_id.item()) if hasattr(item.data.game_id, "item") else int(item.data.game_id)
            windows[game_id].append(item)

        window_group_key: Dict[int, int] = {}
        for game_id, items in windows.items():
            keys_in_window = {item.group_key for item in items}
            if len(keys_in_window) != 1:
                raise PositionQueueError(
                    f"game_id={game_id} ha posizioni con group_key diversi ({sorted(keys_in_window)}). "
                    f"Con game_id generati come uuid questo non dovrebbe accadere per collisione "
                    f"accidentale: verifica se lo stesso oggetto Data e' stato accodato piu' volte, "
                    f"o se c'e' un bug nel chiamante che riusa un game_id tra finestre diverse."
                )
            window_group_key[game_id] = keys_in_window.pop()

        groups_of_windows: Dict[int, List[int]] = defaultdict(list)
        for game_id, key in window_group_key.items():
            groups_of_windows[key].append(game_id)

        generator = torch.Generator().manual_seed(seed)
        train_ratio, val_ratio, _test_ratio = split_ratios
        result: Dict[str, List[Data]] = {"train": [], "val": [], "test": []}
        window_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

        for key in sorted(groups_of_windows.keys()):
            game_ids_in_group = groups_of_windows[key]
            n = len(game_ids_in_group)

            n_train = min(int(train_ratio * n), n)
            n_val = min(int(val_ratio * n), n - n_train)

            perm = torch.randperm(n, generator=generator)
            shuffled_game_ids = [game_ids_in_group[i] for i in perm.tolist()]

            split_assignment = (
                [("train", gid) for gid in shuffled_game_ids[:n_train]]
                + [("val", gid) for gid in shuffled_game_ids[n_train:n_train + n_val]]
                + [("test", gid) for gid in shuffled_game_ids[n_train + n_val:]]
            )

            for split_name, game_id in split_assignment:
                for item in windows[game_id]:
                    result[split_name].append(item.data)
                window_counts[split_name] += 1

        for split_name, data_list in result.items():
            if not data_list:
                continue
            perm = torch.randperm(len(data_list), generator=generator)
            result[split_name] = [data_list[i] for i in perm.tolist()]

        self._persist_enqueued_count()
        self._clear_spool()
        self._log_distribution(result, groups_of_windows, window_counts)

        return result

    def _log_distribution(
        self,
        result: Dict[str, List[Data]],
        groups_of_windows: Dict[int, List[int]],
        window_counts: Dict[str, int],
    ) -> None:
        total_positions = sum(len(v) for v in result.values())
        total_windows = sum(window_counts.values())
        logger.info(
            f"[PositionQueueRegistry] build_splits completato (split-safe per finestra): "
            f"{len(groups_of_windows)} group_key distinti, "
            f"{total_windows} finestre, {total_positions} posizioni totali."
        )
        for split_name in ("train", "val", "test"):
            logger.info(
                f"    {split_name}: {window_counts[split_name]} finestre, "
                f"{len(result[split_name])} posizioni"
            )