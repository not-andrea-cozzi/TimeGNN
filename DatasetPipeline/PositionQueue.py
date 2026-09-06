"""
PositionQueue.py

Sostituisce SequenceQueue.py (schema a grafo-sequenza, superato) con il
contratto aggiornato al nuovo PositionGraphSchema.py: un item in coda e'
ora UNA SINGOLA POSIZIONE (grafo spaziale a 64 nodi), non piu' un'intera
finestra di matto forzato collassata in un solo Data a nodo=ply.

MOTIVO DEL CAMBIO (vedi discussione di design): DualGATModel e
DualGATTimeAwareModel, i due modelli su cui verte l'ablation study
richiesto dal progetto, sono modelli PER-NODO su un grafo SPAZIALE (board),
non modelli di next-event-prediction su sequenze. Ogni ply di una finestra
di matto forzato diventa quindi un CAMPIONE INDIPENDENTE: la board a quel
ply, con la mossa migliore (quella realmente giocata nel PGN) come target.

game_id e ply (entrambi scalari dentro ogni Data, vedi PositionGraphSchema)
restano il modo per tracciare quali posizioni appartengono alla stessa
finestra/partita: lo split e la stratificazione avvengono per POSIZIONE
(ogni posizione e' un campione a se' nel train/val/test), ma un'analisi
futura che voglia raggruppare le posizioni per partita puo' sempre
raggrupparle per game_id (come gia' fa oggi PuzzleSequenceDataset per lo
schema precedente).

CONTRATTO
=========
Il chiamante (GamesBuilder, PuzzleGraphDataset) ha gia' calcolato, per una
finestra di matto forzato accettata, la lista di posizioni che la
compongono (ognuna gia' un torch_geometric.data.Data pronto, costruito con
PositionGraphSchema.build_position_data). Chiama, PER OGNI POSIZIONE:

    local_ref = registry.enqueue(source_tag, position_data, group_key)

group_key resta la chiave di stratificazione (tipicamente mate_n della
finestra di provenienza, replicato su ogni posizione della finestra).

Il game_id NON viene piu' assegnato da questa classe (a differenza della
versione precedente): e' gia' presente dentro position_data.game_id,
assegnato dal chiamante PRIMA di enqueue (una sola volta per finestra,
condiviso da tutte le sue posizioni). Questo e' un cambio deliberato
rispetto a SequenceQueue.py: quando l'unita' di coda era "una sequenza
intera", aveva senso assegnare un id per item in coda; ora che l'unita' e'
"una posizione", il game_id deve invece essere condiviso da PIU' item
(tutte le posizioni della stessa finestra), quindi la sua assegnazione
torna naturalmente a monte, nel builder che gia' conosce l'appartenenza
alla finestra.

PERSISTENZA, SPLIT, SINGLETON: stesso design di SequenceQueue.py (vedi
quel modulo per la discussione completa di persistenza differita e
concorrenza), qui non ripetuta.
"""
from __future__ import annotations

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

logger = logging.getLogger("position_queue")

DEFAULT_STATE_FILENAME = "position_queue_state.json"


class PositionQueueError(RuntimeError):
    """Errore di uso scorretto della coda (es. build_splits senza dati)."""


@dataclass
class _QueuedPosition:
    """Una posizione in coda, in attesa di essere drenata in build_splits."""
    local_ref: int
    source_tag: str
    group_key: int
    data: Data


class PositionQueueRegistry:
    """Singleton: coda in-memory di posizioni + split stratificato.

    A differenza di SequenceQueueRegistry (superato), NON assegna piu' un
    game_id: quello e' gia' dentro ogni Data.game_id, assegnato a monte dal
    chiamante (condiviso da tutte le posizioni della stessa finestra di
    matto forzato). Questa classe si occupa solo di:
        1. accodare le posizioni in arrivo (FIFO, in-memory);
        2. drenare la coda e produrre gli split train/val/test,
           stratificati per group_key (tipicamente mate_n).

    Uso tipico:

        registry = PositionQueueRegistry.instance(state_path="Dataset/position_queue_state.json")

        for position_data in finestra_di_posizioni:
            registry.enqueue(source_tag="lichess", data=position_data, group_key=mate_n)

        splits = registry.build_splits(split_ratios=(0.7, 0.1, 0.2), seed=42)
        # splits = {"train": [Data...], "val": [...], "test": [...]}
    """

    _instance: Optional["PositionQueueRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self, state_path: str) -> None:
        """Non chiamare direttamente: usare PositionQueueRegistry.instance()."""
        self._state_path = state_path
        self._lock = threading.Lock()
        self._queue: "Queue[_QueuedPosition]" = Queue()
        self._next_local_ref = 0
        self._enqueued_count = self._load_enqueued_count()

    # ------------------------------------------------------------------
    # SINGLETON
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls, state_path: Optional[str] = None) -> "PositionQueueRegistry":
        """Ritorna l'unica istanza del registry, creandola al primo uso.

        Args:
            state_path: percorso del file di stato (contatore diagnostico
                di posizioni processate; NON un contatore di id, dato che
                il game_id non e' piu' allocato qui). Usato SOLO alla
                primissima creazione dell'istanza nel processo.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(state_path or DEFAULT_STATE_FILENAME)
            elif state_path is not None and state_path != cls._instance._state_path:
                logger.warning(
                    f"PositionQueueRegistry gia' istanziato con state_path="
                    f"'{cls._instance._state_path}'; ignorato il nuovo "
                    f"state_path='{state_path}' richiesto (il Singleton non "
                    f"puo' cambiare file di stato a runtime)."
                )
            return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        """Distrugge l'istanza Singleton. Da usare SOLO nei test."""
        with cls._instance_lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # PERSISTENZA (solo contatore diagnostico, nessun id da allocare qui)
    # ------------------------------------------------------------------
    def _load_enqueued_count(self) -> int:
        if not os.path.exists(self._state_path):
            return 0
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return int(raw.get("total_enqueued", 0))
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            logger.warning(
                f"Stato coda in {self._state_path} illeggibile ({e}), "
                f"riparto da total_enqueued=0."
            )
            return 0

    def _persist_enqueued_count(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._state_path)) or ".", exist_ok=True)
        tmp_path = self._state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"total_enqueued": self._enqueued_count}, f, indent=2)
        os.replace(tmp_path, self._state_path)

    # ------------------------------------------------------------------
    # ENQUEUE
    # ------------------------------------------------------------------
    def enqueue(self, source_tag: str, data: Data, group_key: int) -> int:
        """Accoda una SINGOLA posizione gia' assemblata.

        Args:
            source_tag: etichetta della sorgente (es. "lichess", "fics",
                "club", "puzzle"). Solo metadato/statistiche.
            data: torch_geometric.data.Data prodotto da
                PositionGraphSchema.build_position_data. Deve gia'
                contenere game_id (assegnato dal chiamante, condiviso da
                tutte le posizioni della stessa finestra di matto forzato).
            group_key: chiave di stratificazione per lo split (tipicamente
                mate_n della finestra di provenienza).

        Returns:
            local_ref: intero opaco, univoco per questa istanza, utile
            solo per correlare log/debug (non e' il game_id: quello e'
            dentro data.game_id).

        Raises:
            PositionQueueError: se data non ha un game_id valido (fail
                fast: un game_id mancante indicherebbe un bug a monte nel
                builder, meglio scoprirlo qui che silenziosamente a valle).
        """
        if not hasattr(data, "game_id") or data.game_id is None:
            raise PositionQueueError(
                f"enqueue rifiutato per source_tag='{source_tag}': il Data "
                f"non ha un game_id valido (deve essere assegnato dal "
                f"chiamante prima di enqueue, condiviso da tutte le "
                f"posizioni della stessa finestra di matto forzato)."
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
            self._enqueued_count += 1

        return local_ref

    def pending_count(self) -> int:
        """Numero di posizioni attualmente in coda, non ancora drenate."""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # DRAIN + SPLIT STRATIFICATO
    # ------------------------------------------------------------------
    def _drain_all(self) -> List[_QueuedPosition]:
        drained: List[_QueuedPosition] = []
        with self._lock:
            while not self._queue.empty():
                drained.append(self._queue.get())
        return drained

    def build_splits(
        self,
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int = 42,
    ) -> Dict[str, List[Data]]:
        """Drena la coda e produce gli split train/val/test, SPLIT-SAFE
        per finestra (game_id) e stratificati per group_key (mate_n).

        FIX LEAKAGE (rispetto alla prima versione di questo modulo): lo
        split NON avviene piu' per singola posizione, ma per FINESTRA
        (game_id). Tutte le posizioni con lo stesso game_id vengono prima
        raggruppate in un unico blocco indivisibile; i blocchi vengono poi
        stratificati per group_key ESATTAMENTE come prima, con la
        differenza che l'unita' che viene assegnata a un singolo split e'
        ora "una finestra intera" invece di "una posizione". Questo
        garantisce che nessuna finestra di matto forzato abbia posizioni
        sparse su piu' split contemporaneamente (niente data leakage tra
        train/val/test).

        Nota implicita: dato che tutte le posizioni di una stessa finestra
        condividono lo stesso group_key (es. mate_n, che e' un attributo
        della finestra, non della singola posizione), raggruppare per
        game_id PRIMA di stratificare per group_key e raggruppare per
        group_key PRIMA di sotto-raggruppare per game_id producono la
        STESSA partizione in gruppi: la differenza pratica sta solo nel
        fatto che qui il campionamento/shuffle per split opera su BLOCCHI
        (finestre), non su singole posizioni.

        Args:
            split_ratios: proporzioni (train, val, test), devono sommare a
                1.0.
            seed: seed per lo shuffle deterministico.

        Returns:
            {"train": [Data...], "val": [Data...], "test": [Data...]}.

        Raises:
            PositionQueueError: se la coda e' vuota, o se split_ratios non
                somma a 1.0.
        """
        if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
            raise PositionQueueError(
                f"split_ratios deve contenere 3 valori che sommano a 1.0 "
                f"(ricevuto {split_ratios}, somma={sum(split_ratios)})."
            )

        drained = self._drain_all()
        if not drained:
            raise PositionQueueError(
                "build_splits chiamato con la coda vuota: nessuna posizione "
                "e' stata accodata (o e' gia' stata drenata da una "
                "build_splits precedente)."
            )

        # Passo 1: raggruppa le posizioni per game_id (una finestra = un
        # blocco indivisibile). Ogni blocco eredita il group_key comune
        # alle sue posizioni (mate_n): verificato esplicitamente che sia
        # davvero comune, per non mascherare un bug a monte nel builder.
        windows: Dict[int, List[_QueuedPosition]] = defaultdict(list)
        for item in drained:
            game_id = int(item.data.game_id.item()) if hasattr(item.data.game_id, "item") else int(item.data.game_id)
            windows[game_id].append(item)

        window_group_key: Dict[int, int] = {}
        for game_id, items in windows.items():
            keys_in_window = {item.group_key for item in items}
            if len(keys_in_window) != 1:
                raise PositionQueueError(
                    f"game_id={game_id} ha posizioni con group_key diversi "
                    f"({sorted(keys_in_window)}): il chiamante deve garantire "
                    f"che tutte le posizioni della stessa finestra condividano "
                    f"lo stesso group_key (es. mate_n)."
                )
            window_group_key[game_id] = keys_in_window.pop()

        # Passo 2: raggruppa i BLOCCHI (finestre) per group_key, per la
        # stratificazione (stesso algoritmo di prima, ma l'unita' e' ora
        # la finestra, non la singola posizione).
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

        # Shuffle finale per split (a livello di POSIZIONE, non di
        # finestra: una volta che le finestre sono assegnate in modo
        # split-safe, mescolare le singole posizioni dentro ogni split e'
        # sicuro e desiderabile per il DataLoader a valle).
        for split_name, data_list in result.items():
            if not data_list:
                continue
            perm = torch.randperm(len(data_list), generator=generator)
            result[split_name] = [data_list[i] for i in perm.tolist()]

        self._persist_enqueued_count()
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