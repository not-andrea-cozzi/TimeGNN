"""

Il chiamante (GamesBuilder, PuzzleBuilder) ha gia' calcolato, per una
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

    A differenza di SequenceQueueRegistry (superato), NON assegna piu' un
    game_id: quello e' gia' dentro ogni Data.game_id, assegnato a monte dal
    chiamante (condiviso da tutte le posizioni della stessa finestra di
    matto forzato). Questa classe si occupa di:
        1. accodare le posizioni in arrivo (FIFO, in-memory + shard su
           disco per sopravvivere a un crash, vedi docstring di modulo);
        2. ricaricare shard non ancora drenati da run precedenti,
           decomprimendoli in modo trasparente (vedi
           position_compression.py);
        3. drenare la coda e produrre gli split train/val/test,
           stratificati per group_key (tipicamente mate_n).

    Uso tipico:

        registry = PositionQueueRegistry.instance(state_path="Dataset/position_queue_state.json")

        for position_data in finestra_di_posizioni:
            registry.enqueue(source_tag="lichess", data=position_data, group_key=mate_n)

        splits = registry.build_splits(split_ratios=(0.7, 0.1, 0.2), seed=42)
        # splits = {"train": [Data...], "val": [...], "test": [...]}
        # (Data nel formato ORIGINALE, non compresso: la compressione e'
        # invisibile al chiamante)
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
        """Ritorna l'unica istanza del registry, creandola al primo uso.

        Args:
            state_path: percorso del file di stato (contatore diagnostico
                di posizioni processate). Usato SOLO alla primissima
                creazione dell'istanza nel processo. La directory di spool
                (shard su disco) viene derivata da questo path.
            shard_size: numero di posizioni accumulate in memoria prima di
                un flush su disco. Usato SOLO alla primissima creazione.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(state_path or DEFAULT_STATE_FILENAME, shard_size=shard_size)
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
    # SPOOL SU DISCO (shard batch): scrittura compressa, reload
    # decompresso, cleanup
    # ------------------------------------------------------------------
    def _existing_shard_paths(self) -> List[str]:
        """Shard presenti sul disco, ordinati per indice crescente (ordine
        di scrittura, irrilevante per la correttezza ma utile per log
        deterministici)."""
        pattern = os.path.join(self._spool_dir, SHARD_GLOB_PATTERN)
        return sorted(glob.glob(pattern))

    def _reload_existing_shards(self) -> None:
        """Ricarica in coda gli shard lasciati da una run precedente
        interrotta prima di un build_splits(). Chiamato SOLO da __init__:
        una volta ricaricati, gli shard restano sul disco finche'
        build_splits() non li consuma con successo (cosi' un secondo
        crash durante il reload stesso non perde nulla).

        Ogni Data letta da shard e' in formato COMPRESSO (vedi
        _flush_pending_shard_locked): viene decompressa qui, prima di
        rientrare nella coda in-memory, cosi' il resto della classe
        (pending_count, drain, build_splits) lavora sempre su Data nel
        formato originale.
        """
        shard_paths = self._existing_shard_paths()
        if not shard_paths:
            self._next_shard_index = 0
            return

        reloaded = 0
        for path in shard_paths:
            try:
                records = torch.load(path, weights_only=False)
            except Exception as e:
                logger.warning(
                    f"Shard {path} presente ma illeggibile ({e}): scartato "
                    f"(le posizioni in questo shard sono perse, ma il resto "
                    f"dello spool resta valido)."
                )
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

        # Prossimo indice shard: continua dopo l'ultimo gia' presente, cosi'
        # non si rischia di sovrascrivere shard esistenti non ancora
        # ripuliti (es. se il reload di uno shard e' fallito sopra).
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
                f"{len(shard_paths)} shard residui in {self._spool_dir} "
                f"(run precedente interrotta prima di build_splits)."
            )

    def _flush_pending_shard_locked(self) -> None:
        """Scrive su disco il buffer pendente come nuovo shard (scrittura
        atomica tmp+replace) e lo svuota. Il chiamante deve gia' detenere
        self._lock.

        Ogni Data viene compressa in modo lossless (vedi
        position_compression.compress_position_data) PRIMA di finire nello
        shard: questo riduce byte su disco/IO per lo spool intermedio
        senza alterare in alcun modo cio' che enqueue()/build_splits()
        espongono al chiamante (la decompressione avviene simmetricamente
        in _reload_existing_shards/_drain_all).

        group_key viene passato anche come mate_n al comprimere: e' gia'
        il valore di stratificazione (tipicamente la profondita' di
        matto), quindi salvarlo come attributo uint8 sul Data compresso e'
        gratuito e rende il dato pronto per un'eventuale stratificazione
        futura per n senza ulteriori modifiche allo spool.
        """
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
        """Rimuove tutti gli shard su disco: chiamato SOLO dopo che
        build_splits() ha gia' prodotto con successo gli split finali,
        quindi le posizioni sono ormai al sicuro nel risultato restituito
        al chiamante (che tipicamente le salva subito in merged_*.pt)."""
        for path in self._existing_shard_paths():
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"Impossibile rimuovere lo shard consumato {path}: {e}")

    # ------------------------------------------------------------------
    # ENQUEUE
    # ------------------------------------------------------------------
    def enqueue(self, source_tag: str, data: Data, group_key: int) -> int:
        """Accoda una SINGOLA posizione gia' assemblata.

        La posizione entra subito nella coda in-memory (visibile
        immediatamente a pending_count()/build_splits(), nel formato
        ORIGINALE non compresso) e viene inoltre accumulata in un buffer
        che, al raggiungimento di shard_size elementi, viene scritto su
        disco come shard COMPRESSO (vedi _flush_pending_shard_locked):
        questo garantisce che un crash del processo perda al massimo le
        ultime shard_size-1 posizioni non ancora flushate, invece
        dell'intera coda, riducendo nel contempo I/O e spazio su disco per
        lo spool.

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
            ValueError: se data contiene valori fuori dal dominio atteso
                per la compressione lossless (vedi
                position_compression.compress_position_data). Propagato al
                momento del flush su disco, non dell'enqueue stesso (la
                validazione avviene quando la posizione lascia il buffer
                in-memory verso lo shard).
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
            self._pending_shard.append(item)
            self._enqueued_count += 1

            if len(self._pending_shard) >= self._shard_size:
                self._flush_pending_shard_locked()

        return local_ref

    def pending_count(self) -> int:
        """Numero di posizioni attualmente in coda, non ancora drenate
        (indipendentemente dal fatto che siano gia' state flushate su
        disco come shard o solo nel buffer in-memory)."""
        return self._queue.qsize()

    def flush(self) -> None:
        """Forza la scrittura su disco del buffer pendente, anche se sotto
        soglia shard_size. Utile per checkpoint espliciti (es. log
        periodici in GamesBuilder.run()) senza dover aspettare
        build_splits()."""
        with self._lock:
            self._flush_pending_shard_locked()

    # ------------------------------------------------------------------
    # DRAIN + SPLIT STRATIFICATO
    # ------------------------------------------------------------------
    def _drain_all(self) -> List[_QueuedPosition]:
        with self._lock:
            # Flush finale del buffer parziale: senza questo, le ultime
            # posizioni sotto shard_size resterebbero SOLO in coda
            # in-memory (drenate correttamente in questa run, ma se
            # build_splits() fallisse DOPO il drain e PRIMA di ritornare,
            # non ci sarebbe piu' alcuno shard su disco da cui recuperarle
            # in una run successiva). Flush prima del drain elimina questa
            # finestra residua. Le posizioni gia' in coda in-memory (mai
            # passate da uno shard) sono gia' nel formato originale, non
            # richiedono decompressione.
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
        """Drena la coda (in-memory + shard su disco residui) e produce
        gli split train/val/test, SPLIT-SAFE per finestra (game_id) e
        stratificati per group_key (mate_n).

        Le Data risultanti sono sempre nel formato ORIGINALE (non
        compresso): quelle rimaste solo in coda in-memory non sono mai
        state compresse; quelle ricaricate da shard residui sono gia'
        state decompresse in _reload_existing_shards al momento del
        reload. Il chiamante non deve fare nulla di diverso rispetto a
        prima di questa modifica.

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

        Lo spool su disco (shard residui) viene ripulito SOLO se lo split
        va a buon fine: un'eccezione durante il calcolo lascia gli shard
        intatti, recuperabili da una run successiva (le posizioni gia'
        drenate dalla coda in-memory in QUESTA chiamata fallita non
        vengono pero' extra-persistite: se build_splits() solleva
        un'eccezione DOPO il drain, si assume che il chiamante rilanci
        l'intero processo, che ricarichera' gli shard rimasti al prossimo
        avvio di instance()).

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
        # Solo ora le posizioni drenate sono al sicuro nel risultato che
        # sta per essere ritornato al chiamante: gli shard su disco che le
        # contenevano non servono piu' come backup.
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