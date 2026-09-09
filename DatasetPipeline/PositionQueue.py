from __future__ import annotations

import glob
import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

# FLUSH PERIODICO: se una run ha throughput basso, il buffer puo' non
# raggiungere mai shard_size elementi per lunghi tratti (es. filtri molto
# selettivi che scartano quasi tutto). Senza un limite di TEMPO, quelle
# poche posizioni accettate restano in _pending_shard per ore: non e' la
# crescita illimitata del vecchio bug (qui il tetto e' shard_size), ma su
# run lunghe e' comunque RAM tenuta inutilmente e un rischio maggiore di
# perdita in caso di crash prima del prossimo shard_size pieno. Un flush
# forzato ogni DEFAULT_FLUSH_INTERVAL_SECONDS elimina entrambi i problemi.
DEFAULT_FLUSH_INTERVAL_SECONDS = 30 * 60  # 20 minuti


class PositionQueueError(RuntimeError):
    """Errore di uso scorretto della coda (es. build_splits senza dati)."""


@dataclass
class _QueuedPosition:
    """Una posizione, usata SOLO come struttura di trasporto per lo shard
    corrente in RAM (`_pending_shard`, al massimo `shard_size` elementi).
    Non esiste piu' una coda in-memory che tiene TUTTE le posizioni della
    run: quella era la causa della crescita lineare di RAM (vedi
    docstring di classe, sezione FIX RAM)."""
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
    """Singleton: accumula posizioni SOLO su disco (shard), mai in una
    coda in-memory persistente per tutta la run.

    ============================================================================
    FIX RAM (motivo della riscrittura rispetto alla versione precedente)
    ============================================================================
    La versione precedente manteneva OGNI posizione accodata in DUE posti
    contemporaneamente:
        1. self._queue (una Queue "coda persistente", MAI svuotata finche'
           non si chiamava build_splits() a fine pipeline)
        2. self._pending_shard (buffer temporaneo, svuotato ogni
           shard_size elementi, scritto su disco)

    Il punto (2) funzionava correttamente (scrive e libera). Il punto (1)
    e' la causa della crescita: enqueue() faceva SEMPRE self._queue.put(item),
    quindi ogni posizione mai piu' liberata dalla RAM per l'intera durata
    della run, anche se gia' scritta su disco in uno shard. Su una run di
    ore/milioni di posizioni, la RAM cresce linearmente con
    enqueued_count e non si stabilizza mai -> rallentamento (GC pressure,
    swap, cache thrashing) che spiega il calo di games/sec osservato.

    FIX: eliminata la coda persistente. enqueue() scrive SOLO nel buffer
    temporaneo (_pending_shard), che viene flushato su disco appena
    raggiunge shard_size e poi SVUOTATO (lista nuova, non piu'
    referenziata). L'unica fonte di verita' per il conteggio/contenuto
    delle posizioni accodate sono gli shard su disco in self._spool_dir:
    build_splits() li legge, decomprime, costruisce gli split e poi li
    cancella (self._clear_spool()), esattamente come prima. Il costo e'
    un giro di I/O in piu' in build_splits() per rileggere l'ultimo/gli
    ultimi shard, trascurabile rispetto al problema di RAM eliminato.

    Se il processo viene interrotto E riavviato, _reload_existing_shards()
    ricostruisce lo stato leggendo gli shard rimasti su disco (comportamento
    INVARIATO rispetto a prima) -- ma ora il "reload" alimenta di nuovo
    SOLO il buffer temporaneo per il prossimo flush, non una coda persistente:
    per questo motivo, allo startup, gli shard pre-esistenti vengono
    ricaricati e RISCRITTI subito in shard "canonici" (stessa logica,
    nessuna sorpresa) invece di restare appesi in RAM.

    ============================================================================
    FLUSH PERIODICO A TEMPO (flush_interval_seconds, default 20 minuti)
    ============================================================================
    Il flush a shard_size copre il caso "alto throughput": il buffer si
    riempie in fretta e va su disco spesso. Su run LUNGHE a basso
    throughput (filtri molto selettivi, Stockfish lento, ecc.) il buffer
    puo' restare sotto shard_size per ore: quelle poche posizioni restano
    in RAM inutilmente a lungo E sono a rischio se il processo crasha
    prima di raggiungere shard_size. Un thread di background chiama
    flush() ogni flush_interval_seconds indipendentemente dal riempimento
    del buffer, cosi' nessuna posizione resta in RAM piu' di
    flush_interval_seconds. Il thread si ferma con _stop_flush_timer()
    (chiamato automaticamente da build_splits() e da shutdown()).
    ============================================================================
    """

    _instance: Optional["PositionQueueRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        state_path: str,
        shard_size: int = DEFAULT_SHARD_SIZE,
        flush_interval_seconds: Optional[float] = DEFAULT_FLUSH_INTERVAL_SECONDS,
    ) -> None:
        """Non chiamare direttamente: usare PositionQueueRegistry.instance().

        Args:
            flush_interval_seconds: se impostato (default 20 minuti), un
                thread di background chiama flush() a intervalli regolari
                indipendentemente da shard_size, cosi' su run a basso
                throughput il buffer non resta pieno per ore. None
                disabilita il flush a tempo (comportamento solo a
                shard_size, come prima).
        """
        self._state_path = state_path
        self._shard_size = max(1, shard_size)
        self._spool_dir = _spool_dir_for(state_path)
        os.makedirs(self._spool_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._pending_shard: List[_QueuedPosition] = []
        self._next_shard_index = 0
        self._enqueued_count = self._load_enqueued_count()

        # A differenza della versione precedente, qui NON ricarichiamo il
        # contenuto degli shard in memoria: verifichiamo solo che esistano
        # e determiniamo il prossimo indice libero. Il contenuto resta su
        # disco fino a build_splits().
        self._bootstrap_shard_index()

        # --- FLUSH PERIODICO A TEMPO ---
        self._flush_interval_seconds = flush_interval_seconds
        self._flush_timer_stop = threading.Event()
        self._flush_timer_thread: Optional[threading.Thread] = None
        if self._flush_interval_seconds is not None and self._flush_interval_seconds > 0:
            self._flush_timer_thread = threading.Thread(
                target=self._flush_timer_loop, daemon=True, name="PositionQueueFlushTimer"
            )
            self._flush_timer_thread.start()
            logger.info(
                f"[PositionQueueRegistry] Flush periodico attivo: ogni "
                f"{self._flush_interval_seconds / 60:.1f} minuti, indipendentemente "
                f"da shard_size={self._shard_size}."
            )

    # ------------------------------------------------------------------
    # SINGLETON
    # ------------------------------------------------------------------
    @classmethod
    def instance(
        cls,
        state_path: Optional[str] = None,
        shard_size: int = DEFAULT_SHARD_SIZE,
        flush_interval_seconds: Optional[float] = DEFAULT_FLUSH_INTERVAL_SECONDS,
    ) -> "PositionQueueRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(
                    state_path or DEFAULT_STATE_FILENAME,
                    shard_size=shard_size,
                    flush_interval_seconds=flush_interval_seconds,
                )
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
            if cls._instance is not None:
                cls._instance._stop_flush_timer()
            cls._instance = None

    # ------------------------------------------------------------------
    # FLUSH PERIODICO A TEMPO (thread di background)
    # ------------------------------------------------------------------
    def _flush_timer_loop(self) -> None:
        """Chiama flush() ogni _flush_interval_seconds, finche' non viene
        segnalato lo stop. wait() con timeout invece di sleep() cosi' lo
        stop e' immediato (non aspetta la fine dell'intervallo corrente)."""
        while not self._flush_timer_stop.wait(self._flush_interval_seconds):
            try:
                pending_before = self.pending_count()
                if pending_before == 0:
                    continue
                self.flush()
                logger.info(
                    f"[PositionQueueRegistry] Flush periodico (timer): "
                    f"{pending_before} posizioni scritte su shard."
                )
            except Exception as e:
                logger.warning(f"[PositionQueueRegistry] Flush periodico fallito: {e}")

    def _stop_flush_timer(self) -> None:
        """Ferma il thread di flush periodico e attende la sua uscita.
        Chiamare esplicitamente a fine pipeline (o da reset_for_testing)
        per non lasciare il thread daemon a girare a vuoto dopo che il
        registry non serve piu'."""
        if self._flush_timer_thread is not None:
            self._flush_timer_stop.set()
            self._flush_timer_thread.join(timeout=5.0)
            self._flush_timer_thread = None

    def shutdown(self) -> None:
        """Ferma il timer di flush periodico e fa un ultimo flush del
        buffer residuo. Da chiamare a fine pipeline, prima o al posto di
        build_splits() se non si vuole drenare subito."""
        self._stop_flush_timer()
        self.flush()

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
    # SPOOL SU DISCO (shard batch) — UNICA FONTE DI VERITA'
    # ------------------------------------------------------------------
    def _existing_shard_paths(self) -> List[str]:
        pattern = os.path.join(self._spool_dir, SHARD_GLOB_PATTERN)
        return sorted(glob.glob(pattern))

    def _bootstrap_shard_index(self) -> None:
        """Determina il prossimo indice di shard libero SENZA caricare il
        contenuto in RAM (a differenza della vecchia _reload_existing_shards,
        che decomprimeva e metteva in coda ogni posizione residua). Se ci
        sono shard residui da una run interrotta, restano su disco cosi'
        come sono: build_splits() li leggera' insieme a quelli nuovi."""
        shard_paths = self._existing_shard_paths()
        if not shard_paths:
            self._next_shard_index = 0
            return

        existing_indices = []
        residual_count = 0
        for path in shard_paths:
            name = os.path.basename(path)
            try:
                idx = int(name[len("shard_"):-len(".pt")])
                existing_indices.append(idx)
            except ValueError:
                continue
            try:
                records = torch.load(path, weights_only=False)
                residual_count += len(records)
            except Exception as e:
                logger.warning(f"Shard {path} illeggibile ({e}): verra' ignorato da build_splits.")

        self._next_shard_index = (max(existing_indices) + 1) if existing_indices else 0

        if residual_count:
            logger.info(
                f"[PositionQueueRegistry] Rilevati {residual_count:,} posizioni residue in "
                f"{len(shard_paths)} shard da run precedente in {self._spool_dir} "
                f"(rimarranno su disco, lette direttamente da build_splits senza "
                f"caricarle ora in RAM)."
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
        # Svuota la lista: questi oggetti Data non sono piu' referenziati
        # da nessuna parte del processo dopo questa riga (a differenza
        # della vecchia versione, dove restavano vivi in self._queue).
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
    def enqueue(self, source_tag: str, data: Data, group_key: int) -> None:
        """Accoda una posizione. Scrive SOLO nel buffer temporaneo
        (_pending_shard, al massimo shard_size elementi in RAM), mai in
        una struttura che vive per tutta la run. Non ritorna piu' un
        local_ref progressivo (non serviva a nessun chiamante osservato
        e teneva un contatore per una coda che ora non esiste piu';
        se serve un ID, usare data.game_id + ply, gia' univoci)."""
        if not hasattr(data, "game_id") or data.game_id is None:
            raise PositionQueueError(
                f"enqueue rifiutato per source_tag='{source_tag}': il Data non ha un game_id valido."
            )

        with self._lock:
            item = _QueuedPosition(
                source_tag=source_tag,
                group_key=int(group_key),
                data=data,
            )
            self._pending_shard.append(item)
            self._enqueued_count += 1

            if len(self._pending_shard) >= self._shard_size:
                self._flush_pending_shard_locked()

    def pending_count(self) -> int:
        """Numero di posizioni nel buffer NON ancora scritte su disco
        (0..shard_size-1). NON e' piu' il totale accodato nella run: quel
        dato, se serve, e' self._enqueued_count (diagnostico, persistito)."""
        with self._lock:
            return len(self._pending_shard)

    def flush(self) -> None:
        with self._lock:
            self._flush_pending_shard_locked()

    # ------------------------------------------------------------------
    # DRAIN (da disco, non da RAM) + SPLIT STRATIFICATO
    # ------------------------------------------------------------------
    def _drain_all_from_disk(self) -> List[_QueuedPosition]:
        """Legge TUTTI gli shard su disco (incluso l'ultimo flush fatto
        qui sopra) e li decomprime in una lista temporanea, SOLO per la
        durata di build_splits(). A differenza della vecchia
        implementazione, questa lista non e' mai accumulata durante la
        run: esiste solo dentro questa chiamata, viene consumata e poi
        scartata dal garbage collector a fine build_splits()."""
        with self._lock:
            self._flush_pending_shard_locked()
            shard_paths = self._existing_shard_paths()

        drained: List[_QueuedPosition] = []
        for path in shard_paths:
            try:
                records = torch.load(path, weights_only=False)
            except Exception as e:
                logger.warning(f"Shard {path} illeggibile in build_splits ({e}): scartato.")
                continue
            for rec in records:
                decompressed_data = decompress_position_data(rec["data"])
                drained.append(
                    _QueuedPosition(
                        source_tag=rec["source_tag"],
                        group_key=rec["group_key"],
                        data=decompressed_data,
                    )
                )
        return drained

    def build_splits(
        self,
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int = 42,
    ) -> Dict[str, List[Data]]:
        if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
            raise PositionQueueError("split_ratios deve contenere 3 valori che sommano a 1.0.")

        # Ferma il flush periodico PRIMA di drenare: altrimenti il timer
        # potrebbe scrivere un nuovo shard mentre _drain_all_from_disk sta
        # leggendo/cancellando la lista di shard esistenti (race tra
        # thread), con rischio di perdere posizioni finite nello shard
        # scritto a meta' della cancellazione.
        self._stop_flush_timer()

        drained = self._drain_all_from_disk()
        if not drained:
            raise PositionQueueError("build_splits chiamato con la coda vuota.")

        windows: Dict[str, List[_QueuedPosition]] = defaultdict(list)
        for item in drained:
            game_id = _extract_game_id(item.data)
            windows[game_id].append(item)

        window_group_key: Dict[str, int] = {}
        for game_id, items in windows.items():
            keys_in_window = {item.group_key for item in items}
            if len(keys_in_window) != 1:
                raise PositionQueueError(
                    f"game_id={game_id!r} ha posizioni con group_key diversi ({sorted(keys_in_window)}). "
                    f"Con game_id generati come stringa leggibile univoca per fonte questo non dovrebbe "
                    f"accadere per collisione accidentale: verifica se lo stesso oggetto Data e' stato "
                    f"accodato piu' volte, o se c'e' un bug nel chiamante che riusa un game_id tra "
                    f"finestre diverse."
                )
            window_group_key[game_id] = keys_in_window.pop()

        groups_of_windows: Dict[int, List[str]] = defaultdict(list)
        for game_id, key in window_group_key.items():
            groups_of_windows[key].append(game_id)

        generator = torch.Generator().manual_seed(seed)
        train_ratio, val_ratio, _test_ratio = split_ratios
        result: Dict[str, List[Data]] = {"train": [], "val": [], "test": []}
        window_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

        for key in sorted(groups_of_windows.keys()):
            game_ids_in_group = sorted(groups_of_windows[key])
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
        groups_of_windows: Dict[int, List[str]],
        window_counts: Dict[str, int],
    ) -> None:
        total_positions = sum(len(v) for v in result.values())
        total_windows = sum(window_counts.values())
        logger.info(
            f"[PositionQueueRegistry] build_splits completato (letto da disco, RAM-safe): "
            f"{len(groups_of_windows)} group_key distinti, "
            f"{total_windows} finestre, {total_positions} posizioni totali."
        )
        for split_name in ("train", "val", "test"):
            logger.info(
                f"    {split_name}: {window_counts[split_name]} finestre, "
                f"{len(result[split_name])} posizioni"
            )