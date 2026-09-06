"""
ipc_safe_data.py

RISOLVE: RuntimeError: received 0 items of ancdata
(vedi traceback in coda al modulo per il caso reale che l'ha originato).

CAUSA DEL BUG
=============
GamesBuilder._worker_main (e PuzzleBuilder in modo analogo) ritorna
oggetti che contengono torch_geometric.data.Data con tensori CPU
(event_ids, x, edge_index, edge_attr, time, y, game_id, ply — vedi
PositionGraphSchema.build_position_data) attraverso un
multiprocessing.Pool STANDARD (non torch.multiprocessing.Pool).

Quando un tensore torch viene messo in coda tra processi, torch
intercetta il pickling di default (torch.multiprocessing.reductions
registra reduction custom per torch.Tensor/Storage anche se importi solo
`torch`, non serve importare esplicitamente torch.multiprocessing: la
patch avviene all'import di torch stesso). La strategia di default su
Linux e' "file_descriptor": ogni tensore condiviso viene passato come fd
via SCM_RIGHTS su un socket Unix (multiprocessing.reduction.recvfds).

Con un flusso di migliaia di piccoli task/secondo (qui: ~17 finestre/sec,
ognuna con N posizioni x 8 tensori ciascuna) si esauriscono i file
descriptor disponibili per il passaggio ancillary-data sul socket del
Pool prima che vengano chiusi lato ricevente: il kernel non riesce a
consegnare i fd e recvfds ritorna 0 elementi invece del 1 atteso ->
`RuntimeError: received 0 items of ancdata`. E' un problema di RISORSE
DI SISTEMA (limite fd/socket buffer), non un bug logico nel builder: si
manifesta in modo intermittente e peggiora con throughput piu' alto,
worker piu' numerosi, o `ulimit -n` basso.

FIX APPLICATO (in ordine di importanza)
========================================
1. torch.multiprocessing.set_sharing_strategy("file_system") a inizio
   processo (main + ogni worker, tramite l'initializer del Pool): usa
   file su /dev/shm con nomi univoci invece di fd via socket. Elimina
   la fonte primaria del problema (nessun consumo di fd sul socket del
   Pool per la condivisione tensori), a scapito di dover ripulire i file
   temporanei di /dev/shm (torch lo fa da solo via refcounting + GC, ma
   in caso di kill -9 possono restare residui: monitorare /dev/shm se il
   processo viene terminato bruscamente in modo ricorrente).

2. SOLUZIONE STRUTTURALE (quella adottata qui, piu' robusta della sola
   sharing strategy): i worker NON ritornano piu' Data con tensori nativi
   attraverso il Pool. Serializzano il risultato con `torch.save` su un
   BytesIO e ritornano `bytes` grezzi. I bytes attraversano l'IPC del
   Pool con il pickling STANDARD di Python (nessuna reduction custom di
   torch coinvolta, nessun fd condiviso, nessun problema di ancdata per
   costruzione). Il processo padre poi decodifica con `torch.load`.
   Costo: una copia in memoria in piu' per task (serializzazione
   esplicita) invece della zero-copy via shared memory — trascurabile
   per la dimensione di queste finestre (poche posizioni x pochi KB
   l'una), e comunque piu' sicuro che tentare di tarare `ulimit -n`
   globalmente sul sistema.

Le due misure sono complementari e vengono applicate entrambe: (1) come
rete di sicurezza per qualunque altro tensore che dovesse attraversare il
Pool altrove nella pipeline, (2) come fix primario per il path
GamesBuilder/PuzzleBuilder.

USO
===
Nel worker (dentro _worker_main o equivalente), invece di:

    return task_local_id, _WindowBuildResult(positions, mate_n, game_id, source_tag, error_counts)

fare:

    result = _WindowBuildResult(positions, mate_n, game_id, source_tag, error_counts)
    return task_local_id, encode_for_ipc(result)

Nel processo padre, invece di:

    for task_local_id, window in pbar:
        ...

fare:

    for task_local_id, payload in pbar:
        window = decode_from_ipc(payload)
        ...

Nell'initializer del Pool (accanto a _init_worker esistente):

    from DatasetPipeline.Utils.ipc_safe_data import harden_process_for_ipc

    def _init_worker(...):
        harden_process_for_ipc()
        ... resto dell'init invariato ...

E nel processo padre, prima di creare il Pool:

    from DatasetPipeline.Utils.ipc_safe_data import harden_process_for_ipc
    harden_process_for_ipc()
    pool = mp.Pool(...)
"""
from __future__ import annotations

import io
import logging
import resource
from typing import Any, Optional

import torch

logger = logging.getLogger("ipc_safe_data")

# Soglia minima di file descriptor consigliata per un Pool con molti
# worker che scambiano oggetti frequentemente. Puramente diagnostica:
# se il limite e' sotto questa soglia solo LOGGIAMO un avviso (alzare
# ulimit -n richiede permessi/pam limits, non lo forziamo qui), dato che
# la vera protezione e' evitare del tutto i tensori nativi sull'IPC.
_RECOMMENDED_MIN_NOFILE = 4096


def harden_process_for_ipc(recommended_min_nofile: int = _RECOMMENDED_MIN_NOFILE) -> None:
    """Applica le mitigazioni note per il crash 'received 0 items of
    ancdata' al processo corrente. Idempotente: chiamabile piu' volte
    senza effetti collaterali negativi (set_sharing_strategy accetta di
    essere richiamato con lo stesso valore).

    Va chiamata:
      - nel processo PADRE, prima di istanziare mp.Pool;
      - in ogni WORKER, dentro l'initializer del Pool (initializer=...),
        perche' la sharing strategy e' per-processo, non ereditata in
        modo affidabile da tutti i backend di avvio (spawn la reimposta
        al default).
    """
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (RuntimeError, ValueError) as e:
        # Non fatale: se per qualche motivo la strategia non è
        # cambiabile in questo contesto, la mitigazione (2) sotto
        # (bytes su Pool standard) resta comunque attiva e sufficiente.
        logger.warning(f"Impossibile impostare sharing_strategy='file_system': {e}")

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < recommended_min_nofile:
            new_soft = min(hard, recommended_min_nofile)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            logger.info(
                f"[ipc_safe_data] RLIMIT_NOFILE alzato da {soft} a {new_soft} "
                f"(hard limit={hard})."
            )
        else:
            logger.debug(f"[ipc_safe_data] RLIMIT_NOFILE gia' sufficiente: {soft}.")
    except (ValueError, OSError) as e:
        # Su alcuni container/sandbox il hard limit e' fisso e basso:
        # logghiamo e proseguiamo, la mitigazione (2) resta la difesa
        # primaria in quel caso.
        logger.warning(
            f"[ipc_safe_data] Impossibile alzare RLIMIT_NOFILE ({e}); "
            f"affidamento esclusivo sulla serializzazione a bytes per l'IPC."
        )


def encode_for_ipc(obj: Any) -> bytes:
    """Serializza un oggetto (tipicamente contenente torch.Tensor /
    torch_geometric.data.Data annidati) in bytes grezzi con torch.save,
    cosi' che attraversi multiprocessing.Pool come un pickle Python
    STANDARD (bytes), senza mai innescare le reduction custom di torch
    per tensori/storage che causano il consumo di file descriptor via
    ancdata.

    Il chiamante nel worker deve ritornare il valore di questa funzione
    al posto dell'oggetto originale.
    """
    buffer = io.BytesIO()
    # weights_only non applicabile qui: stiamo salvando dataclass/Data
    # generiche, non solo state_dict di modelli.
    torch.save(obj, buffer)
    return buffer.getvalue()


def decode_from_ipc(payload: bytes, *, map_location: Optional[str] = "cpu") -> Any:
    """Inversa di encode_for_ipc: ricostruisce l'oggetto originale dai
    bytes ricevuti dal Pool nel processo padre.

    Args:
        payload: bytes prodotti da encode_for_ipc nel worker.
        map_location: dispositivo di destinazione per i tensori
            deserializzati. "cpu" di default: i worker producono sempre
            Data su CPU (build_position_data non usa mai CUDA), quindi
            non c'e' motivo di spostare tensori su device diversi qui —
            l'eventuale .to(device) avviene piu' avanti nella pipeline
            (training loop), non nell'ingestione dei dati grezzi.
    """
    buffer = io.BytesIO(payload)
    return torch.load(buffer, map_location=map_location, weights_only=False)