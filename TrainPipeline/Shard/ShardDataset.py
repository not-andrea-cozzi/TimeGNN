"""
shard_dataset.py

IterableDataset lazy per gli shard prodotti da step1_shard_dataset.py.
Non carica mai l'intero split in RAM: tiene in memoria un solo shard alla
volta (~50MB con shard_size=8000), lo consuma, lo scarta, passa al
successivo. Shuffle a due livelli:
    1. ordine degli shard (permutato ad ogni epoca)
    2. ordine degli elementi dentro ogni shard caricato (permutato)
Non e' uno shuffle globale perfetto (un elemento non puo' finire in un
batch con elementi di uno shard non ancora caricato), ma con shard_size
grande rispetto al batch_size l'approssimazione e' trascurabile ed e' lo
stesso compromesso usato da webdataset/tfrecord per dataset grandi.

Compatibile con DataLoader(num_workers>0): ogni worker riceve una FETTA
DISGIUNTA di shard (via torch.utils.data.get_worker_info), cosi' non c'e'
overlap ne' doppio lavoro tra worker. Su Windows (spawn, non fork) il
dataset intero viene pickled per ogni worker: essendo solo path + piccoli
metadati (manifest), il costo e' trascurabile.

NOTA IMPORTANTE SU persistent_workers + set_epoch():
Con persistent_workers=True (default quando num_workers>0, vedi
build_dataloader) i processi worker vengono creati UNA SOLA VOLTA alla
prima iterazione del DataLoader e non rieseguono __init__: ricevono una
COPIA pickled del dataset a quel momento e la riusano per tutte le epoche
successive. Su Windows (spawn) questo significa che una mutazione fatta
nel processo padre con train_ds.set_epoch(epoch) NON si propaga ai worker
gia' avviati: l'epoca vista dai worker resterebbe sempre quella del primo
avvio. Per questo l'epoca corrente viene ricavata qui non da uno stato
mutabile letto dai worker, ma da un torch.multiprocessing.Value condiviso
tra processo padre e worker (aggiornato da set_epoch, letto in __iter__),
cosi' persistent_workers=True resta sicuro da usare.

USO:
    train_ds = ShardedGraphDataset("Dataset/Train/shards/train", shuffle=True, seed=42)
    loader = DataLoader(
        train_ds,
        batch_size=32,
        collate_fn=custom_collate_graph,   # da timegnn.data.pyg
        num_workers=4,
        persistent_workers=True,
    )
    for epoch in range(num_epochs):
        train_ds.set_epoch(epoch)   # visibile anche ai worker persistenti
        for batch in loader:
            ...
"""
from __future__ import annotations

import json
import logging
import os
import random
from typing import Iterator, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data

logger = logging.getLogger("shard_dataset")

MANIFEST_FILENAME = "manifest.json"
SHARD_FILENAME_TEMPLATE = "shard_{:05d}.pt"


def _label_from_data(data: Data) -> int:
    """
    Estrae la label scalare da un Data per il collate di pyg.py
    (`custom_collate_graph` fa `torch.tensor(labels)`, quindi servono int).

    Solleva ValueError se 'y' manca o non e' scalare.
    """
    y = getattr(data, "y", None)
    if y is None:
        raise ValueError(
            "Data senza campo 'y': impossibile addestrare. "
            "Controlla che clean_file abbia tenuto 'y' (KEEP_FIELDS)."
        )
    if isinstance(y, torch.Tensor):
        if y.numel() == 1:
            return int(y.item())
        raise ValueError(
            f"'y' ha {y.numel()} elementi, atteso scalare per-label di grafo."
        )
    return int(y)


class ShardedGraphDataset(IterableDataset):
    """Dataset lazy su shard di Data, un file alla volta in RAM.

    Args:
        shard_dir: directory contenente shard_NNNNN.pt + manifest.json
            (prodotta da step1_shard_dataset.py).
        shuffle: se True, permuta l'ordine degli shard e degli elementi
            dentro ciascuno shard caricato.
        seed: seed base per la permutazione; combinato con l'epoca
            corrente (set_epoch) per avere un ordine diverso ogni epoca
            restando riproducibile.
    """

    def __init__(self, shard_dir: str, shuffle: bool = True, seed: int = 42) -> None:
        super().__init__()
        self.shard_dir = shard_dir
        self.shuffle = shuffle
        self.seed = seed

        # Value condiviso (non un semplice attributo Python): sopravvive
        # al pickling verso i worker persistenti e le scritture fatte dal
        # processo padre dopo la creazione dei worker restano visibili
        # (vedi nota di modulo su persistent_workers + set_epoch).
        self._epoch = mp.Value("i", 0)

        manifest_path = os.path.join(shard_dir, MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"manifest.json non trovato in '{shard_dir}'. "
                f"Esegui prima step1_shard_dataset.py."
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.num_shards: int = manifest["num_shards"]
        self.total: int = manifest["total"]

    def set_epoch(self, epoch: int) -> None:
        """Da chiamare a inizio di ogni epoca per variare la permutazione
        degli shard mantenendo riproducibilita' (stesso seed -> stessa
        sequenza di epoche). Visibile anche a worker persistenti gia'
        avviati, perche' l'epoca vive in un mp.Value condiviso."""
        with self._epoch.get_lock():
            self._epoch.value = epoch

    def _current_epoch(self) -> int:
        with self._epoch.get_lock():
            return self._epoch.value

    def __len__(self) -> int:
        """Numero totale di elementi nell'intero split. NOTA: se questo
        dataset viene usato con un DataLoader a num_workers>0, PyTorch
        userebbe questo valore per stimare il numero di batch, ma ogni
        worker in realta' itera solo sulla propria fetta disgiunta di
        shard: usare len(loader) per progress bar/step-count con
        num_workers>0 e' quindi FUORVIANTE (sovrastima). Con
        num_workers=0 il valore e' invece esatto."""
        return self.total

    def _shard_order_for_worker(self, epoch: int) -> List[int]:
        """Determina quali indici di shard processa QUESTO worker in
        QUESTA epoca, e in che ordine."""
        shard_indices = list(range(self.num_shards))

        if self.shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(shard_indices)

        worker_info = get_worker_info()
        if worker_info is not None:
            # Fetta disgiunta: worker i prende shard_indices[i::num_workers]
            shard_indices = shard_indices[worker_info.id :: worker_info.num_workers]

        return shard_indices

    def _load_shard(self, shard_idx: int) -> Optional[List[Data]]:
        """Carica uno shard. Uno shard illeggibile (crash a meta'
        scrittura prima della patch atomica in step1, disco danneggiato,
        file rimosso a mano) NON deve fermare l'intero training: viene
        loggato e saltato, stesso principio difensivo usato altrove nel
        progetto (PipelineState, PositionQueueRegistry)."""
        path = os.path.join(self.shard_dir, SHARD_FILENAME_TEMPLATE.format(shard_idx))
        try:
            return torch.load(path, weights_only=False)
        except Exception as e:
            logger.warning(f"Shard '{path}' illeggibile ({type(e).__name__}: {e}): saltato.")
            return None

    def __iter__(self) -> Iterator[Tuple[Data, int]]:
        epoch = self._current_epoch()
        shard_order = self._shard_order_for_worker(epoch)

        # Seed per-worker distinto, cosi' lo shuffle interno agli shard
        # non e' identico tra worker diversi che processano shard diversi.
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        item_rng = random.Random(self.seed + epoch * 1000 + worker_id)

        for shard_idx in shard_order:
            items = self._load_shard(shard_idx)
            if items is None:
                continue
            if self.shuffle:
                item_rng.shuffle(items)
            for item in items:
                yield item, _label_from_data(item)
            del items  # libera esplicitamente prima del prossimo shard


def build_dataloader(
    shard_dir: str,
    batch_size: int,
    collate_fn,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 4,
    persistent_workers: Optional[bool] = None,
):
    """Factory di comodo per un DataLoader su ShardedGraphDataset.

    Nota: shuffle=True qui e' gestito INTERNAMENTE dal dataset (IterableDataset
    non supporta shuffle=True nel DataLoader stesso, va lasciato False).
    """
    from torch.utils.data import DataLoader

    dataset = ShardedGraphDataset(shard_dir, shuffle=shuffle, seed=seed)
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # gestito dal dataset stesso
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    ), dataset