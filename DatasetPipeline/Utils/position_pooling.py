"""
position_pooling.py

DualGATModel e DualGATTimeAwareModel (timegnn/models/gat_basic.py,
timegnn/models/gat_time_decay.py) NON vengono modificati (vincolo di
progetto): il loro forward produce un output per NODO
([N_nodi_nel_batch, output_dim]), mai un output per grafo.

Per ottenere "una mossa per posizione" da un grafo a 64 nodi (caselle) e'
necessario un passo di aggregazione ESTERNO al modello, applicato nel
training/eval loop dopo la chiamata al modello. Questo modulo isola quella
logica, cosi' l'ablation study (DualGATModel vs DualGATTimeAwareModel) usa
esattamente lo stesso pooling per entrambe le run: la sola variabile del
confronto resta la presenza/assenza del decay temporale dentro il modello,
non una differenza nella logica di aggregazione a valle.

USO TIPICO in un training/eval loop:

    out = model(batch)                       # [N_nodi_nel_batch, 4096]
    graph_logits = pool_node_logits(out, batch.batch)   # [batch_size, 4096]
    loss = criterion(graph_logits, batch.y)  # batch.y: [batch_size]

Dove batch.batch e' il vettore di assegnazione nodo->grafo che PyG
Batch.from_data_list produce automaticamente.
"""
from __future__ import annotations

import torch
from torch_geometric.nn import global_mean_pool


def pool_node_logits(node_logits: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    """Aggrega i logit per-nodo prodotti da DualGATModel/DualGATTimeAwareModel
    in un logit per GRAFO (per board), tramite media sui nodi di ciascun
    grafo nel batch.

    Args:
        node_logits: [N_nodi_nel_batch, output_dim], l'output diretto del
            modello (self.fc(x) dentro DualGATModel/DualGATTimeAwareModel).
        batch_index: [N_nodi_nel_batch], vettore che assegna ogni nodo al
            proprio grafo nel batch (attributo `batch` prodotto da
            torch_geometric.data.Batch.from_data_list).

    Returns:
        [batch_size, output_dim]: un logit per board, media dei logit delle
        sue 64 caselle. Usare global_mean_pool (non max/sum) mantiene la
        scala dei logit comparabile a quella del singolo nodo, evitando che
        board con piu' nodi "attivi" dominino artificialmente la softmax
        a valle rispetto a una somma.

    Nota:
        La scelta di MEDIA (non attention-pooling appreso, non max-pooling)
        e' deliberata per restare un passo puramente esterno al modello,
        senza parametri appresi aggiuntivi che snaturerebbero il vincolo
        "modelli invariati": un pooling con pesi appresi richiederebbe un
        nuovo modulo nn.Module con propri parametri da allenare, che di
        fatto sarebbe una modifica architetturale mascherata da
        post-processing.
    """
    return global_mean_pool(node_logits, batch_index)