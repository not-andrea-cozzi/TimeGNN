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

[MODIFICATO] Aggiunta apply_legal_move_mask: il softmax finale valutava
tutte le MOVE_VOCAB_SIZE classi assolute anche quando solo le mosse
legali sulla posizione sono effettivamente giocabili (tipicamente poche
decine su 20480). Mascherare i logit sulle mosse illegali prima della
loss/argmax e' anch'esso un passo puramente esterno al modello, coerente
col principio "modelli invariati" gia' seguito per il pooling: nessun
parametro appreso aggiuntivo, solo un masked_fill deterministico.

USO TIPICO in un training/eval loop:

    out = model(batch)                                  # [N_nodi_nel_batch, 20480]
    graph_logits = pool_node_logits(out, batch.batch)    # [batch_size, 20480]
    graph_logits = apply_legal_move_mask(graph_logits, batch.legal_move_mask)
    loss = criterion(graph_logits, batch.y)              # batch.y: [batch_size]

Dove batch.batch e' il vettore di assegnazione nodo->grafo e
batch.legal_move_mask e' il tensore booleano [batch_size, 20480] prodotto
da Batch.from_data_list a partire dal campo legal_move_mask di ogni Data
(vedi PositionGraphSchema.build_legal_move_mask per la shape [1, N] che
rende possibile questo stacking corretto).
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


def apply_legal_move_mask(
    graph_logits: torch.Tensor,
    legal_move_mask: torch.Tensor,
    fill_value: float = float("-inf"),
) -> torch.Tensor:
    """Azzera (via -inf) i logit corrispondenti a mosse illegali sulla
    posizione, prima di softmax/argmax/loss.

    Args:
        graph_logits: [batch_size, MOVE_VOCAB_SIZE], output di
            pool_node_logits.
        legal_move_mask: [batch_size, MOVE_VOCAB_SIZE], bool. True dove la
            mossa e' legale su quella posizione. Prodotto da
            Batch.from_data_list a partire da Data.legal_move_mask
            (shape [1, MOVE_VOCAB_SIZE] per singolo grafo, vedi
            PositionGraphSchema.build_legal_move_mask).
        fill_value: valore assegnato ai logit mascherati. -inf per
            default: rende quelle classi a probabilita' esattamente zero
            dopo softmax, mai selezionabili da argmax anche in caso di
            pareggio numerico.

    Returns:
        [batch_size, MOVE_VOCAB_SIZE]: stessa shape di graph_logits, con
        i logit sulle mosse illegali sostituiti da fill_value.

    Nota CrossEntropyLoss: nn.CrossEntropyLoss gestisce correttamente
    logit -inf nel calcolo del log-softmax (produce 0 di probabilita'
    per quella classe, senza NaN), a patto che la classe target
    (batch.y) non sia mai essa stessa mascherata come illegale — cosa
    garantita per costruzione: build_position_data solleva ValueError se
    best_move non e' legale su board, quindi y appartiene sempre
    all'insieme delle mosse legali di quella posizione.
    """
    if graph_logits.shape != legal_move_mask.shape:
        raise ValueError(
            f"apply_legal_move_mask: shape mismatch tra graph_logits "
            f"{tuple(graph_logits.shape)} e legal_move_mask "
            f"{tuple(legal_move_mask.shape)}. Verifica che "
            f"PositionGraphSchema.build_legal_move_mask produca [1, N] "
            f"per grafo (necessario per il corretto stacking via "
            f"Batch.from_data_list) e che output_dim del modello coincida "
            f"con MOVE_VOCAB_SIZE."
        )
    return graph_logits.masked_fill(~legal_move_mask, fill_value)