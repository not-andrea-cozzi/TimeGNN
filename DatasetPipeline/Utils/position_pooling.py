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