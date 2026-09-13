from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def pack_legal_logits(
    graph_logits: torch.Tensor,
    legal_move_mask: torch.Tensor,
    labels: torch.Tensor,
    pad_value: float = float("-inf"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compatta i logit alle sole colonne legali per riga, in un tensore
    denso [B, K_max] pronto per cross_entropy.

    Args:
        graph_logits: [B, V] output di pool_node_logits, V=MOVE_VOCAB_SIZE.
        legal_move_mask: [B, V] bool, True sulle mosse legali per quella riga.
        labels: [B] indici (in [0, V)) della mossa target per riga.
        pad_value: valore assegnato alle colonne di padding oltre il
            conteggio di mosse legali della riga (default -inf, cosi'
            softmax le azzera esattamente come nel masking full-size).

    Returns:
        packed_logits: [B, K_max] logit compattati, K_max = max mosse
            legali nel batch. Colonne oltre il conteggio della riga sono
            pad_value.
        target_local: [B] indice LOCALE (in [0, K_max)) della mossa
            target dentro packed_logits[i], da usare come target di
            F.cross_entropy(packed_logits, target_local).

    Raises:
        ValueError: se per qualche riga la mossa target (labels[i]) non
            e' marcata legale in legal_move_mask[i] — indica un
            disallineamento a monte tra data.y e legal_move_mask (bug nei
            dati, non tollerabile silenziosamente: la loss risultante
            sarebbe scorretta senza errore visibile).
    """
    if graph_logits.shape != legal_move_mask.shape:
        raise ValueError(
            f"pack_legal_logits: shape mismatch tra graph_logits "
            f"{tuple(graph_logits.shape)} e legal_move_mask "
            f"{tuple(legal_move_mask.shape)}."
        )

    batch_size, vocab_size = graph_logits.shape
    device = graph_logits.device

    legal_counts = legal_move_mask.sum(dim=1)  # [B]
    if (legal_counts == 0).any():
        bad_rows = (legal_counts == 0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"pack_legal_logits: righe senza alcuna mossa legale (indici "
            f"batch {bad_rows}). Una posizione senza mosse legali non "
            f"dovrebbe mai raggiungere il training loop (sarebbe gia' "
            f"matto/stallo, scartata a monte da build_position_data)."
        )

    target_is_legal = legal_move_mask.gather(1, labels.view(-1, 1)).squeeze(1)
    if not target_is_legal.all():
        bad_rows = (~target_is_legal).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"pack_legal_logits: la mossa target non e' marcata legale per "
            f"le righe batch {bad_rows}. Disallineamento tra data.y e "
            f"legal_move_mask: verificare build_position_data per queste "
            f"posizioni, non e' un caso da mascherare silenziosamente."
        )

    k_max = int(legal_counts.max().item())
    sort_key = (~legal_move_mask).float()
    col_index = torch.argsort(sort_key, dim=1, stable=True)[:, :k_max]  # [B, K_max]

    packed_logits = graph_logits.gather(1, col_index)  # [B, K_max]
    arange_k = torch.arange(k_max, device=device).unsqueeze(0)  # [1, K_max]
    pad_mask = arange_k >= legal_counts.view(-1, 1)  # [B, K_max]
    packed_logits = packed_logits.masked_fill(pad_mask, pad_value)
    target_local = (col_index == labels.view(-1, 1)).float().argmax(dim=1)

    return packed_logits, target_local


def sparse_legal_cross_entropy(
    graph_logits: torch.Tensor,
    legal_move_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Drop-in replacement per:

        masked = apply_legal_move_mask(graph_logits, legal_move_mask)
        loss = F.cross_entropy(masked, labels)

    ma operando su [B, K_max] invece di [B, MOVE_VOCAB_SIZE]. Risultato
    numerico identico (a meno di riordino dei termini in floating point,
    trascurabile) perche' softmax ignora comunque le colonne -inf in
    entrambi i casi; la differenza e' solo quante colonne vengono
    effettivamente processate da softmax/argmax/backward.
    """
    packed_logits, target_local = pack_legal_logits(graph_logits, legal_move_mask, labels)
    return F.cross_entropy(packed_logits, target_local)


def sparse_legal_argmax(
    graph_logits: torch.Tensor,
    legal_move_mask: torch.Tensor,
) -> torch.Tensor:
    """Argmax ristretto alle mosse legali, restituito come indice nel
    vocabolario ORIGINALE (non locale), cosi' e' direttamente comparabile
    a `labels` per il calcolo dell'accuracy senza ulteriore mapping.

    Equivalente a:
        masked = apply_legal_move_mask(graph_logits, legal_move_mask)
        pred = masked.argmax(dim=1)
    ma senza costruire il tensore full-size mascherato.
    """
    masked = graph_logits.masked_fill(~legal_move_mask, float("-inf"))
    return masked.argmax(dim=1)