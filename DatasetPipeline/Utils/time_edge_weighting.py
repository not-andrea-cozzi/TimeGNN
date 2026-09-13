from __future__ import annotations

from typing import Dict, Optional

import torch
from torch_geometric.data import Data

from DatasetPipeline.Model.PositionGraphSchema import (
    EDGE_ATTACK,
    EDGE_LEGAL_MOVE,
    EDGE_PIN,
    NUM_EDGE_TYPES,
)


DEFAULT_EDGE_TYPE_TIME_FACTORS: Dict[int, float] = {
    EDGE_LEGAL_MOVE: 1.0,
    EDGE_ATTACK: 0.5,
    EDGE_PIN: 0.5,
}


def _resolve_edge_types_from_onehot(edge_attr: torch.Tensor) -> torch.Tensor:
    """Ricava l'indice di tipo arco (0..NUM_EDGE_TYPES-1) dal one-hot in
    data.edge_attr prodotto da encode_edge_type_onehot in
    PositionGraphSchema.py, senza richiedere che il chiamante tenga una
    lista edge_type separata.

    Raises:
        ValueError: se edge_attr non ha la shape one-hot attesa
            [E, NUM_EDGE_TYPES], per evitare di derivare silenziosamente
            tipi di arco errati da un tensore che non e' quello previsto.
    """
    if edge_attr.dim() != 2 or edge_attr.shape[1] != NUM_EDGE_TYPES:
        raise ValueError(
            f"apply_edge_type_time_weighting: edge_attr ha shape "
            f"{tuple(edge_attr.shape)}, atteso [E, {NUM_EDGE_TYPES}] "
            f"(one-hot prodotto da encode_edge_type_onehot). Verifica che "
            f"il Data passato provenga da build_position_data invariato."
        )
    return edge_attr.argmax(dim=1)


def apply_edge_type_time_weighting(
    data: Data,
    factors: Optional[Dict[int, float]] = None,
) -> Data:
    """Ricalcola data.time scalando il valore costante prodotto da
    build_position_data per un fattore dipendente dal tipo di arco.

    Non modifica l'oggetto passato in input: ritorna un nuovo Data con
    lo stesso contenuto tranne il campo `time`, per coerenza con lo
    stile "non mutare in place" gia' seguito da
    DatasetPipeline.Utils.position_compression (compress/decompress
    non mutano l'input) e per evitare l'esatto tipo di bug gia' corretto
    in PrefixGCNClassifier.forward (mutazione in-place su un tensore
    potenzialmente condiviso).

    Args:
        data: Data prodotto da build_position_data (deve avere .time
            shape [E] e .edge_attr shape [E, NUM_EDGE_TYPES] one-hot).
        factors: mappa {edge_type_id: fattore_moltiplicativo}. Se un
            edge_type presente nel Data non ha una entry nella mappa,
            si usa fattore 1.0 (nessuna modifica per quel tipo) invece
            di sollevare un errore, cosi' la funzione resta utilizzabile
            anche se in futuro NUM_EDGE_TYPES cresce con nuovi tipi di
            arco non ancora contemplati qui esplicitamente. Default:
            DEFAULT_EDGE_TYPE_TIME_FACTORS.

    Returns:
        Un nuovo Data, copia superficiale di `data` con `time`
        rimpiazzato dal tempo pesato per tipo di arco. Se `data` non ha
        ne' `time` ne' `edge_attr` (Data costruiti con schemi diversi da
        PositionGraphSchema), viene ritornato invariato: questa funzione
        e' un no-op sicuro fuori dal caso d'uso scacchistico.
    """
    if not hasattr(data, "time") or data.time is None:
        return data
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        return data

    time_tensor = data.time
    edge_attr = data.edge_attr

    if time_tensor.numel() == 0:
        return data

    if time_tensor.shape[0] != edge_attr.shape[0]:
        raise ValueError(
            f"apply_edge_type_time_weighting: time ha {time_tensor.shape[0]} "
            f"elementi ma edge_attr ne ha {edge_attr.shape[0]}; devono "
            f"corrispondere 1:1 per arco. Verifica che il Data non sia "
            f"stato compresso (vedi position_compression.py: li' time "
            f"e' compresso a shape [1] e va prima decompresso con "
            f"decompress_position_data)."
        )

    active_factors = dict(DEFAULT_EDGE_TYPE_TIME_FACTORS)
    if factors is not None:
        active_factors.update(factors)

    edge_type_ids = _resolve_edge_types_from_onehot(edge_attr)

    # Fattore per-arco: default 1.0 per qualunque edge_type non presente
    # esplicitamente in active_factors (vedi docstring).
    factor_per_edge = torch.ones_like(time_tensor)
    for edge_type_id, factor_value in active_factors.items():
        mask = edge_type_ids == edge_type_id
        if mask.any():
            factor_per_edge[mask] = factor_value

    weighted_time = time_tensor * factor_per_edge

    new_data = data.clone()
    new_data.time = weighted_time
    return new_data