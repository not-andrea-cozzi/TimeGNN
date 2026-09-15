from __future__ import annotations

from typing import Dict, Optional

import torch
from torch_geometric.data import Data

_COMPRESSIBLE_FLOAT_BINARY_FIELDS = ("edge_attr",)
_MIXED_BINARY_PLUS_CONTINUOUS_FIELD = "x"
_COMPRESSIBLE_LONG_SMALLINT_FIELD = "event_ids"
_COMPRESSIBLE_CONSTANT_EDGE_FIELD = "time"


_PASSTHROUGH_BOOL_FIELDS = ("legal_move_mask",)

_UINT8_MAX = 255


def _assert_fits_uint8(tensor: torch.Tensor, field_name: str) -> None:
    if tensor.numel() == 0:
        return
    min_val = int(tensor.min().item())
    max_val = int(tensor.max().item())
    if min_val < 0 or max_val > _UINT8_MAX:
        raise ValueError(
            f"compress_position_data: campo '{field_name}' ha valori fuori "
            f"dal dominio uint8 [0,{_UINT8_MAX}] (min={min_val}, max={max_val}). "
            f"Compressione rifiutata per evitare perdita silenziosa di dati; "
            f"verifica PositionGraphSchema o disabilita la compressione per "
            f"questo campo."
        )


def _assert_strictly_binary(tensor: torch.Tensor, field_name: str) -> None:
    if tensor.numel() == 0:
        return
    is_binary = torch.all((tensor == 0.0) | (tensor == 1.0)).item()
    if not is_binary:
        offending = tensor[(tensor != 0.0) & (tensor != 1.0)]
        sample = offending.flatten()[:5].tolist()
        raise ValueError(
            f"compress_position_data: campo '{field_name}' contiene valori "
            f"non strettamente binari (esempi: {sample}). Compressione a "
            f"bool rifiutata per evitare perdita silenziosa di dati."
        )


def _assert_constant_along_edges(tensor: torch.Tensor, field_name: str) -> float:
    if tensor.numel() == 0:
        return 0.0
    first_val = tensor.flatten()[0].item()
    if not torch.allclose(tensor, torch.full_like(tensor, first_val)):
        raise ValueError(
            f"compress_position_data: campo '{field_name}' non e' costante "
            f"su tutti gli archi (atteso un unico scalare per-grafo "
            f"ripetuto, come da PositionGraphSchema.build_position_data). "
            f"Compressione a scalare rifiutata per evitare perdita di dati "
            f"reali per-arco che in futuro potrebbero non essere piu' "
            f"costanti."
        )
    return float(first_val)


def _compress_time_per_edge_type(
    time_tensor: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Dict[int, float]:
    edge_type_ids = edge_attr.argmax(dim=1)
    time_by_type: Dict[int, float] = {}

    for type_id in torch.unique(edge_type_ids).tolist():
        mask = edge_type_ids == type_id
        values_for_type = time_tensor[mask]
        first_val = values_for_type.flatten()[0].item()
        if not torch.allclose(values_for_type, torch.full_like(values_for_type, first_val)):
            raise ValueError(
                f"compress_position_data: campo 'time' non e' costante "
                f"per gli archi di tipo {type_id} (trovati valori diversi "
                f"entro lo stesso tipo di arco). Compressione rifiutata: "
                f"verifica che apply_edge_type_time_weighting sia stato "
                f"l'unico punto ad aver modificato time dopo "
                f"build_position_data, e che assegni un unico fattore per "
                f"tipo di arco, non un fattore per-arco individuale."
            )
        time_by_type[int(type_id)] = float(first_val)

    return time_by_type


def _compress_x_mixed(x: torch.Tensor) -> Data:
    """Comprime `x` [64, NUM_EVENT_FEATURES] separando le colonne
    strettamente binarie (occupancy: is_occupied_by_mover,
    is_occupied_by_opponent) dall'ultima colonna continua (clock_norm,
    aggiunta da PositionGraphSchema per portare il clock come node
    feature esplicita). FIX: la versione precedente trattava l'intero
    tensore x come binario e falliva non appena clock_norm (in [0,1]
    ma non ristretto a {0,1}) compariva tra le colonne.

    Assunzione: le prime N-1 colonne sono binarie, l'ultima e' continua.
    Se x ha esattamente 2 colonne (schema legacy senza clock_norm),
    si comprime tutto a bool come prima (nessuna colonna continua).
    """
    num_cols = x.shape[1] if x.dim() == 2 else 0
    if num_cols <= 2:
        _assert_strictly_binary(x, "x")
        return {"x": x.to(torch.bool)}

    binary_part = x[:, :-1]
    continuous_part = x[:, -1:]
    _assert_strictly_binary(binary_part, "x[:, :-1] (occupancy)")

    return {
        "x_binary": binary_part.to(torch.bool),
        "x_continuous": continuous_part.to(torch.float32),
    }


def compress_position_data(data: Data, mate_n: Optional[int] = None) -> Data:
    """Ritorna una NUOVA Data con storage compatto, lossless, per lo shard
    su disco. Non modifica l'oggetto passato in input.
    """
    compressed = Data()
    time_value = None
    edge_attr_value = None

    for key, value in data:
        if key == _COMPRESSIBLE_CONSTANT_EDGE_FIELD and torch.is_tensor(value):
            time_value = value
            continue
        if key == _MIXED_BINARY_PLUS_CONTINUOUS_FIELD and torch.is_tensor(value):
            for out_key, out_value in _compress_x_mixed(value).items():
                compressed[out_key] = out_value
        elif key in _PASSTHROUGH_BOOL_FIELDS and torch.is_tensor(value):
            compressed[key] = value.to(torch.bool)
        elif key == _COMPRESSIBLE_LONG_SMALLINT_FIELD and torch.is_tensor(value):
            _assert_fits_uint8(value, key)
            compressed[key] = value.to(torch.uint8)
        elif key in _COMPRESSIBLE_FLOAT_BINARY_FIELDS and torch.is_tensor(value):
            _assert_strictly_binary(value, key)
            compressed[key] = value.to(torch.bool)
            if key == "edge_attr":
                edge_attr_value = value
        else:
            compressed[key] = value

    if time_value is not None:
        if edge_attr_value is None:
            scalar_value = _assert_constant_along_edges(time_value, _COMPRESSIBLE_CONSTANT_EDGE_FIELD)
            compressed.time = torch.tensor([scalar_value], dtype=torch.float32)
        else:
            time_by_type = _compress_time_per_edge_type(time_value, edge_attr_value)
            type_ids = sorted(time_by_type.keys())
            compressed.time_by_edge_type_ids = torch.tensor(type_ids, dtype=torch.long)
            compressed.time_by_edge_type_values = torch.tensor(
                [time_by_type[t] for t in type_ids], dtype=torch.float32
            )

    if mate_n is not None:
        if not (0 <= mate_n <= _UINT8_MAX):
            raise ValueError(
                f"compress_position_data: mate_n={mate_n} fuori dal dominio "
                f"uint8 [0,{_UINT8_MAX}]."
            )
        compressed.mate_n = torch.tensor(mate_n, dtype=torch.uint8)

    return compressed


def decompress_position_data(data: Data) -> Data:
    """Inversa esatta di compress_position_data."""
    decompressed = Data()
    num_edges: Optional[int] = None

    if hasattr(data, "edge_index") and torch.is_tensor(data.edge_index):
        num_edges = data.edge_index.shape[1]

    has_new_format_time = hasattr(data, "time_by_edge_type_ids") and hasattr(data, "time_by_edge_type_values")
    has_mixed_x = hasattr(data, "x_binary") and hasattr(data, "x_continuous")

    for key, value in data:
        if key == "mate_n":
            continue
        if key in ("time_by_edge_type_ids", "time_by_edge_type_values"):
            continue
        if key in ("x_binary", "x_continuous"):
            continue
        if key in _PASSTHROUGH_BOOL_FIELDS and torch.is_tensor(value):
            decompressed[key] = value.to(torch.bool)
        elif key == _COMPRESSIBLE_LONG_SMALLINT_FIELD and torch.is_tensor(value):
            decompressed[key] = value.to(torch.long)
        elif key == _MIXED_BINARY_PLUS_CONTINUOUS_FIELD and torch.is_tensor(value):
            # formato legacy: x gia' presente non compresso in parti
            decompressed[key] = value.to(torch.float32)
        elif key in _COMPRESSIBLE_FLOAT_BINARY_FIELDS and torch.is_tensor(value):
            decompressed[key] = value.to(torch.float32)
        elif key == _COMPRESSIBLE_CONSTANT_EDGE_FIELD and torch.is_tensor(value):
            if not has_new_format_time:
                if num_edges is not None and num_edges > 0:
                    decompressed[key] = torch.full(
                        (num_edges,), float(value.flatten()[0].item()), dtype=torch.float32
                    )
                else:
                    decompressed[key] = value.to(torch.float32)
        else:
            decompressed[key] = value

    if has_mixed_x:
        x_binary = data.x_binary.to(torch.float32)
        x_continuous = data.x_continuous.to(torch.float32)
        decompressed.x = torch.cat([x_binary, x_continuous], dim=1)

    if has_new_format_time:
        edge_attr = getattr(decompressed, "edge_attr", None)
        if edge_attr is None or num_edges is None or num_edges == 0:
            pass
        else:
            edge_type_ids_lookup = edge_attr.argmax(dim=1)
            type_ids = data.time_by_edge_type_ids.tolist()
            type_values = data.time_by_edge_type_values.tolist()
            value_by_type = dict(zip(type_ids, type_values))

            reconstructed_time = torch.zeros(num_edges, dtype=torch.float32)
            for type_id, value in value_by_type.items():
                mask = edge_type_ids_lookup == type_id
                reconstructed_time[mask] = value
            decompressed.time = reconstructed_time

    return decompressed