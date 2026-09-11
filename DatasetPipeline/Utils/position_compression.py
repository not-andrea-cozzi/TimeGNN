from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Data

_COMPRESSIBLE_FLOAT_BINARY_FIELDS = ("x", "edge_attr")
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


def compress_position_data(data: Data, mate_n: Optional[int] = None) -> Data:
    """Ritorna una NUOVA Data con storage compatto, lossless, per lo shard
    su disco. Non modifica l'oggetto passato in input.

    Args:
        data: Data prodotto da PositionGraphSchema.build_position_data
            (o comunque con gli stessi campi/shape/dtype attesi).
        mate_n: opzionale, profondita' di matto della finestra di
            provenienza. Se fornito, salvato come attributo uint8
            aggiuntivo `mate_n` sul Data compresso (metadato puro, mai
            letto dai modelli).

    Returns:
        Nuova Data con event_ids:uint8, x:bool, edge_attr:bool (se
        presenti), time:float32 scalare [1] invece di [E] (se presente),
        legal_move_mask:bool invariato (se presente). Tutti gli altri
        campi sono copiati invariati.

    Raises:
        ValueError: se un campo non rientra nel dominio atteso per la
            compressione lossless (vedi _assert_* sopra). Nessun fallback
            silenzioso: la pipeline deve fermarsi su un'assunzione violata.
    """
    compressed = Data()

    for key, value in data:
        if key in _PASSTHROUGH_BOOL_FIELDS and torch.is_tensor(value):
            compressed[key] = value.to(torch.bool)
        elif key == _COMPRESSIBLE_LONG_SMALLINT_FIELD and torch.is_tensor(value):
            _assert_fits_uint8(value, key)
            compressed[key] = value.to(torch.uint8)
        elif key in _COMPRESSIBLE_FLOAT_BINARY_FIELDS and torch.is_tensor(value):
            _assert_strictly_binary(value, key)
            compressed[key] = value.to(torch.bool)
        elif key == _COMPRESSIBLE_CONSTANT_EDGE_FIELD and torch.is_tensor(value):
            scalar_value = _assert_constant_along_edges(value, key)
            compressed[key] = torch.tensor([scalar_value], dtype=torch.float32)
        else:
            compressed[key] = value

    if mate_n is not None:
        if not (0 <= mate_n <= _UINT8_MAX):
            raise ValueError(
                f"compress_position_data: mate_n={mate_n} fuori dal dominio "
                f"uint8 [0,{_UINT8_MAX}]."
            )
        compressed.mate_n = torch.tensor(mate_n, dtype=torch.uint8)

    return compressed


def decompress_position_data(data: Data) -> Data:
    """Inversa esatta di compress_position_data: ritorna una NUOVA Data
    con gli stessi dtype/shape dell'originale passato a
    PositionGraphSchema.build_position_data (long/float/float[E]/bool).

    Il numero di archi E per la ri-espansione di `time` e' letto da
    edge_index.shape[1]: se edge_index manca o e' vuoto, `time` viene
    lasciato con shape [1] (nessun arco su cui espandere).

    Idempotente su Data non compresse: se un campo e' gia' nel dtype
    atteso (es. proviene da un builder futuro che scrive gia' float), il
    cast e' un no-op equivalente.
    """
    decompressed = Data()
    num_edges: Optional[int] = None

    if hasattr(data, "edge_index") and torch.is_tensor(data.edge_index):
        num_edges = data.edge_index.shape[1]

    for key, value in data:
        if key == "mate_n":
            continue
        if key in _PASSTHROUGH_BOOL_FIELDS and torch.is_tensor(value):
            decompressed[key] = value.to(torch.bool)
        elif key == _COMPRESSIBLE_LONG_SMALLINT_FIELD and torch.is_tensor(value):
            decompressed[key] = value.to(torch.long)
        elif key in _COMPRESSIBLE_FLOAT_BINARY_FIELDS and torch.is_tensor(value):
            decompressed[key] = value.to(torch.float32)
        elif key == _COMPRESSIBLE_CONSTANT_EDGE_FIELD and torch.is_tensor(value):
            if num_edges is not None and num_edges > 0:
                decompressed[key] = torch.full(
                    (num_edges,), float(value.flatten()[0].item()), dtype=torch.float32
                )
            else:
                decompressed[key] = value.to(torch.float32)
        else:
            decompressed[key] = value

    return decompressed

