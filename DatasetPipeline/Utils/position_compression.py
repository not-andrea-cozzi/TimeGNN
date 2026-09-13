from __future__ import annotations

from typing import Dict, Optional

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


def _compress_time_per_edge_type(
    time_tensor: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Dict[int, float]:
    """Comprime `time` assumendo che sia costante ENTRO ciascun tipo di
    arco (colonna di edge_attr), non necessariamente su tutta la board.

    FIX: la versione precedente (_assert_constant_along_edges applicata
    all'intero tensore time) assumeva un unico scalare per l'intera
    board, coerente con build_position_data quando time e' scritto come
    torch.full((E,), clock_seconds). Da quando
    DatasetPipeline.Utils.time_edge_weighting.apply_edge_type_time_weighting
    puo' scalare time per un fattore diverso a seconda del tipo di arco
    (legal_move/attack/pin), time non e' piu' necessariamente costante
    sull'intera board — ma resta costante DENTRO ciascun tipo di arco
    (tutti gli archi 'legal_move' condividono lo stesso valore, idem per
    'attack' e 'pin'). Questa funzione comprime rispettando quella
    struttura piu' fine, invece di rifiutare la compressione.

    Args:
        time_tensor: shape [E].
        edge_attr: shape [E, NUM_EDGE_TYPES], one-hot del tipo di arco
            (da encode_edge_type_onehot in PositionGraphSchema.py).

    Returns:
        Dizionario {edge_type_id: valore_costante_per_quel_tipo}, con
        una entry solo per i tipi di arco effettivamente presenti sulla
        board (un tipo assente non compare, invece di essere forzato a
        0.0, per non confondere "tipo assente" con "tempo zero").

    Raises:
        ValueError: se time non e' costante DENTRO uno stesso tipo di
            arco. Stesso principio "nessun fallback silenzioso" delle
            altre funzioni _assert_* del modulo.
    """
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


def compress_position_data(data: Data, mate_n: Optional[int] = None) -> Data:
    """Ritorna una NUOVA Data con storage compatto, lossless, per lo shard
    su disco. Non modifica l'oggetto passato in input.

    Args:
        data: Data prodotto da PositionGraphSchema.build_position_data
            (eventualmente post-processato da
            DatasetPipeline.Utils.time_edge_weighting.apply_edge_type_time_weighting),
            o comunque con gli stessi campi/shape/dtype attesi.
        mate_n: opzionale, profondita' di matto della finestra di
            provenienza. Se fornito, salvato come attributo uint8
            aggiuntivo `mate_n` sul Data compresso (metadato puro, mai
            letto dai modelli).

    Returns:
        Nuova Data con event_ids:uint8, x:bool, edge_attr:bool (se
        presenti), time compresso in FORMA COMPATTA per-tipo-di-arco
        (vedi _compress_time_per_edge_type: al posto dello scalare [1]
        della versione precedente, un attributo `time_by_edge_type`
        contenente al massimo NUM_EDGE_TYPES coppie (tipo, valore) —
        vedi decompress_position_data per la ricostruzione),
        legal_move_mask:bool invariato (se presente). Tutti gli altri
        campi sono copiati invariati.

    Raises:
        ValueError: se un campo non rientra nel dominio atteso per la
            compressione lossless (vedi _assert_* sopra). Nessun fallback
            silenzioso: la pipeline deve fermarsi su un'assunzione violata.
    """
    compressed = Data()
    time_value = None
    edge_attr_value = None

    # Prima passata: raccogli time/edge_attr grezzi (servono insieme per
    # la compressione per-tipo-arco), copia tutto il resto invariato o
    # con le regole esistenti.
    for key, value in data:
        if key == _COMPRESSIBLE_CONSTANT_EDGE_FIELD and torch.is_tensor(value):
            time_value = value
            continue  # gestito dopo, richiede edge_attr
        if key in _PASSTHROUGH_BOOL_FIELDS and torch.is_tensor(value):
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
            # Nessun edge_attr disponibile (Data non scacchistico, schema
            # diverso): ricade sul comportamento originale, un unico
            # scalare per l'intera board.
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
    """Inversa esatta di compress_position_data: ritorna una NUOVA Data
    con gli stessi dtype/shape dell'originale passato a
    PositionGraphSchema.build_position_data (long/float/float[E]/bool),
    con `time` ricostruito per-arco a partire dalla forma compatta
    per-tipo-arco scritta da compress_position_data.

    Il numero di archi E e' letto da edge_index.shape[1]: se edge_index
    manca o e' vuoto, `time` viene lasciato con shape [1] o [0] a seconda
    del formato in cui era stato compresso (vedi rami sotto).

    Idempotente su Data non compresse: se un campo e' gia' nel dtype
    atteso (es. proviene da un builder futuro che scrive gia' float), il
    cast e' un no-op equivalente. Gestisce ANCHE i Data compressi con il
    vecchio formato a scalare singolo (attributo `time` shape [1]), per
    retrocompatibilita' con shard scritti prima di questo fix.
    """
    decompressed = Data()
    num_edges: Optional[int] = None

    if hasattr(data, "edge_index") and torch.is_tensor(data.edge_index):
        num_edges = data.edge_index.shape[1]

    has_new_format_time = hasattr(data, "time_by_edge_type_ids") and hasattr(data, "time_by_edge_type_values")

    for key, value in data:
        if key == "mate_n":
            continue
        if key in ("time_by_edge_type_ids", "time_by_edge_type_values"):
            continue  # ricostruiti sotto in un unico attributo `time`
        if key in _PASSTHROUGH_BOOL_FIELDS and torch.is_tensor(value):
            decompressed[key] = value.to(torch.bool)
        elif key == _COMPRESSIBLE_LONG_SMALLINT_FIELD and torch.is_tensor(value):
            decompressed[key] = value.to(torch.long)
        elif key in _COMPRESSIBLE_FLOAT_BINARY_FIELDS and torch.is_tensor(value):
            decompressed[key] = value.to(torch.float32)
        elif key == _COMPRESSIBLE_CONSTANT_EDGE_FIELD and torch.is_tensor(value):
            # Vecchio formato (scalare singolo [1]): mantenuto per
            # retrocompatibilita' con shard scritti prima di questo fix.
            if not has_new_format_time:
                if num_edges is not None and num_edges > 0:
                    decompressed[key] = torch.full(
                        (num_edges,), float(value.flatten()[0].item()), dtype=torch.float32
                    )
                else:
                    decompressed[key] = value.to(torch.float32)
        else:
            decompressed[key] = value

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