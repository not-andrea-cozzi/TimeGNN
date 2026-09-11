from __future__ import annotations

from typing import Dict

import chess

from DatasetPipeline.Model.PositionGraphSchema import (
    EDGE_ATTACK,
    EDGE_LEGAL_MOVE,
    EDGE_PIN,
    EVENT_ID_EMPTY,
    MOVE_VOCAB_SIZE,
    NUM_EDGE_TYPES,
    NUM_EVENT_ID_CATEGORIES,
    NUM_PROMOTION_SLOTS,
)

# ----------------------------------------------------------------------
# Valori materiale (usati per i filtri di compatibilita' e per
# has_mating_material / _material_by_color in GamesBuilder/PuzzleBuilder).
# Re escluso di proposito: non ha "valore materiale" in questi calcoli
# (perdita del Re = fine partita, non un pezzo da valutare).
# ----------------------------------------------------------------------
PIECE_VALUES: Dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# ----------------------------------------------------------------------
# Costanti architetturali lato modello/training.
#
# NUM_EVENT_FEATURES: numero di colonne in Data.x per nodo (casella).
# Storicamente 2 (is_occupied_by_mover, is_occupied_by_opponent). Da
# quando build_position_data aggiunge clock_norm come terza colonna
# (timing come node-feature esplicita, non solo come scalare costante su
# edge_attr/time via TimeAwareGATConv) vale 3. Se in futuro cambia di
# nuovo, va incrementato QUI e in nessun altro posto: build_position_data
# e tutti gli script di training/eval leggono questo valore da qui, non
# lo ridefiniscono piu' localmente.
# ----------------------------------------------------------------------
NUM_EVENT_FEATURES: int = 3

# edge_dim per il modello "basic" (DualGATModel): un one-hot a
# NUM_EDGE_TYPES colonne (vedi encode_edge_type_onehot).
EDGE_DIM_BASIC: int = NUM_EDGE_TYPES

# edge_dim per il modello time-aware (DualGATTimeAwareModel): un solo
# scalare per arco (data.time, costante su tutta la board, vedi nota in
# PositionGraphSchema.build_position_data).
TIME_EDGE_DIM: int = 1

__all__ = [
    "PIECE_VALUES",
    "NUM_EVENT_FEATURES",
    "NUM_EVENT_ID_CATEGORIES",
    "MOVE_VOCAB_SIZE",
    "NUM_EDGE_TYPES",
    "EDGE_DIM_BASIC",
    "TIME_EDGE_DIM",
    "EVENT_ID_EMPTY",
    "EDGE_LEGAL_MOVE",
    "EDGE_ATTACK",
    "EDGE_PIN",
    "NUM_PROMOTION_SLOTS",
]