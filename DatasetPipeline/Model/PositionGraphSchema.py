from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import chess
import torch
from torch_geometric.data import Data

_PROMOTION_TYPES: Tuple[Optional[int], ...] = (None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
_PROMOTION_OFFSET: Dict[Optional[int], int] = {pt: i for i, pt in enumerate(_PROMOTION_TYPES)}
NUM_PROMOTION_SLOTS = len(_PROMOTION_TYPES)

MOVE_VOCAB_SIZE = 64 * 64 * NUM_PROMOTION_SLOTS

EVENT_ID_EMPTY = 0
NUM_EVENT_ID_CATEGORIES = 15

EDGE_LEGAL_MOVE = 0
EDGE_ATTACK = 1
EDGE_PIN = 2
NUM_EDGE_TYPES = 3

NUM_EVENT_FEATURES = 3

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def encode_move(move: "chess.Move") -> int:
    base = move.from_square * 64 + move.to_square
    promo_slot = _PROMOTION_OFFSET[move.promotion]
    return base * NUM_PROMOTION_SLOTS + promo_slot


def decode_move(move_id: int) -> Tuple[int, int, Optional[int]]:
    promo_slot = move_id % NUM_PROMOTION_SLOTS
    base = move_id // NUM_PROMOTION_SLOTS
    from_square = base // 64
    to_square = base % 64
    return from_square, to_square, _PROMOTION_TYPES[promo_slot]


def build_legal_move_mask(board: "chess.Board") -> torch.Tensor:
    mask = torch.zeros(1, MOVE_VOCAB_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        mask[0, encode_move(move)] = True
    return mask


def encode_square_event_id(board: "chess.Board", square: int) -> int:
    piece = board.piece_at(square)
    if piece is None:
        return EVENT_ID_EMPTY
    color_bit = 1 if piece.color == chess.WHITE else 0
    return piece.piece_type * 2 + color_bit + 1


def _build_spatial_edges(board: "chess.Board") -> Tuple[List[int], List[int], List[int]]:
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_type: List[int] = []

    for move in board.legal_moves:
        edge_src.append(move.from_square)
        edge_dst.append(move.to_square)
        edge_type.append(EDGE_LEGAL_MOVE)

    piece_map = board.piece_map()
    for sq, piece in piece_map.items():
        for target_sq in board.attacks(sq):
            edge_src.append(sq)
            edge_dst.append(target_sq)
            edge_type.append(EDGE_ATTACK)

        pin_ray = board.pin(piece.color, sq)
        if len(pin_ray) < 64:
            for ray_sq in pin_ray:
                attacker = piece_map.get(ray_sq)
                if (
                    attacker
                    and attacker.color != piece.color
                    and attacker.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN)
                ):
                    edge_src.append(ray_sq)
                    edge_dst.append(sq)
                    edge_type.append(EDGE_PIN)

    return edge_src, edge_dst, edge_type


def encode_edge_type_onehot(edge_type: List[int]) -> torch.Tensor:
    if not edge_type:
        return torch.zeros((0, NUM_EDGE_TYPES), dtype=torch.float)
    t = torch.tensor(edge_type, dtype=torch.long)
    return torch.nn.functional.one_hot(t, num_classes=NUM_EDGE_TYPES).float()


def _clock_norm(clock_seconds: float, cap_seconds: float = 600.0) -> float:
    denom = math.log1p(cap_seconds)
    if denom <= 0:
        return 0.0
    return min(math.log1p(max(clock_seconds, 0.0)) / denom, 1.0)


def build_position_data(
    board: "chess.Board",
    best_move: "chess.Move",
    clock_seconds: float,
    rating: float,
    game_id: str,
    ply: int,
    mate_n: Optional[int] = None,
) -> Data:
    if best_move not in board.legal_moves:
        raise ValueError(
            f"build_position_data: best_move={best_move.uci()} non e' legale "
            f"sulla posizione data (fen={board.fen()})."
        )

    mover = board.turn

    event_ids = torch.tensor(
        [[encode_square_event_id(board, sq)] for sq in range(64)], dtype=torch.long
    )

    clock_feature = _clock_norm(clock_seconds)
    x_rows = []
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            x_rows.append([0.0, 0.0, clock_feature])
        elif piece.color == mover:
            x_rows.append([1.0, 0.0, clock_feature])
        else:
            x_rows.append([0.0, 1.0, clock_feature])
    x = torch.tensor(x_rows, dtype=torch.float)

    edge_src, edge_dst, edge_type_list = _build_spatial_edges(board)
    if not edge_src:
        raise ValueError(
            f"build_position_data: nessun arco spaziale prodotto per la "
            f"posizione (fen={board.fen()})."
        )

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = encode_edge_type_onehot(edge_type_list)

    num_edges = edge_index.shape[1]
    time_tensor = torch.full((num_edges,), float(clock_feature), dtype=torch.float)

    y = torch.tensor(encode_move(best_move), dtype=torch.long)
    legal_move_mask = build_legal_move_mask(board)

    data = Data(
        event_ids=event_ids,
        x=x,
        edge_index=edge_index,
        num_nodes=64,
    )
    data.edge_attr = edge_attr
    data.time = time_tensor
    data.y = y
    data.legal_move_mask = legal_move_mask
    data.rating = torch.tensor(float(rating), dtype=torch.float16)
    data.game_id = game_id
    data.ply = torch.tensor(int(ply), dtype=torch.int64)

    if mate_n is not None:
        if not (0 <= mate_n <= 255):
            raise ValueError(
                f"build_position_data: mate_n={mate_n} fuori dal dominio "
                f"uint8 [0,255] per position_mate_n (fen={board.fen()})."
            )
        data.position_mate_n = torch.tensor(int(mate_n), dtype=torch.uint8)

    return data