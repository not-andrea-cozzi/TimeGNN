from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess
import torch
from torch_geometric.data import Data

# ============================================================================
# VOCABOLARI FISSI
# ============================================================================

_PROMOTION_TYPES: Tuple[Optional[int], ...] = (None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
_PROMOTION_OFFSET: Dict[Optional[int], int] = {pt: i for i, pt in enumerate(_PROMOTION_TYPES)}
NUM_PROMOTION_SLOTS = len(_PROMOTION_TYPES)  # 5

MOVE_VOCAB_SIZE = 64 * 64 * NUM_PROMOTION_SLOTS  # 20480: (from*64+to)*5 + promo_slot

# event_ids: 0 = casella vuota, 1..12 = piece_type*2+color+1
EVENT_ID_EMPTY = 0
NUM_EVENT_ID_CATEGORIES = 13  # 0 (vuoto) + 12 (6 piece_type x 2 colori)

EDGE_LEGAL_MOVE = 0
EDGE_ATTACK = 1
EDGE_PIN = 2
NUM_EDGE_TYPES = 3

NUM_EVENT_FEATURES = 2  # is_occupied_by_mover, is_occupied_by_opponent

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def encode_move(move: "chess.Move") -> int:
    """Codifica una mossa nel vocabolario globale fisso (0..20479).

    [MODIFICATO] Include lo slot di promozione: due mosse con stesso
    from/to ma promozione diversa (o nessuna) ricevono ora id distinti,
    a differenza della versione precedente che le collassava (vedi
    NUM_PROMOTION_SLOTS sopra). move.promotion e' None per le mosse non
    di promozione e vale chess.QUEEN/ROOK/BISHOP/KNIGHT altrimenti.
    """
    base = move.from_square * 64 + move.to_square
    promo_slot = _PROMOTION_OFFSET[move.promotion]
    return base * NUM_PROMOTION_SLOTS + promo_slot


def decode_move(move_id: int) -> Tuple[int, int, Optional[int]]:
    """Inversa di encode_move: ritorna (from_square, to_square, promotion).

    Utile per debug/ispezione e per ricostruire una chess.Move da un id
    predetto dal modello (chess.Move(from_square, to_square, promotion)).
    """
    promo_slot = move_id % NUM_PROMOTION_SLOTS
    base = move_id // NUM_PROMOTION_SLOTS
    from_square = base // 64
    to_square = base % 64
    return from_square, to_square, _PROMOTION_TYPES[promo_slot]


def build_legal_move_mask(board: "chess.Board") -> torch.Tensor:
    """Costruisce la maschera booleana [1, MOVE_VOCAB_SIZE] delle mosse
    legali sulla posizione data, con la stessa codifica di encode_move.

    Usata a valle (fuori da questo modulo) per mascherare i logit del
    modello prima di softmax/argmax: senza questa maschera il modello
    valuta 20480 classi assolute anche quando solo poche decine sono
    fisicamente giocabili, il che rende il problema di apprendimento
    molto piu' difficile del necessario a parita' di dati.

    NOTA SHAPE [1, MOVE_VOCAB_SIZE] (non [MOVE_VOCAB_SIZE]):
    torch_geometric.data.Batch.from_data_list concatena per default lungo
    dim=0 gli attributi non riconosciuti come node/edge-level. Un tensore
    [MOVE_VOCAB_SIZE] per singolo grafo verrebbe quindi appiattito in un
    unico vettore [batch_size*MOVE_VOCAB_SIZE] invece di uno stack
    [batch_size, MOVE_VOCAB_SIZE]. Salvandolo con una dimensione fittizia
    iniziale, la concatenazione lungo dim=0 produce direttamente la shape
    corretta [batch_size, MOVE_VOCAB_SIZE] attesa dal masking dopo il
    pooling per-grafo (vedi TrainPipeline/Training/Loop.py).

    Returns:
        torch.BoolTensor di shape [1, MOVE_VOCAB_SIZE], True sugli indici
        delle mosse legali sulla board data (board.turn = lato al comando).
    """
    mask = torch.zeros(1, MOVE_VOCAB_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        mask[0, encode_move(move)] = True
    return mask


def encode_square_event_id(board: "chess.Board", square: int) -> int:
    """Categoria pezzo+colore per una casella (0 = vuota, 1..12 = occupata).

    L'ordine di codifica (piece_type*2+color+1) e' arbitrario ma fisso:
    l'importante e' che sia deterministico e coerente in tutto il dataset,
    dato che alimenta un nn.Embedding che impara pesi per ciascun indice.
    """
    piece = board.piece_at(square)
    if piece is None:
        return EVENT_ID_EMPTY
    color_bit = 1 if piece.color == chess.WHITE else 0
    return piece.piece_type * 2 + color_bit + 1


def _build_spatial_edges(board: "chess.Board") -> Tuple[List[int], List[int], List[int]]:
    """Costruisce gli archi spaziali (mossa-legale/attacco/pin) tra caselle
    per la board data, stessa logica del vecchio
    GraphBuilder.board_to_pyg_data._build (qui isolata come funzione pura,
    piu' facile da testare in isolamento).

    Returns:
        (edge_src, edge_dst, edge_type): liste parallele di indici casella
        0-63 e tipo di relazione (EDGE_LEGAL_MOVE/ATTACK/PIN).
    """
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
    """One-hot [E, NUM_EDGE_TYPES] per soddisfare il contratto edge_dim di
    GATConv (vedi NOTA EDGE_ATTR_ENCODING nel docstring di modulo)."""
    if not edge_type:
        return torch.zeros((0, NUM_EDGE_TYPES), dtype=torch.float)
    t = torch.tensor(edge_type, dtype=torch.long)
    return torch.nn.functional.one_hot(t, num_classes=NUM_EDGE_TYPES).float()


def _clock_norm(clock_seconds: float, cap_seconds: float = 600.0) -> float:
    """Normalizzazione log-scale (preserva la differenza tra clock brevi
    senza schiacciare tutto cio' che supera pochi minuti come farebbe una
    scala lineare). Non usata direttamente in x (che qui non contiene una
    colonna tempo, vedi docstring: il tempo entra via `time`, non via x),
    ma esposta per riuso nel builder se serve normalizzare clock_seconds
    prima di passarlo come `time`.
    """
    import math
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
    """Assembla un torch_geometric.data.Data a grana di SINGOLA POSIZIONE
    (64 nodi = caselle), pronto per DualGATModel e DualGATTimeAwareModel
    senza alcuna modifica a quei modelli.

    Args:
        board: posizione corrente (board.turn = lato che deve muovere).
        best_move: la mossa migliore per questa posizione (target).
        clock_seconds: tempo (secondi) impiegato per arrivare a questa
            mossa. Diventa `time`, costante su tutti gli archi della board
            (vedi motivazione nel docstring di modulo).
        rating: rating (Elo) del giocatore di turno (mover) in questa
            posizione. OBBLIGATORIO (mai None): il chiamante deve
            risolvere un valore reale o di fallback prima di chiamare
            questa funzione (vedi docstring di modulo, campo `rating`).
        game_id: identificatore leggibile "{fonte}_{id_originale}" della
            finestra/partita/puzzle di provenienza (es. "lichess_142",
            "puzzle_00sHx"). Stringa, non un tensore.
        ply: indice del ply all'interno della finestra (tracciamento).
        mate_n: profondita' di matto REALE a QUESTA specifica posizione
            (non il group_key di finestra: vedi NOTA POSITION_MATE_N sotto).
            Se fornito, salvato come data.position_mate_n (uint8). Se None
            (default, per non rompere chiamanti esistenti), il campo non
            viene scritto sul Data.

    Returns:
        Data con event_ids/x/edge_index/edge_attr/time/y/legal_move_mask/
        rating/game_id/ply/[position_mate_n] come da docstring di modulo.

    Raises:
        ValueError: se best_move non e' una mossa legale su board (il
            target deve sempre essere verificabile sulla posizione data),
            o se mate_n e' fornito ma fuori dal dominio uint8 [0,255].

    NOTA POSITION_MATE_N (da non confondere con il "mate_n" scritto da
    DatasetPipeline.Utils.position_compression.compress_position_data):
    quest'ultimo e' il group_key di FINESTRA (costante per tutte le
    posizioni di uno stesso game_id, usato per lo split stratificato in
    PositionQueueRegistry.build_splits), passato come parametro separato
    a compress_position_data, non letto da un attributo del Data.
    position_mate_n invece e' la profondita' di matto REALE alla
    posizione specifica: in un puzzle mateIn4, la prima mossa-solver ha
    position_mate_n=4, l'ultima ha position_mate_n=1 (vedi
    PuzzleBuilder.current_mate_n). Sono due numeri diversi per la stessa
    posizione tranne che sulla prima mossa della finestra, dove
    coincidono. Il nome distinto evita di sovrascrivere per errore il
    group_key di finestra quando build_splits legge item.data per
    determinare i bucket di stratificazione.
    """
    if best_move not in board.legal_moves:
        raise ValueError(
            f"build_position_data: best_move={best_move.uci()} non e' legale "
            f"sulla posizione data (fen={board.fen()})."
        )

    mover = board.turn

    event_ids = torch.tensor(
        [[encode_square_event_id(board, sq)] for sq in range(64)], dtype=torch.long
    )

    x_rows = []
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            x_rows.append([0.0, 0.0])
        elif piece.color == mover:
            x_rows.append([1.0, 0.0])
        else:
            x_rows.append([0.0, 1.0])
    x = torch.tensor(x_rows, dtype=torch.float)

    edge_src, edge_dst, edge_type_list = _build_spatial_edges(board)
    if not edge_src:
        raise ValueError(
            f"build_position_data: nessun arco spaziale prodotto per la "
            f"posizione (fen={board.fen()}); una posizione con mate_n "
            f"valido non dovrebbe mai essere priva di mosse legali per il "
            f"lato al comando (sarebbe gia' scacco matto/stallo)."
        )

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = encode_edge_type_onehot(edge_type_list)

    num_edges = edge_index.shape[1]
    time_tensor = torch.full((num_edges,), float(clock_seconds), dtype=torch.float)

    y = torch.tensor(encode_move(best_move), dtype=torch.long)

    # [NUOVO] Maschera delle mosse legali nel vocabolario esteso (20480).
    # Salvata come bool denso: 20480 bit = 2560 byte/posizione, trascurabile
    # rispetto al resto del Data. Consumata a valle nel training/eval loop
    # per mascherare i logit prima della softmax (vedi TrainPipeline).
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