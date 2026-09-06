"""
PositionGraphSchema.py

Schema dati per il sample scacchistico: UNA SINGOLA POSIZIONE = UN GRAFO
SPAZIALE a 64 nodi (caselle), pensato per essere usato SENZA MODIFICHE con
i modelli GIA' ESISTENTI in timegnn.models:

    - DualGATModel            (timegnn/models/gat_basic.py)          "untimed"
    - DualGATTimeAwareModel   (timegnn/models/gat_time_decay.py)     "timed"

QUESTI DUE MODELLI, NON ALTRI: PERCHE'
========================================
Il progetto (proggetto_ai.md) richiede un preciso ablation study: confrontare
un modello CON informazione temporale e uno SENZA, a parita' di tutto il
resto, per rispondere alla research question "does incorporating timing
information improve the model's performance?".

DualGATModel e DualGATTimeAwareModel sono, tra i modelli disponibili nella
libreria, l'UNICA coppia gia' pronta per questo confronto: hanno costruttori
quasi identici (la seconda ha solo lambda_decay in piu'), lo stesso forward
pass a tre path (embed/event/concat), lo stesso output shape (un logit per
NODO, mai pooling). L'unica differenza architetturale reale e' che
DualGATTimeAwareModel usa TimeAwareGATConv al posto di GATConv: la stessa
identica formula di attenzione, con in piu' un decay esponenziale
sull'attenzione pesato da un valore scalare per-arco (chiamato "time" nel
suo forward: `edge_attr = data_event.time`).

VINCOLO DI PROGETTO: i modelli non si toccano. Restano cosi' come sono,
comprese le loro limitazioni:
    - Nessun pooling graph-level: entrambi producono un tensore
      [N_nodi_nel_batch, output_dim], un logit per CASELLA, non un logit
      per l'intera board. Ottenere "una mossa per posizione" richiede un
      passo di aggregazione ESTERNO al modello (vedi
      position_pooling.py), scritto nel training/eval loop, non dentro
      gat_basic.py/gat_time_decay.py.
    - DualGATModel legge data_event.edge_attr (relazione SPAZIALE tra
      caselle: mossa-legale/attacco/pin).
    - DualGATTimeAwareModel legge data_event.time (un valore scalare
      per-arco: qui e' COSTANTE su tutti gli archi della stessa board,
      dato che "quanto tempo il giocatore ha pensato" e' un attributo
      della MOSSA/POSIZIONE nel suo complesso, non di una singola coppia
      di caselle. Il decay esponenziale pesa quindi l'intera rete di
      attenzione della board in base al tempo di riflessione: piu' tempo
      = meno decay = attenzione piena tra le caselle; poco tempo = decay
      forte = l'informazione fluisce meno tra caselle lontane. E'
      un'ipotesi modellistica esplicita, coerente con l'hypothesis di
      progetto "timing helps ... by modeling urgency", non un fatto
      dimostrato.

Un SOLO schema dati (questo modulo) serve ENTRAMBI i modelli: ogni Data
porta sia edge_attr (per DualGATModel) sia time (per DualGATTimeAwareModel)
gia' pronti, cosi' l'ablation study usa esattamente lo stesso dataset per
le due run, isolando la sola variabile "uso o meno del decay temporale".

CAMPI DEL SAMPLE (torch_geometric.data.Data), grafo a 64 nodi:

    event_ids   long[64, 1]  Categoria pezzo+colore per casella:
                                0                        = casella vuota
                                piece_type*2 + color + 1 = casella occupata
                              (piece_type: 1..6 = pawn..king, color: 0=nero
                              1=bianco; range risultante 1..12, quindi 13
                              categorie totali incluso lo 0). Shape [64,1]
                              (non [64]) perche' DualGATModel/DualGATTimeAwareModel
                              derivano da forward comune che fa
                              `self.embedding(data_event.event_ids.view(-1))`:
                              .view(-1) accetta sia [64] sia [64,1], ma [64,1]
                              e' la convenzione scelta qui per coerenza con
                              PrefixGCNClassifier (che invece richiede
                              .squeeze(-1), quindi [N,1] esplicito) nel caso
                              in futuro si voglia riusare lo stesso schema
                              anche li'.

    x           float[64,2]  Feature per casella (path "event", input
                              diretto GAT, NON passa per l'embedding):
                                col 0: is_occupied_by_mover     (0.0/1.0)
                                col 1: is_occupied_by_opponent  (0.0/1.0)
                              Una casella vuota ha entrambe le colonne a
                              0.0. Feature volutamente ridondanti rispetto
                              a event_ids (che gia' codifica il colore):
                              tenerle esplicite in x, che passa per un path
                              GAT NON-embedding, da' al modello un segnale
                              diretto e immediato su "di chi e' questo
                              pezzo" senza dover imparare a decodificarlo
                              dall'embedding.

    edge_index  long[2,E]    Archi SPAZIALI tra caselle: mossa-legale,
                              attacco, pin (stessa logica del vecchio
                              GraphBuilder.board_to_pyg_data). E varia per
                              posizione (non fisso), il batching PyG lo
                              gestisce nativamente concatenando gli
                              edge_index con offset (Batch.from_data_list).

    edge_attr   long[E]      Tipo di relazione spaziale per l'arco:
                              EDGE_LEGAL_MOVE / EDGE_ATTACK / EDGE_PIN
                              (vedi costanti sotto). Letto da DualGATModel
                              come edge_dim per GATConv (occhio: GATConv si
                              aspetta un tensore edge_dim-dimensionale per
                              arco, non un indice categorico grezzo: va
                              quindi passato come float, one-hot o
                              embeddato a monte -- vedi nota
                              EDGE_ATTR_ENCODING sotto per la scelta fatta).

    time        float[E]     Tempo (secondi) impiegato per la mossa che ha
                              PORTATO a questa posizione, ripetuto
                              IDENTICO su tutti gli E archi della board
                              (broadcast di uno scalare per-grafo, non
                              un'informazione per-arco reale: vedi
                              motivazione sopra). Letto da
                              DualGATTimeAwareModel come `data_event.time`.

    y           long (scalare, NON per nodo) Mossa migliore in questa
                              posizione, nello stesso vocabolario globale
                              fisso a 4096 classi (from_sq*64+to_sq) usato
                              altrove nel progetto. E' UN SOLO intero per
                              l'intera board: dato che i modelli producono
                              un logit per NODO, il confronto con y avviene
                              DOPO il pooling esterno (vedi
                              position_pooling.py), non dentro il dataset.

    game_id     int64 (scalare) Id della finestra/partita di provenienza,
                              per tracciamento a valle (queue, split). Non
                              e' letto dal forward dei modelli.

    ply         int64 (scalare) Indice del ply all'interno della finestra
                              di matto forzato di provenienza (0-based).
                              Serve a valle per ricostruire l'ordine delle
                              posizioni della stessa finestra, se in futuro
                              serve un'analisi/aggregazione sequenziale
                              (i modelli attuali non la usano).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess
import torch
from torch_geometric.data import Data

# ============================================================================
# VOCABOLARI FISSI
# ============================================================================

MOVE_VOCAB_SIZE = 64 * 64  # 4096: from_square*64 + to_square

# event_ids: 0 = casella vuota, 1..12 = piece_type*2+color+1
EVENT_ID_EMPTY = 0
NUM_EVENT_ID_CATEGORIES = 13  # 0 (vuoto) + 12 (6 piece_type x 2 colori)

# edge_attr: tipo di relazione spaziale tra caselle (stessa semantica del
# vecchio GraphBuilder: EDGE_LEGAL_MOVE/ATTACK/PIN, senza EDGE_PAD qui
# perche' una board reale ha sempre almeno un arco valido nel nostro caso
# d'uso -- posizioni con mate_n valido non sono mai in stallo).
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
    """Codifica una mossa nel vocabolario globale fisso (0..4095).

    NOTA: la promozione NON e' codificata (vedi NOTA PROMOTION_COLLISION
    nel docstring di modulo).
    """
    return move.from_square * 64 + move.to_square


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
    game_id: int,
    ply: int,
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
        game_id: id della finestra/partita di provenienza (tracciamento).
        ply: indice del ply all'interno della finestra (tracciamento).

    Returns:
        Data con event_ids/x/edge_index/edge_attr/time/y/game_id/ply come
        da docstring di modulo.

    Raises:
        ValueError: se best_move non e' una mossa legale su board (il
            target deve sempre essere verificabile sulla posizione data).
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

    data = Data(
        event_ids=event_ids,
        x=x,
        edge_index=edge_index,
        num_nodes=64,
    )
    data.edge_attr = edge_attr
    data.time = time_tensor
    data.y = y
    data.game_id = torch.tensor(int(game_id), dtype=torch.int64)
    data.ply = torch.tensor(int(ply), dtype=torch.int64)

    return data