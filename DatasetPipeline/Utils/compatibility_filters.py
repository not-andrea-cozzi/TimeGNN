"""
compatibility_filters.py

Filtri di compatibilita' per GamesBuilder, isolati in un modulo dedicato
per due motivi: (1) renderli testabili/tarabili in isolamento senza
toccare la logica di orchestrazione del builder, (2) rendere esplicita la
distinzione tra requisiti HARD (violarli produce un Data incoerente con
PositionGraphSchema o con la specifica di progetto: la finestra va
scartata sempre, non e' negoziabile) e filtri SOFT di qualita'
(riducono rumore/banalita' del dataset ma una partita che li viola e'
comunque strutturalmente valida: il default e' pensato per massimizzare
la resa, tenendo pero' attive le due esclusioni esplicitamente richieste).

===========================================================================
HARD REQUIREMENTS (mai disattivabili, verificati in is_window_hard_valid /
is_game_hard_eligible)
===========================================================================
Motivazione per ciascuno (perche' non e' negoziabile):

  1. Entrambi i Re presenti sulla board
     -> PositionGraphSchema.encode_square_event_id e la ricerca Stockfish
        assumono una board scacchisticamente valida; senza un Re
        board.legal_moves/is_checkmate() sono indefiniti o sollevano.

  2. WhiteElo e BlackElo entrambi presenti e numerici
     -> Richiesto esplicitamente (decisione utente): senza rating la
        partita non entra nel dataset, punto. Non e' un requisito dello
        schema Data (che si costruirebbe comunque), ma e' un requisito di
        PROGETTO qui reso hard su richiesta esplicita.

  3. La finestra di matto termina in board.is_checkmate() reale
     -> Il target del dataset (vedi proggetto_ai.md, "mate in n moves") e'
        un matto VERO sulla scacchiera, non uno score "mate" riportato
        dall'engine che poi la partita reale non realizza (es. avversario
        devia, o la partita PGN finisce prima). Un finto matto
        romperebbe silenziosamente la semantica del label y.

  4. mate_n (da Stockfish) dentro mate_range configurato
     -> Requisito diretto della research question del progetto (n=1..10,
        stratificazione per profondita').

  5. Ogni mossa della finestra e' legale al momento del replay
     -> Se una mossa del PGN non e' legale sulla board ricostruita, il PGN
        e' corrotto o il replay ha perso sincronizzazione: la finestra
        intera e' inaffidabile, non solo quella mossa.

  6. Almeno un arco spaziale costruibile per ogni posizione della finestra
     -> Requisito diretto di PositionGraphSchema.build_position_data (che
        solleva ValueError se non c'e' nessun arco): una posizione senza
        mosse legali per il mover sarebbe gia' matto/stallo, incompatibile
        con "posizione a N ply dalla fine di una sequenza di matto".

===========================================================================
FILTRI SOFT (qualita', configurabili via QualityFilterConfig)
===========================================================================
Le DUE esclusioni esplicitamente richieste sono ON by default:

  - skip_time_forfeit: scarta partite terminate per tempo scaduto
    (Termination contiene "Time forfeit"). Motivazione: un matto raggiunto
    perche' l'avversario ha perso per tempo, non per la sequenza di mosse
    giocata, non e' rappresentativo di "guidance per risolvere un matto
    forzato" -- la partita reale potrebbe non essere mai arrivata al matto
    sulla board per via del gioco.

    ATTENZIONE: questo filtro guarda l'HEADER Termination dell'INTERA
    partita, non la finestra estratta. Una partita con Termination="Time
    forfeit" puo' comunque contenere una sequenza di matto forzato reale
    PRIMA della fine per tempo. Si accetta questo margine di
    imprecisione: verificare "la FINESTRA specifica e' stata raggiunta per
    tempo" non e' determinabile dal solo header PGN senza logica
    addizionale sui timestamp per-mossa a fine finestra, fuori scope di
    un filtro leggero.

  - skip_forced_single_move_window: scarta la finestra se OGNI posizione
    della sequenza di matto ha esattamente una mossa legale per il
    giocatore di turno (sequenza "a senso unico", nessuna scelta reale in
    nessun punto della finestra). Motivazione: una sequenza dove il mover
    non ha mai alternative non e' informativa per un modello che deve
    IMPARARE a scegliere la mossa giusta tra piu' candidate.

    NOTA: la finestra e' scartata solo se TUTTE le posizioni sono a mossa
    forzata. Se anche una sola posizione ha piu' alternative, la finestra
    passa.

Tutti gli altri filtri storici (only_decisive_games, min_material_*,
require_heavy_piece, skip_trivial_endgame, min_rating/max_rating,
max_piece_count, candidate_max_legal_moves, prefiltri extra) sono
disattivati di default in QualityFilterConfig per massimizzare la resa
(target: ~40% delle partite in input compatibili), riattivabili
singolarmente se in futuro serve piu' qualita' a scapito della quantita'.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn

from DatasetPipeline.Model.ChessConstants import PIECE_VALUES as _CLASS_PIECE_VALUES
_PIECE_VALUES: Dict[int, int] = _CLASS_PIECE_VALUES

# ============================================================================
# HARD REQUIREMENTS
# ============================================================================

def validate_kings(board: chess.Board) -> bool:
    """HARD. Vedi motivazione (1) nel docstring di modulo."""
    return board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None


def parse_rating_strict(raw: Optional[str]) -> Optional[int]:
    """Converte un header WhiteElo/BlackElo in int. A differenza di
    chess_replay_utils.parse_rating, qui NON si tenta un recupero
    permissivo (estrazione cifre da stringhe tipo '1500?'): un rating
    hard-required deve essere inequivocabile, non un best-effort."""
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def has_valid_ratings(headers: "chess.pgn.Headers") -> bool:
    """HARD. Vedi motivazione (2). Richiede ENTRAMBI i rating validi."""
    white = parse_rating_strict(headers.get("WhiteElo"))
    black = parse_rating_strict(headers.get("BlackElo"))
    return white is not None and black is not None


def is_real_checkmate(board: chess.Board) -> bool:
    """HARD. Vedi motivazione (3)."""
    return board.is_checkmate()


def mate_n_in_range(mate_n: Optional[int], mate_range: Tuple[int, int]) -> bool:
    """HARD. Vedi motivazione (4)."""
    if mate_n is None:
        return False
    lo, hi = mate_range
    return mate_n > 0 and lo <= mate_n <= hi


def is_game_hard_eligible(game: Optional["chess.pgn.Game"], min_plies: int = 2) -> bool:
    """HARD, a livello di partita intera (prima ancora di cercare la
    finestra di matto): partita presente, non vuota/corrotta, con
    entrambi i rating validi. Il controllo Re/checkmate/mate_n si applica
    invece a livello di finestra (is_window_hard_valid), dato che
    dipendono dalla posizione specifica trovata.
    """
    if game is None:
        return False
    try:
        if game.end().ply() < min_plies:
            return False
    except Exception:
        return False
    return has_valid_ratings(game.headers)


def is_window_hard_valid(
    final_board: chess.Board,
    mate_n: Optional[int],
    mate_range: Tuple[int, int],
) -> bool:
    """HARD, a livello di finestra: la board finale della sequenza deve
    essere un matto reale, con Re presenti, e mate_n nel range. Va
    chiamato DOPO aver rigiocato l'intera finestra fino all'ultima mossa.
    """
    if not validate_kings(final_board):
        return False
    if not is_real_checkmate(final_board):
        return False
    if not mate_n_in_range(mate_n, mate_range):
        return False
    return True


# ============================================================================
# FILTRI SOFT (qualita')
# ============================================================================

@dataclass
class QualityFilterConfig:
    """Configurazione dei filtri soft. Le due esclusioni esplicitamente
    richieste sono ON di default; tutto il resto e' OFF di default per
    massimizzare la resa (vedi docstring di modulo)."""

    # --- Esclusioni esplicitamente richieste (default: attive) ---
    skip_time_forfeit: bool = True
    skip_forced_single_move_window: bool = True

    # --- Filtri storici di qualita' (default: disattivati) ---
    only_decisive_games: bool = False
    min_material_for_mate_attempt: int = 0
    min_material_diff_for_mate_attempt: int = 0
    require_heavy_piece: bool = False
    skip_trivial_endgame: bool = False
    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    max_piece_count: Optional[int] = None
    candidate_min_legal_moves: int = 1
    candidate_max_legal_moves: Optional[int] = None
    skip_if_in_check: bool = False


def game_termination_is_time_forfeit(headers: "chess.pgn.Headers") -> bool:
    termination = headers.get("Termination", "") or ""
    return "Time forfeit" in termination


def passes_time_forfeit_filter(headers: "chess.pgn.Headers", cfg: QualityFilterConfig) -> bool:
    """SOFT (default ON). True = la partita passa il filtro (NON e'
    time-forfeit, o il filtro e' disattivato)."""
    if not cfg.skip_time_forfeit:
        return True
    return not game_termination_is_time_forfeit(headers)


def window_is_forced_single_move_throughout(window_boards: List[chess.Board]) -> bool:
    """True se OGNI board della finestra (una per ply, board PRIMA della
    mossa) ha esattamente una mossa legale per il mover. window_boards deve
    contenere una board per ciascun ply della finestra, nell'ordine di
    gioco (accumulate durante il replay, prima di ogni push).
    """
    if not window_boards:
        return False
    return all(len(list(b.legal_moves)) <= 1 for b in window_boards)


def passes_forced_move_filter(window_boards: List[chess.Board], cfg: QualityFilterConfig) -> bool:
    """SOFT (default ON). True = la finestra passa il filtro (contiene
    almeno un punto di scelta reale, o il filtro e' disattivato)."""
    if not cfg.skip_forced_single_move_window:
        return True
    return not window_is_forced_single_move_throughout(window_boards)


def _material_by_color(board: chess.Board) -> Tuple[int, int]:
    white = 0
    black = 0
    for p in board.piece_map().values():
        val = _PIECE_VALUES.get(p.piece_type, 0)
        if p.color == chess.WHITE:
            white += val
        else:
            black += val
    return white, black


def has_mating_material(board: chess.Board, cfg: QualityFilterConfig) -> bool:
    """SOFT (default: soglie a 0, quindi passa sempre)."""
    mover = board.turn
    white_mat, black_mat = _material_by_color(board)
    mover_mat = white_mat if mover == chess.WHITE else black_mat
    opp_mat = black_mat if mover == chess.WHITE else white_mat
    if mover_mat < cfg.min_material_for_mate_attempt:
        return False
    if (mover_mat - opp_mat) < cfg.min_material_diff_for_mate_attempt:
        return False
    return True


def mover_has_heavy_piece(board: chess.Board) -> bool:
    mover = board.turn
    for pt in (chess.QUEEN, chess.ROOK):
        if board.pieces(pt, mover):
            return True
    return False


def passes_heavy_piece_filter(board: chess.Board, cfg: QualityFilterConfig) -> bool:
    """SOFT (default OFF)."""
    if not cfg.require_heavy_piece:
        return True
    return mover_has_heavy_piece(board)


def is_trivially_drawn_endgame(board: chess.Board) -> bool:
    piece_map = board.piece_map()
    has_heavy_or_pawn = any(
        p.piece_type in (chess.QUEEN, chess.ROOK, chess.PAWN) for p in piece_map.values()
    )
    if has_heavy_or_pawn:
        return False
    white_minors = sum(
        1 for p in piece_map.values() if p.color == chess.WHITE and p.piece_type in (chess.BISHOP, chess.KNIGHT)
    )
    black_minors = sum(
        1 for p in piece_map.values() if p.color == chess.BLACK and p.piece_type in (chess.BISHOP, chess.KNIGHT)
    )
    return white_minors <= 1 and black_minors <= 1


def passes_trivial_endgame_filter(board: chess.Board, cfg: QualityFilterConfig) -> bool:
    """SOFT (default OFF)."""
    if not cfg.skip_trivial_endgame:
        return True
    return not is_trivially_drawn_endgame(board)


def passes_rating_range_filter(headers: "chess.pgn.Headers", cfg: QualityFilterConfig) -> bool:
    """SOFT (default OFF: min_rating/max_rating None -> passa sempre).
    Presuppone rating gia' validati a livello hard (has_valid_ratings)."""
    if cfg.min_rating is None and cfg.max_rating is None:
        return True
    white = parse_rating_strict(headers.get("WhiteElo"))
    black = parse_rating_strict(headers.get("BlackElo"))
    ratings = [r for r in (white, black) if r is not None]
    if not ratings:
        return True
    best_rating = max(ratings)
    worst_rating = min(ratings)
    if cfg.min_rating is not None and best_rating < cfg.min_rating:
        return False
    if cfg.max_rating is not None and worst_rating > cfg.max_rating:
        return False
    return True


def passes_decisive_game_filter(headers: "chess.pgn.Headers", cfg: QualityFilterConfig) -> bool:
    """SOFT (default OFF)."""
    if not cfg.only_decisive_games:
        return True
    return headers.get("Result", "") in ("1-0", "0-1")


def get_candidate_legal_moves(board: chess.Board, cfg: QualityFilterConfig) -> Optional[List[chess.Move]]:
    """SOFT sui limiti numerici (default: min=1, max=None -> quasi sempre
    passa); esclude posizioni gia' terminali (matto/stallo/materiale
    insufficiente), necessario per non sprecare analisi su posizioni
    morte, ma non e' un requisito HARD sulla finestra finale."""
    if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
        return None
    if cfg.max_piece_count is not None and len(board.piece_map()) > cfg.max_piece_count:
        return None
    moves = list(board.legal_moves)
    if len(moves) < cfg.candidate_min_legal_moves:
        return None
    if cfg.candidate_max_legal_moves is not None and len(moves) > cfg.candidate_max_legal_moves:
        return None
    if cfg.skip_if_in_check and board.is_check():
        return None
    return moves


def passes_all_quality_filters_for_candidate(board: chess.Board, cfg: QualityFilterConfig) -> bool:
    """Aggrega i filtri soft applicabili a una posizione CANDIDATA (prima
    di lanciare Stockfish su di essa), per evitare di sprecare analisi su
    posizioni che verrebbero comunque scartate dopo."""
    if get_candidate_legal_moves(board, cfg) is None:
        return False
    if not passes_heavy_piece_filter(board, cfg):
        return False
    if not has_mating_material(board, cfg):
        return False
    if not passes_trivial_endgame_filter(board, cfg):
        return False
    return True