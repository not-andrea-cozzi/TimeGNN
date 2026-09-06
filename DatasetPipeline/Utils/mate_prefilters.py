"""
mate_prefilters.py

Filtri economici AGGIUNTIVI (nessuna chiamata Stockfish) da applicare
PRIMA di GamesBuilder._analyse_position, nello stesso punto della catena
dove oggi girano _get_candidate_legal_moves / _mover_has_heavy_piece /
_has_mating_material / _is_trivially_drawn_endgame / _syzygy_says_no_mate.

QUESTO MODULO NON MODIFICA GamesBuilder.py. Le funzioni qui sono pure
(prendono una board, ritornano bool) e vanno chiamate esplicitamente nel
punto di innesto indicato in fondo al file, quando/se deciderai di
integrarle.

"""
from __future__ import annotations

from typing import Optional

import chess


def king_has_limited_escape_squares(
    board: "chess.Board",
    max_free_squares: int = 2,
) -> bool:
    """True se il re del NON-mover (il re che si cerca di dare matto) ha
    al piu' `max_free_squares` case libere e non attaccate dal mover tra
    le sue case adiacenti.

    Razionale euristico: un matto forzato in poche mosse richiede quasi
    sempre che la rete di fuga del re bersaglio sia gia' ristretta (o lo
    diventi rapidamente). Se il re ha molte case di fuga libere e non
    sotto attacco, un mate_n basso (1-3) e' statisticamente raro. Per
    mate_range piu' alti (6-10) questo filtro e' meno affidabile (il re
    puo' essere braccato in piu' mosse a partire da una posizione con
    ancora spazio): usalo con cautela se il tuo mate_range include valori
    alti, o alza max_free_squares di conseguenza.

    NON e' una condizione necessaria: esistono matti forzati che iniziano
    con il re bersaglio ancora relativamente libero (es. mating net che si
    chiude in 4-5 mosse). E' un filtro di velocita', non di correttezza.

    Args:
        board: posizione da valutare (board.turn = mover).
        max_free_squares: soglia di case libere e non attaccate oltre la
            quale la posizione viene considerata "poco promettente" per un
            mate_n basso. Default 2, tarabile.

    Returns:
        True se la posizione supera il filtro (poche case di fuga: vale
        la pena analizzarla con Stockfish). False se il re ha ancora
        troppo spazio (scartabile senza Stockfish).
    """
    target_color = not board.turn
    king_square = board.king(target_color)
    if king_square is None:
        return True  # board malformata: non e' compito di questo filtro deciderlo, lascia passare

    free_and_safe = 0
    for adjacent_square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]):
        piece_on_square = board.piece_at(adjacent_square)
        if piece_on_square is not None and piece_on_square.color == target_color:
            continue  # occupata da un pezzo proprio: non e' una casa di fuga
        if board.is_attacked_by(board.turn, adjacent_square):
            continue  # attaccata dal mover: non e' una fuga sicura
        free_and_safe += 1
        if free_and_safe > max_free_squares:
            return False

    return True


def mover_has_check_or_heavy_piece_near_king(
    board: "chess.Board",
    king_distance_threshold: int = 3,
) -> bool:
    """True se almeno una mossa legale del mover produce scacco, OPPURE se
    almeno un pezzo pesante (regina/torre) del mover si trova entro
    `king_distance_threshold` case (distanza di Chebyshev) dal re
    avversario.

    Razionale euristico: un attacco di matto in corso quasi sempre ha gia'
    un pezzo pesante vicino al re bersaglio o produce scacco immediato da
    almeno una mossa. Posizioni dove nessuna mossa da' scacco E nessun
    pezzo pesante e' vicino al re avversario sono tipicamente posizioni di
    sviluppo/mediogioco lontane da un matto imminente.

    Piu' costoso di king_has_limited_escape_squares (deve enumerare
    board.legal_moves per il controllo scacco), ma ancora ordini di
    grandezza piu' economico di una chiamata Stockfish.

    NON e' una condizione necessaria per mate_n alti (un matto in 6-10
    mosse puo' iniziare senza scacco immediato ne' pezzi pesanti gia'
    vicini: l'attacco si costruisce nel corso della sequenza). Usalo se il
    tuo mate_range e' concentrato sui valori bassi.

    Args:
        board: posizione da valutare (board.turn = mover).
        king_distance_threshold: distanza di Chebyshev (scacchi: max tra
            differenza di colonna e di riga) entro la quale un pezzo
            pesante del mover conta come "vicino" al re avversario.

    Returns:
        True se la posizione supera il filtro (scacco disponibile o
        pezzo pesante vicino: vale la pena analizzarla). False altrimenti.
    """
    target_color = not board.turn
    king_square = board.king(target_color)
    if king_square is None:
        return True

    for move in board.legal_moves:
        if board.gives_check(move):
            return True

    king_file, king_rank = chess.square_file(king_square), chess.square_rank(king_square)
    for piece_type in (chess.QUEEN, chess.ROOK):
        for square in board.pieces(piece_type, board.turn):
            file_dist = abs(chess.square_file(square) - king_file)
            rank_dist = abs(chess.square_rank(square) - king_rank)
            if max(file_dist, rank_dist) <= king_distance_threshold:
                return True

    return False


def material_advantage_sufficient_for_mate_range(
    board: "chess.Board",
    mate_n_upper_bound: int,
    piece_values: Optional[dict] = None,
) -> bool:
    """True se il vantaggio materiale del mover e' coerente con un matto
    forzato entro mate_n_upper_bound mosse, con soglia SCALATA su
    mate_n_upper_bound (piu' stringente di min_material_diff_for_mate_attempt
    fisso gia' presente in GamesBuilder, che usa una soglia costante
    indipendente da quanto e' basso il mate_range richiesto).

    Razionale euristico: un mate forzato in 1-2 mosse richiede quasi
    sempre un vantaggio materiale netto significativo o un attacco gia'
    decisivo; un mate in 8-10 mosse puo' invece avvenire anche con
    materiale sostanzialmente pari (mating net posizionale). La soglia qui
    scala inversamente con mate_n_upper_bound: piu' basso il mate_n
    cercato, piu' alto il vantaggio richiesto per non scartare la
    posizione.

    Pensato per essere usato IN AGGIUNTA a
    GamesBuilder._has_mating_material (non in sostituzione): quello
    esistente resta il filtro di baseline con soglia fissa
    (min_material_for_mate_attempt, min_material_diff_for_mate_attempt);
    questo aggiunge una soglia piu' stringente quando il mate_range
    massimo cercato in quella run e' basso.

    Args:
        board: posizione da valutare (board.turn = mover).
        mate_n_upper_bound: il valore massimo di mate_n che la ricerca
            corrente accetta (tipicamente games_cfg.mate_range_max).
        piece_values: mappa piece_type->valore. Se None, usa gli standard
            (pawn=1, knight=3, bishop=3, rook=5, queen=9), stesso schema
            gia' usato in GamesBuilder._PIECE_VALUES.

    Returns:
        True se il vantaggio materiale e' coerente con mate_n_upper_bound
        (vale la pena analizzarla). False se il vantaggio e' troppo basso
        per la profondita' di matto massima cercata.
    """
    values = piece_values or {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    mover_material = 0
    opponent_material = 0
    for piece in board.piece_map().values():
        value = values.get(piece.piece_type, 0)
        if piece.color == board.turn:
            mover_material += value
        else:
            opponent_material += value

    material_diff = mover_material - opponent_material

    # Scala lineare inversa: mate_n=1 -> richiede diff >= 6; mate_n=10 ->
    # richiede diff >= 0 (nessun vincolo aggiuntivo oltre il baseline
    # esistente). Soglie scelte come punto di partenza ragionevole, non
    # derivate da dati: se troppo aggressive/permissive, vanno tarate
    # osservando il tasso di finestre accettate perse.
    min_required_diff = max(0, 7 - mate_n_upper_bound)
    return material_diff >= min_required_diff


def passes_all_prefilters(
    board: "chess.Board",
    mate_n_upper_bound: int,
    max_free_squares: int = 2,
    king_distance_threshold: int = 3,
    piece_values: Optional[dict] = None,
) -> bool:
    """Combina i tre filtri sopra in AND logico, con ordine scelto per
    fail-fast sul filtro piu' economico prima (king_has_limited_escape_squares
    non enumera legal_moves, gli altri due si').

    Ritorna False alla prima condizione non soddisfatta (short-circuit):
    non e' necessario chiamare tutti e tre i filtri se il primo gia'
    scarta la posizione.
    """
    if not king_has_limited_escape_squares(board, max_free_squares=max_free_squares):
        return False
    if not material_advantage_sufficient_for_mate_range(board, mate_n_upper_bound, piece_values=piece_values):
        return False
    if not mover_has_check_or_heavy_piece_near_king(board, king_distance_threshold=king_distance_threshold):
        return False
    return True


# ----------------------------------------------------------------------
# PUNTO DI INNESTO (non applicato qui, da fare esplicitamente in
# GamesBuilder._find_mate_window_start se decidi di adottarlo):
#
#   dentro il ciclo while, DOPO self._has_mating_material(board) e PRIMA
#   di self._analyse_position(board):
#
#       from DatasetPipeline.Utils.mate_prefilters import passes_all_prefilters
#       ...
#       if not passes_all_prefilters(board, mate_n_upper_bound=self.mate_range[1]):
#           node = next_node
#           continue
#
#   Posizionato dopo _has_mating_material apposta: quel filtro e' il piu'
#   economico in assoluto (somma su piece_map, nessuna enumerazione di
#   mosse/attacchi), quindi resta la prima barriera. I filtri qui sono
#   piu' costosi (enumerano legal_moves o square adiacenti) ma sempre
#   ordini di grandezza sotto una chiamata Stockfish.
#
# COME MISURARE L'IMPATTO (raccomandato PRIMA di integrare in produzione):
#   contare, con un Counter separato per run di prova su un sottoinsieme
#   di partite (es. le prime 5000), quante volte si arriva a
#   _analyse_position CON e SENZA questi prefiltri attivi. Se il numero di
#   chiamate Stockfish non scende in modo significativo, il collo di
#   bottiglia e' altrove (parsing PGN, I/O, _game_is_eligible) e questi
#   filtri da soli non velocizzeranno l'estrazione in modo percepibile.
# ----------------------------------------------------------------------