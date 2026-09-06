import logging
from typing import Any, List, Optional
import chess
import chess.pgn

logger = logging.getLogger("game_filters")


class GameFilter:
    """Modulo di filtraggio essenziale: rimuove solo stati non validi o già terminati."""

    @staticmethod
    def validate_kings(board: chess.Board) -> bool:
        """Verifica la presenza di entrambi i Re per evitare errori dell'engine."""
        return board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None

    @staticmethod
    def is_game_eligible(game: chess.pgn.Game, min_plies: int = 2) -> bool:
        """Filtra solo partite vuote o corrotte."""
        if game is None:
            return False
        try:
            return game.end().ply() >= min_plies
        except Exception:
            return False

    @staticmethod
    def get_candidate_legal_moves(board: chess.Board) -> Optional[List[chess.Move]]:
        """Restituisce le mosse legali solo se la posizione è giocabile."""
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            return None
        return list(board.legal_moves)

    @staticmethod
    def syzygy_says_no_mate(board: chess.Board, tablebase: Any) -> bool:
        """Consulta Syzygy per scartare posizioni matematicamente senza matto."""
        if tablebase is None:
            return False
        if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
            return False
        try:
            wdl = tablebase.probe_wdl(board)
            return wdl is not None and wdl <= 0
        except Exception:
            return False