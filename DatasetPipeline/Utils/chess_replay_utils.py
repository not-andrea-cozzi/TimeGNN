"""
chess_replay_utils.py

Funzioni di parsing PGN/clock/rating condivise, estratte dalla
duplicazione quasi identica presente in GamesBuilder.py e
ClubGamesTimedBuilder.py (entrambi definivano _parse_clk, _parse_emt,
_parse_time_control, _compute_move_duration, _parse_rating come metodi
statici privati con corpo identico).
Un'unica fonte di verita' qui evita
che un fix futuro (es. un nuovo formato di TimeControl, un edge case nel
parsing del rating) debba essere applicato in piu' punti e rischi di
disallinearsi tra i due builder.

"""
from __future__ import annotations

import re
from typing import Optional, Tuple

CLK_RE = re.compile(r"\[\s*%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]")
EMT_RE = re.compile(r"\[\s*%emt\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]")


def parse_clk(comment: str) -> Optional[float]:
    """Estrae %clk (tempo RIMANENTE) da un commento PGN, in secondi."""
    if not comment:
        return None
    match = CLK_RE.search(comment)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_emt(comment: str) -> Optional[float]:
    """Estrae %emt (tempo SPESO sulla mossa, gia' una durata) da un
    commento PGN, in secondi. A differenza di %clk, il valore NON richiede
    sottrazione con lo stato precedente: e' gia' move_duration."""
    if not comment:
        return None
    match = EMT_RE.search(comment)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_time_control(time_control: str) -> Tuple[float, float]:
    """Estrae (base_time_seconds, increment_seconds) dall'header PGN
    TimeControl (formato "base+increment" o solo "base"). Ritorna (0.0,
    0.0) se il formato non e' riconosciuto o e' assente ("-")."""
    if not time_control or time_control == "-":
        return 0.0, 0.0
    match = re.match(r"^(\d+)\+(\d+)$", time_control)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.match(r"^(\d+)$", time_control)
    if match:
        return float(match.group(1)), 0.0
    return 0.0, 0.0


def compute_move_duration(
    previous_clock: Optional[float], current_clock: Optional[float], increment: float
) -> Optional[float]:
    """Deriva la durata di una mossa da due letture consecutive di %clk
    (tempo rimanente prima/dopo). Ritorna None se manca uno dei due
    clock. Il risultato non e' mai negativo (clamp a 0.0)."""
    if previous_clock is None or current_clock is None:
        return None
    spent = previous_clock - current_clock + increment
    return max(0.0, spent)


def parse_rating(raw: Optional[str]) -> Optional[int]:
    """Converte un header WhiteElo/BlackElo in int, tollerando valori
    mancanti o non numerici (es. "?", stringa vuota, "1500?")."""
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in raw if ch.isdigit())
        return int(digits) if digits else None


def closest_bucket_time(rating: Optional[int], avg_time_by_rating: dict) -> Optional[float]:
    """Trova il tempo medio del bucket di rating piu' vicino in
    avg_time_by_rating ({bucket_rating: secondi_medi}, tipicamente
    prodotto da TimeStatBuilder.build_and_save). Ritorna None se il
    dizionario e' vuoto o il rating non e' disponibile, cosi' il
    chiamante puo' ricadere su un default costante."""
    if rating is None or not avg_time_by_rating:
        return None
    closest = min(avg_time_by_rating.keys(), key=lambda b: abs(b - rating))
    return avg_time_by_rating[closest]