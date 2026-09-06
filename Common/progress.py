"""
progress.py

Modulo unico per tutte le progress bar della pipeline (GamesBuilder,
PuzzleBuilder, TimeStatsBuilder, decompressione .zst, ecc.), cosi' che
stile, postfix e comportamento (refresh rate, dynamic_ncols, unit) siano
coerenti ovunque invece di essere ridefiniti ad-hoc in ogni builder.

USO TIPICO — barra "a conteggio" (partite processate, righe lette):

    from DatasetPipeline.Utils.progress import make_bar

    with make_bar(total=n_games, desc="Elaborazione partite", unit="game") as pbar:
        for game in games:
            ...
            pbar.update(1)

USO TIPICO — barra "a iteratore" (wrappa un generatore/pool.imap):

    from DatasetPipeline.Utils.progress import wrap_iter

    for item in wrap_iter(pool.imap_unordered(fn, tasks), desc="[GamesBuilder] Ricerca finestre matto"):
        ...

USO TIPICO — barra con contatori live (accepted/positions), sostituendo
pbar.set_postfix(...) sparso nel codice chiamante:

    pbar = make_bar(total=None, desc="...", unit="game")
    stats = LiveStats(pbar, fields=("accepted", "positions"))
    ...
    stats.update(accepted=accepted_windows, positions=enqueued_positions)

Nessuna dipendenza da GamesBuilder/PuzzleBuilder: puo' essere importato
anche da script standalone (fix_dataset.py, notebook di analisi, ecc.).
"""
from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional, TypeVar

from tqdm import tqdm

T = TypeVar("T")

_DEFAULT_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)
_DEFAULT_BAR_FORMAT_NO_TOTAL = "{l_bar}{bar}{n_fmt} [{elapsed}, {rate_fmt}{postfix}]"


def make_bar(
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    *,
    leave: bool = True,
    position: Optional[int] = None,
    disable: bool = False,
    mininterval: float = 0.3,
) -> tqdm:
    """Crea una tqdm bar con impostazioni coerenti per l'intera pipeline.

    Args:
        total: numero totale di elementi attesi. None per barre "a
            conteggio infinito" (es. wrap di un generatore di lunghezza
            ignota, come pool.imap_unordered su uno stream PGN).
        desc: etichetta mostrata a sinistra della barra.
        unit: unita' mostrata accanto al contatore (es. "game", "row",
            "position").
        leave: se False, la barra viene rimossa a fine ciclo (utile per
            barre annidate/temporanee).
        position: riga verticale della barra, per barre multiple
            concorrenti (0 = piu' in alto). None = auto.
        disable: disabilita la barra (utile per test o esecuzione non
            interattiva/log-only).
        mininterval: secondi minimi tra un refresh e l'altro, per non
            saturare stdout con migliaia di item/sec.

    Returns:
        Istanza tqdm pronta all'uso (come context manager o iterabile).
    """
    bar_format = _DEFAULT_BAR_FORMAT if total else _DEFAULT_BAR_FORMAT_NO_TOTAL
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        position=position,
        disable=disable,
        mininterval=mininterval,
        bar_format=bar_format,
        file=sys.stdout,
    )


def wrap_iter(
    iterable: Iterable[T],
    desc: str = "",
    unit: str = "it",
    *,
    total: Optional[int] = None,
    leave: bool = True,
    position: Optional[int] = None,
    disable: bool = False,
) -> Iterator[T]:
    """Wrappa un iterabile (generatore, pool.imap_unordered, DataLoader)
    con una tqdm bar coerente. Equivalente a `tqdm(iterable, ...)` ma con
    le stesse impostazioni di default di `make_bar`, cosi' non serve
    ripetere `dynamic_ncols=True` etc. in ogni punto della pipeline.
    """
    bar_format = _DEFAULT_BAR_FORMAT if total else _DEFAULT_BAR_FORMAT_NO_TOTAL
    return tqdm(
        iterable,
        desc=desc,
        unit=unit,
        total=total,
        dynamic_ncols=True,
        leave=leave,
        position=position,
        disable=disable,
        mininterval=0.3,
        bar_format=bar_format,
        file=sys.stdout,
    )


@dataclass
class LiveStats:
    """Aggiorna il postfix di una tqdm bar con contatori live, in modo
    thread-safe (utile quando la barra viene aggiornata sia dal thread
    principale sia da callback/worker).

    A differenza di chiamare `pbar.set_postfix(...)` sparso nel codice,
    centralizza il refresh (con throttling via `refresh=False` + refresh
    esplicito ogni `refresh_every` update) per non rallentare il loop
    principale con troppi redraw quando gli update sono molto frequenti.
    """
    pbar: tqdm
    refresh_every: int = 25
    _counter: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def update(self, **fields: Any) -> None:
        """Aggiorna i contatori mostrati nel postfix della barra.

        Args:
            **fields: coppie chiave/valore da mostrare (es.
                accepted=123, positions=4567). I valori numerici grandi
                vengono formattati con separatore delle migliaia.
        """
        formatted: Dict[str, str] = {}
        for key, value in fields.items():
            if isinstance(value, int):
                formatted[key] = f"{value:,}"
            elif isinstance(value, float):
                formatted[key] = f"{value:.3f}"
            else:
                formatted[key] = str(value)

        with self._lock:
            self._counter += 1
            do_refresh = (self._counter % self.refresh_every) == 0
            self.pbar.set_postfix(formatted, refresh=do_refresh)

    def force_refresh(self) -> None:
        """Forza un refresh immediato (es. da chiamare a fine loop, cosi'
        l'ultimo stato mostrato non resta quello dell'ultimo batch di
        `refresh_every`)."""
        with self._lock:
            self.pbar.refresh()


@contextmanager
def stage_bar(
    desc: str,
    total: Optional[int] = None,
    unit: str = "it",
    **kwargs: Any,
):
    pbar = make_bar(total=total, desc=desc, unit=unit, **kwargs)
    try:
        yield pbar
    finally:
        pbar.close()