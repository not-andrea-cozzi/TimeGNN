"""
MetricsLogger - Logging CSV delle metriche di training/validazione.

Un file CSV per modello: metrics_<model_name>.csv nella directory indicata.
Modalità append: se il file esiste, i nuovi record vengono aggiunti in coda
mantenendo lo stesso header. Se cambiano le colonne rispetto all'header
esistente, viene loggato un warning e i nuovi campi mancanti restano vuoti
(i campi extra vengono scartati) per non corrompere la struttura del file.

Scrittura riga-per-riga con flush immediato: un crash a metà training non
perde le epoche già completate.
"""
from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger("metrics_logger")


class MetricsLogger:
    """Logger CSV per-modello, un record per epoca."""

    def __init__(
        self,
        model_name: str,
        output_dir: str = "metrics",
        filename: Optional[str] = None,
    ):
        """
        Args:
            model_name: Nome del modello (usato per il filename e come colonna).
            output_dir: Directory dove salvare il CSV (creata se non esiste).
            filename: Nome file custom. Se None, usa 'metrics_<model_name>.csv'.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        fname = filename or f"metrics_{self._sanitize(model_name)}.csv"
        self.path = os.path.join(output_dir, fname)

        # Se il file esiste, leggiamo l'header per coerenza in append.
        self._existing_header: Optional[list] = None
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            try:
                with open(self.path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    self._existing_header = next(reader, None)
                logger.info(
                    f"[{model_name}] CSV esistente trovato in {self.path} "
                    f"({len(self._existing_header or [])} colonne), modalità append."
                )
            except Exception as e:
                logger.warning(f"[{model_name}] Impossibile leggere header esistente: {e}")

    @staticmethod
    def _sanitize(name: str) -> str:
        """Rende un nome sicuro per il filesystem."""
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_top1: float,
        val_top3: float,
        lr: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Scrive una riga di metriche per l'epoca appena completata.

        Args:
            epoch: Numero epoca (1-based tipicamente).
            train_loss, train_acc: metriche di training.
            val_loss, val_top1, val_top3: metriche di validazione.
            lr: Learning rate corrente (opzionale).
            extra: Dizionario di metriche aggiuntive (es. grad_norm, tempo epoca).
        """
        record: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_name": self.model_name,
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_top1": f"{val_top1:.6f}",
            "val_top3": f"{val_top3:.6f}",
            "lr": f"{lr:.2e}" if lr is not None else "",
        }
        if extra:
            for k, v in extra.items():
                record[k] = f"{v:.6f}" if isinstance(v, float) else v

        # Determina header: prima scrittura -> usa chiavi correnti;
        # append -> usa header esistente e allinea i campi.
        if self._existing_header is None:
            header = list(record.keys())
            write_header = True
            self._existing_header = header
        else:
            header = self._existing_header
            write_header = False
            new_keys = set(record.keys()) - set(header)
            missing_keys = set(header) - set(record.keys())
            if new_keys:
                logger.warning(
                    f"[{self.model_name}] Colonne extra ignorate (non in header CSV): {new_keys}"
                )
            if missing_keys:
                for k in missing_keys:
                    record.setdefault(k, "")

        # Scrittura con flush immediato per resistere ai crash.
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(record)
            f.flush()
            os.fsync(f.fileno())

        logger.debug(f"[{self.model_name}] Epoca {epoch} loggata in {self.path}")