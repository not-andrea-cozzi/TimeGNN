"""
TrainState - Gestione dello stato di training con supporto per last/best checkpoint.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger("train_state")


class TrainState:
    """
    Mantiene lo stato del training: epoca corrente, history, best_val_loss.
    Supporta il salvataggio/caricamento di checkpoint con percorsi personalizzati.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        epoch: int = 0,
        history: Optional[List[Dict[str, float]]] = None,
        best_val_loss: float = float("inf"),
    ):
        """
        Args:
            checkpoint_path: Percorso base per i checkpoint (usato come default per save/load).
            epoch: Epoca di partenza (0 = nessuna epoca completata).
            history: Lista di dizionari con metriche per epoca.
            best_val_loss: Miglior loss di validazione finora.
        """
        self.checkpoint_path = checkpoint_path
        self.epoch = epoch
        self.history = history if history is not None else []
        self.best_val_loss = best_val_loss

    def try_resume(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        map_location: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> bool:
        """
        Tenta di caricare lo stato da un checkpoint. Se il file esiste, aggiorna
        model, optimizer, scaler, self.epoch, self.history, self.best_val_loss.
        Restituisce True se il caricamento è riuscito.

        Args:
            checkpoint_path: Se None, usa self.checkpoint_path.
        """
        path = checkpoint_path or self.checkpoint_path
        if path is None or not os.path.exists(path):
            logger.info("Nessun checkpoint trovato, partenza da zero.")
            return False

        logger.info(f"Caricamento checkpoint da {path}...")
        try:
            state = torch.load(path, map_location=map_location or "cpu")
        except Exception as e:
            logger.error(f"Errore nel caricamento del checkpoint: {e}")
            return False

        # Aggiorna il modello
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            # Se il checkpoint contiene direttamente il state_dict del modello
            model.load_state_dict(state)

        # Aggiorna l'ottimizzatore
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        # Aggiorna lo scaler (AMP)
        if scaler is not None and "scaler_state_dict" in state and state["scaler_state_dict"]:
            scaler.load_state_dict(state["scaler_state_dict"])

        # Aggiorna i campi di TrainState
        self.epoch = state.get("epoch", 0)
        self.history = state.get("history", [])
        self.best_val_loss = state.get("best_val_loss", float("inf"))

        logger.info(
            f"Checkpoint caricato: epoca {self.epoch}, best_val_loss={self.best_val_loss:.4f}, "
            f"{len(self.history)} epoche in history."
        )
        return True

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Salva lo stato corrente in un file.

        Args:
            checkpoint_path: Se None, usa self.checkpoint_path.
        """
        path = checkpoint_path or self.checkpoint_path
        if path is None:
            raise ValueError("checkpoint_path non specificato né in self né come argomento.")

        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        state = {
            "epoch": self.epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        # Salvataggio atomico (scrive su .tmp poi rinomina)
        tmp_path = path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        logger.debug(f"Checkpoint salvato in {path} (epoca {self.epoch})")