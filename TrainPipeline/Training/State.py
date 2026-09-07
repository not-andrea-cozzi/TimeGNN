"""
train_state.py

Gestione checkpoint/resume per training lunghi, stesso principio di
DatasetPipeline/PipelineState.py (scrittura atomica via file .tmp +
os.replace, mai un file a metà scritto che possa corrompere il resume).

Salva: model.state_dict, optimizer.state_dict, scaler.state_dict (AMP),
epoch corrente, global_step, best_val_loss, history. Resume automatico se
il checkpoint esiste: riprende da global_step successivo, non ricomincia
l'epoca da capo (importante con dataset da 300k Data / epoche lunghe).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("train_state")


@dataclass
class TrainState:
    checkpoint_path: str
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    history: List[Dict[str, Any]] = field(default_factory=list)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> None:
        """Scrittura atomica: mai un checkpoint parziale su disco."""
        os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_path)) or ".", exist_ok=True)
        payload = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
        }
        tmp_path = self.checkpoint_path + ".tmp"
        t0 = time.monotonic()
        torch.save(payload, tmp_path)
        os.replace(tmp_path, self.checkpoint_path)
        logger.debug(
            f"Checkpoint salvato in {time.monotonic() - t0:.2f}s "
            f"(epoch={self.epoch}, step={self.global_step}) -> '{self.checkpoint_path}'."
        )

    def try_resume(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None,
        map_location: str = "cpu",
    ) -> bool:
        """Se checkpoint_path esiste ed e' leggibile, ripristina stato IN
        PLACE su model/optimizer/scaler e aggiorna epoch/global_step/
        best_val_loss/history. Ritorna True se il resume e' avvenuto.

        Un checkpoint illeggibile (crash durante scrittura di una versione
        precedente, prima della patch atomica, o disco corrotto) NON blocca
        la pipeline: si logga un warning e si riparte da zero, stesso
        comportamento difensivo di PipelineState.load().
        """
        if not os.path.exists(self.checkpoint_path):
            logger.info(f"Nessun checkpoint trovato in '{self.checkpoint_path}': training da zero.")
            return False

        try:
            payload = torch.load(self.checkpoint_path, map_location=map_location, weights_only=False)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            if scaler is not None and payload.get("scaler_state") is not None:
                scaler.load_state_dict(payload["scaler_state"])

            self.epoch = payload["epoch"]
            self.global_step = payload["global_step"]
            self.best_val_loss = payload["best_val_loss"]
            self.history = payload.get("history", [])

            logger.info(
                f"Resume da '{self.checkpoint_path}': epoch={self.epoch}, "
                f"global_step={self.global_step}, best_val_loss={self.best_val_loss:.4f}."
            )
            return True
        except Exception as e:
            logger.warning(
                f"Checkpoint in '{self.checkpoint_path}' illeggibile ({type(e).__name__}: {e}): "
                f"training ripartira' da zero."
            )
            return False