"""Early stopping utility for training loops."""
from __future__ import annotations


class EarlyStopping:
    """Early stopping helper based on validation loss.

    Args:
        patience: Number of epochs to wait for improvement.
        delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, delta: float = 0.0) -> None:
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_loss_updated = False

    def __call__(self, val_loss: float) -> bool:
        """Check whether training should stop.

        Returns:
            True if training should stop.
        """
        self.best_loss_updated = False
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_loss_updated = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
