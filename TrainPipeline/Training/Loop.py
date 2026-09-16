from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from Common.progress import LiveStats, stage_bar
from Common.sparse_legal_moves import sparse_legal_cross_entropy
from TrainPipeline.Training.State import TrainState

logger = logging.getLogger("train_loop")


def pool_node_logits(node_logits: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    return global_mean_pool(node_logits, batch_index)


def _masked_for_accuracy(graph_logits: torch.Tensor, legal_move_mask: torch.Tensor) -> torch.Tensor:
    return graph_logits.masked_fill(~legal_move_mask, float("-inf"))


def _topk_correct_tensor(masked_logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    topk = masked_logits.topk(k, dim=1).indices
    return topk.eq(targets.view(-1, 1)).any(dim=1).sum()


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: Optional[torch.amp.GradScaler] = None,
    train_state: Optional[TrainState] = None,
    checkpoint_every: Optional[int] = None,
    use_amp: bool = True,
    max_grad_norm: Optional[float] = 5.0,
    total_items: Optional[int] = None,
    epoch_label: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
    warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[float, float]:
   
    model.train()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0
    skipped_batches = 0
    amp_enabled = use_amp and device.startswith("cuda")

    desc = epoch_label or "Training"
    with stage_bar(desc, total=total_items, unit="pos") as pbar:
        stats = LiveStats(pbar, refresh_every=10)

        for batch_event, labels in loader:
            batch_event = batch_event.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda" if amp_enabled else "cpu", enabled=amp_enabled):
                node_logits = model(batch_event)
                graph_logits = pool_node_logits(node_logits, batch_event.batch)
                loss = sparse_legal_cross_entropy(
                    graph_logits, batch_event.legal_move_mask, labels,
                    class_weights=class_weights,
                )

            if not torch.isfinite(loss):
                skipped_batches += 1
                logger.warning(f"Loss non finita ({loss.item()}) al batch: step saltato.")
                pbar.update(labels.size(0))
                continue

            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            if warmup_scheduler is not None:
                warmup_scheduler.step()

            batch_size = labels.size(0)
            total_loss += loss.detach() * batch_size

            with torch.no_grad():
                masked_logits = _masked_for_accuracy(graph_logits, batch_event.legal_move_mask)
                pred = masked_logits.argmax(dim=1)
                correct += pred.eq(labels).sum()

            total += batch_size

            pbar.update(batch_size)
            stats.update(
                loss=(total_loss / total).item(),
                acc=(correct / total).item(),
            )

            if train_state is not None:
                train_state.global_step += 1
                if checkpoint_every and train_state.global_step % checkpoint_every == 0:
                    train_state.save(model, optimizer, scaler)
                    stats.force_refresh()
                    logger.info(
                        f"Checkpoint automatico a step {train_state.global_step} "
                        f"(loss corrente batch={loss.item():.4f})."
                    )

        stats.force_refresh()

    if skipped_batches:
        logger.warning(f"Epoca completata con {skipped_batches} batch saltati per loss non finita.")

    avg_loss = (total_loss / total).item() if total else 0.0
    accuracy = (correct / total).item() if total else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = True,
    total_items: Optional[int] = None,
    epoch_label: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    
    model.eval()
    total_loss = torch.zeros((), device=device)
    correct_top1 = torch.zeros((), device=device)
    correct_top3 = torch.zeros((), device=device)
    total = 0
    amp_enabled = use_amp and device.startswith("cuda")

    desc = epoch_label or "Validazione"
    with stage_bar(desc, total=total_items, unit="pos") as pbar:
        stats = LiveStats(pbar, refresh_every=10)

        for batch_event, labels in loader:
            batch_event = batch_event.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda" if amp_enabled else "cpu", enabled=amp_enabled):
                node_logits = model(batch_event)
                graph_logits = pool_node_logits(node_logits, batch_event.batch)
                loss = sparse_legal_cross_entropy(
                    graph_logits, batch_event.legal_move_mask, labels,
                    class_weights=class_weights,
                )

            masked_logits = _masked_for_accuracy(graph_logits, batch_event.legal_move_mask)

            batch_size = labels.size(0)
            total_loss += loss.detach() * batch_size
            correct_top1 += masked_logits.argmax(dim=1).eq(labels).sum()
            correct_top3 += _topk_correct_tensor(masked_logits, labels, k=3)
            total += batch_size

            pbar.update(batch_size)
            stats.update(
                loss=(total_loss / total).item(),
                top1=(correct_top1 / total).item(),
                top3=(correct_top3 / total).item(),
            )

        stats.force_refresh()

    avg_loss = (total_loss / total).item() if total else 0.0
    top1 = (correct_top1 / total).item() if total else 0.0
    top3 = (correct_top3 / total).item() if total else 0.0
    return avg_loss, top1, top3