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
    """Media dei logit per-nodo entro ciascun grafo del batch. Stessa
    funzione di DatasetPipeline/Utils/position_pooling.py, duplicata qui
    per non introdurre una dipendenza da quel modulo (path diverso nel
    progetto training vs dataset-building)."""
    return global_mean_pool(node_logits, batch_index)


def _masked_for_accuracy(graph_logits: torch.Tensor, legal_move_mask: torch.Tensor) -> torch.Tensor:
    """Masking full-size in spazio-vocabolario-originale, SOLO per
    argmax/topk (accuracy), mai per la loss (vedi sparse_legal_cross_entropy
    sopra). Calcolato una volta per batch e riusato da top-1 e top-3."""
    return graph_logits.masked_fill(~legal_move_mask, float("-inf"))


def _topk_correct(masked_logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    topk = masked_logits.topk(k, dim=1).indices
    return topk.eq(targets.view(-1, 1)).any(dim=1).sum().item()


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
) -> Tuple[float, float]:
    """
    NOTA: `criterion` e' mantenuto nella firma per compatibilita' con i
    chiamanti esistenti (TrainMain.py costruisce nn.CrossEntropyLoss()
    fuori dal loop e lo passa qui), ma NON viene piu' usato per la loss
    di training: sparse_legal_cross_entropy la sostituisce internamente.
    Se in futuro criterion smette di essere costruito a monte, questo
    parametro puo' diventare Optional senza rompere nulla qui dentro.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    skipped_batches = 0
    amp_enabled = use_amp and device.startswith("cuda")

    desc = epoch_label or "Training"
    with stage_bar(desc, total=total_items, unit="pos") as pbar:
        stats = LiveStats(pbar, refresh_every=10)

        for batch_event, labels in loader:
            batch_event = batch_event.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda" if amp_enabled else "cpu", enabled=amp_enabled):
                node_logits = model(batch_event)
                graph_logits = pool_node_logits(node_logits, batch_event.batch)
                loss = sparse_legal_cross_entropy(graph_logits, batch_event.legal_move_mask, labels)

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

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            with torch.no_grad():
                masked_logits = _masked_for_accuracy(graph_logits, batch_event.legal_move_mask)
                pred = masked_logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()

            total += batch_size

            pbar.update(batch_size)
            stats.update(loss=total_loss / total, acc=correct / total)

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

    avg_loss = total_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
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
) -> Tuple[float, float, float]:
    """Valutazione: loss, top-1 accuracy, top-3 accuracy (per-grafo).

    total_items: vedi docstring di train_epoch (stesso motivo: len(loader)
    e' inaffidabile con num_workers>0 su un IterableDataset shardato).

    NOTA su `criterion`: stessa considerazione di train_epoch, non piu'
    usato per calcolare la loss (sparse_legal_cross_entropy la sostituisce),
    mantenuto in firma per compatibilita' con i chiamanti esistenti.
    """
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    amp_enabled = use_amp and device.startswith("cuda")

    desc = epoch_label or "Validazione"
    with stage_bar(desc, total=total_items, unit="pos") as pbar:
        stats = LiveStats(pbar, refresh_every=10)

        for batch_event, labels in loader:
            batch_event = batch_event.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda" if amp_enabled else "cpu", enabled=amp_enabled):
                node_logits = model(batch_event)
                graph_logits = pool_node_logits(node_logits, batch_event.batch)
                loss = sparse_legal_cross_entropy(graph_logits, batch_event.legal_move_mask, labels)

            masked_logits = _masked_for_accuracy(graph_logits, batch_event.legal_move_mask)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_top1 += masked_logits.argmax(dim=1).eq(labels).sum().item()
            correct_top3 += _topk_correct(masked_logits, labels, k=3)
            total += batch_size

            pbar.update(batch_size)
            stats.update(loss=total_loss / total, top1=correct_top1 / total, top3=correct_top3 / total)

        stats.force_refresh()

    avg_loss = total_loss / total if total else 0.0
    top1 = correct_top1 / total if total else 0.0
    top3 = correct_top3 / total if total else 0.0
    return avg_loss, top1, top3