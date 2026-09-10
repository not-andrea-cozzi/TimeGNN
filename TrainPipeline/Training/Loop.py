"""
train_loop.py

Train/eval epoch DEDICATI per modelli che producono output per-NODO
(DualGATModel / DualGATTimeAwareModel, mai modificati) usati su un target
per-GRAFO (data.y = mossa scacchistica in [0, 4096)).

Non riusa timegnn/models/training.py (train_epoch/evaluate_epoch): quello
e' scritto per next-event prediction con label per-nodo e ignore_index=-1,
incompatibile con un singolo scalare per board. Qui invece:

    node_logits = model(batch)                      # [N_nodi_batch, 4096]
    graph_logits = pool_node_logits(node_logits, batch.batch)  # [B, 4096]
    loss = CrossEntropyLoss(graph_logits, batch.y)   # [B] target scalari

Supporta AMP (torch.cuda.amp) e checkpoint automatico ogni `checkpoint_every`
step tramite TrainState, con salvataggio ASINCRONO rispetto al training?
No: il salvataggio e' sincrono (torch.save blocca), ma e' comunque un'
operazione rara (ogni N step) e la scrittura atomica (file.tmp + replace)
garantisce che un crash a meta' salvataggio non comprometta il checkpoint
precedente.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from Common.progress import LiveStats, stage_bar
from TrainPipeline.Training.State import TrainState

logger = logging.getLogger("train_loop")


def pool_node_logits(node_logits: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    """Media dei logit per-nodo entro ciascun grafo del batch. Stessa
    funzione di DatasetPipeline/Utils/position_pooling.py, duplicata qui
    per non introdurre una dipendenza da quel modulo (path diverso nel
    progetto training vs dataset-building)."""
    return global_mean_pool(node_logits, batch_index)


def _topk_correct(graph_logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    topk = graph_logits.topk(k, dim=1).indices
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
    """Un'epoca di training con pooling per-grafo e AMP opzionale.

    Se train_state e checkpoint_every sono forniti, salva un checkpoint
    ogni `checkpoint_every` step (non solo a fine epoca): su un'epoca
    lunga (300k Data) un crash a meta' non fa perdere tutto il lavoro.
    NOTA: il resume da un checkpoint step-based riparte comunque dall'
    inizio dell'epoca corrente (l'IterableDataset non supporta skip a
    metà epoca): il salvataggio frequente protegge i PESI del modello da
    un crash, non fa un resume esatto infra-epoca.

    max_grad_norm: se non None, applica gradient clipping (norma L2)
    prima dello step di ottimizzazione. Con GAT + AMP i gradienti possono
    esplodere in fp16; il clipping e' una salvaguardia a basso costo.

    total_items: numero di POSIZIONI (non batch) nell'intero split, per
    dimensionare la progress bar in modo esatto. Va passato esplicitamente
    (es. len(dataset)) e non dedotto da len(loader): con un
    IterableDataset e num_workers>0 ogni worker vede solo una fetta
    disgiunta di shard, quindi len(loader) sovrastimerebbe il totale reale
    (vedi nota in ShardedGraphDataset.__len__). Se None, la barra mostra
    solo un contatore progressivo senza percentuale/ETA.
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
                loss = criterion(graph_logits, labels)

            if not torch.isfinite(loss):
                # Batch corrotto o istabilita' numerica: si salta lo step
                # invece di propagare NaN nei pesi (che un checkpoint
                # successivo salverebbe irreversibilmente).
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
            pred = graph_logits.argmax(dim=1)
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
                loss = criterion(graph_logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_top1 += graph_logits.argmax(dim=1).eq(labels).sum().item()
            correct_top3 += _topk_correct(graph_logits, labels, k=3)
            total += batch_size

            pbar.update(batch_size)
            stats.update(loss=total_loss / total, top1=correct_top1 / total, top3=correct_top3 / total)

        stats.force_refresh()

    avg_loss = total_loss / total if total else 0.0
    top1 = correct_top1 / total if total else 0.0
    top3 = correct_top3 / total if total else 0.0
    return avg_loss, top1, top3