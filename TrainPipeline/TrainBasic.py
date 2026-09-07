from __future__ import annotations

import argparse
import logging
import os
import time

import torch
import torch.nn as nn

from shard_dataset import ShardedGraphDataset
from timegnn.data.pyg import custom_collate_graph
from timegnn.models.gat_basic import DualGATModel
from timegnn.train.early_stopping import EarlyStopping
from TrainPipeline.Training.Loop import evaluate_epoch, train_epoch
from TrainPipeline.Training.State import TrainState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_basic")

# Vocabolario fisso dello schema scacchistico (PositionGraphSchema.py):
# NON un iperparametro, deriva dalla codifica board 64x64.
NUM_EVENT_ID_CATEGORIES = 13   # 0=vuota, 1..12=piece_type*2+color+1
NUM_EVENT_FEATURES = 2         # is_occupied_by_mover, is_occupied_by_opponent
MOVE_VOCAB_SIZE = 64 * 64      # 4096: output_dim del modello
NUM_EDGE_TYPES = 3             # edge_attr = one-hot(EDGE_LEGAL_MOVE/ATTACK/PIN)


def build_model(args: argparse.Namespace, device: str) -> DualGATModel:
    model = DualGATModel(
        num_event_features=NUM_EVENT_FEATURES,
        num_embedding_features=NUM_EVENT_ID_CATEGORIES,
        embedding_dims=args.embedding_dims,
        gat_hidden_dim_event=args.gat_hidden_dim_event,
        gat_hidden_dim_embed=args.gat_hidden_dim_embed,
        gat_hidden_dim_concat=args.gat_hidden_dim_concat,
        output_dim=MOVE_VOCAB_SIZE,
        num_heads=args.num_heads,
        edge_dim=NUM_EDGE_TYPES,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        activation=args.activation,
    ).to(device)
    return model


def build_dataloaders(args: argparse.Namespace):
    from torch.utils.data import DataLoader

    train_dir = os.path.join(args.shards_dir, "train")
    val_dir = os.path.join(args.shards_dir, "val")

    train_ds = ShardedGraphDataset(train_dir, shuffle=True, seed=args.seed)
    val_ds = ShardedGraphDataset(val_dir, shuffle=False, seed=args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,  # gestito dal dataset (IterableDataset)
        collate_fn=custom_collate_graph,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_graph,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, train_ds, val_loader, val_ds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training DualGATModel (basic) su dataset scacchistico.")
    p.add_argument("--shards-dir", default="Dataset/Train/shards")
    p.add_argument("--checkpoint", default="Dataset/Train/checkpoints/basic.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=500, help="Salva checkpoint ogni N step (0=disabilita)")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--no-amp", action="store_true", help="Disabilita mixed precision")

    # Iperparametri modello (default coerenti con GATBasicConfig)
    p.add_argument("--embedding-dims", type=int, default=64)
    p.add_argument("--gat-hidden-dim-event", type=int, default=32)
    p.add_argument("--gat-hidden-dim-embed", type=int, default=128)
    p.add_argument("--gat-hidden-dim-concat", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use-batch-norm", action="store_true")
    p.add_argument("--activation", default="elu")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (not args.no_amp) and device == "cuda"

    logger.info("=" * 60)
    logger.info(f"Training DualGATModel (basic) | device={device} | AMP={use_amp}")
    logger.info("=" * 60)

    torch.manual_seed(args.seed)

    train_loader, train_ds, val_loader, val_ds = build_dataloaders(args)
    logger.info(f"Train: {len(train_ds):,} posizioni | Val: {len(val_ds):,} posizioni.")

    model = build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # API non deprecata: torch.amp.GradScaler(device, ...) al posto di
    # torch.cuda.amp.GradScaler(...) (deprecata dalle versioni recenti di
    # PyTorch, emette FutureWarning ad ogni istanziazione).
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    state = TrainState(checkpoint_path=args.checkpoint)
    state.try_resume(model, optimizer, scaler, map_location=device)

    early_stopping = EarlyStopping(patience=args.patience)

    for epoch in range(state.epoch, args.epochs):
        train_ds.set_epoch(epoch)
        t0 = time.monotonic()

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            train_state=state,
            checkpoint_every=args.checkpoint_every or None,
            use_amp=use_amp,
            total_items=len(train_ds),
            epoch_label=f"Epoch {epoch + 1}/{args.epochs} [train]",
        )
        val_loss, val_top1, val_top3 = evaluate_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
            total_items=len(val_ds),
            epoch_label=f"Epoch {epoch + 1}/{args.epochs} [val]",
        )

        elapsed = time.monotonic() - t0
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_top1={val_top1:.4f} val_top3={val_top3:.4f}"
        )

        state.epoch = epoch + 1
        state.history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top3": val_top3,
            }
        )
        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
        state.save(model, optimizer, scaler)

        if early_stopping(val_loss):
            logger.info(f"Early stopping a epoch {epoch + 1} (patience={args.patience}).")
            break

    logger.info("Training completato.")


if __name__ == "__main__":
    main()