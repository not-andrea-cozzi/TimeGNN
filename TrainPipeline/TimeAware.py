from __future__ import annotations

import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DatasetPipeline.Model.ChessConstants import (
    NUM_EVENT_FEATURES,
    NUM_EVENT_ID_CATEGORIES,
    MOVE_VOCAB_SIZE,
    TIME_EDGE_DIM,
)
from TrainPipeline.Shard.ShardDataset import ShardedGraphDataset
from timegnn.data.pyg import custom_collate_graph
from timegnn.models.gat_time_decay import DualGATTimeAwareModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_time_aware")


def build_model(args: argparse.Namespace, device: str) -> DualGATTimeAwareModel:
    model = DualGATTimeAwareModel(
        num_event_features=NUM_EVENT_FEATURES,
        num_embedding_features=NUM_EVENT_ID_CATEGORIES,
        embedding_dims=args.embedding_dims,
        gat_hidden_dim_event=args.gat_hidden_dim_event,
        gat_hidden_dim_embed=args.gat_hidden_dim_embed,
        gat_hidden_dim_concat=args.gat_hidden_dim_concat,
        output_dim=MOVE_VOCAB_SIZE,
        num_heads=args.num_heads,
        lambda_decay=args.lambda_decay,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        activation=args.activation,
    ).to(device)
    return model


def build_dataloaders(args: argparse.Namespace):
    train_dir = os.path.join(args.shards_dir, "train")
    val_dir = os.path.join(args.shards_dir, "val")

    train_ds = ShardedGraphDataset(train_dir, shuffle=True, seed=args.seed)
    val_ds = ShardedGraphDataset(val_dir, shuffle=False, seed=args.seed)
    use_cuda = torch.cuda.is_available() and getattr(args, "device", "cpu") != "cpu"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,  # gestito dal dataset (IterableDataset)
        collate_fn=custom_collate_graph,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=use_cuda,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_graph,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=use_cuda,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    return train_loader, train_ds, val_loader, val_ds