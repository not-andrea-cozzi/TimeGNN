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

from DatasetPipeline.Model.ChessConstants import (
    NUM_EVENT_FEATURES,
    NUM_EVENT_ID_CATEGORIES,
    MOVE_VOCAB_SIZE,
    NUM_EDGE_TYPES,
)

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

