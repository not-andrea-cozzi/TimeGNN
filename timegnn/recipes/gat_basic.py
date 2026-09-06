from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.encoding import (
    encode_label_event,
    encode_pad_event,
    encode_pad_sequence,
    length_stratified_split,
    node_time_list,
    scale_time_differences_fast_fixed,
)
from ..data.pyg import CustomDataset, custom_collate_fn, prepare_data_core_timedif, prepare_data_y
from ..models.gat_basic import DualGATModel
from ..models.training import train_epoch, evaluate_epoch
from ..train.early_stopping import EarlyStopping
from .utils import resolve_config, build_sequence_table


@dataclass
class GATBasicConfig:
    """Configuration for the basic GAT recipe."""
    embedding_dims: int = 64
    gat_hidden_dim_event: int = 32
    gat_hidden_dim_embed: int = 128
    gat_hidden_dim_concat: int = 256
    num_heads: int = 4
    num_layers: int = 1
    dropout: float = 0.0
    use_batch_norm: bool = False
    activation: str = "elu"
    batch_size: int = 16
    lr: float = 1e-3
    num_epochs: int = 10
    patience: int = 3
    delta: float = 0.0
    test_size: float = 0.2


def train_gat_basic(
    event: pd.DataFrame,
    case_index: str,
    core_event: str,
    start_time_col: str,
    cat_col_event: List[str],
    num_col_event: List[str],
    seq_cols: List[str],
    cat_col_seq: List[str],
    num_col_seq: List[str],
    config: Optional[GATBasicConfig] = None,
    device: Optional[str] = None,
    **overrides,
):
    """Train the basic GAT model using the legacy notebook workflow.

    Args:
        event: Event log dataframe.
        case_index: Column identifying sequences/cases.
        core_event: Column with event labels.
        start_time_col: Timestamp column.
        cat_col_event: Categorical event-level columns.
        num_col_event: Numerical event-level columns.
        seq_cols: Columns used to build sequence-level features.
        cat_col_seq: Categorical sequence-level columns.
        num_col_seq: Numerical sequence-level columns.
        config: Optional configuration dataclass.
        device: Torch device string.
        **overrides: Config overrides (e.g., num_epochs=5).

    Returns:
        Dict with trained model and training history.
    """
    cfg = resolve_config(config, GATBasicConfig, overrides)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    sequence = build_sequence_table(event, case_index, seq_cols)

    core_encode, y_encode, core_size, output_size, _ = encode_label_event(
        event, core_event, case_index
    )
    event_encode = encode_pad_event(
        event, cat_col_event, num_col_event, case_index, cat_mask=True, num_mask=True, eos=False
    )
    sequence_encode = encode_pad_sequence(sequence, cat_col_seq, num_col_seq)

    scaled_time_diffs = scale_time_differences_fast_fixed(event, sequence, start_time_col, case_index)
    node_times = node_time_list(event, start_time_col, case_index)

    max_num_events = event_encode.shape[1]
    if sequence_encode.size > 0 and sequence_encode.ndim == 2:
        sequence_features_expanded = np.expand_dims(sequence_encode, axis=1)
        sequence_features_expanded = np.repeat(sequence_features_expanded, max_num_events, axis=1)
        combined_features = np.concatenate((event_encode, sequence_features_expanded), axis=2)
    else:
        combined_features = event_encode

    event_feature_list = prepare_data_core_timedif(
        combined_features, core_encode, scaled_time_diffs, node_times
    )
    y_list = prepare_data_y(combined_features, y_encode)

    train_indices, test_indices = length_stratified_split(
        event_feature_list, test_size=cfg.test_size, n_bins=10
    )

    train_event_features = [event_feature_list[i] for i in train_indices]
    test_event_features = [event_feature_list[i] for i in test_indices]
    train_y = [y_list[i] for i in train_indices]
    test_y = [y_list[i] for i in test_indices]

    train_dataset = CustomDataset(train_event_features, train_y)
    test_dataset = CustomDataset(test_event_features, test_y)

    num_event_features = train_event_features[0].x.shape[1]
    num_embedding_features = core_size

    model = DualGATModel(
        num_event_features=num_event_features,
        num_embedding_features=num_embedding_features,
        embedding_dims=cfg.embedding_dims,
        gat_hidden_dim_event=cfg.gat_hidden_dim_event,
        gat_hidden_dim_embed=cfg.gat_hidden_dim_embed,
        gat_hidden_dim_concat=cfg.gat_hidden_dim_concat,
        output_dim=output_size,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        use_batch_norm=cfg.use_batch_norm,
        activation=cfg.activation,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    early_stopping = EarlyStopping(patience=cfg.patience, delta=cfg.delta)
    history = []
    for epoch in range(cfg.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate_epoch(model, test_loader, criterion, device)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        if early_stopping(test_loss):
            break

    return {"model": model, "history": history}
