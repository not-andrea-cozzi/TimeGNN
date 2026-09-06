from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from ..data.encoding import encode_event_prefix_label, encode_pad_sequence, scale_time_differences_fast_fixed
from ..data.pyg import prepare_data_prefix, PrefixDataset, custom_collate_prefix
from ..models.prefix_gcn import PrefixGCNClassifier, evaluate_epoch, train_epoch
from ..train.early_stopping import EarlyStopping
from .utils import resolve_config


def _filter_singletons(labels: torch.Tensor):
    """Return mask filtering classes with only one sample."""
    unique, counts = torch.unique(labels, return_counts=True)
    singletons = unique[counts == 1]
    mask = ~torch.isin(labels, singletons)
    return mask


@dataclass
class PrefixGCNConfig:
    """Configuration for the prefix GCN recipe."""
    prefix_size: int = 10
    gcn_hidden_dims: int = 64
    embedding_dims: int = 64
    gcn_hidden_dims_embedding: int = 64
    gcn_hidden_dims_concat: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    use_batch_norm: bool = False
    activation: str = "relu"
    fc_hidden_dims: int = 64
    fc_hidden_dims_concat: int = 128
    batch_size: int = 32
    lr: float = 1e-3
    num_epochs: int = 10
    patience: int = 3
    delta: float = 0.0
    test_size: float = 0.2
    stratify: bool = True
    filter_singletons: bool = True


def train_prefix_gcn(
    event: pd.DataFrame,
    case_index: str,
    core_event: str,
    start_time_col: str,
    cat_col_event: List[str],
    num_col_event: List[str],
    cat_col_seq: List[str],
    num_col_seq: List[str],
    config: Optional[PrefixGCNConfig] = None,
    device: Optional[str] = None,
    **overrides,
):
    """Train the prefix-based GCN model.

    Args:
        event: Event log dataframe.
        case_index: Column identifying sequences/cases.
        core_event: Column with event labels.
        start_time_col: Timestamp column.
        cat_col_event: Categorical event-level columns.
        num_col_event: Numerical event-level columns.
        cat_col_seq: Categorical sequence-level columns.
        num_col_seq: Numerical sequence-level columns.
        config: Optional configuration dataclass.
        device: Torch device string.
        **overrides: Config overrides (e.g., num_epochs=5).

    Returns:
        Dict with trained model and training history.
    """
    cfg = resolve_config(config, PrefixGCNConfig, overrides)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    event = event[event.groupby(case_index)[case_index].transform("size") >= cfg.prefix_size].reset_index(drop=True)

    text_encode, event_encode, y_encode, text_size, output_dim = encode_event_prefix_label(
        event,
        core_event,
        cat_col_event,
        num_col_event,
        case_index,
        cfg.prefix_size,
        cat_mask=False,
        num_mask=False,
    )

    sequence = pd.concat(
        [g.iloc[cfg.prefix_size - 1 :] for _, g in event.groupby(case_index, sort=False)],
        ignore_index=True,
    )
    sequence_encode = encode_pad_sequence(sequence, cat_col_seq, num_col_seq)

    scaled_time_diffs = scale_time_differences_fast_fixed(event, sequence, start_time_col, case_index)

    event_feature_list = prepare_data_prefix(event_encode, text_encode, scaled_time_diffs)
    sequence_features = torch.tensor(sequence_encode, dtype=torch.float)
    y_encode_tensor = torch.tensor(y_encode, dtype=torch.long)

    if cfg.filter_singletons:
        mask = _filter_singletons(y_encode_tensor)
        filtered_indices = torch.where(mask)[0].cpu().numpy()
        event_feature_list = [event_feature_list[i] for i in filtered_indices]
        sequence_features = sequence_features[filtered_indices]
        y_encode_tensor = y_encode_tensor[filtered_indices]

    indices = np.arange(len(event_feature_list))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=cfg.test_size,
        stratify=y_encode_tensor.numpy() if cfg.stratify else None,
        random_state=42,
    )

    train_event_features = [event_feature_list[i] for i in train_indices]
    test_event_features = [event_feature_list[i] for i in test_indices]
    train_sequence_features = sequence_features[train_indices]
    test_sequence_features = sequence_features[test_indices]
    train_y = y_encode_tensor[train_indices]
    test_y = y_encode_tensor[test_indices]

    train_dataset = PrefixDataset(train_event_features, train_sequence_features, train_y)
    test_dataset = PrefixDataset(test_event_features, test_sequence_features, test_y)

    num_event_features = event_encode.shape[2]
    num_sequence_features = sequence_encode.shape[1]
    num_embedding_features = output_dim

    model = PrefixGCNClassifier(
        num_event_features=num_event_features,
        gcn_hidden_dims=cfg.gcn_hidden_dims,
        num_embedding_features=num_embedding_features,
        embedding_dims=cfg.embedding_dims,
        gcn_hidden_dims_embedding=cfg.gcn_hidden_dims_embedding,
        gcn_hidden_dims_concat=cfg.gcn_hidden_dims_concat,
        num_sequence_features=num_sequence_features,
        fc_hidden_dims=cfg.fc_hidden_dims,
        fc_hidden_dims_concat=cfg.fc_hidden_dims_concat,
        output_dim=output_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        use_batch_norm=cfg.use_batch_norm,
        activation=cfg.activation,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=custom_collate_prefix
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=custom_collate_prefix
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
