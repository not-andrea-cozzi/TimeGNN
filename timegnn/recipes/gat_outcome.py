from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool

from ..data.encoding import length_stratified_split
from ..data.transformer import EventLogTransformer
from ..data.pyg import CustomDataset, custom_collate_graph
from ..models.gat_basic import DualGATModel
from ..models.gat_status_emb import DualGAT2EdgesModel
from ..models.gat_time_decay import DualGATTimeAwareModel
from ..models.gat_time_decay_status_emb import DualGATTimeAwareETModel
from ..train.early_stopping import EarlyStopping
from .utils import resolve_config


@dataclass
class GATOutcomeConfig:
    """Configuration for outcome (case-level) GAT training."""
    embedding_dims: int = 64
    gat_hidden_dim_event: int = 32
    gat_hidden_dim_embed: int = 128
    gat_hidden_dim_concat: int = 256
    num_heads: int = 4
    num_layers: int = 1
    dropout: float = 0.0
    use_batch_norm: bool = False
    activation: str = "elu"
    edge_type_dim: int = 32
    lambda_decay: float = 0.01
    batch_size: int = 16
    lr: float = 1e-3
    num_epochs: int = 10
    patience: int = 3
    delta: float = 0.0
    test_size: float = 0.2


def _prepare_outcome_labels(event: pd.DataFrame, case_index: str, outcome_col: str):
    if outcome_col not in event.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in dataframe.")

    case_labels = event.groupby(case_index)[outcome_col].last()
    case_labels = case_labels.dropna()
    event = event[event[case_index].isin(case_labels.index)].copy()

    case_order = event.groupby(case_index)[case_index].first().index
    ordered_labels = case_labels.loc[case_order]

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(ordered_labels.astype(str))

    return event, labels.tolist(), label_encoder


def _build_outcome_model(
    *,
    mode: str,
    num_event_features: int,
    num_embedding_features: int,
    output_dim: int,
    trans_size: Optional[int],
    cfg: GATOutcomeConfig,
):
    if mode == "gat_basic":
        return DualGATModel(
            num_event_features=num_event_features,
            num_embedding_features=num_embedding_features,
            embedding_dims=cfg.embedding_dims,
            gat_hidden_dim_event=cfg.gat_hidden_dim_event,
            gat_hidden_dim_embed=cfg.gat_hidden_dim_embed,
            gat_hidden_dim_concat=cfg.gat_hidden_dim_concat,
            output_dim=output_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_batch_norm=cfg.use_batch_norm,
            activation=cfg.activation,
        )
    if mode == "gat_time_decay":
        return DualGATTimeAwareModel(
            num_event_features=num_event_features,
            num_embedding_features=num_embedding_features,
            embedding_dims=cfg.embedding_dims,
            gat_hidden_dim_event=cfg.gat_hidden_dim_event,
            gat_hidden_dim_embed=cfg.gat_hidden_dim_embed,
            gat_hidden_dim_concat=cfg.gat_hidden_dim_concat,
            output_dim=output_dim,
            num_heads=cfg.num_heads,
            lambda_decay=cfg.lambda_decay,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_batch_norm=cfg.use_batch_norm,
            activation=cfg.activation,
        )
    if mode == "gat_status":
        if trans_size is None:
            raise ValueError("trans_size is required for gat_status outcome model.")
        return DualGAT2EdgesModel(
            num_event_features=num_event_features,
            num_embedding_features=num_embedding_features,
            embedding_dims=cfg.embedding_dims,
            gat_hidden_dim_event=cfg.gat_hidden_dim_event,
            gat_hidden_dim_embed=cfg.gat_hidden_dim_embed,
            gat_hidden_dim_concat=cfg.gat_hidden_dim_concat,
            output_dim=output_dim,
            num_heads=cfg.num_heads,
            num_edge_types=trans_size,
            edge_type_dim=cfg.edge_type_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_batch_norm=cfg.use_batch_norm,
            activation=cfg.activation,
        )
    if mode == "gat_time_decay_status":
        if trans_size is None:
            raise ValueError("trans_size is required for gat_time_decay_status outcome model.")
        return DualGATTimeAwareETModel(
            num_event_features=num_event_features,
            num_embedding_features=num_embedding_features,
            embedding_dims=cfg.embedding_dims,
            gat_hidden_dim_event=cfg.gat_hidden_dim_event,
            gat_hidden_dim_embed=cfg.gat_hidden_dim_embed,
            gat_hidden_dim_concat=cfg.gat_hidden_dim_concat,
            output_dim=output_dim,
            num_heads=cfg.num_heads,
            num_edge_types=trans_size,
            edge_type_dim=cfg.edge_type_dim,
            lambda_decay=cfg.lambda_decay,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_batch_norm=cfg.use_batch_norm,
            activation=cfg.activation,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def _train_epoch_outcome(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for event_data, labels in loader:
        event_data = event_data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        node_logits = model(event_data)
        graph_logits = global_mean_pool(node_logits, event_data.batch)

        loss = criterion(graph_logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = graph_logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total else 0.0
    loss = total_loss / total if total else 0.0
    return loss, accuracy


def _eval_epoch_outcome(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            labels = labels.to(device)

            node_logits = model(event_data)
            graph_logits = global_mean_pool(node_logits, event_data.batch)

            loss = criterion(graph_logits, labels)
            total_loss += loss.item() * labels.size(0)

            pred = graph_logits.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total else 0.0
    loss = total_loss / total if total else 0.0
    return loss, accuracy


def train_gat_outcome(
    event: pd.DataFrame,
    case_index: str,
    core_event: str,
    start_time_col: str,
    outcome_col: str,
    *,
    status_col: Optional[str] = None,
    cat_col_event: Optional[List[str]] = None,
    num_col_event: Optional[List[str]] = None,
    seq_cols: Optional[List[str]] = None,
    cat_col_seq: Optional[List[str]] = None,
    num_col_seq: Optional[List[str]] = None,
    mode: str = "gat_basic",
    config: Optional[GATOutcomeConfig] = None,
    device: Optional[str] = None,
    **overrides,
):
    """Train a case-level outcome classifier using GAT graph inputs.

    Args:
        event: Event log dataframe.
        case_index: Column identifying sequences/cases.
        core_event: Column with event labels.
        start_time_col: Timestamp column.
        outcome_col: Column with outcome labels (case-level).
        status_col: Column used for transition edge types (status modes).
        cat_col_event: Categorical event-level columns.
        num_col_event: Numerical event-level columns.
        seq_cols: Columns used to build sequence-level features.
        cat_col_seq: Categorical sequence-level columns.
        num_col_seq: Numerical sequence-level columns.
        mode: One of gat_basic, gat_time_decay, gat_status, gat_time_decay_status.
        config: Optional configuration dataclass.
        device: Torch device string.
        **overrides: Config overrides (e.g., num_epochs=5).

    Returns:
        Dict with trained model, history, and label encoder.
    """
    cfg = resolve_config(config, GATOutcomeConfig, overrides)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    cat_col_event = cat_col_event or []
    num_col_event = num_col_event or []
    seq_cols = seq_cols or []
    cat_col_seq = cat_col_seq or []
    num_col_seq = num_col_seq or []

    event, labels, label_encoder = _prepare_outcome_labels(event, case_index, outcome_col)

    transformer = EventLogTransformer(
        case_col=case_index,
        event_col=core_event,
        time_col=start_time_col,
        status_col=status_col,
        cat_event=cat_col_event,
        num_event=num_col_event,
        seq_cols=seq_cols,
        cat_seq=cat_col_seq,
        num_seq=num_col_seq,
        mode=mode,
    ).fit(event)

    transformed = transformer.transform(event)
    event_feature_list = transformed.event_features
    if len(event_feature_list) != len(labels):
        raise RuntimeError("Outcome labels do not align with event features.")

    train_indices, test_indices = length_stratified_split(
        event_feature_list, test_size=cfg.test_size, n_bins=10
    )

    train_event_features = [event_feature_list[i] for i in train_indices]
    test_event_features = [event_feature_list[i] for i in test_indices]
    train_y = [labels[i] for i in train_indices]
    test_y = [labels[i] for i in test_indices]

    train_dataset = CustomDataset(train_event_features, train_y)
    test_dataset = CustomDataset(test_event_features, test_y)

    num_event_features = train_event_features[0].x.shape[1]
    num_embedding_features = transformed.core_size
    output_dim = len(label_encoder.classes_)

    model = _build_outcome_model(
        mode=mode,
        num_event_features=num_event_features,
        num_embedding_features=num_embedding_features,
        output_dim=output_dim,
        trans_size=transformed.trans_size,
        cfg=cfg,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=custom_collate_graph
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=custom_collate_graph
    )

    early_stopping = EarlyStopping(patience=cfg.patience, delta=cfg.delta)
    history = []
    for epoch in range(cfg.num_epochs):
        train_loss, train_acc = _train_epoch_outcome(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = _eval_epoch_outcome(model, test_loader, criterion, device)

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

    return {
        "model": model,
        "history": history,
        "label_encoder": label_encoder,
        "classes": label_encoder.classes_.tolist(),
    }
