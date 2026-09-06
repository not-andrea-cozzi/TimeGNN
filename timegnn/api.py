from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .recipes import (
    GATBasicConfig,
    GATOutcomeConfig,
    GATStatusEmbConfig,
    GATTimeDecayConfig,
    GATTimeDecayStatusConfig,
    PrefixGCNConfig,
    train_gat_outcome,
    train_gat_basic,
    train_gat_status_emb,
    train_gat_time_decay,
    train_gat_time_decay_status_emb,
    train_prefix_gcn,
)


def gat_basic(
    event: pd.DataFrame,
    *,
    case_col: str,
    event_col: str,
    time_col: str,
    cat_event: Optional[List[str]] = None,
    num_event: Optional[List[str]] = None,
    seq_cols: Optional[List[str]] = None,
    cat_seq: Optional[List[str]] = None,
    num_seq: Optional[List[str]] = None,
    config: Optional[GATBasicConfig] = None,
    device: Optional[str] = None,
    **overrides,
):
    """Train the basic GAT model with a simple, user-friendly interface.

    Args:
        event: Event log dataframe.
        case_col: Column identifying sequences/cases.
        event_col: Column with event labels.
        time_col: Column with timestamps.
        cat_event: Categorical event-level columns.
        num_event: Numerical event-level columns.
        seq_cols: Columns used to build sequence-level features.
        cat_seq: Categorical sequence-level columns.
        num_seq: Numerical sequence-level columns.
        config: Optional config dataclass.
        device: Torch device string.
        **overrides: Simple overrides for config fields (e.g., num_layers,
            dropout, use_batch_norm, activation, patience, delta).

    Returns:
        Dict with trained model and training history.
    """
    return train_gat_basic(
        event,
        case_index=case_col,
        core_event=event_col,
        start_time_col=time_col,
        cat_col_event=cat_event or [],
        num_col_event=num_event or [],
        seq_cols=seq_cols or [],
        cat_col_seq=cat_seq or [],
        num_col_seq=num_seq or [],
        config=config,
        device=device,
        **overrides,
    )


def gat_status(
    event: pd.DataFrame,
    *,
    case_col: str,
    event_col: str,
    time_col: str,
    status_col: str,
    cat_event: Optional[List[str]] = None,
    num_event: Optional[List[str]] = None,
    seq_cols: Optional[List[str]] = None,
    cat_seq: Optional[List[str]] = None,
    num_seq: Optional[List[str]] = None,
    config: Optional[GATStatusEmbConfig] = None,
    device: Optional[str] = None,
    **overrides,
):
    """Train the GAT model with edge-type/status embeddings.

    Args:
        event: Event log dataframe.
        case_col: Column identifying sequences/cases.
        event_col: Column with event labels.
        time_col: Column with timestamps.
        status_col: Column used to build transition edge types.
        cat_event: Categorical event-level columns.
        num_event: Numerical event-level columns.
        seq_cols: Columns used to build sequence-level features.
        cat_seq: Categorical sequence-level columns.
        num_seq: Numerical sequence-level columns.
        config: Optional config dataclass.
        device: Torch device string.
        **overrides: Simple overrides for config fields (e.g., num_layers,
            dropout, use_batch_norm, activation, edge_type_dim,
            patience, delta).

    Returns:
        Dict with trained model and training history.
    """
    return train_gat_status_emb(
        event,
        case_index=case_col,
        core_event=event_col,
        start_time_col=time_col,
        status_col=status_col,
        cat_col_event=cat_event or [],
        num_col_event=num_event or [],
        seq_cols=seq_cols or [],
        cat_col_seq=cat_seq or [],
        num_col_seq=num_seq or [],
        config=config,
        device=device,
        **overrides,
    )


def gat_time_decay(
    event: pd.DataFrame,
    *,
    case_col: str,
    event_col: str,
    time_col: str,
    cat_event: Optional[List[str]] = None,
    num_event: Optional[List[str]] = None,
    seq_cols: Optional[List[str]] = None,
    cat_seq: Optional[List[str]] = None,
    num_seq: Optional[List[str]] = None,
    config: Optional[GATTimeDecayConfig] = None,
    device: Optional[str] = None,
    **overrides,
):
    """Train the time-decay GAT model.

    Args:
        event: Event log dataframe.
        case_col: Column identifying sequences/cases.
        event_col: Column with event labels.
        time_col: Column with timestamps.
        cat_event: Categorical event-level columns.
        num_event: Numerical event-level columns.
        seq_cols: Columns used to build sequence-level features.
        cat_seq: Categorical sequence-level columns.
        num_seq: Numerical sequence-level columns.
        config: Optional config dataclass.
        device: Torch device string.
        **overrides: Simple overrides for config fields (e.g., num_layers,
            dropout, use_batch_norm, activation, lambda_decay,
            patience, delta).

    Returns:
        Dict with trained model, history, and attention.
    """
    return train_gat_time_decay(
        event,
        case_index=case_col,
        core_event=event_col,
        start_time_col=time_col,
        cat_col_event=cat_event or [],
        num_col_event=num_event or [],
        seq_cols=seq_cols or [],
        cat_col_seq=cat_seq or [],
        num_col_seq=num_seq or [],
        config=config,
        device=device,
        **overrides,
    )


def gat_time_decay_status(
    event: pd.DataFrame,
    *,
    case_col: str,
    event_col: str,
    time_col: str,
    status_col: str,
    cat_event: Optional[List[str]] = None,
    num_event: Optional[List[str]] = None,
    seq_cols: Optional[List[str]] = None,
    cat_seq: Optional[List[str]] = None,
    num_seq: Optional[List[str]] = None,
    config: Optional[GATTimeDecayStatusConfig] = None,
    device: Optional[str] = None,
    **overrides,
):
    """Train the time-decay GAT model with edge-type embeddings.

    Args:
        event: Event log dataframe.
        case_col: Column identifying sequences/cases.
        event_col: Column with event labels.
        time_col: Column with timestamps.
        status_col: Column used to build transition edge types.
        cat_event: Categorical event-level columns.
        num_event: Numerical event-level columns.
        seq_cols: Columns used to build sequence-level features.
        cat_seq: Categorical sequence-level columns.
        num_seq: Numerical sequence-level columns.
        config: Optional config dataclass.
        device: Torch device string.
        **overrides: Simple overrides for config fields (e.g., num_layers,
            dropout, use_batch_norm, activation, edge_type_dim,
            lambda_decay, patience, delta).

    Returns:
        Dict with trained model, history, and attention.
    """
    return train_gat_time_decay_status_emb(
        event,
        case_index=case_col,
        core_event=event_col,
        start_time_col=time_col,
        status_col=status_col,
        cat_col_event=cat_event or [],
        num_col_event=num_event or [],
        seq_cols=seq_cols or [],
        cat_col_seq=cat_seq or [],
        num_col_seq=num_seq or [],
        config=config,
        device=device,
        **overrides,
    )


def prefix_gcn(
    event: pd.DataFrame,
    *,
    case_col: str,
    event_col: str,
    time_col: str,
    cat_event: Optional[List[str]] = None,
    num_event: Optional[List[str]] = None,
    cat_seq: Optional[List[str]] = None,
    num_seq: Optional[List[str]] = None,
    config: Optional[PrefixGCNConfig] = None,
    device: Optional[str] = None,
    **overrides,
):
    """Train the prefix-based GCN classifier.

    Args:
        event: Event log dataframe.
        case_col: Column identifying sequences/cases.
        event_col: Column with event labels.
        time_col: Column with timestamps.
        cat_event: Categorical event-level columns.
        num_event: Numerical event-level columns.
        cat_seq: Categorical sequence-level columns.
        num_seq: Numerical sequence-level columns.
        config: Optional config dataclass.
        device: Torch device string.
        **overrides: Simple overrides for config fields (e.g., prefix_size,
            num_layers, dropout, use_batch_norm, activation,
            patience, delta).

    Returns:
        Dict with trained model and training history.
    """
    return train_prefix_gcn(
        event,
        case_index=case_col,
        core_event=event_col,
        start_time_col=time_col,
        cat_col_event=cat_event or [],
        num_col_event=num_event or [],
        cat_col_seq=cat_seq or [],
        num_col_seq=num_seq or [],
        config=config,
        device=device,
        **overrides,
    )


def gat_outcome(
    event: pd.DataFrame,
    *,
    case_col: str,
    event_col: str,
    time_col: str,
    outcome_col: str,
    status_col: Optional[str] = None,
    cat_event: Optional[List[str]] = None,
    num_event: Optional[List[str]] = None,
    seq_cols: Optional[List[str]] = None,
    cat_seq: Optional[List[str]] = None,
    num_seq: Optional[List[str]] = None,
    mode: str = "gat_basic",
    config: Optional[GATOutcomeConfig] = None,
    device: Optional[str] = None,
    **overrides,
):
    """Train a case-level outcome classifier using GAT graph inputs.

    Args:
        event: Event log dataframe.
        case_col: Column identifying sequences/cases.
        event_col: Column with event labels.
        time_col: Column with timestamps.
        outcome_col: Column with outcome labels (case-level).
        status_col: Column used to build transition edge types.
        cat_event: Categorical event-level columns.
        num_event: Numerical event-level columns.
        seq_cols: Columns used to build sequence-level features.
        cat_seq: Categorical sequence-level columns.
        num_seq: Numerical sequence-level columns.
        mode: One of gat_basic, gat_time_decay, gat_status, gat_time_decay_status.
        config: Optional config dataclass.
        device: Torch device string.
        **overrides: Simple overrides for config fields (e.g., num_layers,
            dropout, use_batch_norm, activation, edge_type_dim,
            lambda_decay, patience, delta).

    Returns:
        Dict with trained model, history, and label encoder.
    """
    return train_gat_outcome(
        event,
        case_index=case_col,
        core_event=event_col,
        start_time_col=time_col,
        outcome_col=outcome_col,
        status_col=status_col,
        cat_col_event=cat_event or [],
        num_col_event=num_event or [],
        seq_cols=seq_cols or [],
        cat_col_seq=cat_seq or [],
        num_col_seq=num_seq or [],
        mode=mode,
        config=config,
        device=device,
        **overrides,
    )
