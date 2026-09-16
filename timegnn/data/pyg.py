from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch.nn.utils.rnn import pad_sequence


def _chain_edge_index(num_events: int) -> torch.Tensor:
    """Build a [2, num_events-1] linear-chain edge_index (j -> j+1) using
    vectorized torch.arange instead of a Python list comprehension + .t().
    """
    if num_events <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    src = torch.arange(0, num_events - 1, dtype=torch.long)
    dst = torch.arange(1, num_events, dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def _to_float_tensor_batch(array_like) -> torch.Tensor:
    arr = np.asarray(array_like)
    return torch.from_numpy(arr).float()


def _to_long_tensor_batch(array_like) -> torch.Tensor:
    """Same idea as _to_float_tensor_batch but for long/int tensors
    (core_encode / event id columns)."""
    arr = np.asarray(array_like)
    return torch.from_numpy(arr).long()


def prepare_data_core_2edges(
    event_encode,
    core_encode,
    scaled_time_diffs,
    edge_types_encoded,
    node_times=None,
):
    """Build PyG Data objects with edge types and time diffs."""
    data_list_event = []

    node_features_all = _to_float_tensor_batch(event_encode)
    node_core_all = _to_long_tensor_batch(core_encode)
    time_all = None if node_times is None else _to_float_tensor_batch(node_times)

    for i in range(len(event_encode)):
        node_features = node_features_all[i]
        node_core = node_core_all[i]
        time = None if time_all is None else time_all[i]
        num_events = int((node_core[:, 0] != -1).sum())

        edge_index = _chain_edge_index(num_events)

        time_diffs = scaled_time_diffs[i][: num_events - 1]
        edge_types = edge_types_encoded[i][: num_events - 1]

        edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
        edge_time_tensor = torch.tensor(time_diffs, dtype=torch.float).view(-1, 1)

        event_ids = node_core[:num_events]

        graph_data = Data(
            x=node_features[:num_events],
            edge_index=edge_index,
            event_ids=event_ids,
            num_nodes=num_events,
        )
        graph_data.edge_type = edge_type_tensor
        graph_data.edge_time_diff = edge_time_tensor
        if time is not None:
            graph_data.time = time

        data_list_event.append(graph_data)

    return data_list_event


def prepare_data_y(event_encode, y_encode):
    """Prepare per-node labels aligned with event sequences."""
    data_list = []

    node_features_all = _to_float_tensor_batch(event_encode)
    labels_all = _to_long_tensor_batch(y_encode)

    for i in range(len(event_encode)):
        node_features = node_features_all[i]
        labels = labels_all[i]
        num_events = int((node_features[:, 0] != -1).sum())
        data_list.append(labels[:num_events])
    return data_list


def prepare_data_core_timedif(
    event_encode,
    core_encode,
    scaled_time_diffs,
    node_times: Optional[list] = None,
):
    data_list_event = []

    node_features_all = _to_float_tensor_batch(event_encode)
    node_core_all = _to_long_tensor_batch(core_encode)
    time_all = None if node_times is None else _to_float_tensor_batch(node_times)

    for i in range(len(event_encode)):
        node_features = node_features_all[i]
        node_core = node_core_all[i]
        num_events = int((node_core[:, 0] != -1).sum())

        edge_index = _chain_edge_index(num_events)
        edge_attr = torch.tensor(
            scaled_time_diffs[i][: num_events - 1], dtype=torch.float
        ).view(-1, 1)

        event_ids = node_core[:num_events]
        graph_data = Data(
            x=node_features[:num_events],
            edge_index=edge_index,
            edge_attr=edge_attr,
            event_ids=event_ids,
        )
        graph_data.num_nodes = num_events
        if time_all is not None:
            graph_data.time = time_all[i]
        data_list_event.append(graph_data)
    return data_list_event


def prepare_data_core(event_encode, core_encode, node_times):
    """Build PyG Data objects without explicit edge attributes."""
    data_list_event = []

    node_features_all = _to_float_tensor_batch(event_encode)
    node_core_all = _to_long_tensor_batch(core_encode)
    time_all = _to_float_tensor_batch(node_times)

    for i in range(len(event_encode)):
        node_features = node_features_all[i]
        node_core = node_core_all[i]
        time = time_all[i]
        num_events = int((node_core[:, 0] != -1).sum())

        edge_index = _chain_edge_index(num_events)
        event_ids = node_core[:num_events]

        graph_data = Data(
            x=node_features[:num_events],
            edge_index=edge_index,
            event_ids=event_ids,
        )
        graph_data.num_nodes = num_events
        graph_data.time = time
        data_list_event.append(graph_data)
    return data_list_event


def prepare_data_prefix(event_encode, core_encode, scaled_time_diffs):
    """Build PyG Data objects for prefix-based GCN models."""
    data_list_event = []

    node_features_all = _to_float_tensor_batch(event_encode)
    node_core_all = _to_long_tensor_batch(core_encode)

    for i in range(len(event_encode)):
        node_features = node_features_all[i]
        node_core = node_core_all[i]
        num_events = int((node_core[:, 0] != -1).sum())

        edge_index = _chain_edge_index(num_events)
        edge_attr = torch.tensor(
            scaled_time_diffs[i][: num_events - 1], dtype=torch.float
        ).view(-1, 1)

        event_ids = node_core[:num_events]
        graph_data = Data(
            x=node_features[:num_events],
            edge_index=edge_index,
            edge_attr=edge_attr,
            event_ids=event_ids,
        )
        graph_data.num_nodes = num_events
        data_list_event.append(graph_data)
    return data_list_event


class CustomDataset(Dataset):
    """Dataset wrapper for event graphs and labels."""
    def __init__(self, event_features, y):
        self.event_features = event_features
        self.y = y

    def __len__(self):
        return len(self.event_features)

    def __getitem__(self, idx):
        return self.event_features[idx], self.y[idx]


class PrefixDataset(Dataset):
    """Dataset wrapper for prefix graphs, sequence features, and labels."""
    def __init__(self, event_features, sequence_features, y):
        self.event_features = event_features
        self.sequence_features = sequence_features
        self.y = y

    def __len__(self):
        return len(self.event_features)

    def __getitem__(self, idx):
        return self.event_features[idx], self.sequence_features[idx], self.y[idx]


def custom_collate_fn(batch):
    """Collate function for variable-length sequence labels."""
    event_data_list, label_list = zip(*batch)
    batch_event = Batch.from_data_list(event_data_list)
    padded_labels = pad_sequence(
        [lbl.squeeze(1) for lbl in label_list],
        batch_first=True,
        padding_value=-1,
    )
    return batch_event, padded_labels


def custom_collate_prefix(batch):
    """Collate function for prefix models with sequence-level features."""
    event_data, seq_features, labels = zip(*batch)
    return (
        Batch.from_data_list(event_data),
        torch.stack(seq_features),
        torch.tensor(labels),
    )


def custom_collate_graph(batch):
    """Collate function for graph-level labels."""
    event_data_list, labels = zip(*batch)
    return Batch.from_data_list(event_data_list), torch.tensor(labels)