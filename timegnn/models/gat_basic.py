from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


def _to_list(val: Union[int, List[int]], length: int) -> List[int]:
    """Expand a scalar to a list of the given length."""
    if isinstance(val, int):
        return [val] * length
    return list(val)


def _resolve_activation(name: str):
    activations = {
        "relu": F.relu,
        "elu": F.elu,
        "gelu": F.gelu,
        "leaky_relu": F.leaky_relu,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation '{name}'. Choose one of: {sorted(activations)}")
    return activations[key]


class DualGATModel(nn.Module):
    """Dual-path GAT model using event embeddings and raw features.

    Args:
        num_layers: Number of GAT layers per path (embed, event) and for the
            concat path.  Defaults to 1 for backward compatibility.
        dropout: Dropout rate applied between layers (0 = no dropout).
        use_batch_norm: Apply BatchNorm1d between hidden GAT layers.
        activation: Hidden-layer activation (relu, elu, gelu, leaky_relu).
    """
    def __init__(
        self,
        num_event_features: int,
        num_embedding_features: int,
        embedding_dims: int,
        gat_hidden_dim_event: int,
        gat_hidden_dim_embed: int,
        gat_hidden_dim_concat: int,
        output_dim: int,
        num_heads: int,
        edge_dim: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.activation = _resolve_activation(activation)

        self.embedding = nn.Embedding(
            num_embeddings=num_embedding_features, embedding_dim=embedding_dims
        )

        # --- embed path ---
        self.gat_embed = nn.ModuleList()
        in_dim = embedding_dims
        for _ in range(num_layers):
            self.gat_embed.append(
                GATConv(in_dim, gat_hidden_dim_embed, heads=num_heads, concat=True, edge_dim=edge_dim)
            )
            in_dim = gat_hidden_dim_embed * num_heads
        self.bn_embed = nn.ModuleList(
            [nn.BatchNorm1d(gat_hidden_dim_embed * num_heads) for _ in range(num_layers)]
        )

        # --- event path ---
        self.gat_event = nn.ModuleList()
        in_dim = num_event_features
        for _ in range(num_layers):
            self.gat_event.append(
                GATConv(in_dim, gat_hidden_dim_event, heads=num_heads, concat=True, edge_dim=edge_dim)
            )
            in_dim = gat_hidden_dim_event * num_heads
        self.bn_event = nn.ModuleList(
            [nn.BatchNorm1d(gat_hidden_dim_event * num_heads) for _ in range(num_layers)]
        )

        # --- concat path ---
        concat_input_dim = (gat_hidden_dim_embed + gat_hidden_dim_event) * num_heads
        self.gat_concat = nn.ModuleList()
        in_dim = concat_input_dim
        for _ in range(num_layers):
            self.gat_concat.append(
                GATConv(in_dim, gat_hidden_dim_concat, heads=num_heads, concat=True, edge_dim=edge_dim)
            )
            in_dim = gat_hidden_dim_concat * num_heads
        self.bn_concat = nn.ModuleList(
            [nn.BatchNorm1d(gat_hidden_dim_concat * num_heads) for _ in range(num_layers)]
        )

        final_dim = gat_hidden_dim_concat * num_heads
        self.fc = nn.Linear(final_dim, output_dim)

    def _run_path(self, layers: nn.ModuleList, norms: nn.ModuleList, x, edge_index, edge_attr):
        for i, layer in enumerate(layers):
            x = layer(x, edge_index, edge_attr=edge_attr)
            if i < len(layers) - 1:
                if self.use_batch_norm:
                    x = norms[i](x)
                x = self.activation(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, data_event):
        """Forward pass for batched event graphs."""
        edge_attr = data_event.edge_attr

        x_embed = self.embedding(data_event.event_ids.view(-1))
        x_embed = self._run_path(self.gat_embed, self.bn_embed, x_embed, data_event.edge_index, edge_attr)

        x_event = self._run_path(self.gat_event, self.bn_event, data_event.x, data_event.edge_index, edge_attr)

        x = torch.cat([x_embed, x_event], dim=1)
        x = self._run_path(self.gat_concat, self.bn_concat, x, data_event.edge_index, edge_attr)

        out = self.fc(x)
        return out
