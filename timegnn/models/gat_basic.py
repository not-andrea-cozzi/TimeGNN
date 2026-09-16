from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .norm_layers import GraphAwareNormList, make_norm_layer


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
        use_batch_norm: Apply a normalization layer between hidden GAT
            layers. Kept for backward compatibility: when norm_kind is
            not explicitly set, use_batch_norm=True maps to "batch_norm"
            (the original behaviour) and False maps to "none".
        norm_kind: Explicit choice of normalization ("batch_norm",
            "layer_norm", "graph_norm", "none"). Takes precedence over
            use_batch_norm when provided. See timegnn.models.norm_layers
            and TrainPipeline.Steps.TuningStep.recommend_norm_kind for
            guidance (layer_norm/graph_norm are generally more stable
            than batch_norm on small/variable-composition graph batches).
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
        norm_kind: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.activation = _resolve_activation(activation)

        if norm_kind is None:
            norm_kind = "batch_norm" if use_batch_norm else "none"
        self.norm_kind = norm_kind
        self.use_norm = norm_kind != "none"

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
            [make_norm_layer(norm_kind, gat_hidden_dim_embed * num_heads) for _ in range(num_layers)]
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
            [make_norm_layer(norm_kind, gat_hidden_dim_event * num_heads) for _ in range(num_layers)]
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
            [make_norm_layer(norm_kind, gat_hidden_dim_concat * num_heads) for _ in range(num_layers)]
        )

        final_dim = gat_hidden_dim_concat * num_heads
        self.fc = nn.Linear(final_dim, output_dim)

    def _set_batch_index(self, batch_index: torch.Tensor) -> None:
        """Propaga data.batch ai layer GraphAwareNormList (no-op per gli
        altri norm_kind, che non richiedono l'indice di grafo)."""
        if self.norm_kind != "graph_norm":
            return
        for module_list in (self.bn_embed, self.bn_event, self.bn_concat):
            for norm in module_list:
                if isinstance(norm, GraphAwareNormList):
                    norm.set_batch_index(batch_index)

    def _run_path(self, layers: nn.ModuleList, norms: nn.ModuleList, x, edge_index, edge_attr):
        for i, layer in enumerate(layers):
            x = layer(x, edge_index, edge_attr=edge_attr)
            if i < len(layers) - 1:
                if self.use_norm:
                    x = norms[i](x)
                x = self.activation(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, data_event):
        """Forward pass for batched event graphs."""
        edge_attr = data_event.edge_attr

        if self.norm_kind == "graph_norm" and hasattr(data_event, "batch") and data_event.batch is not None:
            self._set_batch_index(data_event.batch)

        x_embed = self.embedding(data_event.event_ids.view(-1))
        x_embed = self._run_path(self.gat_embed, self.bn_embed, x_embed, data_event.edge_index, edge_attr)

        x_event = self._run_path(self.gat_event, self.bn_event, data_event.x, data_event.edge_index, edge_attr)

        x = torch.cat([x_embed, x_event], dim=1)
        x = self._run_path(self.gat_concat, self.bn_concat, x, data_event.edge_index, edge_attr)

        out = self.fc(x)
        return out