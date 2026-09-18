from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import scatter


def _to_list(val: Union[int, List[int]], length: int) -> List[int]:
    if isinstance(val, int):
        return [val] * length
    return list(val)


def _resolve_activation(name: str):
    activations = {"relu": F.relu, "elu": F.elu, "gelu": F.gelu, "leaky_relu": F.leaky_relu}
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation '{name}'. Choose one of: {sorted(activations)}")
    return activations[key]


class _GraphNorm(nn.Module):
    """Normalizza per-grafo (media/varianza calcolate su ciascun grafo del
    batch, non sull'intero batch di nodi). Richiede che venga impostato
    batch_index prima del forward (vedi DualGATModel._set_batch_index)."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self._batch_index: Optional[torch.Tensor] = None

    def set_batch_index(self, batch_index: torch.Tensor) -> None:
        self._batch_index = batch_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_index = self._batch_index
        if batch_index is None:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, unbiased=False, keepdim=True)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            return x_norm * self.weight + self.bias

        num_graphs = int(batch_index.max().item()) + 1
        mean = scatter(x, batch_index, dim=0, dim_size=num_graphs, reduce="mean")
        mean_per_node = mean[batch_index]
        centered = x - mean_per_node
        var = scatter(centered * centered, batch_index, dim=0, dim_size=num_graphs, reduce="mean")
        var_per_node = var[batch_index]
        x_norm = centered / torch.sqrt(var_per_node + self.eps)
        return x_norm * self.weight + self.bias


def _resolve_norm(norm_kind: str, use_batch_norm: bool):
    """Factory di normalizzazione. Accetta sia i nomi storici ('batch',
    'layer', 'none') sia quelli prodotti da TuningStep.recommend_norm_kind
    ('batch_norm', 'layer_norm', 'graph_norm', 'none'), cosi' la
    raccomandazione del tuning non viene piu' silenziosamente ignorata."""
    kind = "batch" if use_batch_norm and norm_kind == "none" else norm_kind
    kind = kind.replace("_norm", "")  # "layer_norm" -> "layer", "graph_norm" -> "graph"
    if kind == "batch":
        return lambda dim: nn.BatchNorm1d(dim)
    if kind == "layer":
        return lambda dim: nn.LayerNorm(dim)
    if kind == "graph":
        return lambda dim: _GraphNorm(dim)
    return lambda dim: nn.Identity()


class DualGATModel(nn.Module):
    """Dual-path GAT: embedding dei pezzi + feature di casella.

    Args:
        num_layers: layer GAT per path. >=2 consigliato (con 1 ogni nodo
            vede solo i vicini diretti).
        dropout: dropout tra i layer.
        norm_kind: 'none' | 'batch'/'batch_norm' | 'layer'/'layer_norm' |
            'graph'/'graph_norm'.
        use_batch_norm: deprecato, equivale a norm_kind='batch'.
        pool_before_head: se True (default) la fc finale viene applicata
            DOPO il mean-pool per grafo: forward ritorna [B, output_dim]
            invece di [N_nodi, output_dim]. Matematicamente identico (fc e'
            lineare, pool e' una media) ma evita di moltiplicare N_nodi
            righe per una matrice da output_dim colonne (con
            MOVE_VOCAB_SIZE=20480 e' una differenza enorme).

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
        num_layers: int = 2,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "elu",
        norm_kind: str = "none",
        pool_before_head: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers deve essere >= 1 (ricevuto {num_layers}).")
        if num_heads < 1:
            raise ValueError(f"num_heads deve essere >= 1 (ricevuto {num_heads}).")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout deve essere in [0, 1) (ricevuto {dropout}).")

        self.dropout = dropout
        self.activation = _resolve_activation(activation)
        self.pool_before_head = pool_before_head
        self.norm_kind = norm_kind
        Norm = _resolve_norm(norm_kind, use_batch_norm)

        self.embedding = nn.Embedding(num_embedding_features, embedding_dims)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        def _build_path(in_dim: int, hidden: int):
            layers, norms = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_layers):
                layers.append(
                    GATConv(in_dim, hidden, heads=num_heads, concat=True, edge_dim=edge_dim)
                )
                norms.append(Norm(hidden * num_heads))
                in_dim = hidden * num_heads
            return layers, norms

        self.gat_embed, self.bn_embed = _build_path(embedding_dims, gat_hidden_dim_embed)
        self.gat_event, self.bn_event = _build_path(num_event_features, gat_hidden_dim_event)
        concat_in = (gat_hidden_dim_embed + gat_hidden_dim_event) * num_heads
        self.gat_concat, self.bn_concat = _build_path(concat_in, gat_hidden_dim_concat)

        self.fc = nn.Linear(gat_hidden_dim_concat * num_heads, output_dim)
        # Init piccola sul layer finale: con output_dim potenzialmente grande
        # (MOVE_VOCAB_SIZE=20480), l'init di default di nn.Linear produce
        # logit iniziali con varianza troppo alta per una softmax stabile.
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        nn.init.zeros_(self.fc.bias)

    def _set_batch_index(self, batch_index: torch.Tensor) -> None:
        """Propaga data.batch ai layer _GraphNorm (no-op se norm_kind non e'
        graph)."""
        for module_list in (self.bn_embed, self.bn_event, self.bn_concat):
            for norm in module_list:
                if isinstance(norm, _GraphNorm):
                    norm.set_batch_index(batch_index)

    def _run_path(self, layers, norms, x, edge_index, edge_attr):
        for i, layer in enumerate(layers):
            h = layer(x, edge_index, edge_attr=edge_attr)
            h = self.activation(norms[i](h))
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
            if h.shape == x.shape:
                h = h + x
            x = h
        return x

    def forward(self, data_event):
        edge_index, edge_attr = data_event.edge_index, data_event.edge_attr

        if hasattr(data_event, "batch") and data_event.batch is not None:
            self._set_batch_index(data_event.batch)

        x_embed = self.embedding(data_event.event_ids.view(-1))
        x_embed = self._run_path(self.gat_embed, self.bn_embed, x_embed, edge_index, edge_attr)

        x_event = self._run_path(
            self.gat_event, self.bn_event, data_event.x, edge_index, edge_attr
        )

        x = torch.cat([x_embed, x_event], dim=1)
        x = self._run_path(self.gat_concat, self.bn_concat, x, edge_index, edge_attr)

        if self.pool_before_head:
            x = global_mean_pool(x, data_event.batch)
        return self.fc(x)