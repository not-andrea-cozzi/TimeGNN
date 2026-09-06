from __future__ import annotations

import inspect
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from ..train.early_stopping import EarlyStopping  # noqa: F401 – re-export


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


class TimeAwareETGATConv(GATConv):
    """GAT layer with time decay and edge-type attention."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        lambda_decay: float = 0.1,
        edge_type_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(in_channels, out_channels, heads=heads, concat=concat, **kwargs)
        self.lambda_decay = lambda_decay
        self.edge_type_dim = edge_type_dim

        self._decay = None
        self._edge_type_score = None

        self.att = nn.Parameter(torch.Tensor(heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.att)

        if edge_type_dim is None:
            raise ValueError("edge_type_dim must be specified for TimeAwareETGATConv.")

        self.edge_type_att = nn.Parameter(torch.Tensor(heads, edge_type_dim))
        nn.init.xavier_uniform_(self.edge_type_att)

    def edge_attention(self, x_i, x_j, time, edge_attr, edge_index):
        """Compute time- and edge-type-aware attention logits."""
        if edge_attr is inspect._empty:
            edge_attr = None
        cat_ij = torch.cat([x_i, x_j], dim=-1)
        alpha = torch.einsum("ehc,hc->eh", cat_ij, self.att)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if edge_attr is not None:
            edge_type_score = torch.einsum("ed,hd->eh", edge_attr, self.edge_type_att)
            alpha = alpha + edge_type_score

            delta_t = time
            decay = torch.exp(-self.lambda_decay * delta_t).unsqueeze(-1)
            alpha = alpha * decay

            self._decay = decay.detach().cpu()
            self._edge_type_score = edge_type_score.detach().cpu()

        return alpha

    def message(
        self,
        x_i=None,
        x_j=None,
        time=None,
        edge_attr=None,
        edge_index=None,
        index=None,
        ptr=None,
        size_i=None,
        alpha=None,
        **kwargs,
    ):
        """Message passing with time and edge-type attention weights."""
        if x_j is None:
            x_j = kwargs.get("x_j")
        if x_j is None:
            return None
        if x_i is None:
            x_i = kwargs.get("x_i")
        if time is None:
            time = kwargs.get("time")
        if edge_attr is None:
            edge_attr = kwargs.get("edge_attr")
        if edge_index is None:
            edge_index = kwargs.get("edge_index")
        if alpha is inspect._empty:
            alpha = None
        if alpha is None:
            if x_i is None:
                alpha = torch.ones(x_j.size(0), x_j.size(1), device=x_j.device)
            else:
                if time is None:
                    time = 0.0
                alpha = self.edge_attention(x_i, x_j, time, edge_attr, edge_index)
        self._alpha = alpha
        return x_j * alpha.unsqueeze(-1)

    def forward(self, x, edge_index, edge_attr=None, time=None, return_attention=False):
        """Forward pass with optional attention return."""
        h, c = self.heads, self.out_channels
        x = self.lin(x)
        x = x.view(-1, h, c)
        kwargs = {"x": x, "edge_index": edge_index, "size": None}
        sig = inspect.signature(self.propagate)
        if "edge_attr" in sig.parameters:
            kwargs["edge_attr"] = edge_attr
        if "time" in sig.parameters:
            kwargs["time"] = time
        if "alpha" in sig.parameters:
            kwargs["alpha"] = None
        out = self.propagate(**kwargs)

        if self.concat:
            out = out.view(-1, h * c)
        else:
            out = out.mean(dim=1)

        if return_attention:
            return out, self._alpha
        return out


class DualGATTimeAwareETModel(nn.Module):
    """Dual-path GAT with time decay and edge-type embeddings.

    Args:
        num_layers: Number of GAT layers per path.  Defaults to 1.
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
        num_edge_types: int,
        edge_type_dim: int,
        lambda_decay: float,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.activation = _resolve_activation(activation)

        self.embedding = nn.Embedding(num_embeddings=num_embedding_features, embedding_dim=embedding_dims)
        self.edge_type_emb = nn.Embedding(num_embeddings=num_edge_types, embedding_dim=edge_type_dim)
        edge_attr_dim = edge_type_dim

        self.gat_embed = nn.ModuleList()
        in_dim = embedding_dims
        for _ in range(num_layers):
            self.gat_embed.append(
                TimeAwareETGATConv(in_dim, gat_hidden_dim_embed, heads=num_heads, concat=True, edge_dim=edge_attr_dim, lambda_decay=lambda_decay, edge_type_dim=edge_type_dim)
            )
            in_dim = gat_hidden_dim_embed * num_heads
        self.bn_embed = nn.ModuleList(
            [nn.BatchNorm1d(gat_hidden_dim_embed * num_heads) for _ in range(num_layers)]
        )

        self.gat_event = nn.ModuleList()
        in_dim = num_event_features
        for _ in range(num_layers):
            self.gat_event.append(
                TimeAwareETGATConv(in_dim, gat_hidden_dim_event, heads=num_heads, concat=True, edge_dim=edge_attr_dim, lambda_decay=lambda_decay, edge_type_dim=edge_type_dim)
            )
            in_dim = gat_hidden_dim_event * num_heads
        self.bn_event = nn.ModuleList(
            [nn.BatchNorm1d(gat_hidden_dim_event * num_heads) for _ in range(num_layers)]
        )

        concat_input_dim = (gat_hidden_dim_embed + gat_hidden_dim_event) * num_heads
        self.gat_concat = nn.ModuleList()
        in_dim = concat_input_dim
        for _ in range(num_layers):
            self.gat_concat.append(
                TimeAwareETGATConv(in_dim, gat_hidden_dim_concat, heads=num_heads, concat=True, edge_dim=edge_attr_dim, lambda_decay=lambda_decay, edge_type_dim=edge_type_dim)
            )
            in_dim = gat_hidden_dim_concat * num_heads
        self.bn_concat = nn.ModuleList(
            [nn.BatchNorm1d(gat_hidden_dim_concat * num_heads) for _ in range(num_layers)]
        )

        final_dim = gat_hidden_dim_concat * num_heads
        self.fc = nn.Linear(final_dim, output_dim)

    def _run_path(self, layers, norms, x, edge_index, edge_attr, time, return_attention: bool = False):
        attn_last = None
        for i, layer in enumerate(layers):
            if return_attention and i == len(layers) - 1:
                x, attn_last = layer(
                    x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    time=time,
                    return_attention=True,
                )
            else:
                x = layer(x, edge_index=edge_index, edge_attr=edge_attr, time=time)

            if i < len(layers) - 1:
                if self.use_batch_norm:
                    x = norms[i](x)
                x = self.activation(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x, attn_last

    def forward(self, data_event, return_attention: bool = False):
        """Forward pass for batched event graphs with optional attention."""
        edge_type = data_event.edge_type
        edge_attr = self.edge_type_emb(edge_type)
        time = data_event.time
        edge_index = data_event.edge_index

        x_embed = self.embedding(data_event.event_ids.view(-1))
        x_embed, attn_embed = self._run_path(
            self.gat_embed,
            self.bn_embed,
            x_embed,
            edge_index,
            edge_attr,
            time,
            return_attention=return_attention,
        )

        x_event, attn_event = self._run_path(
            self.gat_event,
            self.bn_event,
            data_event.x,
            edge_index,
            edge_attr,
            time,
            return_attention=return_attention,
        )

        x = torch.cat([x_embed, x_event], dim=1)
        x, attn_final = self._run_path(
            self.gat_concat,
            self.bn_concat,
            x,
            edge_index,
            edge_attr,
            time,
            return_attention=return_attention,
        )

        out = self.fc(x)

        if return_attention:
            return out, {
                "alpha_embed": attn_embed.detach().cpu(),
                "alpha_event": attn_event.detach().cpu(),
                "alpha_final": attn_final.detach().cpu(),
                "edge_index": data_event.edge_index.detach().cpu(),
                "time": time.detach().cpu(),
                "edge_type": edge_type.detach().cpu(),
                "batch": data_event.batch.detach().cpu(),
                "decay_embed": self.gat_embed[-1]._decay.detach().cpu() if self.gat_embed[-1]._decay is not None else None,
                "decay_event": self.gat_event[-1]._decay.detach().cpu() if self.gat_event[-1]._decay is not None else None,
                "decay_final": self.gat_concat[-1]._decay.detach().cpu() if self.gat_concat[-1]._decay is not None else None,
                "edge_type_score_embed": self.gat_embed[-1]._edge_type_score.detach().cpu() if self.gat_embed[-1]._edge_type_score is not None else None,
                "edge_type_score_event": self.gat_event[-1]._edge_type_score.detach().cpu() if self.gat_event[-1]._edge_type_score is not None else None,
                "edge_type_score_final": self.gat_concat[-1]._edge_type_score.detach().cpu() if self.gat_concat[-1]._edge_type_score is not None else None,
            }
        return out


def evaluate_epoch(model, loader, criterion, device, return_attention: bool = False):
    """Evaluate the time-decay + edge-type model for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_tokens = 0
    all_attn_maps = []

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            labels = labels.to(device)

            if return_attention:
                output, attn_data = model(event_data, return_attention=True)
            else:
                output = model(event_data)

            output = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            mask = labels != -1
            labels = labels[mask]

            loss = criterion(output, labels)
            total_loss += loss.item() * labels.size(0)

            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total_tokens += labels.size(0)

            if return_attention:
                batch_vector = attn_data["batch"]
                num_graphs = batch_vector.max().item() + 1

                for graph_idx in range(num_graphs):
                    node_mask = batch_vector == graph_idx
                    node_indices = node_mask.nonzero(as_tuple=True)[0]

                    edge_mask = (
                        node_mask[attn_data["edge_index"][0]]
                        & node_mask[attn_data["edge_index"][1]]
                    )
                    edge_indices = edge_mask.nonzero(as_tuple=True)[0]

                    if edge_indices.numel() == 0:
                        continue

                    old2new = {old.item(): new for new, old in enumerate(node_indices)}
                    edge_index_sub = attn_data["edge_index"][:, edge_indices].clone()
                    for j in range(edge_index_sub.size(1)):
                        edge_index_sub[0, j] = old2new[edge_index_sub[0, j].item()]
                        edge_index_sub[1, j] = old2new[edge_index_sub[1, j].item()]

                    all_attn_maps.append(
                        {
                            "alpha_embed": attn_data["alpha_embed"][edge_indices],
                            "alpha_event": attn_data["alpha_event"][edge_indices],
                            "alpha_final": attn_data["alpha_final"][edge_indices],
                            "edge_index": edge_index_sub,
                            "time": attn_data["time"][edge_indices],
                            "edge_type": attn_data["edge_type"][edge_indices],
                            "decay_embed": (
                                attn_data["decay_embed"][edge_indices]
                                if attn_data.get("decay_embed") is not None
                                else None
                            ),
                            "decay_event": (
                                attn_data["decay_event"][edge_indices]
                                if attn_data.get("decay_event") is not None
                                else None
                            ),
                            "decay_final": (
                                attn_data["decay_final"][edge_indices]
                                if attn_data.get("decay_final") is not None
                                else None
                            ),
                            "edge_type_score_embed": (
                                attn_data["edge_type_score_embed"][edge_indices]
                                if attn_data.get("edge_type_score_embed") is not None
                                else None
                            ),
                            "edge_type_score_event": (
                                attn_data["edge_type_score_event"][edge_indices]
                                if attn_data.get("edge_type_score_event") is not None
                                else None
                            ),
                            "edge_type_score_final": (
                                attn_data["edge_type_score_final"][edge_indices]
                                if attn_data.get("edge_type_score_final") is not None
                                else None
                            ),
                        }
                    )

    accuracy = correct / total_tokens if total_tokens else 0.0
    loss = total_loss / total_tokens if total_tokens else 0.0
    return loss, accuracy, all_attn_maps
