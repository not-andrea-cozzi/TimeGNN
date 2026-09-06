from __future__ import annotations

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


def normalize_attention_minmax(attn_raw, edge_indices):
    """Normalize attention values per edge using min-max."""
    selected = attn_raw[edge_indices]
    min_vals = selected.min(dim=0, keepdim=True)[0]
    max_vals = selected.max(dim=0, keepdim=True)[0]
    normed = (selected - min_vals) / (max_vals - min_vals + 1e-8)
    return normed.mean(dim=1)


def min_max_normalize(x):
    """Min-max normalize a tensor to [0, 1]."""
    x = x.squeeze()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


class TimeAwareGATConv(GATConv):
    """GAT layer with exponential time decay on attention."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        lambda_decay: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(in_channels, out_channels, heads=heads, concat=concat, **kwargs)
        self.lambda_decay = lambda_decay
        self.att = nn.Parameter(torch.Tensor(heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.att)
        self._decay = None

    def edge_attention(self, x_i, x_j, edge_attr):
        """Compute time-decayed attention logits."""
        cat_ij = torch.cat([x_i, x_j], dim=-1)
        alpha = torch.einsum("ehc,hc->eh", cat_ij, self.att)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if edge_attr is not None:
            time_diff = edge_attr
            decay = torch.exp(-self.lambda_decay * time_diff).unsqueeze(-1)
            alpha = alpha * decay
            self._decay = decay.detach().cpu()
        return alpha

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        """Message passing with time-decayed attention weights."""
        alpha = self.edge_attention(x_i, x_j, edge_attr)
        self._alpha = alpha
        return x_j * alpha.unsqueeze(-1)

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        """Forward pass with optional attention return."""
        h, c = self.heads, self.out_channels
        x = self.lin(x)
        x = x.view(-1, h, c)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.concat:
            out = out.view(-1, h * c)
        else:
            out = out.mean(dim=1)

        if return_attention:
            return out, self._alpha
        return out


class DualGATTimeAwareModel(nn.Module):
    """Dual-path GAT model with time-decayed attention.

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

        edge_dim = 1
        self.embedding = nn.Embedding(
            num_embeddings=num_embedding_features, embedding_dim=embedding_dims
        )

        # --- embed path ---
        self.gat_embed = nn.ModuleList()
        in_dim = embedding_dims
        for _ in range(num_layers):
            self.gat_embed.append(
                TimeAwareGATConv(in_dim, gat_hidden_dim_embed, heads=num_heads, concat=True, edge_dim=edge_dim, lambda_decay=lambda_decay)
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
                TimeAwareGATConv(in_dim, gat_hidden_dim_event, heads=num_heads, concat=True, edge_dim=edge_dim, lambda_decay=lambda_decay)
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
                TimeAwareGATConv(in_dim, gat_hidden_dim_concat, heads=num_heads, concat=True, edge_dim=edge_dim, lambda_decay=lambda_decay)
            )
            in_dim = gat_hidden_dim_concat * num_heads
        self.bn_concat = nn.ModuleList(
            [nn.BatchNorm1d(gat_hidden_dim_concat * num_heads) for _ in range(num_layers)]
        )

        final_dim = gat_hidden_dim_concat * num_heads
        self.fc = nn.Linear(final_dim, output_dim)

    def _run_path(self, layers, norms, x, edge_index, edge_attr, return_attention: bool = False):
        attn_last = None
        for i, layer in enumerate(layers):
            if return_attention and i == len(layers) - 1:
                x, attn_last = layer(
                    x, edge_index=edge_index, edge_attr=edge_attr, return_attention=True
                )
            else:
                x = layer(x, edge_index=edge_index, edge_attr=edge_attr)

            if i < len(layers) - 1:
                if self.use_batch_norm:
                    x = norms[i](x)
                x = self.activation(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x, attn_last

    def forward(self, data_event, return_attention: bool = False):
        """Forward pass for batched event graphs with optional attention."""
        edge_attr = data_event.time
        edge_index = data_event.edge_index

        x_embed = self.embedding(data_event.event_ids.view(-1))
        x_embed, attn_embed = self._run_path(
            self.gat_embed,
            self.bn_embed,
            x_embed,
            edge_index,
            edge_attr,
            return_attention=return_attention,
        )

        x_event, attn_event = self._run_path(
            self.gat_event,
            self.bn_event,
            data_event.x,
            edge_index,
            edge_attr,
            return_attention=return_attention,
        )

        x = torch.cat([x_embed, x_event], dim=1)
        x, attn_final = self._run_path(
            self.gat_concat,
            self.bn_concat,
            x,
            edge_index,
            edge_attr,
            return_attention=return_attention,
        )

        out = self.fc(x)

        if return_attention:
            return out, {
                "alpha_embed": attn_embed.detach().cpu(),
                "alpha_event": attn_event.detach().cpu(),
                "alpha_final": attn_final.detach().cpu(),
                "edge_index": edge_index.detach().cpu(),
                "time": edge_attr.detach().cpu(),
                "decay_embed": self.gat_embed[-1]._decay.detach().cpu()
                if self.gat_embed[-1]._decay is not None
                else None,
                "decay_event": self.gat_event[-1]._decay.detach().cpu()
                if self.gat_event[-1]._decay is not None
                else None,
                "decay_final": self.gat_concat[-1]._decay.detach().cpu()
                if self.gat_concat[-1]._decay is not None
                else None,
                "batch": data_event.batch.detach().cpu(),
            }
        return out


def evaluate_epoch(model, loader, criterion, device, return_attention: bool = False):
    """Evaluate the time-decay GAT model for one epoch."""
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

                    alpha_embed_norm = normalize_attention_minmax(
                        attn_data["alpha_embed"], edge_indices
                    )
                    alpha_event_norm = normalize_attention_minmax(
                        attn_data["alpha_event"], edge_indices
                    )
                    alpha_final_norm = normalize_attention_minmax(
                        attn_data["alpha_final"], edge_indices
                    )

                    decay_embed = (
                        min_max_normalize(attn_data["decay_embed"][edge_indices])
                        if attn_data.get("decay_embed") is not None
                        else None
                    )
                    decay_event = (
                        min_max_normalize(attn_data["decay_event"][edge_indices])
                        if attn_data.get("decay_event") is not None
                        else None
                    )
                    decay_final = (
                        min_max_normalize(attn_data["decay_final"][edge_indices])
                        if attn_data.get("decay_final") is not None
                        else None
                    )

                    graph_attn = {
                        "alpha_embed": alpha_embed_norm,
                        "alpha_event": alpha_event_norm,
                        "alpha_final": alpha_final_norm,
                        "decay_embed": decay_embed,
                        "decay_event": decay_event,
                        "decay_final": decay_final,
                        "edge_index": edge_index_sub,
                        "time": attn_data["time"][edge_indices],
                        "batch": batch_vector[node_indices],
                        "graph_idx": graph_idx,
                    }
                    all_attn_maps.append(graph_attn)

    accuracy = correct / total_tokens if total_tokens else 0.0
    loss = total_loss / total_tokens if total_tokens else 0.0

    if return_attention:
        return loss, accuracy, all_attn_maps
    return loss, accuracy
