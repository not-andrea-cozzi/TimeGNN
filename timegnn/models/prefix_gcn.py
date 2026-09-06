from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from torch_geometric.nn import GCNConv, global_mean_pool


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


class PrefixGCNClassifier(nn.Module):
    """Prefix-based GCN classifier with sequence-level features.

    Args:
        num_layers: Number of GCN layers per path.  Defaults to 1.
        dropout: Dropout rate applied between layers (0 = no dropout).
        use_batch_norm: Apply BatchNorm1d between hidden GCN layers.
        activation: Hidden-layer activation (relu, elu, gelu, leaky_relu).
    """
    def __init__(
        self,
        num_event_features: int,
        gcn_hidden_dims: int,
        num_embedding_features: int,
        embedding_dims: int,
        gcn_hidden_dims_embedding: int,
        gcn_hidden_dims_concat: int,
        num_sequence_features: int,
        fc_hidden_dims: int,
        fc_hidden_dims_concat: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.activation = _resolve_activation(activation)

        self.embedding = nn.Embedding(
            num_embeddings=num_embedding_features, embedding_dim=embedding_dims
        )
        self.gcn_embed = nn.ModuleList()
        in_dim = embedding_dims
        for _ in range(num_layers):
            self.gcn_embed.append(GCNConv(in_dim, gcn_hidden_dims_embedding))
            in_dim = gcn_hidden_dims_embedding
        self.bn_embed = nn.ModuleList(
            [nn.BatchNorm1d(gcn_hidden_dims_embedding) for _ in range(num_layers)]
        )

        self.gcn_event = nn.ModuleList()
        in_dim = num_event_features
        for _ in range(num_layers):
            self.gcn_event.append(GCNConv(in_dim, gcn_hidden_dims))
            in_dim = gcn_hidden_dims
        self.bn_event = nn.ModuleList(
            [nn.BatchNorm1d(gcn_hidden_dims) for _ in range(num_layers)]
        )

        self.gcn_concat = nn.ModuleList()
        in_dim = gcn_hidden_dims + gcn_hidden_dims_embedding
        for _ in range(num_layers):
            self.gcn_concat.append(GCNConv(in_dim, gcn_hidden_dims_concat))
            in_dim = gcn_hidden_dims_concat
        self.bn_concat = nn.ModuleList(
            [nn.BatchNorm1d(gcn_hidden_dims_concat) for _ in range(num_layers)]
        )

        self.seq_proj = nn.Linear(num_sequence_features, fc_hidden_dims)
        self.concat_proj = nn.Linear(gcn_hidden_dims_concat + fc_hidden_dims, fc_hidden_dims_concat)

        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(fc_hidden_dims_concat, output_dim))

    def _run_path(self, layers: nn.ModuleList, norms: nn.ModuleList, x, edge_index, edge_weight):
        for i, layer in enumerate(layers):
            x = layer(x, edge_index, edge_weight=edge_weight)
            if i < len(layers) - 1:
                if self.use_batch_norm:
                    x = norms[i](x)
                x = self.activation(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, data, sequence_features):
        """Forward pass for event graphs and sequence features."""
        d = self.embedding(data.event_ids.squeeze(-1))
        d = self._run_path(self.gcn_embed, self.bn_embed, d, data.edge_index, data.edge_attr)

        f = data.x
        f[f == -1] = 0
        f = self._run_path(self.gcn_event, self.bn_event, f, data.edge_index, data.edge_attr)

        x = torch.cat([d, f], dim=1)
        x = self._run_path(self.gcn_concat, self.bn_concat, x, data.edge_index, data.edge_attr)
        graph_emb = global_mean_pool(x, data.batch)

        seq_out = self.seq_proj(sequence_features)
        seq_out_concat = torch.cat([graph_emb, seq_out], dim=1)
        seq_out_concat = self.concat_proj(seq_out_concat)

        out = self.classifier(seq_out_concat)
        return out


def train_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch for prefix GCN."""
    model.train()
    total_loss = 0.0
    correct = 0

    for event_data, sequence_features, labels in loader:
        event_data = event_data.to(device)
        sequence_features = sequence_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(event_data, sequence_features)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()

    accuracy = correct / len(loader.dataset)
    loss = total_loss / len(loader.dataset)
    return loss, accuracy


def evaluate_epoch(model, loader, criterion, device):
    """Evaluate prefix GCN for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for event_data, sequence_features, labels in loader:
            event_data = event_data.to(device)
            sequence_features = sequence_features.to(device)
            labels = labels.to(device)

            output = model(event_data, sequence_features)
            loss = criterion(output, labels)
            total_loss += loss.item() * labels.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()

    accuracy = correct / len(loader.dataset)
    loss = total_loss / len(loader.dataset)
    return loss, accuracy


def f1_eva(model, loader, device, k: int = 3):
    """Compute classification report and top-k accuracy for prefix GCN."""
    model.eval()
    correct_topk = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for event_data, sequence_features, labels in loader:
            event_data = event_data.to(device)
            sequence_features = sequence_features.to(device)
            labels = labels.to(device)

            output = model(event_data, sequence_features)
            pred = output.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

            topk_preds = torch.topk(output, k=k, dim=1).indices
            correct_topk += sum([labels[i] in topk_preds[i] for i in range(labels.size(0))])
            total += labels.size(0)

    class_report = classification_report(all_labels, all_preds, digits=4)
    topk_accuracy = correct_topk / total if total else 0.0

    return class_report, topk_accuracy


def get_misclassified_samples(model, loader, device):
    """Collect misclassified samples for error analysis."""
    model.eval()
    errors = []

    with torch.no_grad():
        for event_data, sequence_features, labels in loader:
            event_data = event_data.to(device)
            sequence_features = sequence_features.to(device)
            labels = labels.to(device)

            outputs = model(event_data, sequence_features)
            preds = outputs.argmax(dim=1)

            for i in range(labels.size(0)):
                if preds[i] != labels[i]:
                    errors.append(
                        {
                            "pred": preds[i].item(),
                            "label": labels[i].item(),
                            "sequence_feats": sequence_features[i].cpu().numpy(),
                            "event_feats": event_data[i].x.cpu().numpy()
                            if hasattr(event_data[i], "x")
                            else None,
                            "embedding_feats": event_data[i].event_ids.cpu().numpy()
                            if hasattr(event_data[i], "x")
                            else None,
                        }
                    )
    return errors


def cluster_errors(errors, num_clusters, use: str = "sequence_feats", method: str = "pca"):
    """Cluster misclassified samples with PCA or t-SNE."""
    features = []
    labels = []

    for e in errors:
        if e[use] is not None:
            features.append(e[use].flatten())
            labels.append((e["label"], e["pred"]))

    if not features:
        return None

    features = np.array(features)
    n_samples, n_features = features.shape
    effective_components = min(2, n_features, n_samples - 1) if n_samples > 1 else 1

    if method == "tsne":
        reducer = TSNE(n_components=effective_components, random_state=42)
        reduced = reducer.fit_transform(features)
    else:
        reducer = PCA(n_components=effective_components)
        reduced = reducer.fit_transform(features)

    return reduced, labels
