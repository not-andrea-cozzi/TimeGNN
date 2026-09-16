from __future__ import annotations

from typing import List, Optional, Union

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


def _make_residual_proj(in_dim: int, out_dim: int) -> nn.Module:
    """Proiezione per lo skip connection tra un layer e il successivo.

    Identity se le dimensioni combaciano, altrimenti una Linear (senza
    bias: il residuo somma solo una trasformazione lineare del segnale
    in ingresso, il bias del layer principale a valle e' gia' sufficiente).
    Con num_layers=1 questa proiezione non ha alcun effetto visibile: lo
    skip si applica solo tra un layer e il successivo all'interno dello
    stesso path (embed/event/concat), e con un solo layer non c'e' un
    "successivo" a cui sommare nulla (vedi _run_path).
    """
    if in_dim == out_dim:
        return nn.Identity()
    return nn.Linear(in_dim, out_dim, bias=False)


class DualGATModel(nn.Module):
    """Dual-path GAT model using event embeddings and raw features.

    Args:
        num_layers: Number of GAT layers per path (embed, event) and for the
            concat path.  Defaults to 1 for backward compatibility. Quando
            num_layers > 1, ogni layer (tranne l'ultimo del path) applica
            uno skip connection residuo verso il layer successivo (vedi
            _run_path), per mitigare over-smoothing e gradienti deboli
            nelle configurazioni piu' profonde.
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
        if num_layers < 1:
            raise ValueError(f"num_layers deve essere >= 1 (ricevuto {num_layers}).")
        if num_heads < 1:
            raise ValueError(f"num_heads deve essere >= 1 (ricevuto {num_heads}).")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout deve essere in [0, 1) (ricevuto {dropout}).")

        self.dropout = dropout
        self.activation = _resolve_activation(activation)

        if norm_kind is None:
            norm_kind = "batch_norm" if use_batch_norm else "none"
        self.norm_kind = norm_kind
        self.use_norm = norm_kind != "none"

        self.embedding = nn.Embedding(
            num_embeddings=num_embedding_features, embedding_dim=embedding_dims
        )
        # Init esplicita: il default di nn.Embedding (N(0,1)) produce
        # vettori con norma ~sqrt(embedding_dims), grande abbastanza da
        # destabilizzare i primi step di un GAT che riceve l'embedding
        # come feature di input diretta, senza normalizzazione a monte.
        # std=0.02 e' lo standard usato per gli embedding nei transformer:
        # un punto di partenza numericamente stabile che non limita la
        # capacita' di apprendimento successiva.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # --- embed path ---
        self.gat_embed = nn.ModuleList()
        self.res_embed = nn.ModuleList()
        in_dim = embedding_dims
        for _ in range(num_layers):
            self.gat_embed.append(
                GATConv(in_dim, gat_hidden_dim_embed, heads=num_heads, concat=True, edge_dim=edge_dim)
            )
            out_dim = gat_hidden_dim_embed * num_heads
            self.res_embed.append(_make_residual_proj(in_dim, out_dim))
            in_dim = out_dim
        self.bn_embed = nn.ModuleList(
            [make_norm_layer(norm_kind, gat_hidden_dim_embed * num_heads) for _ in range(num_layers)]
        )

        # --- event path ---
        self.gat_event = nn.ModuleList()
        self.res_event = nn.ModuleList()
        in_dim = num_event_features
        for _ in range(num_layers):
            self.gat_event.append(
                GATConv(in_dim, gat_hidden_dim_event, heads=num_heads, concat=True, edge_dim=edge_dim)
            )
            out_dim = gat_hidden_dim_event * num_heads
            self.res_event.append(_make_residual_proj(in_dim, out_dim))
            in_dim = out_dim
        self.bn_event = nn.ModuleList(
            [make_norm_layer(norm_kind, gat_hidden_dim_event * num_heads) for _ in range(num_layers)]
        )

        # --- concat path ---
        concat_input_dim = (gat_hidden_dim_embed + gat_hidden_dim_event) * num_heads
        self.gat_concat = nn.ModuleList()
        self.res_concat = nn.ModuleList()
        in_dim = concat_input_dim
        for _ in range(num_layers):
            self.gat_concat.append(
                GATConv(in_dim, gat_hidden_dim_concat, heads=num_heads, concat=True, edge_dim=edge_dim)
            )
            out_dim = gat_hidden_dim_concat * num_heads
            self.res_concat.append(_make_residual_proj(in_dim, out_dim))
            in_dim = out_dim
        self.bn_concat = nn.ModuleList(
            [make_norm_layer(norm_kind, gat_hidden_dim_concat * num_heads) for _ in range(num_layers)]
        )

        final_dim = gat_hidden_dim_concat * num_heads
        self.fc = nn.Linear(final_dim, output_dim)
        # Init piccola sul layer finale: con output_dim potenzialmente
        # grande (es. MOVE_VOCAB_SIZE=20480), l'init di default di
        # nn.Linear (Kaiming uniform, pensata per ReLU) produce logit
        # iniziali con varianza che cresce con final_dim, spingendo la
        # softmax a valle verso distribuzioni gia' molto piccate prima
        # ancora che il modello abbia imparato nulla. Una gain ridotta
        # mantiene i logit iniziali vicini a zero (softmax quasi
        # uniforme), un punto di partenza piu' stabile per la loss.
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        nn.init.zeros_(self.fc.bias)

    def _set_batch_index(self, batch_index: torch.Tensor) -> None:
        """Propaga data.batch ai layer GraphAwareNormList (no-op per gli
        altri norm_kind, che non richiedono l'indice di grafo)."""
        if self.norm_kind != "graph_norm":
            return
        for module_list in (self.bn_embed, self.bn_event, self.bn_concat):
            for norm in module_list:
                if isinstance(norm, GraphAwareNormList):
                    norm.set_batch_index(batch_index)

    def _run_path(self, layers: nn.ModuleList, norms: nn.ModuleList, residuals: nn.ModuleList, x, edge_index, edge_attr):
        """Esegue un path (embed/event/concat) applicando, tra un layer e
        il successivo, uno skip connection residuo: x_out = layer(x) +
        proj(x_in). La proiezione e' Identity se le dimensioni combaciano,
        altrimenti una Linear (vedi _make_residual_proj). Con num_layers=1
        il ciclo esegue una sola iterazione e non applica mai lo skip
        (nessun "layer successivo" a cui sommare), quindi il comportamento
        per la configurazione di default resta identico a prima.
        """
        for i, layer in enumerate(layers):
            residual = residuals[i](x)
            x = layer(x, edge_index, edge_attr=edge_attr)
            if i < len(layers) - 1:
                x = x + residual
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
        x_embed = self._run_path(self.gat_embed, self.bn_embed, self.res_embed, x_embed, data_event.edge_index, edge_attr)

        x_event = self._run_path(self.gat_event, self.bn_event, self.res_event, data_event.x, data_event.edge_index, edge_attr)

        x = torch.cat([x_embed, x_event], dim=1)
        x = self._run_path(self.gat_concat, self.bn_concat, self.res_concat, x, data_event.edge_index, edge_attr)

        out = self.fc(x)
        return out