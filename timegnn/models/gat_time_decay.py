from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax as pyg_softmax

from .norm_layers import GraphAwareNormList, make_norm_layer
from ..train.early_stopping import EarlyStopping  # noqa: F401 – re-export


# Limite superiore sull'argomento di exp(-lambda_decay * delta_t): con
# delta_t normalizzato in [0,1] (vedi node_time_list) e lambda_decay
# tipico ~1e-2, l'esponente resta piccolo. Ma se in futuro arrivano
# delta_t non normalizzati o lambda_decay piu' aggressivo, un esponente
# molto negativo produce comunque un decay~0 corretto; il vero rischio e'
# l'opposto: lambda_decay negativo o delta_t negativo che spingerebbe
# l'esponente verso +inf e decay verso +inf, propagando NaN/inf nei
# logit. Il clamp taglia l'esponente PRIMA dell'exp, in entrambe le
# direzioni, cosi' decay resta sempre in un range finito e stabile.
_MAX_DECAY_EXPONENT = 30.0


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

    Identity se le dimensioni combaciano (caso comune quando
    gat_hidden_dim_* * num_heads == in_dim di quel layer), altrimenti una
    Linear che porta il residuo alla dimensione giusta prima di sommarlo.
    Con num_layers=1 questa proiezione non ha alcun effetto visibile: lo
    skip si applica solo tra un layer e il successivo all'interno dello
    stesso path (embed/event/concat), e con un solo layer non c'e' un
    "successivo" a cui sommare nulla (vedi _run_path).
    """
    if in_dim == out_dim:
        return nn.Identity()
    return nn.Linear(in_dim, out_dim, bias=False)


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
    """GAT layer with exponential time decay on attention.

    Args:
        in_channels: input feature dimension.
        out_channels: output feature dimension per head.
        heads: number of attention heads.
        concat: whether to concatenate or average head outputs.
        lambda_decay: decay rate applied to time_diff before exp(-.).
        **kwargs: forwarded to GATConv (e.g. edge_dim).
    """
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
        """Compute time-decayed, softmax-normalized attention logits.

        Ordine delle operazioni (importante per la correttezza, non solo
        per stile):
          1. logit grezzi content-based (leaky_relu su [x_i || x_j] · att)
          2. scala per il decay temporale (edge piu' vecchi pesano meno)
          3. softmax PER NODO DESTINAZIONE (index = nodo target dell'arco)

        Il passo 3 mancava nella versione precedente: GATConv.message()
        standard applica softmax(alpha, index, ...) prima di pesare x_j,
        cosa che qui viene bypassata perche' message() e' overridden.
        Senza questo softmax, alpha non e' mai vincolato a sommare 1 sui
        vicini di un nodo: l'aggregazione diventa una somma pesata a
        scala libera (dipendente dal grado del nodo) invece di una media
        pesata, e il decay agisce come fattore di scala assoluto invece
        che come redistribuzione relativa dell'importanza tra vicini.
        Applicare il softmax DOPO il decay (non prima) e' la scelta
        corretta: cosi' un vicino "vecchio" perde peso relativo a favore
        di uno "recente" nella stessa softmax, che e' l'effetto voluto
        dal meccanismo di decay.

        Se edge_attr e' None (nessuna informazione temporale disponibile
        per questo batch di archi), il decay e' un no-op (1.0 ovunque):
        l'attenzione resta puramente content-based, poi comunque
        softmax-normalizzata.
        """
        cat_ij = torch.cat([x_i, x_j], dim=-1)
        alpha = torch.einsum("ehc,hc->eh", cat_ij, self.att)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if edge_attr is not None:
            time_diff = edge_attr
            exponent = (-self.lambda_decay * time_diff).clamp(
                min=-_MAX_DECAY_EXPONENT, max=_MAX_DECAY_EXPONENT
            )
            decay = torch.exp(exponent).unsqueeze(-1)
            alpha = alpha * decay
            self._decay = decay.detach().cpu()
        else:
            decay = torch.ones(alpha.size(0), 1, device=alpha.device)
            self._decay = decay.detach().cpu()

        return alpha

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        """Message passing con attenzione time-decayed e softmax-normalizzata
        per nodo destinazione (stesso contratto di GATConv standard)."""
        alpha = self.edge_attention(x_i, x_j, edge_attr)
        alpha = pyg_softmax(alpha, index, ptr, size_i)
        if self.dropout > 0 and self.training:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        self._alpha = alpha
        return x_j * alpha.unsqueeze(-1)

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        """Forward pass with optional attention return.

        Gestisce esplicitamente grafi senza archi (0 o 1 nodo, edge_index
        vuoto): propagate() su un edge_index vuoto e' comunque valido in
        PyG e produce output a zero, ma senza questo branch self._alpha
        resterebbe quello dell'ultima chiamata precedente (stato residuo
        tra grafi diversi nello stesso batch), con il rischio di
        restituire un'attenzione non pertinente quando return_attention=True.
        """
        h, c = self.heads, self.out_channels
        x = self.lin(x)
        x = x.view(-1, h, c)

        if edge_index.numel() == 0:
            out = torch.zeros(x.size(0), h, c, device=x.device, dtype=x.dtype)
            self._alpha = torch.zeros(0, h, device=x.device, dtype=x.dtype)
            self._decay = torch.zeros(0, 1)
        else:
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
        use_batch_norm: Apply a normalization layer between hidden GAT
            layers. Kept for backward compatibility: when norm_kind is
            not explicitly set, use_batch_norm=True maps to "batch_norm"
            (the original behaviour) and False maps to "none".
        norm_kind: Explicit choice of normalization ("batch_norm",
            "layer_norm", "graph_norm", "none"). Takes precedence over
            use_batch_norm when provided, mirroring DualGATModel in
            gat_basic.py. layer_norm/graph_norm are generally more stable
            than batch_norm on small/variable-composition graph batches
            (see TrainPipeline.Steps.TuningStep.recommend_norm_kind).
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

        edge_dim = 1
        self.embedding = nn.Embedding(
            num_embeddings=num_embedding_features, embedding_dim=embedding_dims
        )
        # Init esplicita: il default di nn.Embedding (N(0,1)) produce
        # vettori con norma ~sqrt(embedding_dims), grande abbastanza da
        # destabilizzare i primi step di un GAT (che riceve l'embedding
        # come feature di input diretta, senza alcuna normalizzazione
        # a monte). std=0.02 e' lo standard usato per gli embedding nei
        # transformer, un punto di partenza numericamente stabile che
        # lascia comunque piena liberta' di apprendimento.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # --- embed path ---
        self.gat_embed = nn.ModuleList()
        self.res_embed = nn.ModuleList()
        in_dim = embedding_dims
        for _ in range(num_layers):
            self.gat_embed.append(
                TimeAwareGATConv(in_dim, gat_hidden_dim_embed, heads=num_heads, concat=True, edge_dim=edge_dim, lambda_decay=lambda_decay)
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
                TimeAwareGATConv(in_dim, gat_hidden_dim_event, heads=num_heads, concat=True, edge_dim=edge_dim, lambda_decay=lambda_decay)
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
                TimeAwareGATConv(in_dim, gat_hidden_dim_concat, heads=num_heads, concat=True, edge_dim=edge_dim, lambda_decay=lambda_decay)
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

    def _run_path(self, layers, norms, residuals, x, edge_index, edge_attr, return_attention: bool = False):
        """Esegue un path (embed/event/concat) applicando, tra un layer e
        il successivo, uno skip connection residuo: x_out = layer(x) +
        proj(x_in). La proiezione e' Identity se le dimensioni combaciano,
        altrimenti una Linear (vedi _make_residual_proj). Con num_layers=1
        il ciclo esegue una sola iterazione e non applica mai lo skip
        (nessun "layer successivo" a cui sommare), quindi il comportamento
        per la configurazione di default resta identico a prima.
        """
        attn_last = None
        for i, layer in enumerate(layers):
            residual = residuals[i](x)
            if return_attention and i == len(layers) - 1:
                x, attn_last = layer(
                    x, edge_index=edge_index, edge_attr=edge_attr, return_attention=True
                )
            else:
                x = layer(x, edge_index=edge_index, edge_attr=edge_attr)

            if i < len(layers) - 1:
                x = x + residual
                if self.use_norm:
                    x = norms[i](x)
                x = self.activation(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x, attn_last

    def forward(self, data_event, return_attention: bool = False):
        """Forward pass for batched event graphs with optional attention."""
        edge_attr = getattr(data_event, "time", None)
        edge_index = data_event.edge_index

        if self.norm_kind == "graph_norm" and hasattr(data_event, "batch") and data_event.batch is not None:
            self._set_batch_index(data_event.batch)

        x_embed = self.embedding(data_event.event_ids.view(-1))
        x_embed, attn_embed = self._run_path(
            self.gat_embed,
            self.bn_embed,
            self.res_embed,
            x_embed,
            edge_index,
            edge_attr,
            return_attention=return_attention,
        )

        x_event, attn_event = self._run_path(
            self.gat_event,
            self.bn_event,
            self.res_event,
            data_event.x,
            edge_index,
            edge_attr,
            return_attention=return_attention,
        )

        x = torch.cat([x_embed, x_event], dim=1)
        x, attn_final = self._run_path(
            self.gat_concat,
            self.bn_concat,
            self.res_concat,
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
                "time": edge_attr.detach().cpu() if edge_attr is not None else None,
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

                    time_vals = (
                        attn_data["time"][edge_indices]
                        if attn_data.get("time") is not None
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
                        "time": time_vals,
                        "batch": batch_vector[node_indices],
                        "graph_idx": graph_idx,
                    }
                    all_attn_maps.append(graph_attn)

    accuracy = correct / total_tokens if total_tokens else 0.0
    loss = total_loss / total_tokens if total_tokens else 0.0

    if return_attention:
        return loss, accuracy, all_attn_maps
    return loss, accuracy