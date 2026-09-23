from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class GraphNorm(nn.Module):
    """Normalizza le feature dei nodi usando media/varianza calcolate per
    ciascun grafo del batch (non sull'intero batch di nodi come
    BatchNorm1d, non per singolo nodo come LayerNorm).

    Riferimento concettuale: Cai et al., "GraphNorm: A Principled Approach
    to Accelerating Graph Neural Network Training". Qui usata una
    variante minimale (senza il termine di shift appreso alpha separato
    dalla media, che nel paper originale stabilizza ulteriormente reti
    molto profonde): sufficiente per i num_layers=1-3 tipici di questa
    pipeline, e piu' semplice da mantenere allineata a PyG che cambia
    versione.

    Args:
        num_features: dimensione delle feature dei nodi (ultima dim di x).
        eps: costante di stabilita' numerica nella divisione per std.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, batch_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            x: [N, num_features], nodi concatenati di tutti i grafi del batch.
            batch_index: [N], indice del grafo di appartenenza per ciascun
                nodo (attributo `batch` di torch_geometric.data.Batch). Se
                None, tratta l'intero x come un unico grafo (equivalente a
                LayerNorm su tutto il batch, utile per compatibilita' con
                chiamate fuori da un contesto batched).
        """
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


class GraphAwareNormList(nn.Module):
    """Wrapper che rende GraphNorm compatibile con l'interfaccia usata
    dai modelli esistenti (norms[i](x) posizionale, senza batch_index) —
    i modelli GAT chiamano norms[i](x) dentro _run_path senza passare
    data.batch. Questo wrapper tiene un riferimento al batch_index
    corrente, impostato una volta per forward dal chiamante, cosi' le
    firme di _run_path in gat_basic.py/gat_status_emb.py restano
    invariate (nessuna modifica alla logica di forward esistente).
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.graph_norm = GraphNorm(num_features, eps=eps)
        self._batch_index: Optional[torch.Tensor] = None

    def set_batch_index(self, batch_index: torch.Tensor) -> None:
        self._batch_index = batch_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.graph_norm(x, self._batch_index)


def make_norm_layer(norm_kind: str, num_features: int) -> nn.Module:
    """Factory usata dai modelli invece di istanziare nn.BatchNorm1d
    direttamente.

    Args:
        norm_kind: uno tra:
            "batch_norm"  -> nn.BatchNorm1d (comportamento originale,
                default per non rompere config esistenti).
            "layer_norm"  -> nn.LayerNorm, normalizza per-nodo,
                indipendente dalla composizione del batch. Raccomandato
                per batch_size < 32 (vedi TuningStep.recommend_norm_kind).
            "graph_norm"  -> GraphAwareNormList (vedi sopra), normalizza
                per-grafo usando data.batch. Raccomandato per
                batch_size >= 32.
            "none"        -> nn.Identity, nessuna normalizzazione (per
                use_batch_norm=False esplicito, evita di istanziare e
                includere nell'ottimizzatore parametri mai usati).
        num_features: dimensione delle feature su cui normalizzare.

    Returns:
        Il modulo nn.Module corrispondente.
    """
    key = norm_kind.lower()
    if key == "batch_norm":
        return nn.BatchNorm1d(num_features)
    if key == "layer_norm":
        return nn.LayerNorm(num_features)
    if key == "graph_norm":
        return GraphAwareNormList(num_features)
    if key == "none":
        return nn.Identity()
    raise ValueError(
        f"norm_kind non supportato: '{norm_kind}'. Usa 'batch_norm', "
        f"'layer_norm', 'graph_norm' o 'none'."
    )