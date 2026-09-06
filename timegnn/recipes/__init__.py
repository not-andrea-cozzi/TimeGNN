"""High-level training recipes."""

from .gat_basic import train_gat_basic, GATBasicConfig
from .gat_status_emb import train_gat_status_emb, GATStatusEmbConfig
from .gat_time_decay import train_gat_time_decay, GATTimeDecayConfig
from .gat_time_decay_status_emb import (
	train_gat_time_decay_status_emb,
	GATTimeDecayStatusConfig,
)
from .prefix_gcn import train_prefix_gcn, PrefixGCNConfig
from .gat_outcome import train_gat_outcome, GATOutcomeConfig

__all__ = [
	"train_gat_basic",
	"GATBasicConfig",
	"train_gat_status_emb",
	"GATStatusEmbConfig",
	"train_gat_time_decay",
	"GATTimeDecayConfig",
	"train_gat_time_decay_status_emb",
	"GATTimeDecayStatusConfig",
	"train_prefix_gcn",
	"PrefixGCNConfig",
	"train_gat_outcome",
	"GATOutcomeConfig",
]
