from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Import delle classi dei modelli
from timegnn.models.gat_basic import DualGATModel
from timegnn.models.gat_time_decay import DualGATTimeAwareModel
from timegnn.data.pyg import custom_collate_graph
from DatasetPipeline.Utils.position_pooling import apply_legal_move_mask

# Dataset shardato (stesso usato da TrainMain.py per train/val/test)
from TrainPipeline.Shard.ShardDataset import ShardedGraphDataset
from Common.sparse_legal_moves import sparse_legal_argmax

# Plotter
from Common.EvaluatorPlotter import EvaluatorPlotter

from DatasetPipeline.Model.ChessConstants import (
    NUM_EVENT_FEATURES,
    NUM_EVENT_ID_CATEGORIES,
    MOVE_VOCAB_SIZE,
    NUM_EDGE_TYPES,
    TIME_EDGE_DIM,
)

logger = logging.getLogger("evaluate_models")


class ConfigError(Exception):
    pass


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise ConfigError(f"File YAML non trovato: {config_path}")
    try:
        import yaml
    except ImportError:
        raise ConfigError("Modulo 'pyyaml' non installato.")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError("Il file YAML deve definire un dizionario.")
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    required = ["evaluation", "model_params"]
    for section in required:
        if section not in cfg:
            raise ConfigError(f"Sezione mancante: '{section}'")
    eval_cfg = cfg["evaluation"]
    for key in ["test_dir", "checkpoint_basic", "checkpoint_time"]:
        if key not in eval_cfg:
            raise ConfigError(f"Chiave '{key}' mancante in 'evaluation'.")


def _require_sharded_dir(path: str, label: str) -> None:
    """Verifica che `path` sia una cartella shardata con manifest.json.

    Stessa funzione di TrainMain.py: il test set qui e' shardato
    (ShardedGraphDataset), non un singolo file .pt caricabile con
    torch.load — quel path andava bene per un test set piccolo/legacy,
    ma non per la nuova struttura a shard usata da tutta la pipeline.
    """
    if not os.path.isdir(path):
        raise ConfigError(f"{label}: cartella non trovata: {path}")
    manifest = os.path.join(path, "manifest.json")
    if not os.path.exists(manifest):
        raise ConfigError(f"{label}: manifest.json non trovato in {path}")


def build_dataloader(
    dataset,
    section: Dict[str, Any],
    collate_fn,
    shuffle: bool,
    device: str = "cpu",
) -> DataLoader:
    """Identica a build_dataloader in TrainMain.py, duplicata qui per
    mantenere questo script eseguibile in modo standalone."""
    num_workers = int(section.get("num_workers", 0))
    persistent = bool(section.get("persistent_workers", False)) and num_workers > 0
    prefetch = int(section.get("prefetch_factor", 4)) if num_workers > 0 else None
    pin_memory = bool(section.get("pin_memory", device == "cuda"))

    kwargs = dict(
        dataset=dataset,
        batch_size=section["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    if prefetch is not None:
        kwargs["prefetch_factor"] = prefetch

    return DataLoader(**kwargs)


def load_model(
    checkpoint_path: str,
    model_class,
    model_params: Dict[str, Any],
    extra_kwargs: Optional[Dict] = None,
    device: str = "cuda",
) -> nn.Module:
    """Carica un modello dal checkpoint.

    NOTA su edge_dim: e' un parametro reale del costruttore di
    DualGATModel (default 1), MA NON di DualGATTimeAwareModel, che lo
    fissa internamente a 1 (vedi gat_time_decay.py). Per questo edge_dim
    non e' tra i parametri comuni letti da model_params qui sotto: va
    passato esplicitamente via extra_kwargs solo per il modello basic,
    con lo stesso valore usato in training (run_training/TrainMain.py
    passa edge_dim=NUM_EDGE_TYPES per "basic"). Se omesso, il modello
    basic viene ricostruito con edge_dim=1 di default e load_state_dict
    fallisce con size mismatch su gat_*.lin_edge.weight ([*, N] nel
    checkpoint vs [*, 1] nel modello ricostruito).

    norm_kind viene invece letto da model_params cosi' l'architettura
    ricostruita in valutazione combacia con quella usata in training
    (run_training passa norm_kind = tuning_meta["recommended_norm"]):
    se qui restasse None/"none" mentre il checkpoint e' stato allenato
    con, ad es., "graph_norm", i moduli di normalizzazione nel modello
    ricostruito sarebbero nn.Identity invece di GraphAwareNormList/
    LayerNorm, e load_state_dict fallirebbe (shape/keys mismatch) o
    caricherebbe pesi fuori posto.
    """
    if extra_kwargs is None:
        extra_kwargs = {}

    model = model_class(
        num_event_features=NUM_EVENT_FEATURES,
        num_embedding_features=NUM_EVENT_ID_CATEGORIES,
        embedding_dims=model_params.get("embedding_dims", 64),
        gat_hidden_dim_event=model_params.get("gat_hidden_dim_event", 32),
        gat_hidden_dim_embed=model_params.get("gat_hidden_dim_embed", 128),
        gat_hidden_dim_concat=model_params.get("gat_hidden_dim_concat", 128),
        output_dim=MOVE_VOCAB_SIZE,
        num_heads=model_params.get("num_heads", 4),
        num_layers=model_params.get("num_layers", 1),
        dropout=model_params.get("dropout", 0.0),
        use_batch_norm=model_params.get("use_batch_norm", False),
        activation=model_params.get("activation", "elu"),
        norm_kind=model_params.get("norm_kind"),
        **extra_kwargs,
    ).to(device)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Il checkpoint può contenere l'intero stato del training (con optimizer, ecc.)
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        logger.info(f"Modello caricato da {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint non trovato: {checkpoint_path}, uso modello non addestrato.")
    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    use_amp: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Valuta un modello sul dataloader.
    Restituisce un dizionario con:
        - move_correct: array booleano per accuratezza della mossa
        - mate_correct: array booleano per accuratezza della profondità (se disponibile)
        - mate_true: array di profondità reali
        - mate_pred: array di profondità predette (placeholder se non disponibile)
        - mate_n: array di profondità reali (per stratificazione)

    NOTA: entrambi i modelli (DualGATModel, DualGATTimeAwareModel) hanno
    pool_before_head=True di default e restituiscono direttamente
    graph_logits [B, output_dim] da model(batch_event). Non va richiamato
    pool_node_logits/global_mean_pool sull'output: farlo di nuovo qui
    causerebbe lo stesso RuntimeError da doppio pooling gia' risolto in
    Loop.py/TrainMain.py ("Expected index [...] to be no larger than
    self [...]").

    Usa sparse_legal_argmax (stessa funzione di TrainMain.py) quando la
    maschera delle mosse legali e' disponibile, invece di un argmax
    seguito da un mascheramento manuale via apply_legal_move_mask: le due
    strade sono equivalenti nel risultato finale, ma sparse_legal_argmax
    e' la funzione gia' validata e usata nel resto della pipeline.
    """
    model.eval()
    move_correct_list = []
    mate_correct_list = []
    mate_true_list = []
    mate_pred_list = []
    mate_n_list = []

    with torch.no_grad():
        for batch_event, labels in dataloader:
            batch_event = batch_event.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp and device == "cuda":
                with torch.autocast(device_type="cuda"):
                    graph_logits = model(batch_event)
            else:
                graph_logits = model(batch_event)

            if hasattr(batch_event, "legal_move_mask") and batch_event.legal_move_mask is not None:
                pred = sparse_legal_argmax(graph_logits, batch_event.legal_move_mask)
            else:
                pred = graph_logits.argmax(dim=1)

            correct_move = (pred == labels).cpu().numpy()
            move_correct_list.extend(correct_move)

            if hasattr(batch_event, "position_mate_n") and batch_event.position_mate_n is not None:
                mate_n = batch_event.position_mate_n.cpu().numpy()
                mate_n_list.extend(mate_n)
                mate_true_list.extend(mate_n)
                mate_pred_list.extend(np.zeros_like(mate_n))
                mate_correct_list.extend(np.zeros_like(mate_n, dtype=bool))

            del batch_event, labels, graph_logits, pred

    results = {
        "move_correct": np.array(move_correct_list),
        "mate_correct": np.array(mate_correct_list) if mate_correct_list else np.array([]),
        "mate_true": np.array(mate_true_list) if mate_true_list else np.array([]),
        "mate_pred": np.array(mate_pred_list) if mate_pred_list else np.array([]),
        "mate_n": np.array(mate_n_list) if mate_n_list else np.array([]),
    }
    return results


def main(config_path: str = "Yaml/evaluate_models.yaml") -> None:
    cfg = load_yaml_config(config_path)
    validate_config(cfg)

    eval_cfg = cfg["evaluation"]
    model_params = cfg["model_params"]

    # Logging
    log_level = eval_cfg.get("log_level", "INFO")
    log_file = eval_cfg.get("log_file")
    setup_logging(log_level, log_file)

    logger.info("=" * 60)
    logger.info("AVVIO VALUTAZIONE MODELLI TIMEGNN")
    logger.info("=" * 60)

    # Device
    device_str = eval_cfg.get("device", "auto")
    if device_str == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str
    use_amp = eval_cfg.get("use_amp", False) and device == "cuda"
    logger.info(f"Device: {device}, AMP: {use_amp}")

    # Caricamento dataset di test (shardato, stessa struttura di train/val)
    test_dir = eval_cfg["test_dir"]
    _require_sharded_dir(test_dir, "test")

    logger.info(f"Caricamento test set shardato da {test_dir}...")
    test_ds = ShardedGraphDataset(test_dir, shuffle=False, seed=eval_cfg.get("seed", 42))
    test_loader = build_dataloader(
        test_ds, eval_cfg, custom_collate_graph, shuffle=False, device=device
    )
    logger.info(
        f"Test set: {len(test_ds):,} campioni in {len(test_loader)} batch | "
        f"num_workers={eval_cfg.get('num_workers', 0)} | "
        f"pin_memory={test_loader.pin_memory}"
    )

    # Caricamento dei modelli
    logger.info("Caricamento modello basic...")
    model_basic = load_model(
        eval_cfg["checkpoint_basic"],
        DualGATModel,
        model_params,
        # edge_dim e' un parametro reale del costruttore di DualGATModel
        # (default 1), a differenza di DualGATTimeAwareModel dove edge_dim
        # e' fissato internamente a 1. Il checkpoint qui e' stato allenato
        # con NUM_EDGE_TYPES=3 (vedi run_training/TrainMain.py, che passa
        # edge_dim=NUM_EDGE_TYPES per il modello basic): senza specificarlo
        # qui, load_state_dict fallisce per size mismatch su gat_*.lin_edge.weight
        # (shape [*, 3] nel checkpoint vs [*, 1] nel modello ricostruito col default).
        extra_kwargs={"edge_dim": NUM_EDGE_TYPES},
        device=device,
    )

    logger.info("Caricamento modello time-aware...")
    model_time = load_model(
        eval_cfg["checkpoint_time"],
        DualGATTimeAwareModel,
        model_params,
        extra_kwargs={"lambda_decay": model_params.get("lambda_decay", 0.01)},
        device=device,
    )

    # Valutazione
    logger.info("Valutazione modello basic...")
    t0 = time.monotonic()
    res_basic = evaluate_model(model_basic, test_loader, device, use_amp)
    logger.info(f"Basic valutato in {time.monotonic() - t0:.2f}s, campioni: {len(res_basic['move_correct'])}")

    # skip_time_aware: stesso meccanismo di TrainMain.py/evaluate_models().
    # Il try/except sotto e' comunque il guard primario: se la valutazione
    # time_aware fallisce per qualsiasi motivo (es. incompatibilita' di
    # propagate() con la versione di torch_geometric installata), il
    # basic gia' valutato non va perso e lo script prosegue generando solo
    # i grafici/metriche disponibili, invece di crashare tutto lo script.
    skip_time_aware = bool(eval_cfg.get("skip_time_aware", False))
    res_time = None
    if not skip_time_aware:
        logger.info("Valutazione modello time-aware...")
        t0 = time.monotonic()
        try:
            res_time = evaluate_model(model_time, test_loader, device, use_amp)
            logger.info(
                f"Time-aware valutato in {time.monotonic() - t0:.2f}s, "
                f"campioni: {len(res_time['move_correct'])}"
            )
        except TypeError as e:
            logger.error(
                f"[eval] Valutazione del modello time_aware saltata: {e}. "
                f"TimeAwareGATConv.propagate() non e' compatibile con la versione di "
                f"torch_geometric installata su questa macchina. Imposta "
                f"'evaluation.skip_time_aware: true' nello YAML per saltarla "
                f"esplicitamente senza passare da qui."
            )
    else:
        logger.info("Valutazione del modello time_aware saltata (skip_time_aware=true).")

    # Generazione dei plot e metriche usando EvaluatorPlotter
    plots_dir = eval_cfg.get("plots_dir", "Dataset/Test/plots")
    metrics_dir = eval_cfg.get("metrics_dir", "Dataset/Test/metrics")
    max_n = eval_cfg.get("max_n", 10)

    plotter = EvaluatorPlotter(plots_dir=plots_dir, out_dir=metrics_dir)

    if res_time is not None:
        # Plot a barre per profondità
        plotter.plot_depth_bars(res_time, res_basic, max_n=max_n, filename="bars_per_n.png")
        # Curve di accuracy
        plotter.plot_depth_curves(res_time, res_basic, max_n=max_n, filename="curves_per_n.png")
        # CSV con metriche per n
        plotter.save_depth_metrics(res_time, res_basic, max_n=max_n, filename="metrics_per_n.csv")
        # Barre aggregate globali
        plotter.plot_aggregate_bars(res_time, res_basic, filename="aggregate_bars.png")

        # Matrici di confusione per la profondità (se disponibili)
        if len(res_time.get("mate_true", [])) > 0:
            plotter.plot_confusion_matrix(
                res_time["mate_true"],
                res_time["mate_pred"],
                num_classes=max_n + 1,
                title="Confusion matrix - Timed model",
                filename="cm_timed.png"
            )
            plotter.plot_confusion_matrix(
                res_basic["mate_true"],
                res_basic["mate_pred"],
                num_classes=max_n + 1,
                title="Confusion matrix - Basic model",
                filename="cm_untimed.png"
            )
        else:
            logger.warning("Nessun dato di profondità (mate_n) disponibile nel test set; matrici di confusione non generate.")
    else:
        logger.warning(
            "Solo il modello basic e' stato valutato "
            f"(move accuracy = {np.mean(res_basic['move_correct']):.4f}). "
            "Grafici comparativi basic/time_aware non generati."
        )
        os.makedirs(plots_dir, exist_ok=True)
        with open(os.path.join(plots_dir, "bars_per_n.png"), "wb"):
            pass

    # Opzionalmente salvare i risultati in un file pickle per analisi successive
    results_file = os.path.join(metrics_dir, "evaluation_results.pkl")
    torch.save({"timed": res_time, "untimed": res_basic}, results_file)
    logger.info(f"Risultati salvati in {results_file}")

    logger.info("=" * 60)
    logger.info("VALUTAZIONE COMPLETATA.")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Valutazione standalone dei modelli TimeGNN.")
    parser.add_argument("--config", default="Yaml/evaluate_models.yaml", help="Percorso del file YAML di configurazione.")
    args = parser.parse_args()
    main(args.config)