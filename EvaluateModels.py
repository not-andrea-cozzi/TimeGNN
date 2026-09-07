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

# Utilità per la gestione dei dati
from timegnn.data.pyg import custom_collate_graph

# Plotter
from Common.EvaluatorPlotter import EvaluatorPlotter

# Costanti derivate da PositionGraphSchema (devono coincidere con quelle del training)
NUM_EVENT_ID_CATEGORIES = 13
NUM_EVENT_FEATURES = 2
MOVE_VOCAB_SIZE = 64 * 64
NUM_EDGE_TYPES = 3
TIME_EDGE_DIM = 1

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
    for key in ["test_data", "checkpoint_basic", "checkpoint_time"]:
        if key not in eval_cfg:
            raise ConfigError(f"Chiave '{key}' mancante in 'evaluation'.")


class SimpleTestDataset(torch.utils.data.Dataset):
    """Dataset semplice che avvolge una lista di campioni (dizionari o tensori)."""
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_model(
    checkpoint_path: str,
    model_class,
    model_params: Dict[str, Any],
    edge_dim: int,
    extra_kwargs: Optional[Dict] = None,
    device: str = "cuda",
) -> nn.Module:
    """Carica un modello dal checkpoint."""
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
        edge_dim=edge_dim,
        num_layers=model_params.get("num_layers", 1),
        dropout=model_params.get("dropout", 0.0),
        use_batch_norm=model_params.get("use_batch_norm", False),
        activation=model_params.get("activation", "elu"),
        **extra_kwargs,
    ).to(device)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
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
    """
    model.eval()
    move_correct_list = []
    mate_correct_list = []
    mate_true_list = []
    mate_pred_list = []
    mate_n_list = []

    with torch.no_grad():
        for batch in dataloader:
            # batch è un oggetto con attributi: x, edge_index, edge_attr, y, (eventualmente mate_n)
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            y = batch.y.to(device)  # target della mossa (classe)

            # Forward pass (con AMP se richiesto)
            if use_amp and device == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(x, edge_index, edge_attr)
            else:
                logits = model(x, edge_index, edge_attr)

            pred = logits.argmax(dim=1)
            correct_move = (pred == y).cpu().numpy()
            move_correct_list.extend(correct_move)

            # Se il batch ha l'attributo mate_n, lo registriamo per eventuali stratificazioni
            if hasattr(batch, "mate_n") and batch.mate_n is not None:
                mate_n = batch.mate_n.cpu().numpy()
                mate_n_list.extend(mate_n)
                mate_true_list.extend(mate_n)
                mate_pred_list.extend(np.zeros_like(mate_n))
                mate_correct_list.extend(np.zeros_like(mate_n, dtype=bool))
            else:
                # Se non c'è mate_n, usiamo array vuoti
                pass

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

    # Caricamento dataset di test
    test_path = eval_cfg["test_data"]
    if not os.path.exists(test_path):
        raise ConfigError(f"File di test non trovato: {test_path}")

    logger.info(f"Caricamento test set da {test_path}...")
    test_data = torch.load(test_path, map_location="cpu")
    if not isinstance(test_data, list):
        # Se è un tensore o un dizionario, proviamo a convertirlo in lista
        logger.warning("Il test set non è una lista; provo a convertirlo in lista di campioni.")
        # Assumiamo che sia un tensore di grafi o un dizionario con chiavi; per semplicità,
        # se è un tensore, lo dividiamo in campioni separati (dimensione batch).
        if isinstance(test_data, torch.Tensor):
            # Se è un tensore 3D (N, features, ...), lo spacchettiamo
            # Ma qui ci aspettiamo una lista di dizionari, quindi meglio sollevare errore.
            raise ConfigError("Il dataset di test deve essere una lista di dizionari, non un tensore.")
        else:
            # Se è un dizionario unico, lo mettiamo in una lista
            test_data = [test_data]

    test_ds = SimpleTestDataset(test_data)
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_cfg.get("batch_size", 64),
        shuffle=False,
        collate_fn=custom_collate_graph,
        num_workers=eval_cfg.get("num_workers", 2),
        persistent_workers=eval_cfg.get("num_workers", 2) > 0,
    )
    logger.info(f"Test set caricato: {len(test_ds)} campioni.")

    # Caricamento dei modelli
    logger.info("Caricamento modello basic...")
    model_basic = load_model(
        eval_cfg["checkpoint_basic"],
        DualGATModel,
        model_params,
        edge_dim=NUM_EDGE_TYPES,
        extra_kwargs={},
        device=device,
    )

    logger.info("Caricamento modello time-aware...")
    model_time = load_model(
        eval_cfg["checkpoint_time"],
        DualGATTimeAwareModel,
        model_params,
        edge_dim=TIME_EDGE_DIM,
        extra_kwargs={"lambda_decay": model_params.get("lambda_decay", 0.01)},
        device=device,
    )

    # Valutazione
    logger.info("Valutazione modello basic...")
    t0 = time.monotonic()
    res_basic = evaluate_model(model_basic, test_loader, device, use_amp)
    logger.info(f"Basic valutato in {time.monotonic() - t0:.2f}s, campioni: {len(res_basic['move_correct'])}")

    logger.info("Valutazione modello time-aware...")
    t0 = time.monotonic()
    res_time = evaluate_model(model_time, test_loader, device, use_amp)
    logger.info(f"Time-aware valutato in {time.monotonic() - t0:.2f}s, campioni: {len(res_time['move_correct'])}")

    # Generazione dei plot e metriche usando EvaluatorPlotter
    plots_dir = eval_cfg.get("plots_dir", "Dataset/Test/plots")
    metrics_dir = eval_cfg.get("metrics_dir", "Dataset/Test/metrics")
    max_n = eval_cfg.get("max_n", 10)

    plotter = EvaluatorPlotter(plots_dir=plots_dir, out_dir=metrics_dir)

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