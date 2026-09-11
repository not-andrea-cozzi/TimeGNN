from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

import torch.multiprocessing

# ----------------------------------------------------------------------
# Stato pipeline
# ----------------------------------------------------------------------
from DatasetPipeline.PipelineState import PipelineState, file_ready
from TrainPipeline.Shard.Sharding import shard_split
from TrainPipeline.Training.State import TrainState
from TrainPipeline.Training.Loop import train_epoch, evaluate_epoch
from TrainPipeline.Shard.ShardDataset import ShardedGraphDataset
from timegnn.models.gat_basic import DualGATModel
from timegnn.models.gat_time_decay import DualGATTimeAwareModel
from timegnn.data.pyg import custom_collate_graph
from timegnn.train.early_stopping import EarlyStopping
from Common.EvaluatorPlotter import EvaluatorPlotter
from TrainPipeline.CleanDataset import clean_file
from DatasetPipeline.Utils.position_pooling import pool_node_logits, apply_legal_move_mask

# ----------------------------------------------------------------------
# Costanti
# ----------------------------------------------------------------------
from DatasetPipeline.Model.ChessConstants import (
    NUM_EVENT_FEATURES,
    NUM_EVENT_ID_CATEGORIES,
    MOVE_VOCAB_SIZE,
    NUM_EDGE_TYPES,
    TIME_EDGE_DIM,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helper memoria
# ----------------------------------------------------------------------
def free_memory(verbose: bool = False) -> None:
    """Forza garbage collection e svuota la cache CUDA."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if verbose:
        try:
            import psutil
            rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
            logger.debug(f"RAM processo dopo free_memory: {rss:.2f} GB")
        except ImportError:
            pass


def apply_memory_limit(max_ram_gb: Optional[float]) -> None:
    """
    Limita la memoria virtuale del processo.
    ATTENZIONE: può interferire con CUDA. Usare solo se necessario.
    """
    if max_ram_gb is None or max_ram_gb <= 0:
        return
    try:
        import resource
        limit_bytes = int(max_ram_gb * 1024**3)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_hard = limit_bytes if hard == resource.RLIM_INFINITY else min(limit_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, new_hard))
        logger.info(f"Limite RAM impostato a {max_ram_gb} GB (RLIMIT_AS).")
    except (ImportError, ValueError, OSError) as e:
        logger.warning(f"Impossibile impostare il limite RAM: {e}")


# ----------------------------------------------------------------------
# Eccezioni e helper
# ----------------------------------------------------------------------
class PipelineConfigError(Exception):
    pass


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,
    backup_count: int = 3,
) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise PipelineConfigError(f"File YAML non trovato: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise PipelineConfigError("Il file YAML deve definire un dizionario.")
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    required = ["pipeline", "clean", "shard", "train_basic", "train_time_aware", "evaluate"]
    for section in required:
        if section not in cfg:
            raise PipelineConfigError(f"Sezione mancante: '{section}'.")


def run_step(state: PipelineState, step_name: str, is_ready_fn, do_fn) -> None:
    """Esegue uno step della pipeline con logging di stato e tempi."""
    if state.is_done(step_name) and is_ready_fn():
        logger.info(f"[SKIP] Step '{step_name}' già completato.")
        return
    if state.is_done(step_name) and not is_ready_fn():
        logger.warning(f"[REDO] Step '{step_name}' marcato ma output mancante.")
    logger.info(f"[RUN] Avvio step '{step_name}'...")
    t0 = time.monotonic()
    try:
        do_fn()
    except Exception as e:
        state.mark_failed(step_name, str(e))
        logger.error(f"[FAILED] Step '{step_name}': {e}", exc_info=True)
        raise
    finally:
        # Libera sempre la memoria dopo ogni step
        free_memory()
    elapsed = time.monotonic() - t0
    state.mark_done(step_name)
    logger.info(f"[DONE] Step '{step_name}' in {elapsed:.2f}s.")


def log_config(cfg: Dict[str, Any], heading: str = "Configurazione") -> None:
    logger.info("=" * 70)
    logger.info(f"{heading}:")
    logger.info("=" * 70)
    for section, values in cfg.items():
        logger.info(f"[{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"  {values}")
    logger.info("=" * 70)


# ----------------------------------------------------------------------
# Helper per costruire DataLoader in modo sicuro
# ----------------------------------------------------------------------
def build_dataloader(dataset, section: Dict[str, Any], collate_fn, shuffle: bool) -> DataLoader:
    """
    Costruisce un DataLoader con parametri sicuri rispetto alla RAM.
    - num_workers: da config (default 0 = nessun worker, nessuna shared memory)
    - persistent_workers: solo se num_workers > 0
    - prefetch_factor: solo se num_workers > 0
    - pin_memory: da config (default False)
    """
    num_workers = int(section.get("num_workers", 0))
    persistent = bool(section.get("persistent_workers", False)) and num_workers > 0
    prefetch = section.get("prefetch_factor", 2) if num_workers > 0 else None
    pin_memory = bool(section.get("pin_memory", False))

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


# ----------------------------------------------------------------------
# Training (basic / time-aware)
# ----------------------------------------------------------------------
def run_training(
    cfg: Dict[str, Any],
    model_type: str,
    shards_dir: str,
    checkpoint_base: str,
    device: str,
    use_amp: bool,
) -> None:
    section = cfg["train_basic"] if model_type == "basic" else cfg["train_time_aware"]
    logger.info(f"Avvio training {model_type} con configurazione: {section}")

    # 1. Modello e dati ----------------------------------------------------
    if model_type == "basic":
        model_class = DualGATModel
        edge_dim = NUM_EDGE_TYPES
        extra_kwargs: Dict[str, Any] = {}
    else:
        model_class = DualGATTimeAwareModel
        edge_dim = TIME_EDGE_DIM
        extra_kwargs = {"lambda_decay": float(section.get("lambda_decay", 0.01))}

    train_dir = os.path.join(shards_dir, "train")
    val_dir = os.path.join(shards_dir, "val")

    train_ds = ShardedGraphDataset(train_dir, shuffle=True, seed=section.get("seed", 42))
    val_ds = ShardedGraphDataset(val_dir, shuffle=False, seed=section.get("seed", 42))

    train_loader = build_dataloader(train_ds, section, custom_collate_graph, shuffle=False)
    val_loader = build_dataloader(val_ds, section, custom_collate_graph, shuffle=False)

    logger.info(
        f"Train: {len(train_ds):,} samples in {len(train_loader)} batches | "
        f"Val: {len(val_ds):,} samples in {len(val_loader)} batches | "
        f"num_workers={section.get('num_workers', 0)}"
    )

    model = model_class(
        num_event_features=NUM_EVENT_FEATURES,
        num_embedding_features=NUM_EVENT_ID_CATEGORIES,
        embedding_dims=section.get("embedding_dims", 64),
        gat_hidden_dim_event=section.get("gat_hidden_dim_event", 32),
        gat_hidden_dim_embed=section.get("gat_hidden_dim_embed", 128),
        gat_hidden_dim_concat=section.get("gat_hidden_dim_concat", 128),
        output_dim=MOVE_VOCAB_SIZE,
        num_heads=section.get("num_heads", 4),
        edge_dim=edge_dim,
        num_layers=section.get("num_layers", 1),
        dropout=section.get("dropout", 0.0),
        use_batch_norm=section.get("use_batch_norm", False),
        activation=section.get("activation", "elu"),
        **extra_kwargs,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(section.get("lr", 1e-3)),
        weight_decay=float(section.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device == "cuda" else None

    # 2. Checkpoint --------------------------------------------------------
    base_dir = os.path.dirname(checkpoint_base) or "."
    base_name = os.path.basename(checkpoint_base)
    if base_name.endswith(".pt"):
        base_name = base_name[:-3]
    last_path = os.path.join(base_dir, f"{base_name}_last.pt")
    best_path = os.path.join(base_dir, f"{base_name}_best.pt")

    resume_path = None
    if os.path.exists(last_path):
        resume_path = last_path
        logger.info(f"Ripresa da checkpoint last: {last_path}")
    elif os.path.exists(checkpoint_base):
        resume_path = checkpoint_base
        logger.info(f"Ripresa da checkpoint base: {checkpoint_base}")
    else:
        logger.info("Nessun checkpoint esistente, partenza da zero.")

    train_state = TrainState(checkpoint_path=resume_path)
    if resume_path:
        train_state.try_resume(model, optimizer, scaler, map_location=device)
        logger.info(
            f"Checkpoint caricato: epoca {train_state.epoch}, "
            f"best_val_loss={train_state.best_val_loss:.4f}"
        )

    early_stopping = EarlyStopping(patience=section.get("patience", 5))
    epochs = section.get("epochs", 20)
    best_val_loss = train_state.best_val_loss

    # 3. Loop di training --------------------------------------------------
    try:
        for epoch in range(train_state.epoch, epochs):
            train_ds.set_epoch(epoch)

            t0 = time.monotonic()

            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler=scaler,
                train_state=None,
                checkpoint_every=None,
                use_amp=use_amp,
                total_items=len(train_ds),
                epoch_label=f"Epoch {epoch+1}/{epochs} [train]",
            )

            val_loss, val_top1, val_top3 = evaluate_epoch(
                model,
                val_loader,
                criterion,
                device,
                use_amp=use_amp,
                total_items=len(val_ds),
                epoch_label=f"Epoch {epoch+1}/{epochs} [val]",
            )

            elapsed = time.monotonic() - t0

            logger.info(
                f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_top1={val_top1:.4f} val_top3={val_top3:.4f}"
            )

            if device == "cuda":
                mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                logger.debug(
                    f"GPU memoria: allocata {mem_alloc:.2f} GB, riservata {mem_reserved:.2f} GB"
                )

            train_state.epoch = epoch + 1
            train_state.history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top3": val_top3,
            })

            train_state.save(model, optimizer, scaler, checkpoint_path=last_path)
            logger.debug(f"Last checkpoint salvato: {last_path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                train_state.best_val_loss = best_val_loss
                train_state.save(model, optimizer, scaler, checkpoint_path=best_path)
                logger.info(
                    f"Nuovo best checkpoint: {best_path} (val_loss={best_val_loss:.4f})"
                )

            early_stopping(val_loss)
            if early_stopping.early_stop:
                logger.info(f"Early stopping attivato all'epoca {epoch+1}.")
                if os.path.exists(best_path):
                    logger.info(f"Caricamento del best modello da {best_path}")
                    best_state = torch.load(best_path, map_location=device)
                    model.load_state_dict(best_state["model_state_dict"])
                    del best_state
                break

            free_memory()

        logger.info(
            f"Training {model_type} completato. Best val loss: {best_val_loss:.4f}"
        )
    finally:
        # Libera esplicitamente dataset, loader, modello e ottimizzatore
        del train_loader, val_loader, train_ds, val_ds
        del model, optimizer, criterion, scaler, train_state, early_stopping
        free_memory(verbose=True)


def evaluate_models(cfg: Dict[str, Any], device: str, use_amp: bool) -> None:
    eval_cfg = cfg["evaluate"]
    if not eval_cfg.get("enabled", True):
        logger.info("Valutazione disabilitata.")
        return

    test_path = eval_cfg["test_data"]
    if not os.path.exists(test_path):
        logger.warning(f"Test set non trovato: {test_path}. Salto la valutazione.")
        return

    logger.info(f"Caricamento test set da {test_path}")
    test_data = torch.load(test_path, map_location="cpu", weights_only=False)

    class SimpleTestDataset(torch.utils.data.Dataset):
        def __init__(self, data_list):
            self.data = data_list

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    test_ds = SimpleTestDataset(test_data)
    test_loader = build_dataloader(test_ds, eval_cfg, custom_collate_graph, shuffle=False)
    logger.info(
        f"Test set: {len(test_ds):,} samples in {len(test_loader)} batches | "
        f"num_workers={eval_cfg.get('num_workers', 0)}"
    )

    def load_model(checkpoint_path, model_class, edge_dim, extra_kwargs):
        logger.debug(f"Caricamento modello da {checkpoint_path}")
        model = model_class(
            num_event_features=NUM_EVENT_FEATURES,
            num_embedding_features=NUM_EVENT_ID_CATEGORIES,
            embedding_dims=cfg["train_basic"].get("embedding_dims", 64),
            gat_hidden_dim_event=cfg["train_basic"].get("gat_hidden_dim_event", 32),
            gat_hidden_dim_embed=cfg["train_basic"].get("gat_hidden_dim_embed", 128),
            gat_hidden_dim_concat=cfg["train_basic"].get("gat_hidden_dim_concat", 128),
            output_dim=MOVE_VOCAB_SIZE,
            num_heads=cfg["train_basic"].get("num_heads", 4),
            edge_dim=edge_dim,
            num_layers=cfg["train_basic"].get("num_layers", 1),
            dropout=cfg["train_basic"].get("dropout", 0.0),
            use_batch_norm=cfg["train_basic"].get("use_batch_norm", False),
            activation=cfg["train_basic"].get("activation", "elu"),
            **extra_kwargs,
        ).to(device)
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)
            del state_dict
            logger.info(f"Modello caricato da {checkpoint_path}")
        else:
            logger.warning(
                f"Checkpoint non trovato: {checkpoint_path}, uso modello non addestrato."
            )
        return model

    model_basic = load_model(
        eval_cfg["model_basic_checkpoint"], DualGATModel, NUM_EDGE_TYPES, {}
    )
    model_basic.eval()

    model_time = load_model(
        eval_cfg["model_time_aware_checkpoint"],
        DualGATTimeAwareModel,
        TIME_EDGE_DIM,
        {"lambda_decay": cfg["train_time_aware"].get("lambda_decay", 0.01)},
    )
    model_time.eval()

    def evaluate_model(model, loader, name: str):
        logger.info(f"Valutazione del modello {name}...")
        move_correct_list = []
        mate_correct_list = []
        mate_true_list = []
        mate_pred_list = []
        mate_n_list = []

        with torch.no_grad():
            for batch_idx, (batch_event, labels) in enumerate(loader):
                batch_event = batch_event.to(device)
                labels = labels.to(device)

                node_logits = model(batch_event)
                graph_logits = pool_node_logits(node_logits, batch_event.batch)

                if hasattr(batch_event, "legal_move_mask") and batch_event.legal_move_mask is not None:
                    graph_logits = apply_legal_move_mask(graph_logits, batch_event.legal_move_mask)

                pred = graph_logits.argmax(dim=1)
                correct_move = (pred == labels).cpu().numpy()
                move_correct_list.extend(correct_move)

                if hasattr(batch_event, "position_mate_n") and batch_event.position_mate_n is not None:
                    mate_n = batch_event.position_mate_n.cpu().numpy()
                    mate_true_list.extend(mate_n)
                    mate_pred_list.extend(np.zeros_like(mate_n))
                    mate_correct_list.extend(np.zeros_like(mate_n, dtype=bool))
                    mate_n_list.extend(mate_n)

                if batch_idx % 10 == 0:
                    logger.debug(f"  batch {batch_idx+1}/{len(loader)} processato")

                del batch_event, labels, node_logits, graph_logits, pred

        results = {
            "move_correct": np.array(move_correct_list),
            "mate_correct": np.array(mate_correct_list) if mate_correct_list else np.array([]),
            "mate_true": np.array(mate_true_list) if mate_true_list else np.array([]),
            "mate_pred": np.array(mate_pred_list) if mate_pred_list else np.array([]),
            "mate_n": np.array(mate_n_list) if mate_n_list else np.array([]),
        }
        logger.info(
            f"Modello {name}: move accuracy = {np.mean(move_correct_list):.4f}"
        )
        return results

    try:
        res_basic = evaluate_model(model_basic, test_loader, "basic")
        res_time = evaluate_model(model_time, test_loader, "time_aware")

        plotter = EvaluatorPlotter(
            plots_dir=eval_cfg["plots_dir"], out_dir=eval_cfg["out_dir"]
        )
        max_n = eval_cfg.get("max_n", 5)

        logger.info("Generazione dei grafici di valutazione...")
        plotter.plot_depth_bars(res_time, res_basic, max_n=max_n, filename="bars_per_n.png")
        plotter.plot_depth_curves(res_time, res_basic, max_n=max_n, filename="curves_per_n.png")
        plotter.save_depth_metrics(res_time, res_basic, max_n=max_n, filename="metrics_per_n.csv")
        plotter.plot_aggregate_bars(res_time, res_basic, filename="aggregate_bars.png")

        if len(res_time.get("mate_true", [])) > 0:
            plotter.plot_confusion_matrix(
                res_time["mate_true"],
                res_time["mate_pred"],
                num_classes=max_n + 1,
                title="Confusion matrix - Timed",
                filename="cm_timed.png",
            )
            plotter.plot_confusion_matrix(
                res_basic["mate_true"],
                res_basic["mate_pred"],
                num_classes=max_n + 1,
                title="Confusion matrix - Untimed",
                filename="cm_untimed.png",
            )

        logger.info(
            f"Valutazione completata. Output salvati in {eval_cfg['plots_dir']} e {eval_cfg['out_dir']}"
        )
    finally:
        del test_loader, test_ds, test_data
        del model_basic, model_time
        free_memory(verbose=True)
        
        
# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(config_path: str = "Yaml/train_main.yaml") -> None:
    cfg = load_yaml_config(config_path)
    validate_config(cfg)

    pipe_cfg = cfg["pipeline"]
    setup_logging(pipe_cfg.get("log_level", "INFO"), pipe_cfg.get("log_file"))

    log_config(cfg, "Configurazione pipeline")

    logger.info("=" * 70)
    logger.info("AVVIO PIPELINE TRAINING TIMEGNN")
    logger.info("=" * 70)

    # --- Limite RAM opzionale (configurabile da YAML) ---
    apply_memory_limit(pipe_cfg.get("max_ram_gb", None))

    # --- Strategia di condivisione PyTorch (configurabile) ---
    # 'file_descriptor' (default PyTorch) usa fd (richiede ulimit -n alto)
    # 'file_system' usa /dev/shm (RAM!) -> può saturare la RAM
    sharing = pipe_cfg.get("sharing_strategy", "file_descriptor")
    if sharing not in ("file_descriptor", "file_system"):
        logger.warning(
            f"sharing_strategy '{sharing}' non valida, uso 'file_descriptor'."
        )
        sharing = "file_descriptor"
    try:
        torch.multiprocessing.set_sharing_strategy(sharing)
        logger.info(f"Strategia condivisione PyTorch: {sharing}")
    except RuntimeError as e:
        logger.warning(f"Impossibile impostare sharing strategy '{sharing}': {e}")

    step_filter = pipe_cfg.get("step")
    valid_steps = ["clean", "shard", "train_basic", "train_time_aware", "evaluate"]
    if step_filter is not None and step_filter not in valid_steps:
        raise PipelineConfigError(f"'pipeline.step' non valido: {step_filter}")

    dataset_dir = pipe_cfg.get("dataset_dir", "Dataset")
    shards_dir = os.path.join(
        dataset_dir, pipe_cfg.get("shards_subfolder", "Train/shards")
    )
    checkpoints_dir = os.path.join(
        dataset_dir, pipe_cfg.get("checkpoints_subfolder", "Train/checkpoints")
    )
    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    state_file = pipe_cfg.get("state_file", "train_pipeline_state.json")
    state_path = os.path.join(dataset_dir, state_file)

    if pipe_cfg.get("force_recompute", False) and os.path.exists(state_path):
        logger.warning(f"Rimozione stato precedente ({state_path})")
        os.remove(state_path)

    state = PipelineState(state_path)
    logger.info(f"Stato caricato da {state_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (not pipe_cfg.get("no_amp", False)) and device == "cuda"
    logger.info(f"Device: {device}, AMP: {use_amp}")

    # ------------------------------------------------------------------
    # STEP 0: Clean dataset
    # ------------------------------------------------------------------
    if step_filter is None or step_filter == "clean":
        clean_cfg = cfg.get("clean", {})
        if clean_cfg.get("enabled", True):
            logger.info("-" * 70)
            logger.info("STEP 0/4: clean (rimozione campi superflui)")
            logger.info("-" * 70)

            in_train = clean_cfg["input_train"]
            in_val = clean_cfg["input_val"]
            in_test = clean_cfg.get("input_test", None)
            out_train = clean_cfg["output_train"]
            out_val = clean_cfg["output_val"]
            out_test = clean_cfg.get("output_test", None)
            workers = int(clean_cfg.get("workers", 1))

            for f in [in_train, in_val] + ([in_test] if in_test else []):
                if not os.path.exists(f):
                    raise PipelineConfigError(f"File di input non trovato: {f}")

            def _is_clean_ready():
                ready = file_ready(out_train) and file_ready(out_val)
                if out_test:
                    ready = ready and file_ready(out_test)
                return ready

            def _do_clean():
                logger.info(f"Pulizia train: {in_train} -> {out_train}")
                clean_file(in_train, out_train, workers)
                free_memory()
                logger.info(f"Pulizia val: {in_val} -> {out_val}")
                clean_file(in_val, out_val, workers)
                free_memory()
                if in_test and out_test:
                    logger.info(f"Pulizia test: {in_test} -> {out_test}")
                    clean_file(in_test, out_test, workers)
                    free_memory()

            run_step(state, "clean", _is_clean_ready, _do_clean)
        else:
            logger.info("clean disabilitato.")

    # ------------------------------------------------------------------
    # STEP 1: Sharding
    # ------------------------------------------------------------------
    if step_filter is None or step_filter == "shard":
        shard_cfg = cfg["shard"]
        train_clean = shard_cfg.get("train_clean", "Dataset/Train/train_clean.pt")
        val_clean = shard_cfg.get("val_clean", "Dataset/Train/val_clean.pt")
        shard_size = shard_cfg.get("shard_size", 8000)
        train_shard_dir = os.path.join(shards_dir, "train")
        val_shard_dir = os.path.join(shards_dir, "val")

        def _is_shard_ready():
            return file_ready(os.path.join(train_shard_dir, "manifest.json")) and file_ready(
                os.path.join(val_shard_dir, "manifest.json")
            )

        def _do_shard():
            logger.info(
                f"Sharding train: {train_clean} -> {train_shard_dir} con shard_size={shard_size}"
            )
            shard_split(train_clean, train_shard_dir, shard_size)
            free_memory()
            logger.info(f"Sharding val: {val_clean} -> {val_shard_dir}")
            shard_split(val_clean, val_shard_dir, shard_size)
            free_memory()

        run_step(state, "shard", _is_shard_ready, _do_shard)

    # ------------------------------------------------------------------
    # STEP 2: Train Basic
    # ------------------------------------------------------------------
    if step_filter is None or step_filter == "train_basic":
        basic_cfg = cfg["train_basic"]
        if basic_cfg.get("enabled", True):
            checkpoint = basic_cfg.get("checkpoint", os.path.join(checkpoints_dir, "basic.pt"))
            shards = basic_cfg.get("shards_dir", shards_dir)

            def _is_basic_ready():
                return os.path.exists(checkpoint) and os.path.getsize(checkpoint) > 0

            def _do_basic():
                run_training(cfg, "basic", shards, checkpoint, device, use_amp)

            run_step(state, "train_basic", _is_basic_ready, _do_basic)
        else:
            logger.info("train_basic disabilitato.")

    # ------------------------------------------------------------------
    # STEP 3: Train Time-Aware
    # ------------------------------------------------------------------
    if step_filter is None or step_filter == "train_time_aware":
        time_cfg = cfg["train_time_aware"]
        if time_cfg.get("enabled", True):
            checkpoint = time_cfg.get(
                "checkpoint", os.path.join(checkpoints_dir, "time_aware.pt")
            )
            shards = time_cfg.get("shards_dir", shards_dir)

            def _is_time_ready():
                return os.path.exists(checkpoint) and os.path.getsize(checkpoint) > 0

            def _do_time():
                run_training(cfg, "time_aware", shards, checkpoint, device, use_amp)

            run_step(state, "train_time_aware", _is_time_ready, _do_time)
        else:
            logger.info("train_time_aware disabilitato.")

    # ------------------------------------------------------------------
    # STEP 4: Evaluation & Plots
    # ------------------------------------------------------------------
    if step_filter is None or step_filter == "evaluate":
        eval_cfg = cfg["evaluate"]
        if eval_cfg.get("enabled", True):

            def _is_eval_ready():
                plots_dir = eval_cfg.get("plots_dir", "Dataset/Test/plots")
                return file_ready(os.path.join(plots_dir, "bars_per_n.png"))

            def _do_eval():
                evaluate_models(cfg, device, use_amp)

            run_step(state, "evaluate", _is_eval_ready, _do_eval)
        else:
            logger.info("Valutazione disabilitata.")

    free_memory(verbose=True)
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETATA.")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Yaml/train_main.yaml")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Override del livello di log (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args()
    # Applica il livello di log passato da CLI PRIMA di main
    if args.log_level:
        os.environ["LOG_LEVEL_OVERRIDE"] = args.log_level
    main(args.config)