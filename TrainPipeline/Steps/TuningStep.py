from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger("step0_tuning")

TUNING_META_FILENAME = "tuning_meta.json"
CLASS_WEIGHTS_FILENAME = "class_weights.pt"


# ---------------------------------------------------------------------------
# Hardware setup — RTX 5070 (esecuzione singola-GPU, sempre la stessa scheda)
# ---------------------------------------------------------------------------
def configure_for_rtx_5070() -> Dict[str, Any]:

    settings: Dict[str, Any] = {"device": "cpu", "amp_dtype": "float32"}

    if not torch.cuda.is_available():
        logger.warning("[tuning] CUDA non disponibile: configurazione RTX 5070 saltata (esecuzione su CPU).")
        return settings

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    bf16_supported = torch.cuda.is_bf16_supported()
    amp_dtype = "bfloat16" if bf16_supported else "float16"

    settings.update({
        "device": "cuda",
        "amp_dtype": amp_dtype,
        "tf32_enabled": True,
        "cudnn_benchmark": True,
        "gpu_name": torch.cuda.get_device_name(0),
    })

    logger.info(
        f"[tuning] RTX 5070 configurata: TF32=on, cudnn.benchmark=on, "
        f"amp_dtype={amp_dtype} (bf16_supported={bf16_supported})."
    )
    return settings


# ---------------------------------------------------------------------------
# Class weighting — distribuzione reale delle mosse target nel train set
# ---------------------------------------------------------------------------
def compute_class_weights_streaming(
    train_dir: str,
    move_vocab_size: int,
    scheme: str = "inverse_sqrt_freq",
    smoothing: float = 1.0,
    max_weight: float = 50.0,
) -> torch.Tensor:
    manifest_path = os.path.join(train_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    counts = torch.zeros(move_vocab_size, dtype=torch.float64)
    num_shards = manifest["num_shards"]

    logger.info(f"[tuning] Calcolo class_weights su {num_shards} shard di train (streaming)...")

    for shard_i in range(num_shards):
        shard_path = os.path.join(train_dir, f"shard_{shard_i:05d}.pt")
        if not os.path.exists(shard_path):
            logger.warning(f"[tuning] Shard mancante {shard_path}: saltato nel conteggio classi.")
            continue
        try:
            data_list = torch.load(shard_path, weights_only=False)
        except Exception as e:
            logger.warning(f"[tuning] Shard illeggibile {shard_path} ({e}): saltato.")
            continue

        for d in data_list:
            y = getattr(d, "y", None)
            if y is None:
                continue
            label = int(y.item()) if torch.is_tensor(y) else int(y)
            if 0 <= label < move_vocab_size:
                counts[label] += 1

        del data_list

    total = counts.sum().item()
    if total == 0:
        raise RuntimeError("[tuning] Nessuna label valida trovata nel train set: impossibile calcolare class_weights.")

    freq = counts + smoothing

    if scheme == "inverse_freq":
        raw_weights = 1.0 / freq
    elif scheme == "inverse_sqrt_freq":
        raw_weights = 1.0 / torch.sqrt(freq)
    else:
        raise ValueError(f"scheme non supportato: {scheme}. Usa 'inverse_freq' o 'inverse_sqrt_freq'.")

    raw_weights = torch.clamp(raw_weights, max=max_weight)

    observed_mean = (raw_weights * counts).sum() / total
    normalized_weights = raw_weights / observed_mean

    num_unseen = int((counts == 0).sum().item())
    if num_unseen:
        logger.info(
            f"[tuning] {num_unseen}/{move_vocab_size} mosse del vocabolario mai viste come "
            f"target nel train set (peso comunque finito grazie a smoothing={smoothing})."
        )

    return normalized_weights.to(torch.float32)


# ---------------------------------------------------------------------------
# Warmup + cosine LR schedule
# ---------------------------------------------------------------------------
def build_warmup_cosine_lambda(
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> Callable[[int], float]:
    if warmup_steps < 0:
        raise ValueError("warmup_steps deve essere >= 0.")
    if total_steps <= warmup_steps:
        raise ValueError("total_steps deve essere maggiore di warmup_steps.")

    cosine_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = min(1.0, float(step - warmup_steps) / float(cosine_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

    return lr_lambda


def estimate_warmup_steps(steps_per_epoch: int, warmup_epochs_equivalent: float = 0.5) -> int:
    return max(1, int(steps_per_epoch * warmup_epochs_equivalent))


# ---------------------------------------------------------------------------
# Normalizzazione raccomandata: GraphNorm/LayerNorm al posto di BatchNorm1d
# ---------------------------------------------------------------------------
def recommend_norm_kind(batch_size: int) -> str:
    """Raccomanda un norm_kind per i modelli timegnn in base alla dimensione
    del batch usata in training.

    Nota importante su `batch_size`: nei DataLoader PyG rappresenta il numero
    di GRAFI per batch, non di nodi. La normalizzazione problematica è
    BatchNorm1d applicata sui nodi concatenati di tutti i grafi del batch: la
    sua statistica dipende dalla composizione (quanti nodi per grafo)
    del batch, che cambia ad ogni iterazione. Sotto la soglia di 32 grafi
    la varianza inter-batch è tipicamente troppo alta e LayerNorm (per-nodo,
    indipendente dal batch) è più stabile; sopra, GraphNorm (per-grafo via
    scatter su data.batch) sfrutta meglio la struttura del grafo.

    I valori restituiti ("layer_norm", "graph_norm") sono quelli accettati
    da timegnn.models.norm_layers.make_norm_layer; "batch_norm" e "none"
    restano disponibili per retrocompatibilità ma non vengono raccomandati.
    """
    return "layer_norm" if batch_size < 32 else "graph_norm"


def _resolve_batch_size_for_norm(cfg: Dict[str, Any]) -> int:
    """Estrae il batch_size effettivo dai config dei training step.

    Guarda prima in train_basic, poi in train_time_aware; il fallback 16
    riflette il default reale di GATBasicConfig/GATTimeDecayConfig (vedi
    docstring di timegnn.models.norm_layers), non un valore arbitrario.
    """
    train_basic = cfg.get("train_basic", {}) or {}
    train_time_aware = cfg.get("train_time_aware", {}) or {}
    batch_size = train_basic.get("batch_size") or train_time_aware.get("batch_size")
    return int(batch_size) if batch_size is not None else 16


# ---------------------------------------------------------------------------
# Orchestrazione step
# ---------------------------------------------------------------------------
@dataclass
class TuningStepConfig:
    enabled: bool = True
    move_vocab_size: int = 20480
    class_weight_scheme: str = "inverse_sqrt_freq"
    class_weight_smoothing: float = 1.0
    class_weight_max: float = 50.0
    warmup_epochs_equivalent: float = 0.5
    min_lr_ratio: float = 0.1
    force_recompute: bool = False


def run_tuning_step(
    cfg: Dict[str, Any],
    state,
    dataset_dir: str,
    train_dir: str,
    steps_per_epoch: Optional[int] = None,
    total_planned_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """STEP 0: prepara class_weights, warmup schedule e raccomandazione di
    norm, da consumare nei successivi step train_basic/train_time_aware.

    Idempotente: se class_weights.pt e tuning_meta.json esistono gia' e
    force_recompute=False, li ricarica invece di ricalcolare.

    Args:
        cfg: sezione "tuning" letta dallo YAML (vedi TuningStepConfig).
        state: PipelineState del chiamante (per is_done/mark_done).
        dataset_dir: cartella base dataset (dove salvare i metadati).
        train_dir: cartella shardata di train (train_clean).
        steps_per_epoch: se noto (len(train_ds)//batch_size), usato per
            calcolare warmup_steps; se None, warmup_steps e' lasciato a
            None e va ricalcolato dal chiamante con estimate_warmup_steps.
        total_planned_epochs: epoche pianificate, per total_steps (coseno).

    Returns:
        Dict con:
            - hw_settings: settings hardware (device, amp_dtype, ...).
            - class_weights_path: path del file .pt con i pesi di classe.
            - warmup_steps: int se steps_per_epoch e' noto, altrimenti None.
            - total_steps: int se SIA steps_per_epoch SIA total_planned_epochs
              sono noti, altrimenti None. Quando None, il chiamante NON puo'
              usare build_warmup_cosine_lambda (che richiede total_steps >
              warmup_steps) e deve ricadere su uno schedule alternativo
              (es. LR costante o cosine senza warmup).
            - recommended_norm: "layer_norm" o "graph_norm", da passare come
              norm_kind al costruttore dei modelli. NOTA: passarlo
              esplicitamente e' necessario perche' altrimenti un config
              legacy con use_batch_norm=True resterebbe attivo (il mapping
              retrocompatibile in DualGATModel usa use_batch_norm solo
              quando norm_kind e' None).
            - class_weight_scheme, move_vocab_size: eco della config usata.
    """
    tuning_cfg = TuningStepConfig(**{**TuningStepConfig().__dict__, **(cfg.get("tuning", {}) or {})})

    out_dir = os.path.join(dataset_dir, "Tuning")
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, TUNING_META_FILENAME)
    weights_path = os.path.join(out_dir, CLASS_WEIGHTS_FILENAME)

    if not tuning_cfg.enabled:
        logger.info("[tuning] Step disabilitato da config: skip.")
        return {}

    already_done = state.is_done("tuning", skip=tuning_cfg.force_recompute)
    outputs_ready = os.path.exists(meta_path) and os.path.exists(weights_path)

    if already_done and outputs_ready:
        logger.info("[tuning] Gia' completato: carico metadati esistenti.")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["class_weights_path"] = weights_path
        return meta

    logger.info("=" * 70)
    logger.info("STEP 0: tuning (class_weights + warmup schedule + norm consigliata)")
    logger.info("=" * 70)

    hw_settings = configure_for_rtx_5070()

    try:
        class_weights = compute_class_weights_streaming(
            train_dir=train_dir,
            move_vocab_size=tuning_cfg.move_vocab_size,
            scheme=tuning_cfg.class_weight_scheme,
            smoothing=tuning_cfg.class_weight_smoothing,
            max_weight=tuning_cfg.class_weight_max,
        )
        tmp_weights_path = weights_path + ".tmp"
        torch.save({"weights": class_weights, "scheme": tuning_cfg.class_weight_scheme}, tmp_weights_path)
        os.replace(tmp_weights_path, weights_path)
        logger.info(
            f"[tuning] class_weights salvati -> {weights_path} "
            f"(min={class_weights.min().item():.3f}, max={class_weights.max().item():.3f}, "
            f"mean={class_weights.mean().item():.3f})."
        )
    except Exception as e:
        state.mark_failed("tuning", str(e))
        raise

    warmup_steps: Optional[int] = None
    total_steps: Optional[int] = None
    if steps_per_epoch is not None:
        warmup_steps = estimate_warmup_steps(steps_per_epoch, tuning_cfg.warmup_epochs_equivalent)
        if total_planned_epochs is not None:
            total_steps = steps_per_epoch * total_planned_epochs
        else:
            logger.info(
                "[tuning] total_planned_epochs non fornito: total_steps resta None, "
                "il chiamante dovra' usare uno schedule LR senza coseno."
            )
    elif total_planned_epochs is not None:
        logger.info(
            "[tuning] steps_per_epoch non fornito: warmup_steps e total_steps restano None. "
            "Il chiamante puo' calcolare warmup_steps con estimate_warmup_steps(len(loader), ...)."
        )

    batch_size_for_norm = _resolve_batch_size_for_norm(cfg)
    recommended_norm = recommend_norm_kind(batch_size=batch_size_for_norm)
    logger.info(
        f"[tuning] norm raccomandata: {recommended_norm} "
        f"(batch_size={batch_size_for_norm} grafi/batch)."
    )

    meta: Dict[str, Any] = {
        "hw_settings": hw_settings,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "min_lr_ratio": tuning_cfg.min_lr_ratio,
        "recommended_norm": recommended_norm,
        "batch_size_for_norm": batch_size_for_norm,
        "class_weight_scheme": tuning_cfg.class_weight_scheme,
        "move_vocab_size": tuning_cfg.move_vocab_size,
    }

    tmp_meta_path = meta_path + ".tmp"
    with open(tmp_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_meta_path, meta_path)

    state.mark_done("tuning", recommended_norm=recommended_norm, warmup_steps=warmup_steps or 0)
    logger.info(f"[tuning] Completato: {meta}")

    meta["class_weights_path"] = weights_path
    return meta