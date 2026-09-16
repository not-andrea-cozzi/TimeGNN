from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from typing import List, Optional

import torch
from torch_geometric.data import Data

# Riusa la pipeline di training reale, non la reimplementa: se
# TrainMain.py viene modificato, questo test resta allineato.
import TrainMain

from DatasetPipeline.Model.ChessConstants import MOVE_VOCAB_SIZE

logger = logging.getLogger("train_main_test")

SHARD_FILENAME_TEMPLATE = "shard_{:05d}.pt"
MANIFEST_FILENAME = "manifest.json"

# Modelli supportati dallo smoke test. Il valore passato a
# TrainMain.run_training e' esattamente questa stringa (secondo argomento).
SUPPORTED_MODELS = ("basic", "time_aware")


# ----------------------------------------------------------------------
# Mini-state in-memory per il TuningStep
# ----------------------------------------------------------------------
class _MiniState:
    """State minimale per lo smoke test: espone solo i tre metodi che
    TuningStep.run_tuning_step usa (is_done / mark_done / mark_failed).

    Non persiste nulla su disco: ogni run dello smoke test riparte da
    zero (is_done ritorna sempre False), cosi' il tuning viene
    effettivamente ricalcolato ad ogni invocazione.
    """

    def __init__(self) -> None:
        self._done: dict = {}

    def is_done(self, step: str, skip: bool = False) -> bool:
        return (not skip) and step in self._done

    def mark_done(self, step: str, **kwargs) -> None:
        self._done[step] = kwargs

    def mark_failed(self, step: str, reason: str) -> None:
        logger.error(f"[tuning] step '{step}' fallito: {reason}")


# ----------------------------------------------------------------------
# Import robusto di run_tuning_step
# ----------------------------------------------------------------------
def _import_run_tuning_step():
    """Importa run_tuning_step dal percorso canonico della pipeline.
    Prova prima il percorso usato da TrainMain, poi un fallback diretto.
    Solleva ImportError con messaggio chiaro se non lo trova."""
    candidates = (
        "TrainPipeline.Steps.TuningStep",
        "TrainPipeline.Steps.tuning_step",
        "TrainPipeline.TuningStep",
    )
    last_err: Optional[Exception] = None
    for mod_path in candidates:
        try:
            mod = __import__(mod_path, fromlist=["run_tuning_step"])
            fn = getattr(mod, "run_tuning_step", None)
            if fn is not None:
                return fn
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise ImportError(
        "Impossibile importare run_tuning_step da TrainPipeline.Steps.TuningStep. "
        "Verifica il percorso del modulo con: "
        "  grep -rn 'def run_tuning_step' TrainPipeline/"
        f" Ultimo errore: {last_err}"
    )


# ----------------------------------------------------------------------
# Costruzione del mini-dataset shardato
# ----------------------------------------------------------------------
def _read_manifest(shard_dir: str) -> dict:
    manifest_path = os.path.join(shard_dir, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"manifest.json non trovato in '{shard_dir}'. Verifica che la "
            f"pipeline di dataset (DatasetMain.py, step 'clean') sia gia' "
            f"stata eseguita e abbia prodotto questa cartella."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_n_items(shard_dir: str, n_items: int, label: str) -> List[Data]:
    """Legge shard in ordine (shard_00000.pt, shard_00001.pt, ...) finche'
    non accumula almeno n_items elementi, poi tronca esattamente a
    n_items. Non carica l'intero dataset: si ferma appena ha abbastanza.
    """
    manifest = _read_manifest(shard_dir)
    num_shards = manifest["num_shards"]
    total_available = manifest["total"]

    if total_available < n_items:
        raise ValueError(
            f"'{shard_dir}' contiene solo {total_available} campioni "
            f"({label}), ma ne sono richiesti {n_items}. Riduci --n-{label} "
            f"o usa un dataset piu' grande."
        )

    collected: List[Data] = []
    for shard_idx in range(num_shards):
        if len(collected) >= n_items:
            break
        shard_path = os.path.join(shard_dir, SHARD_FILENAME_TEMPLATE.format(shard_idx))
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard mancante: {shard_path}")
        items = torch.load(shard_path, weights_only=False)
        collected.extend(items)
        del items

    collected = collected[:n_items]
    logger.info(f"[{label}] Raccolti {len(collected)}/{n_items} campioni da '{shard_dir}'.")
    return collected


def _write_single_shard_dataset(items: List[Data], out_dir: str) -> None:
    """Scrive `items` come un UNICO shard + manifest.json, nel formato
    atteso da ShardedGraphDataset (stesso formato di
    TrainPipeline/Shard/Sharding.py e TrainPipeline/CleanDataset.py)."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    shard_path = os.path.join(out_dir, SHARD_FILENAME_TEMPLATE.format(0))
    tmp_path = shard_path + ".tmp"
    torch.save(items, tmp_path)
    os.replace(tmp_path, shard_path)

    manifest = {"num_shards": 1, "shard_size": len(items), "total": len(items)}
    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_manifest, manifest_path)

    logger.info(
        f"Mini-shard scritto: {shard_path} ({len(items)} campioni) "
        f"-> manifest in {manifest_path}"
    )


def build_mini_dataset(
    train_dir: str,
    val_dir: str,
    out_root: str,
    n_train: int,
    n_val: int,
) -> tuple:
    """Costruisce Dataset/_smoke_test/{train_mini,val_mini} a partire dagli
    shard reali, e ritorna (train_mini_dir, val_mini_dir)."""
    train_items = _collect_n_items(train_dir, n_train, "train")
    val_items = _collect_n_items(val_dir, n_val, "val")

    train_mini_dir = os.path.join(out_root, "train_mini")
    val_mini_dir = os.path.join(out_root, "val_mini")

    _write_single_shard_dataset(train_items, train_mini_dir)
    _write_single_shard_dataset(val_items, val_mini_dir)

    del train_items, val_items
    return train_mini_dir, val_mini_dir


# ----------------------------------------------------------------------
# Config smoke test per i due modelli
# ----------------------------------------------------------------------
# Campi comuni a basic e time_aware. Tenuti in un unico punto per evitare
# che le due config divergano silenziosamente (es. dropout o num_layers
# diversi tra i due smoke test, che renderebbero i risultati non
# confrontabili).
_COMMON_SMOKE_FIELDS: dict = {
    "enabled": True,
    "train_dir": None,  # sovrascritto dal chiamante di run_training
    "val_dir": None,
    "num_workers": 0,
    "persistent_workers": False,
    "prefetch_factor": None,
    "pin_memory": False,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "seed": 42,
    "patience": 5,
    "embedding_dims": 16,
    "gat_hidden_dim_event": 8,
    "gat_hidden_dim_embed": 16,
    "gat_hidden_dim_concat": 16,
    "num_heads": 2,
    "num_layers": 1,
    "dropout": 0.0,
    "use_batch_norm": False,
    "activation": "elu",
    "compile": False,
    "memory_cleanup_threshold_gb": 1.5,
}


def build_basic_smoke_test_cfg(checkpoint_path: str, batch_size: int, epochs: int) -> dict:
    """Config 'train_basic' minimale. Valori scelti per essere leggeri su
    CPU: batch_size piccolo, hidden dims ridotte, niente BatchNorm
    (rischioso con batch piccoli/ultimo batch da 1 elemento), niente
    num_workers, niente compile.
    """
    return {
        **_COMMON_SMOKE_FIELDS,
        "checkpoint": checkpoint_path,
        "epochs": epochs,
        "batch_size": batch_size,
    }


def build_time_aware_smoke_test_cfg(
    checkpoint_path: str,
    batch_size: int,
    epochs: int,
    lambda_decay: float = 0.1,
) -> dict:
    """Config 'train_time_aware' minimale.

    Identica a train_basic piu' lambda_decay, il parametro che controlla
    il decay esponenziale dell'attenzione in TimeAwareGATConv:
        decay = exp(-lambda_decay * time_diff)
    Con lambda_decay=0.1 e time_diff in [0, qualche decina di secondi]
    il decay resta in un range numerico ragionevole (exp(-1) ~ 0.37 a
    time_diff=10), evitando saturazione a 0 o 1 che renderebbe
    l'attenzione degenere proprio nel test che dovrebbe verificarla.
    """
    return {
        **_COMMON_SMOKE_FIELDS,
        "checkpoint": checkpoint_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "lambda_decay": lambda_decay,
    }


def build_tuning_cfg() -> dict:
    """Sezione 'tuning' per lo smoke test.

    move_vocab_size NON viene mai inferito dal dataset: e' una costante
    architetturale del problema (MOVE_VOCAB_SIZE = 64*64*5 = 20480,
    vedi DatasetPipeline/Model/PositionGraphSchema.py), non qualcosa che
    dipende da quante mosse distinte compaiono nel mini-campione. Un
    mini-dataset da poche centinaia di posizioni puo' tranquillamente non
    contenere tutte le 20480 mosse possibili, ma il vocabolario resta
    quello: class_weights e legal_move_mask devono avere sempre la stessa
    dimensione (MOVE_VOCAB_SIZE), altrimenti sparse_legal_cross_entropy
    solleva IndexError non appena una label supera la size del tensore
    dei pesi.
    """
    return {"enabled": True, "move_vocab_size": MOVE_VOCAB_SIZE}


def _build_model_cfg(model_name: str, checkpoint_path: str, batch_size: int, epochs: int) -> dict:
    """Dispatch tra le due config smoke test in base al modello scelto."""
    if model_name == "basic":
        return build_basic_smoke_test_cfg(checkpoint_path, batch_size, epochs)
    if model_name == "time_aware":
        return build_time_aware_smoke_test_cfg(checkpoint_path, batch_size, epochs)
    raise ValueError(f"Modello non supportato: {model_name!r}. Attesi: {SUPPORTED_MODELS}.")


def _cleanup_stale_checkpoints(checkpoint_dir: str, stem: str) -> None:
    """Rimuove i checkpoint di un test precedente per lo stesso stem, cosi'
    la run parte sempre da zero (niente resume accidentale tra run di
    test diverse con parametri diversi, es. batch_size cambiato)."""
    for suffix in ("_last.pt", "_best.pt", "_last_scheduler.pt", "_best_scheduler.pt"):
        stale = os.path.join(checkpoint_dir, stem + suffix)
        if os.path.exists(stale):
            os.remove(stale)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test: verifica che il training giri end-to-end su un piccolo sottoinsieme."
    )
    parser.add_argument("--train-dir", default="Dataset/Train/train_clean", help="Cartella shardata di train reale.")
    parser.add_argument("--val-dir", default="Dataset/Train/val_clean", help="Cartella shardata di val reale.")
    parser.add_argument("--out-root", default="Dataset/_smoke_test", help="Dove scrivere mini-dataset e checkpoint di test.")
    parser.add_argument("--n-train", type=int, default=10000, help="Numero di campioni di train da usare.")
    parser.add_argument("--n-val", type=int, default=1000, help="Numero di campioni di val da usare.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (piccolo, adatto a CPU).")
    parser.add_argument("--epochs", type=int, default=5, help="Numero di epoche (default 1: solo verifica che giri).")
    parser.add_argument("--keep-output", action="store_true", help="Non cancellare --out-root a fine test.")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="basic",
        help="Modello da testare: 'basic' (DualGATModel) o 'time_aware' "
             "(DualGATTimeAwareModel, con time decay sull'attenzione).",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Disattiva lo step di tuning (utile per isolare problemi di training).",
    )
    args = parser.parse_args()

    TrainMain.setup_logging("INFO")
    logger.info("=" * 70)
    logger.info(f"SMOKE TEST TRAINING [{args.model}] (solo verifica funzionamento, NON valuta qualita')")
    logger.info("=" * 70)

    # Esecuzione forzata su CPU: nessuna diramazione device=cuda in questo
    # script. Se in futuro serve testare anche su GPU, e' piu' sicuro
    # farlo con un secondo script dedicato piuttosto che riintrodurre qui
    # rami condizionali che nessuno esercita piu'.
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} (smoke test forzato su CPU).")

    # AMP disabilitato: autocast bf16/fp16 non ha senso su CPU per questo
    # test (nessun beneficio, solo complessita' di debug in piu').
    use_amp = True
    logger.info(f"AMP: {use_amp} (disattivato, non applicabile su CPU).")

    if not os.path.isdir(args.train_dir):
        raise TrainMain.PipelineConfigError(
            f"--train-dir non trovato: '{args.train_dir}'. Esegui prima la "
            f"pipeline di dataset (DatasetMain.py) fino allo step 'clean'."
        )
    if not os.path.isdir(args.val_dir):
        raise TrainMain.PipelineConfigError(f"--val-dir non trovato: '{args.val_dir}'.")

    t0 = time.monotonic()
    train_mini_dir, val_mini_dir = build_mini_dataset(
        args.train_dir, args.val_dir, args.out_root, args.n_train, args.n_val
    )
    logger.info(f"Mini-dataset pronto in {time.monotonic() - t0:.2f}s.")

    logger.info(f"[tuning] move_vocab_size = {MOVE_VOCAB_SIZE} (costante MOVE_VOCAB_SIZE, non inferita dal dataset).")

    checkpoint_dir = os.path.join(args.out_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Nome del checkpoint differenziato per modello: basic e time_aware
    # possono coesistere in out_root senza che l'uno sovrascriva l'altro
    # (utile se in futuro si aggiunge --model both).
    checkpoint_stem = f"smoke_{args.model}"
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_stem}.pt")

    _cleanup_stale_checkpoints(checkpoint_dir, checkpoint_stem)

    # Config del modello selezionato.
    model_cfg = _build_model_cfg(
        args.model, checkpoint_path, args.batch_size, args.epochs
    )

    # La sezione "train_<model>" e' quella letta da run_training quando
    # model_name == args.model. L'altra resta disabilitata esplicitamente.
    smoke_cfg = {
        "train_basic": (
            model_cfg if args.model == "basic" else {"enabled": False}
        ),
        "train_time_aware": (
            model_cfg if args.model == "time_aware" else {"enabled": False}
        ),
        "evaluate": {"enabled": False},
        "pipeline": {},
        "tuning": {"enabled": False} if args.no_tuning else build_tuning_cfg(),
    }

    # ------------------------------------------------------------------
    # Tuning step (class_weights + warmup schedule + norm raccomandata)
    # ------------------------------------------------------------------
    # TrainTest.py non passa da TrainMain.main(), che e' l'unico punto in
    # cui run_tuning_step verrebbe chiamato normalmente: lo invochiamo
    # qui esplicitamente, cosi' lo smoke test copre anche questo step.
    #
    # Il tuning e' identico per basic e time_aware (dipende dal dataset,
    # non dall'architettura): viene calcolato una volta per run. Se in
    # futuro serve --model both, il tuning va fattorizzato fuori dal loop.
    tuning_meta: dict = {}
    if not args.no_tuning:
        logger.info("-" * 70)
        logger.info("Esecuzione tuning step (class_weights + warmup + norm)...")
        logger.info("-" * 70)

        try:
            run_tuning_step = _import_run_tuning_step()
        except ImportError as e:
            logger.error(f"Impossibile importare run_tuning_step: {e}")
            raise

        t_tuning_0 = time.monotonic()
        try:
            tuning_meta = run_tuning_step(
                cfg=smoke_cfg,
                state=_MiniState(),
                dataset_dir=args.out_root,
                train_dir=train_mini_dir,
                steps_per_epoch=None,
                total_planned_epochs=None,
            )
        except Exception:
            logger.error(
                "TUNING STEP FALLITO: run_tuning_step ha sollevato un'eccezione. "
                "Vedi traceback sopra/sotto per la causa.",
                exc_info=True,
            )
            raise

        logger.info(
            f"Tuning completato in {time.monotonic() - t_tuning_0:.2f}s: {tuning_meta}"
        )
    else:
        logger.info("Tuning disattivato (--no-tuning): salto run_tuning_step.")

    logger.info("-" * 70)
    logger.info(
        f"Avvio run_training('{args.model}', ...) su {args.n_train} train / {args.n_val} val, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, device={device}."
    )
    logger.info("-" * 70)

    t0 = time.monotonic()
    try:
        TrainMain.run_training(
            smoke_cfg,
            args.model,
            train_mini_dir,
            val_mini_dir,
            checkpoint_path,
            device,
            use_amp,
            tuning_meta=tuning_meta,
        )
    except Exception:
        logger.error(
            "SMOKE TEST FALLITO: run_training ha sollevato un'eccezione. "
            "Vedi traceback sopra/sotto per la causa.",
            exc_info=True,
        )
        raise

    elapsed = time.monotonic() - t0
    logger.info("=" * 70)
    logger.info(f"SMOKE TEST OK: run_training('{args.model}') completata senza eccezioni in {elapsed:.2f}s.")
    logger.info("=" * 70)

    best_path = os.path.join(checkpoint_dir, f"{checkpoint_stem}_best.pt")
    last_path = os.path.join(checkpoint_dir, f"{checkpoint_stem}_last.pt")
    for p, label in ((last_path, "last"), (best_path, "best")):
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / 1024**2
            logger.info(f"Checkpoint {label} scritto: {p} ({size_mb:.2f} MB)")
        else:
            logger.warning(f"Checkpoint {label} atteso ma non trovato: {p}")

    if not args.keep_output:
        logger.info(
            f"Rimozione output di test in '{args.out_root}' "
            f"(usa --keep-output per conservarli)."
        )
        shutil.rmtree(args.out_root, ignore_errors=True)
    else:
        logger.info(f"Output di test conservati in '{args.out_root}' (--keep-output).")


if __name__ == "__main__":
    try:
        main()
    except TrainMain.PipelineConfigError as e:
        logging.getLogger("train_main_test").error(f"Errore di configurazione: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)