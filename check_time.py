"""
Script diagnostico: verifica che i grafi nel dataset abbiano davvero
l'informazione temporale attesa dal modello time-aware
(DualGATTimeAwareModel legge `data_event.time`, uno scalare per arco,
atteso normalizzato in [0,1] — vedi gat_time_decay.py).

Uso:
    python check_time_data.py --data path/al/test_set.pt
    python check_time_data.py --data path/al/shard_dir --sharded

Cosa controlla, su un campione di grafi e su un batch collato:
  1. l'attributo `time` esiste ed è non-None
  2. la sua shape è allineata a edge_index (uno scalare per arco)
  3. range/statistiche (min, max, mean, std) — deve stare in [0,1] o
     comunque in un range piccolo, altrimenti il clamp su
     _MAX_DECAY_EXPONENT (=30.0) satura e il decay diventa
     costante (0 o 1), rendendo il path time-aware equivalente al
     path basic senza errori visibili.
  4. quanti campioni hanno `time` mancante o vuoto
  5. quanti valori sono negativi (un time_diff negativo spinge
     l'esponente verso +inf, rischio di comportamento degenerato anche
     dopo il clamp).
"""

from __future__ import annotations

import argparse
import sys

import torch


def describe_tensor(name: str, t: torch.Tensor) -> None:
    t = t.float()
    n_neg = (t < 0).sum().item()
    n_zero = (t == 0).sum().item()
    print(
        f"    {name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={t.min().item():.6g} max={t.max().item():.6g} "
        f"mean={t.mean().item():.6g} std={t.std().item():.6g} "
        f"n_negativi={n_neg} n_zero={n_zero}"
    )


def check_single_graph(data, idx: int) -> dict:
    info = {"idx": idx, "has_time": False, "aligned": None, "n_edges": None}

    if not hasattr(data, "edge_index") or data.edge_index is None:
        print(f"[{idx}] ATTENZIONE: nessun edge_index sul grafo.")
        return info

    n_edges = data.edge_index.size(1)
    info["n_edges"] = n_edges

    time_attr = getattr(data, "time", None)
    if time_attr is None:
        print(f"[{idx}] time MANCANTE (None) — decay sarà no-op per questo grafo.")
        return info

    info["has_time"] = True
    n_time = time_attr.view(-1).size(0)
    info["aligned"] = (n_time == n_edges)

    if n_time != n_edges:
        print(
            f"[{idx}] DISALLINEATO: edge_index ha {n_edges} archi ma "
            f"time ha {n_time} valori."
        )
    return info


def main():
    parser = argparse.ArgumentParser(description="Verifica presenza/qualità del dato temporale nel dataset.")
    parser.add_argument("--data", required=True, help="Path al file .pt (lista di Data) o alla directory di shard.")
    parser.add_argument("--sharded", action="store_true", help="Se il dataset è organizzato a shard (ShardedGraphDataset) invece di un singolo file .pt.")
    parser.add_argument("--max-samples", type=int, default=200, help="Numero massimo di grafi singoli da ispezionare in dettaglio (default 200).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size da usare per il check collato (default 64).")
    args = parser.parse_args()

    if args.sharded:
        print("Modalità --sharded non ancora cablata a un loader specifico.")
        print("Se usi TrainPipeline.Shard.ShardDataset.ShardedGraphDataset, importalo qui")
        print("e sostituisci il blocco sotto con la sua istanziazione, poi rilancia.")
        sys.exit(1)

    print(f"Caricamento {args.data} ...")
    data_list = torch.load(args.data, map_location="cpu", weights_only=False)
    if not isinstance(data_list, list):
        data_list = [data_list]
    print(f"Campioni totali: {len(data_list)}")

    # --- 1. Check per-grafo su un sottoinsieme ---
    n_check = min(args.max_samples, len(data_list))
    print(f"\n=== Check per-grafo (primi {n_check} campioni) ===")

    n_has_time = 0
    n_aligned = 0
    n_missing = 0

    for i in range(n_check):
        sample = data_list[i]
        # alcuni dataset restituiscono (Data, label) invece di Data puro
        data = sample[0] if isinstance(sample, (tuple, list)) else sample
        info = check_single_graph(data, i)
        if info["has_time"]:
            n_has_time += 1
            if info["aligned"]:
                n_aligned += 1
        else:
            n_missing += 1

    print(f"\nRiepilogo: {n_has_time}/{n_check} hanno 'time' presente, "
          f"{n_aligned}/{n_check} allineati a edge_index, "
          f"{n_missing}/{n_check} MANCANTI.")

    if n_missing == n_check:
        print("\n>>> NESSUN campione ha il dato temporale. Il modello time-aware")
        print(">>> equivale in pratica al modello basic su questo dataset.")
        return

    # --- 2. Statistiche aggregate su tutti i valori di time disponibili ---
    print("\n=== Statistiche aggregate su tutti i valori 'time' disponibili ===")
    all_time_vals = []
    for sample in data_list:
        data = sample[0] if isinstance(sample, (tuple, list)) else sample
        t = getattr(data, "time", None)
        if t is not None and t.numel() > 0:
            all_time_vals.append(t.view(-1).float())

    if all_time_vals:
        all_time = torch.cat(all_time_vals)
        describe_tensor("time (tutti i grafi)", all_time)

        if all_time.min().item() < 0:
            print("\n>>> ATTENZIONE: valori negativi presenti in 'time'.")
            print(">>> Un time_diff negativo spinge l'esponente di decay verso +inf,")
            print(">>> rischio di decay degenere anche dopo il clamp a _MAX_DECAY_EXPONENT.")

        if all_time.max().item() > 1.0 or all_time.min().item() < 0.0:
            print("\n>>> ATTENZIONE: valori fuori dal range [0,1] atteso da gat_time_decay.py.")
            print(">>> Se lambda_decay è tipico (~1e-2) e time non è normalizzato,")
            print(">>> l'effetto del decay può essere trascurabile o saturato.")
    else:
        print("Nessun valore 'time' trovato in nessun campione.")

    # --- 3. Check su un batch collato reale (come lo vede il modello) ---
    print("\n=== Check su un batch collato (via custom_collate_graph) ===")
    try:
        from timegnn.data.pyg import custom_collate_graph
    except ImportError as e:
        print(f"Impossibile importare custom_collate_graph: {e}")
        print("Salto il check sul batch collato.")
        return

    batch_samples = data_list[: args.batch_size]
    batch_event, labels = custom_collate_graph(batch_samples)

    print(f"batch_event.edge_index shape: {tuple(batch_event.edge_index.shape)}")
    has_time = hasattr(batch_event, "time") and batch_event.time is not None
    print(f"batch_event ha 'time': {has_time}")
    if has_time:
        describe_tensor("batch_event.time", batch_event.time)
        n_edges = batch_event.edge_index.size(1)
        n_time = batch_event.time.view(-1).size(0)
        if n_edges != n_time:
            print(f"\n>>> DISALLINEATO nel batch collato: {n_edges} archi vs {n_time} valori di time.")
            print(">>> custom_collate_graph potrebbe non concatenare 'time' correttamente.")
        else:
            print(f"\nOK: {n_edges} archi, {n_time} valori di time, allineati.")
    else:
        print("\n>>> Il batch collato NON contiene 'time': il modello time-aware")
        print(">>> userà decay=1.0 ovunque (no-op) per l'intero training/valutazione.")


if __name__ == "__main__":
    main()