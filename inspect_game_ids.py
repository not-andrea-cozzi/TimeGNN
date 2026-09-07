"""
Script per l'ispezione dei `data.game_id` gestiti da PositionQueueRegistry.
"""

import glob
import os
import torch
from torch_geometric.data import Data

from DatasetPipeline.Utils.position_compression import decompress_position_data
from DatasetPipeline.PositionQueue import (
    SHARD_GLOB_PATTERN,
    PositionQueueRegistry,
    _extract_game_id,
    _spool_dir_for,
)


def print_game_ids_from_splits(splits: dict) -> None:
    """Ripercorre tutti i dataset generati (train, val, test) e stampa il game_id di ogni singolo Data object."""
    print("\n" + "=" * 60)
    print(" 📊 VISUALIZZAZIONE GAME_ID DAGLI SPLIT COSTRUITI")
    print("=" * 60)

    total_items = 0
    for split_name, data_list in splits.items():
        print(f"\n--- [SPLIT: {split_name.upper()}] ({len(data_list)} elementi) ---")
        for idx, data in enumerate(data_list, start=1):
            game_id = _extract_game_id(data)
            print(f"  [{idx:04d}] game_id: {game_id}")
            total_items += 1

    print(f"\nTotale elementi ispezionati negli split: {total_items}")


def print_game_ids_from_spool_shards(state_path: str) -> None:
    """Legge direttamente i file shard (.pt) presenti nella cartella di spool,

    decomprime i dati e stampa i game_id senza svuotare la coda in memoria.
    """
    spool_dir = _spool_dir_for(state_path)
    pattern = os.path.join(spool_dir, SHARD_GLOB_PATTERN)
    shard_paths = sorted(glob.glob(pattern))

    print("\n" + "=" * 60)
    print(f" 💾 VISUALIZZAZIONE GAME_ID DAI FILE SHARD SU DISCO ({spool_dir})")
    print("=" * 60)

    if not shard_paths:
        print("Nessun file shard trovato su disco.")
        return

    total_items = 0
    for shard_path in shard_paths:
        shard_name = os.path.basename(shard_path)
        print(f"\n📂 File Shard: {shard_name}")

        try:
            records = torch.load(shard_path, weights_only=False)
            for idx, rec in enumerate(records, start=1):
                decompressed_data = decompress_position_data(rec["data"])
                game_id = _extract_game_id(decompressed_data)
                source_tag = rec.get("source_tag", "N/A")
                group_key = rec.get("group_key", "N/A")

                print(
                    f"  [{idx:03d}] game_id: {game_id:<25} | "
                    f"source_tag: {source_tag:<15} | group_key: {group_key}"
                )
                total_items += 1
        except Exception as e:
            print(f"  ❌ Errore nella lettura dello shard {shard_name}: {e}")

    print(f"\nTotale elementi trovati negli shard su disco: {total_items}")


def main():
    state_path = "test_position_queue_state.json"

    PositionQueueRegistry.reset_for_testing()

    queue = PositionQueueRegistry.instance(
        state_path=state_path, shard_size=3
    )

    print("\n1. Accodamento elementi con log in tempo reale...")

    sample_items = [
        ("lichess_puzzles", "lichess_puzzle_1001", 1),
        ("lichess_puzzles", "lichess_puzzle_1002", 1),
        ("lichess_puzzles", "lichess_puzzle_1003", 2),
        ("game_generator", "game_run_01_pos_0", 2),
        ("game_generator", "game_run_01_pos_1", 2),
    ]

    for source_tag, game_id_str, group_key in sample_items:
        # Generazione matrice x strettamente binaria (0.0 o 1.0)
        data = Data(x=torch.randint(0, 2, (4, 16), dtype=torch.float32))
        data.game_id = game_id_str

        ref = queue.enqueue(source_tag=source_tag, data=data, group_key=group_key)
        print(f"  [ENQUEUE] ref: {ref:<2} | game_id: {_extract_game_id(data)}")

    queue.flush()

    print_game_ids_from_spool_shards(state_path)

    print("\nCostruzione degli split con build_splits()...")
    splits = queue.build_splits(split_ratios=(0.6, 0.2, 0.2), seed=42)

    print_game_ids_from_splits(splits)


if __name__ == "__main__":
    main()