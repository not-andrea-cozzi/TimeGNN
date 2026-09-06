"""
fix_dataset.py — pulizia one-shot dello spool residuo.

A differenza della versione precedente, QUESTO script NON chiama
build_splits() e non scrive train.pt/val.pt/test.pt: si limita a
scaricare lo spool esistente, scartare le finestre con group_key
incoerente (collisione di game_id tra run diversi), e RI-SCRIVERE lo
spool pulito tramite chiamate REALI a registry.enqueue() — cosi' la
idmap persistente (source_tag, orig_id) -> safe_id viene popolata in modo
coerente con la stessa logica che DatasetMain.py usera' per i prossimi
games.

Da usare quando Dataset/Train/ e' ancora vuota (nessun salvataggio finale
gia' avvenuto): dopo questo script, DatasetMain.py puo' essere rilanciato
da games_pipeline in poi per aggiungere nuove partite, e finalize_splits
finale (unico, a fine pipeline) includera' sia i dati vecchi ripuliti sia
quelli nuovi.
"""
import torch
from collections import defaultdict

from DatasetPipeline.PositionQueue import PositionQueueRegistry


def pulisci_spool():
    print("1. Caricamento dei dati dalla spool dir...")
    registry = PositionQueueRegistry.instance(state_path="Dataset/position_queue_state.json")

    drained = registry._drain_all()
    print(f"   Trovate {len(drained)} posizioni totali.")

    print("2. Rilevamento game_id con group_key incoerente (scarto, non fusione)...")
    by_gid = defaultdict(list)
    for item in drained:
        orig_id = int(item.data.game_id.item()) if hasattr(item.data.game_id, "item") else int(item.data.game_id)
        by_gid[orig_id].append(item)

    dropped_positions = 0
    dropped_windows = 0
    surviving_windows = []
    for gid, items in by_gid.items():
        keys = {it.group_key for it in items}
        if len(keys) != 1:
            dropped_windows += 1
            dropped_positions += len(items)
            continue
        surviving_windows.append(items)

    print(
        f"   Scartate {dropped_windows} finestre collisive ({dropped_positions} posizioni). "
        f"Finestre superstiti: {len(surviving_windows)}."
    )

    print("3. Reset traduzione id e re-inserimento tramite enqueue() reale...")
    # Coda e pending shard gia' svuotati da _drain_all(). Azzeriamo anche
    # la idmap: i game_id grezzi originali (assegnati da run precedenti,
    # gia' persi/collisivi) non sono piu' significativi. Ogni finestra
    # superstite viene trattata come una nuova unita' con un source_tag
    # dedicato, cosi' la sua traduzione (source_tag, orig_id) -> safe_id
    # non potra' mai collidere con quella che GamesBuilder/PuzzleBuilder
    # genereranno per le partite nuove (che usano i tag
    # "lichess"/"fics"/"club"/"puzzle").
    registry._id_map = {}
    registry._next_safe_game_id = 0

    new_game_counter = 0
    for items in surviving_windows:
        group_key = items[0].group_key
        for it in items:
            it.data.game_id = torch.tensor([new_game_counter], dtype=torch.long)
            registry.enqueue(
                source_tag="fix_dataset_recovered",
                data=it.data,
                group_key=group_key,
            )
        new_game_counter += 1

    registry.flush()  # forza la scrittura dell'ultimo shard parziale, se presente

    print(f"   {new_game_counter} finestre re-inserite in coda/spool, idmap persistita.")
    print(
        "Pulizia completata. Lo spool ora contiene solo dati coerenti: "
        "puoi rilanciare DatasetMain.py (dopo aver azzerato/aggiornato lo step "
        "'games_pipeline' in pipeline_state.json) per aggiungere nuove partite "
        "e fare un UNICO finalize_splits finale su tutto l'insieme."
    )


if __name__ == "__main__":
    pulisci_spool()