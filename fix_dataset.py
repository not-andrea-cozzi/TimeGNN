import os
import torch
from DatasetPipeline.PositionQueue import PositionQueueRegistry

def recupera_e_salva_dataset():
    print("1. Caricamento dei dati salvati dalla spool dir...")
    # Inizializza il registro che caricherà in automatico i 26 shard residui
    registry = PositionQueueRegistry.instance(state_path="Dataset/position_queue_state.json")
    
    # Estrae tutto il contenuto per manipolarlo
    drained = registry._drain_all()
    print(f"   Trovate {len(drained)} posizioni totali da sistemare.")

    print("2. Correzione delle collisioni dei game_id...")
    last_seen_idx = {}
    assigned_run = {}
    new_game_id_counter = 0
    assigned_new_ids = {}

    for i, item in enumerate(drained):
        # Estrai il game_id originale (sia che sia un tensor o un int)
        orig_id = int(item.data.game_id.item()) if hasattr(item.data.game_id, "item") else int(item.data.game_id)
        
        # Se abbiamo visto questo ID di recente (entro le ultime 5000 iterazioni), appartiene alla stessa partita
        if orig_id in last_seen_idx and (i - last_seen_idx[orig_id]) < 5000:
            run_id = assigned_run[orig_id]
        else:
            # È una nuova partita (o il salto causato dall'unione della vecchia run con la nuova)
            run_id = i
            assigned_run[orig_id] = run_id
            
        last_seen_idx[orig_id] = i
        
        # Genera una chiave composta e assegna il nuovo ID univoco
        key = (orig_id, run_id)
        if key not in assigned_new_ids:
            assigned_new_ids[key] = new_game_id_counter
            new_game_id_counter += 1
            
        # Applica il nuovo ID univoco alla posizione
        item.data.game_id = torch.tensor([assigned_new_ids[key]], dtype=torch.long)

    print("3. Reinserimento in coda per lo split finale...")
    registry._queue.queue.clear()
    registry._pending_shard = []
    
    for item in drained:
        registry._queue.put(item)
        
    print("4. Generazione split (Train, Val, Test)...")
    # Usa il sistema nativo, ora che i conflitti sono risolti
    splits = registry.build_splits(split_ratios=(0.7, 0.1, 0.2), seed=42)

    print("5. Salvataggio su disco...")
    os.makedirs("Dataset/Train", exist_ok=True)
    for split_name, data_list in splits.items():
        out_path = f"Dataset/Train/{split_name}.pt"
        torch.save(data_list, out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"   Salvato {split_name}: {len(data_list)} posizioni in {out_path} ({size_mb:.2f} MB)")
        
    print("Ripristino completato! I file sono pronti in Dataset/Train/ e la cartella temporanea è stata ripulita.")

if __name__ == "__main__":
    recupera_e_salva_dataset()