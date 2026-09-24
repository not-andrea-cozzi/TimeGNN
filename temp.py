import sys
import torch

def check_shard(path):
    data_list = torch.load(path, weights_only=False)
    print(f"Shard: {path}")
    print(f"Numero elementi: {len(data_list)}")

    if len(data_list) == 0:
        print("Shard vuoto.")
        return

    sample = data_list[0]
    print(f"Campi presenti nel primo elemento: {list(sample.keys())}")
    has_fen = hasattr(sample, "fen") and sample.fen is not None
    print(f"Ha 'fen'? {has_fen}")
    if has_fen:
        print(f"Esempio fen: {sample.fen}")

    # conteggio su tutto lo shard, non solo il primo elemento
    n_with_fen = sum(1 for d in data_list if hasattr(d, "fen") and d.fen is not None)
    print(f"Elementi con fen valido: {n_with_fen}/{len(data_list)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python check_fen.py path/a/shard_00000.pt")
        sys.exit(1)
    check_shard(sys.argv[1])