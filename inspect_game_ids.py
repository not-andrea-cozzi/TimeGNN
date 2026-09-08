import torch

path = "Dataset/Train/test.pt"

# Quanti elementi mostrare per i Tensor grandi
MAX_ELEMENTS = 200


def print_value(name, value):
    """Stampa nome, tipo, shape e contenuto del valore."""

    print(f"\n{'-' * 60}")
    print(f"KEY: {name}")
    print(f"TIPO: {type(value)}")

    # Tensor
    if isinstance(value, torch.Tensor):
        print(f"SHAPE: {tuple(value.shape)}")
        print(f"DTYPE: {value.dtype}")
        print(f"DEVICE: {value.device}")

        # Tensor scalare
        if value.numel() == 1:
            print(f"VALORE: {value.item()}")

        # Tensor piccolo
        elif value.numel() <= MAX_ELEMENTS:
            print("VALORE:")
            print(value)

        # Tensor grande
        else:
            print(f"NUMERO ELEMENTI: {value.numel()}")
            print(f"PRIMI {MAX_ELEMENTS} ELEMENTI:")
            print(value.flatten()[:MAX_ELEMENTS])

    # Stringhe
    elif isinstance(value, str):
        print(f"VALORE: {value}")

    # Numeri / bool
    elif isinstance(value, (int, float, bool)):
        print(f"VALORE: {value}")

    # Liste / tuple
    elif isinstance(value, (list, tuple)):
        print(f"LUNGHEZZA: {len(value)}")

        if len(value) <= MAX_ELEMENTS:
            print("VALORE:")
            print(value)
        else:
            print(f"PRIMI {MAX_ELEMENTS} ELEMENTI:")
            print(value[:MAX_ELEMENTS])

    # Altro
    else:
        print(f"VALORE: {value}")


# ============================================================
# CARICAMENTO
# ============================================================

data = torch.load(
    path,
    map_location="cpu",
    weights_only=False
)


print("=" * 60)
print("ISPEZIONE FILE PT")
print("=" * 60)

print(f"FILE: {path}")
print(f"TIPO: {type(data)}")


# ============================================================
# CASO 1: SINGOLO torch_geometric.data.Data
# ============================================================

if hasattr(data, "keys") and not isinstance(data, (list, tuple)):

    print("\n")
    print("=" * 60)
    print("SINGOLO OGGETTO DATA")
    print("=" * 60)

    keys = list(data.keys())

    print(f"\nNUMERO KEYS: {len(keys)}")

    print("\nKEYS:")
    for key in keys:
        print(f"  - {key}")

    print("\n")
    print("=" * 60)
    print("VALORI")
    print("=" * 60)

    for key in keys:
        value = getattr(data, key)
        print_value(key, value)


# ============================================================
# CASO 2: LISTA / TUPLA DI Data
# ============================================================

elif isinstance(data, (list, tuple)):

    print("\n")
    print("=" * 60)
    print("COLLEZIONE")
    print("=" * 60)

    print(f"NUMERO ELEMENTI: {len(data)}")

    if len(data) == 0:
        print("\nLa lista è vuota.")

    else:

        # ----------------------------------------------------
        # STAMPA TIPO ELEMENTI
        # ----------------------------------------------------

        print("\nTIPI DEGLI ELEMENTI:")

        for i, element in enumerate(data[:MAX_ELEMENTS]):
            print(f"  [{i}] {type(element)}")

        # ----------------------------------------------------
        # PRIMO ELEMENTO
        # ----------------------------------------------------

        first = data[0]

        print("\n")
        print("=" * 60)
        print("PRIMO ELEMENTO")
        print("=" * 60)

        print(f"TIPO: {type(first)}")

        if hasattr(first, "keys"):

            keys = list(first.keys())

            print(f"\nNUMERO KEYS: {len(keys)}")

            print("\nKEYS:")
            for key in keys:
                print(f"  - {key}")

            # ------------------------------------------------
            # VALORI DEL PRIMO ELEMENTO
            # ------------------------------------------------

            print("\n")
            print("=" * 60)
            print("VALORI DEL PRIMO ELEMENTO")
            print("=" * 60)

            for key in keys:
                value = getattr(first, key)
                print_value(key, value)

        else:
            print("\nIl primo elemento non possiede keys().")
            print("VALORE:")
            print(first)


# ============================================================
# ALTRO TIPO DI CONTENUTO
# ============================================================

else:

    print("\n")
    print("=" * 60)
    print("CONTENUTO")
    print("=" * 60)

    print_value("data", data)


# ============================================================
# GAME ID
# ============================================================

print("\n")
print("=" * 60)
print("GAME_ID")
print("=" * 60)


if isinstance(data, (list, tuple)):

    game_ids = []

    for i, element in enumerate(data):

        if hasattr(element, "game_id"):

            game_id = element.game_id

            game_ids.append(game_id)

            print(f"[{i}] game_id = {game_id}")

        else:
            print(f"[{i}] game_id NON PRESENTE")


elif hasattr(data, "game_id"):

    print(f"game_id = {data.game_id}")

else:

    print("game_id NON PRESENTE")


# ============================================================
# FINE
# ============================================================

print("\n")
print("=" * 60)
print("FINE ISPEZIONE")
print("=" * 60)
