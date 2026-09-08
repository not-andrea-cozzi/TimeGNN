def conta_partite_pgn(nome_file):
    with open(nome_file, "r", encoding="utf-8") as f:
        contenuto = f.read()

    # Ogni partita PGN normalmente inizia con [Event
    partite = contenuto.count("[Event ")

    return partite


numero = conta_partite_pgn("RawData/ficsgamesdb_2017_chess2000_movetimes_4339914.pgn")
print(f"Numero di partite: {numero}")
