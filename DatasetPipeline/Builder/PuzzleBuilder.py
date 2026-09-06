"""
Sostituisce PuzzleGraphDataset.py (schema compresso a sequenza, superato,
basato su GraphBuilder/SequenceIdAllocator) con un BUILDER allineato al
contratto di GamesBuilder.py: usa PositionGraphSchema.build_position_data
per costruire ogni singola posizione (grafo spaziale a 64 caselle) e
PositionQueueRegistry.enqueue per accodarla, condividendo la STESSA coda
di GamesBuilder cosi' che PositionQueueRegistry.build_splits() unisca e
splitti insieme partite reali e puzzle in un'unica pipeline coerente.

"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import chess
import pandas as pd
from tqdm import tqdm

from DatasetPipeline.Model.PositionGraphSchema import build_position_data
from DatasetPipeline.PositionQueue import PositionQueueRegistry

logger = logging.getLogger("puzzle_builder")


class PuzzleBuilder:
    """Builder per il dataset puzzle Lichess, allineato al contratto
    board-level di GamesBuilder (vedi docstring di modulo).

    Uso tipico (dopo GamesBuilder.run(), per condividere il registry e
    conoscere il prossimo game_id libero):

        games_result = games_builder.run()

        puzzle_builder = PuzzleBuilder(
            csv_path="lichess_puzzles.csv",
            mate_range=(1, 5),
            max_puzzles=100_000,
            avg_time_by_rating=avg_time_by_rating,
            queue_state_path="Dataset/position_queue_state.json",
            debug_jsonl_path="Dataset/Puzzles/puzzle_debug.jsonl",
        )
        puzzle_result = puzzle_builder.run(game_id_start=games_result["accepted_windows"])

        splits = PositionQueueRegistry.instance().build_splits(split_ratios=(0.7, 0.1, 0.2))
    """

    def __init__(
        self,
        csv_path: str,
        mate_range: Tuple[int, int] = (1, 5),
        max_puzzles: Optional[int] = None,
        avg_time_by_rating: Optional[Dict[int, float]] = None,
        chunksize: int = 50_000,
        queue_state_path: Optional[str] = None,
        debug_jsonl_path: Optional[str] = None,
        config_error_cls: type = ValueError,
    ) -> None:
        """
        Args:
            csv_path: percorso del CSV puzzle Lichess (PuzzleId, FEN,
                Moves, Rating, Themes, ...).
            mate_range: (min, max) inclusivi di N per il tema "mateInN"
                da includere.
            max_puzzles: limite superiore di righe CSV processate (dopo il
                filtro tema), None = nessun limite.
            avg_time_by_rating: mappa {bucket_rating: secondi_medi} da
                TimeStatBuilder, usata per simulare clock_seconds quando il
                puzzle non ha un tempo reale (i puzzle non hanno mai clock
                reale: e' sempre simulato, vedi _simulated_clock).
            chunksize: dimensione dei chunk di lettura del CSV (righe).
            queue_state_path: path del file di stato di
                PositionQueueRegistry, per ottenere la STESSA istanza
                condivisa (via PositionQueueRegistry.instance()) usata da
                GamesBuilder. Se None, usa l'istanza gia' eventualmente
                creata nel processo (o il default della classe).
            debug_jsonl_path: se fornito, scrive un file .jsonl di audit
                (fen, best_move_uci, puzzle_id, mate_n, rating, ply_idx,
                game_id) accanto ai dati accodati, un record per posizione,
                nello stesso ordine di accodamento.
            config_error_cls: classe di eccezione da sollevare per errori
                di configurazione (permette all'orchestratore di
                distinguere errori di config da altri errori).
        """
        self.csv_path = csv_path
        self.mate_range = mate_range
        self.max_puzzles = max_puzzles
        self.avg_time_by_rating = avg_time_by_rating or {}
        self.chunksize = chunksize
        self.debug_jsonl_path = debug_jsonl_path
        self._config_error_cls = config_error_cls

        self._queue_registry = PositionQueueRegistry.instance(state_path=queue_state_path)

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        lo, hi = self.mate_range
        if lo < 1:
            raise self._config_error_cls("mate_range deve iniziare da almeno 1.")
        if hi < lo:
            raise self._config_error_cls("mate_range non valido.")
        if not os.path.exists(self.csv_path):
            raise self._config_error_cls(f"CSV puzzle non trovato: {self.csv_path}.")
        if self.max_puzzles is not None and self.max_puzzles < 1:
            raise self._config_error_cls("max_puzzles deve essere >= 1 se specificato.")

    # ------------------------------------------------------------------
    # LETTURA E FILTRO CSV (invariati nella logica rispetto alla versione
    # precedente: filtro per tema mateInN dentro mate_range, limite
    # max_puzzles applicato DOPO il filtro).
    # ------------------------------------------------------------------
    def _load_filtered_rows(self) -> List[dict]:
        lo, hi = self.mate_range
        theme_pattern = "|".join(f"mateIn{n}" for n in range(lo, hi + 1))
        rows: List[dict] = []
        reader = pd.read_csv(self.csv_path, chunksize=self.chunksize)
        pbar = tqdm(desc="Lettura CSV puzzle", unit=" righe valide")

        for chunk in reader:
            mask = chunk["Themes"].str.contains(theme_pattern, na=False)
            filtered = chunk[mask]
            rows.extend(filtered.to_dict("records"))
            pbar.update(len(filtered))

            if self.max_puzzles and len(rows) >= self.max_puzzles:
                rows = rows[: self.max_puzzles]
                break
        pbar.close()
        return rows

    @staticmethod
    def _extract_mate_n(themes: str) -> int:
        for t in themes.split():
            if t.startswith("mateIn"):
                return int(t.replace("mateIn", ""))
        return 0

    def _simulated_clock(self, rating: float) -> float:
        """Tempo simulato per un puzzle (i puzzle non hanno clock reale):
        se disponibili statistiche per rating (da TimeStatBuilder), usa il
        bucket piu' vicino; altrimenti una rampa lineare semplice in
        funzione del rating come fallback grezzo."""
        if self.avg_time_by_rating:
            bucket = round(rating / 100) * 100
            return self.avg_time_by_rating.get(bucket, 15.0)
        return 5.0 + (rating / 3000.0) * 55.0

    # ------------------------------------------------------------------
    # PROCESSING: per ogni puzzle, replay ply-per-ply -> build_position_data
    # -> enqueue, esattamente come GamesBuilder fa per le finestre di
    # matto forzato da partite reali.
    # ------------------------------------------------------------------
    def run(self, game_id_start: int = 0) -> Dict[str, Any]:
        """Processa il CSV puzzle e accoda ogni posizione risultante sul
        PositionQueueRegistry condiviso.

        Args:
            game_id_start: primo game_id da assegnare (vedi docstring di
                modulo per il contratto anti-collisione con GamesBuilder).

        Returns:
            Dict con "processed_puzzles", "accepted_puzzles",
            "enqueued_positions", "mate_n_counts", "next_game_id" (utile
            per incatenare un'altra sorgente senza collisioni).
        """
        all_rows = self._load_filtered_rows()

        debug_records: List[Dict[str, Any]] = []
        next_game_id = game_id_start

        processed_puzzles = 0
        accepted_puzzles = 0
        enqueued_positions = 0
        mate_n_counts: Dict[int, int] = {}

        for row in tqdm(all_rows, desc="Costruzione posizioni puzzle"):
            processed_puzzles += 1
            uci_moves = str(row["Moves"]).split()
            if not uci_moves:
                continue

            try:
                board = chess.Board(row["FEN"])
            except ValueError as e:
                logger.warning(f"PuzzleId={row.get('PuzzleId')}: FEN non valido ({e}), scartato.")
                continue

            mate_n_iniziale = self._extract_mate_n(str(row.get("Themes", "")))
            if mate_n_iniziale <= 0:
                continue

            rating_raw = row.get("Rating")
            puzzle_rating = float(rating_raw) if pd.notna(rating_raw) else 1500.0
            clock_base = self._simulated_clock(puzzle_rating)

            first_move = chess.Move.from_uci(uci_moves[0])
            if first_move not in board.legal_moves:
                continue
            board.push(first_move)

            game_id = next_game_id
            next_game_id += 1
            puzzle_enqueued = 0

            for ply_idx, uci in enumerate(uci_moves[1:], start=1):
                move = chess.Move.from_uci(uci)

                if ply_idx % 2 == 0:
                    if move in board.legal_moves:
                        board.push(move)
                    continue

                if move not in board.legal_moves:
                    break

                current_mate_n = max(1, mate_n_iniziale - (ply_idx // 2))
                clock_seconds = clock_base * (1 + 0.1 * ply_idx)

                try:
                    data = build_position_data(
                        board=board,
                        best_move=move,
                        clock_seconds=clock_seconds,
                        game_id=game_id,
                        ply=ply_idx,
                    )
                except ValueError as e:
                    logger.warning(
                        f"PuzzleId={row.get('PuzzleId')} ply={ply_idx}: "
                        f"scarto la posizione ({e})."
                    )
                    board.push(move)
                    continue

                self._queue_registry.enqueue(
                    source_tag="puzzle",
                    data=data,
                    group_key=current_mate_n,
                )
                puzzle_enqueued += 1
                mate_n_counts[current_mate_n] = mate_n_counts.get(current_mate_n, 0) + 1

                if self.debug_jsonl_path is not None:
                    debug_records.append(
                        {
                            "puzzle_id": row.get("PuzzleId"),
                            "fen": board.fen(),
                            "best_move_uci": move.uci(),
                            "mate_n": current_mate_n,
                            "rating": puzzle_rating,
                            "ply_idx": ply_idx,
                            "game_id": game_id,
                        }
                    )

                board.push(move)

            if puzzle_enqueued > 0:
                accepted_puzzles += 1
                enqueued_positions += puzzle_enqueued

        if self.debug_jsonl_path is not None and debug_records:
            self._write_debug_jsonl(debug_records, self.debug_jsonl_path)

        self._log_summary(processed_puzzles, accepted_puzzles, enqueued_positions, mate_n_counts)

        return {
            "processed_puzzles": processed_puzzles,
            "accepted_puzzles": accepted_puzzles,
            "enqueued_positions": enqueued_positions,
            "mate_n_counts": mate_n_counts,
            "next_game_id": next_game_id,
        }

    @staticmethod
    def _write_debug_jsonl(records: List[Dict[str, Any]], path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)

    def _log_summary(
        self,
        processed_puzzles: int,
        accepted_puzzles: int,
        enqueued_positions: int,
        mate_n_counts: Dict[int, int],
    ) -> None:
        logger.info("=" * 60)
        logger.info("PUZZLE BUILDER — RIEPILOGO (schema board-level per posizione)")
        logger.info("=" * 60)
        logger.info(f"Puzzle processati: {processed_puzzles:,}")
        logger.info(f"Puzzle accettati (almeno una posizione accodata): {accepted_puzzles:,}")
        logger.info(f"Posizioni accodate: {enqueued_positions:,}")
        if mate_n_counts:
            logger.info("Per profondita' mate (N, mosse intere):")
            for n in sorted(mate_n_counts.keys()):
                logger.info(f"  n={n}: {mate_n_counts[n]:,}")
        logger.info("=" * 60)