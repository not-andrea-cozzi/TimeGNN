from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import chess
import pandas as pd
import torch
from tqdm import tqdm

from DatasetPipeline.Model.PositionGraphSchema import build_position_data
from DatasetPipeline.PositionQueue import PositionQueueRegistry

logger = logging.getLogger("puzzle_builder")


@dataclass(frozen=True)
class PuzzleBuilderConfig:
    """Configurazione per PuzzleBuilder (allineata a GamesBuilderConfig)."""
    csv_path: str
    mate_range: Tuple[int, int] = (1, 5)          # mateInN da includere
    max_puzzles: Optional[int] = None             # limite dopo il filtro tematico
    avg_time_by_rating: Dict[int, float] = field(default_factory=dict)  # da TimeStatBuilder
    chunksize: int = 50_000                       # lettura chunk CSV

    # Queue condivisa
    queue_state_path: Optional[str] = None        # se None usa default di PositionQueueRegistry
    shard_size: int = 500

    # Debug
    save_debug_jsonl: bool = True
    debug_jsonl_dir: Optional[str] = None         # se None usa dir del queue_state_path

    # Split per debug (usato solo per scrivere JSONL separati, lo split reale è in registry)
    split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2)
    split_seed: int = 42

    # Numero massimo di posizioni per puzzle (None = tutte)
    max_positions_per_puzzle: Optional[int] = None


class PuzzleBuilder:
    """
    Builder per dataset puzzle Lichess, allineato al contratto di GamesBuilder.
    Ogni puzzle viene trattato come una "finestra" di posizioni (i ply alterni)
    e ogni posizione viene accodata in PositionQueueRegistry con un game_id univoco
    (uuid) per puzzle, in modo da garantire split coerenti (tutte le posizioni
    di uno stesso puzzle vanno nello stesso split).

    Uso tipico:
        config = PuzzleBuilderConfig(csv_path="lichess_puzzles.csv", ...)
        builder = PuzzleBuilder(config)
        result = builder.run()
        # poi, insieme a GamesBuilder, chiamare:
        registry = PositionQueueRegistry.instance()
        splits = registry.build_splits(...)
    """

    def __init__(self, config: PuzzleBuilderConfig):
        self.config = config
        self._validate_config()

        # Ottieni istanza condivisa del registry
        self._registry = PositionQueueRegistry.instance(
            state_path=config.queue_state_path,
            shard_size=config.shard_size
        )

        # Preparazione debug JSONL
        self._debug_records: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
        self._debug_jsonl_path = None
        if config.save_debug_jsonl:
            if config.debug_jsonl_dir:
                os.makedirs(config.debug_jsonl_dir, exist_ok=True)
                self._debug_jsonl_path = os.path.join(config.debug_jsonl_dir, "puzzle_debug.jsonl")
            else:
                state_dir = os.path.dirname(config.queue_state_path) if config.queue_state_path else "."
                os.makedirs(state_dir, exist_ok=True)
                self._debug_jsonl_path = os.path.join(state_dir, "puzzle_debug.jsonl")

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.mate_range[0] < 1:
            raise ValueError("mate_range deve iniziare da almeno 1.")
        if cfg.mate_range[1] < cfg.mate_range[0]:
            raise ValueError("mate_range non valido.")
        if not os.path.exists(cfg.csv_path):
            raise ValueError(f"CSV puzzle non trovato: {cfg.csv_path}.")
        if cfg.max_puzzles is not None and cfg.max_puzzles < 1:
            raise ValueError("max_puzzles deve essere >= 1 se specificato.")
        if cfg.chunksize < 1:
            raise ValueError("chunksize deve essere >= 1.")

    # ------------------------------------------------------------------
    # LETTURA E FILTRO CSV
    # ------------------------------------------------------------------
    def _load_filtered_rows(self) -> List[Dict]:
        lo, hi = self.config.mate_range
        theme_pattern = "|".join(f"mateIn{n}" for n in range(lo, hi + 1))
        rows: List[Dict] = []
        reader = pd.read_csv(self.config.csv_path, chunksize=self.config.chunksize)
        pbar = tqdm(desc="Lettura CSV puzzle", unit=" righe valide")
        for chunk in reader:
            mask = chunk["Themes"].str.contains(theme_pattern, na=False)
            filtered = chunk[mask]
            rows.extend(filtered.to_dict("records"))
            pbar.update(len(filtered))
            if self.config.max_puzzles and len(rows) >= self.config.max_puzzles:
                rows = rows[:self.config.max_puzzles]
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
        """Tempo simulato per puzzle (i puzzle non hanno clock reale)."""
        if self.config.avg_time_by_rating:
            bucket = round(rating / 100) * 100
            return self.config.avg_time_by_rating.get(bucket, 15.0)
        # Fallback lineare
        return 5.0 + (rating / 3000.0) * 55.0

    def _assign_split(self, game_id: int) -> str:
        """Split deterministico per debug JSONL (usa lo stesso seed di GamesBuilder)."""
        import random
        rng = random.Random(self.config.split_seed + game_id)
        val = rng.random()
        train, val_ratio, _ = self.config.split_ratios
        if val < train:
            return "train"
        if val < train + val_ratio:
            return "val"
        return "test"

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Processa il CSV, accoda le posizioni nel registry condiviso."""
        all_rows = self._load_filtered_rows()
        processed = 0
        accepted_puzzles = 0
        enqueued_positions = 0
        mate_n_counts: Dict[int, int] = defaultdict(int)

        for row in tqdm(all_rows, desc="Costruzione posizioni puzzle"):
            processed += 1
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

            # Prima mossa (quella del puzzle) – la applichiamo subito per partire dalla posizione successiva
            first_move = chess.Move.from_uci(uci_moves[0])
            if first_move not in board.legal_moves:
                continue
            board.push(first_move)

            # Game ID univoco per questo puzzle (come in GamesBuilder)
            game_id = uuid.uuid4().int & ((1 << 63) - 1)

            # I puzzle hanno una sequenza di mosse: la soluzione.
            # Prendiamo solo i ply alterni (quelli in cui il solver deve muovere)
            puzzle_enqueued = 0
            for ply_idx, uci in enumerate(uci_moves[1:], start=1):
                move = chess.Move.from_uci(uci)

                # Se è una mossa del solver (ply dispari nel contesto del puzzle)
                if ply_idx % 2 == 0:
                    # È la risposta dell'avversario: la applichiamo e continuiamo
                    if move in board.legal_moves:
                        board.push(move)
                    continue

                # Questa è una mossa che il solver deve trovare (ply dispari)
                if move not in board.legal_moves:
                    break

                current_mate_n = max(1, mate_n_iniziale - (ply_idx // 2))
                # Simuliamo clock crescente con il numero di mosse
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
                    # Continuiamo comunque con la prossima mossa
                    board.push(move)
                    continue

                # Accoda la posizione nel registry condiviso
                self._registry.enqueue(
                    source_tag="puzzle",
                    data=data,
                    group_key=current_mate_n,
                )
                puzzle_enqueued += 1
                mate_n_counts[current_mate_n] += 1

                # Accumula debug
                if self.config.save_debug_jsonl:
                    split_name = self._assign_split(game_id)
                    self._debug_records[split_name].append({
                        "puzzle_id": row.get("PuzzleId"),
                        "fen": board.fen(),
                        "best_move_uci": move.uci(),
                        "mate_n": current_mate_n,
                        "rating": puzzle_rating,
                        "ply_idx": ply_idx,
                        "game_id": game_id,
                        "source": "puzzle",
                    })

                board.push(move)

            if puzzle_enqueued > 0:
                accepted_puzzles += 1
                enqueued_positions += puzzle_enqueued

            # Limite opzionale per puzzle (posizioni)
            if (self.config.max_positions_per_puzzle is not None and
                enqueued_positions >= self.config.max_positions_per_puzzle):
                break

        # Flush coda (scrive shard residui)
        self._registry.flush()

        # Scrive debug JSONL
        if self.config.save_debug_jsonl and self._debug_jsonl_path:
            self._write_debug_jsonl()

        self._log_summary(processed, accepted_puzzles, enqueued_positions, mate_n_counts)

        return {
            "processed_puzzles": processed,
            "accepted_puzzles": accepted_puzzles,
            "enqueued_positions": enqueued_positions,
            "mate_n_counts": dict(mate_n_counts),
        }

    def _write_debug_jsonl(self) -> None:
        all_records = []
        for split in ("train", "val", "test"):
            all_records.extend(self._debug_records.get(split, []))
        if not all_records:
            return
        tmp_path = self._debug_jsonl_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self._debug_jsonl_path)
        logger.info(f"Debug JSONL puzzle scritto in {self._debug_jsonl_path} ({len(all_records)} record)")

    def _log_summary(self, processed, accepted, enqueued, mate_n_counts):
        logger.info("=" * 60)
        logger.info("PUZZLE BUILDER — RIEPILOGO (allineato a GamesBuilder)")
        logger.info("=" * 60)
        logger.info(f"Puzzle processati: {processed:,}")
        logger.info(f"Puzzle accettati (almeno una posizione): {accepted:,}")
        logger.info(f"Posizioni accodate: {enqueued:,}")
        if mate_n_counts:
            logger.info("Per profondità mate (N, mosse intere):")
            for n in sorted(mate_n_counts.keys()):
                logger.info(f"  n={n}: {mate_n_counts[n]:,}")
        logger.info("=" * 60)


