from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import chess
import pandas as pd
import torch
from tqdm import tqdm

from DatasetPipeline.Model.PositionGraphSchema import build_position_data
from DatasetPipeline.PositionQueue import PositionQueueRegistry
from DatasetPipeline.Utils.compatibility_filters import (
    has_mating_material,
    is_trivially_drawn_endgame,
    mover_has_heavy_piece,
    parse_rating_strict,
)

logger = logging.getLogger("puzzle_builder")


@dataclass(frozen=True)
class PuzzleBuilderConfig:
    csv_path: str
    mate_range: Tuple[int, int] = (1, 5)          # mateInN da includere
    max_puzzles: Optional[int] = None             # limite TOTALE dopo il filtro tematico (usato solo se max_puzzles_per_theme è None)
    max_puzzles_per_theme: Optional[int] = None   # tetto per singolo tema mateInN, per campionamento stratificato
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
    source_tag: str = "puzzle"

    # --- Filtri di compatibilita' applicabili ai puzzle (vedi docstring) ---
    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    max_piece_count: Optional[int] = None
    min_material_for_mate_attempt: int = 0
    min_material_diff_for_mate_attempt: int = 0
    require_heavy_piece: bool = False
    skip_trivial_endgame: bool = False
    dedupe_positions: bool = True


class PuzzleBuilder:

    _PIECE_VALUES: Dict[int, int] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

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
        if cfg.max_puzzles_per_theme is not None and cfg.max_puzzles_per_theme < 1:
            raise ValueError("max_puzzles_per_theme deve essere >= 1 se specificato.")
        if cfg.chunksize < 1:
            raise ValueError("chunksize deve essere >= 1.")
        if cfg.min_rating is not None and cfg.max_rating is not None and cfg.min_rating > cfg.max_rating:
            raise ValueError("min_rating non puo' essere maggiore di max_rating.")
        if cfg.max_piece_count is not None and cfg.max_piece_count < 2:
            raise ValueError("max_piece_count deve essere >= 2 se specificato (servono almeno i due Re).")

    # ------------------------------------------------------------------
    # LETTURA E FILTRO CSV
    # ------------------------------------------------------------------
    def _load_filtered_rows(self) -> List[Dict]:
        lo, hi = self.config.mate_range
        themes_wanted = [f"mateIn{n}" for n in range(lo, hi + 1)]
        theme_pattern = "|".join(themes_wanted)

        reader = pd.read_csv(self.config.csv_path, chunksize=self.config.chunksize)

        if self.config.max_puzzles_per_theme is not None:
            return self._load_filtered_rows_stratified(reader, themes_wanted, theme_pattern)
        return self._load_filtered_rows_flat(reader, theme_pattern)

    def _row_passes_rating_filter(self, row: Dict) -> bool:
        """min_rating/max_rating a livello di PUZZLE INTERO (un solo campo
        Rating per riga, a differenza di WhiteElo/BlackElo di GamesBuilder).
        Nessun bound configurato -> passa sempre."""
        cfg = self.config
        if cfg.min_rating is None and cfg.max_rating is None:
            return True
        rating = parse_rating_strict(row.get("Rating"))
        if rating is None:
            return False
        if cfg.min_rating is not None and rating < cfg.min_rating:
            return False
        if cfg.max_rating is not None and rating > cfg.max_rating:
            return False
        return True

    def _load_filtered_rows_flat(self, reader, theme_pattern: str) -> List[Dict]:
        """Comportamento storico: taglio secco a max_puzzles righe totali."""
        rows: List[Dict] = []
        pbar = tqdm(desc="Lettura CSV puzzle (flat)", unit=" righe valide")
        for chunk in reader:
            mask = chunk["Themes"].str.contains(theme_pattern, na=False)
            filtered = chunk[mask]
            for record in filtered.to_dict("records"):
                if not self._row_passes_rating_filter(record):
                    continue
                rows.append(record)
                pbar.update(1)
            if self.config.max_puzzles and len(rows) >= self.config.max_puzzles:
                rows = rows[:self.config.max_puzzles]
                break
        pbar.close()
        return rows

    def _load_filtered_rows_stratified(self, reader, themes_wanted: List[str], theme_pattern: str) -> List[Dict]:
        """Campionamento stratificato: fino a max_puzzles_per_theme righe per
        CIASCUN tema mateInN, cosi' ogni bucket di profondita' mate riceve
        una quota garantita di puzzle sorgente (vedi NOTA CAMPIONAMENTO
        STRATIFICATO nel docstring di classe)."""
        cap = self.config.max_puzzles_per_theme
        rows_by_theme: Dict[str, List[Dict]] = {t: [] for t in themes_wanted}

        pbar = tqdm(desc="Lettura CSV puzzle (stratificato)", unit=" righe valide")
        for chunk in reader:
            mask = chunk["Themes"].str.contains(theme_pattern, na=False)
            filtered = chunk[mask]
            if filtered.empty:
                continue

            for record in filtered.to_dict("records"):
                if not self._row_passes_rating_filter(record):
                    continue

                theme_found = self._extract_theme_tag(str(record.get("Themes", "")), themes_wanted)
                if theme_found is None:
                    continue
                bucket = rows_by_theme[theme_found]
                if len(bucket) < cap:
                    bucket.append(record)
                    pbar.update(1)

            # Stop anticipato se TUTTI i bucket sono pieni
            if all(len(v) >= cap for v in rows_by_theme.values()):
                break
        pbar.close()

        all_rows: List[Dict] = []
        for theme in themes_wanted:
            found = len(rows_by_theme[theme])
            if found < cap:
                logger.warning(
                    f"Tema '{theme}': solo {found}/{cap} puzzle trovati nel CSV "
                    f"(il dataset Lichess ne contiene meno di quanti richiesti, "
                    f"anche considerando il filtro rating configurato)."
                )
            all_rows.extend(rows_by_theme[theme])

        if self.config.max_puzzles is not None and len(all_rows) > self.config.max_puzzles:
            logger.info(
                f"Campionamento stratificato ha prodotto {len(all_rows)} righe, "
                f"troncate a max_puzzles={self.config.max_puzzles} (taglio finale, "
                f"puo' rompere la stratificazione se applicato qui: valuta di "
                f"alzare max_puzzles invece)."
            )
            all_rows = all_rows[: self.config.max_puzzles]

        logger.info(
            "Distribuzione puzzle sorgente per tema (prima della generazione posizioni): "
            + ", ".join(f"{t}={len(rows_by_theme[t])}" for t in themes_wanted)
        )
        return all_rows

    @staticmethod
    def _extract_theme_tag(themes: str, themes_wanted: List[str]) -> Optional[str]:
        """Ritorna il PRIMO tema tra quelli cercati (mateIn1..mateInN) presente
        nella stringa Themes della riga, o None se nessuno matcha (non
        dovrebbe succedere se la riga è già passata dal filtro .str.contains,
        ma per sicurezza in caso di match parziale/overlap)."""
        tokens = set(themes.split())
        for t in themes_wanted:
            if t in tokens:
                return t
        return None

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

    def _assign_split(self, game_id: str) -> str:
        """Split deterministico per debug JSONL (usa lo stesso seed di GamesBuilder)."""
        import random
        rng = random.Random(self.config.split_seed + hash(game_id))
        val = rng.random()
        train, val_ratio, _ = self.config.split_ratios
        if val < train:
            return "train"
        if val < train + val_ratio:
            return "val"
        return "test"

    # ------------------------------------------------------------------
    # FILTRI DI COMPATIBILITA' SU SINGOLA POSIZIONE SOLVER
    # ------------------------------------------------------------------
    def _material_by_color(self, board: "chess.Board") -> Tuple[int, int]:
        white_mat = black_mat = 0
        for p in board.piece_map().values():
            val = self._PIECE_VALUES.get(p.piece_type, 0)
            if p.color == chess.WHITE:
                white_mat += val
            else:
                black_mat += val
        return white_mat, black_mat

    def _position_passes_quality_filters(self, board: "chess.Board") -> bool:
        cfg = self.config

        if cfg.max_piece_count is not None and len(board.piece_map()) > cfg.max_piece_count:
            return False

        from DatasetPipeline.Utils.compatibility_filters import QualityFilterConfig
        quality_cfg = QualityFilterConfig(
            min_material_for_mate_attempt=cfg.min_material_for_mate_attempt,
            min_material_diff_for_mate_attempt=cfg.min_material_diff_for_mate_attempt,
            require_heavy_piece=cfg.require_heavy_piece,
            skip_trivial_endgame=cfg.skip_trivial_endgame,
        )

        if not has_mating_material(board, quality_cfg):
            return False
        if cfg.require_heavy_piece and not mover_has_heavy_piece(board):
            return False
        if cfg.skip_trivial_endgame and is_trivially_drawn_endgame(board):
            return False
        return True

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
        source_mate_n_counts: Dict[int, int] = defaultdict(int)  # quanti PUZZLE sorgente per mate_n_iniziale (diagnostico)
        quality_filtered_positions = 0  # diagnostico: posizioni scartate dai filtri di compatibilita'
        deduped_positions = 0  # diagnostico: posizioni scartate perche' gia' viste nello stesso puzzle

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

            source_mate_n_counts[mate_n_iniziale] += 1

            puzzle_id_raw = row.get("PuzzleId")
            if not puzzle_id_raw:
                logger.warning("Riga puzzle senza PuzzleId, scartata.")
                continue

            rating_raw = row.get("Rating")
            puzzle_rating = float(rating_raw) if pd.notna(rating_raw) else 1500.0
            clock_base = self._simulated_clock(puzzle_rating)

            # Prima mossa (quella del puzzle) – la applichiamo subito per partire dalla posizione successiva
            first_move = chess.Move.from_uci(uci_moves[0])
            if first_move not in board.legal_moves:
                continue
            board.push(first_move)

            game_id = f"{self.config.source_tag}_{puzzle_id_raw}"

            window_group_key = mate_n_iniziale

            # I puzzle hanno una sequenza di mosse: la soluzione.
            # Prendiamo solo i ply alterni (quelli in cui il solver deve muovere)
            puzzle_enqueued = 0
            seen_positions: set = set()  # dedupe_positions: FEN troncato gia' visto in QUESTO puzzle
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

                if self.config.dedupe_positions:
                    position_key = " ".join(board.fen().split(" ")[:4])
                    if position_key in seen_positions:
                        deduped_positions += 1
                        board.push(move)
                        continue
                    seen_positions.add(position_key)

                if not self._position_passes_quality_filters(board):
                    quality_filtered_positions += 1
                    board.push(move)
                    continue

                current_mate_n = max(1, mate_n_iniziale - (ply_idx // 2))
                # Simuliamo clock crescente con il numero di mosse
                clock_seconds = clock_base * (1 + 0.1 * ply_idx)

                try:
                    data = build_position_data(
                        board=board,
                        best_move=move,
                        clock_seconds=clock_seconds,
                        rating=puzzle_rating,
                        game_id=game_id,
                        ply=ply_idx,
                        mate_n=int(current_mate_n),
                    )
                except ValueError as e:
                    logger.warning(
                        f"PuzzleId={row.get('PuzzleId')} ply={ply_idx}: "
                        f"scarto la posizione ({e})."
                    )
                    # Continuiamo comunque con la prossima mossa
                    board.push(move)
                    continue

                self._registry.enqueue(
                    source_tag=self.config.source_tag,   # era "puzzle"
                    data=data,
                    group_key=window_group_key,
                )
                puzzle_enqueued += 1
                mate_n_counts[current_mate_n] += 1

                if self.config.save_debug_jsonl:
                    split_name = self._assign_split(game_id)
                    self._debug_records[split_name].append({
                        "puzzle_id": row.get("PuzzleId"),
                        "fen": board.fen(),
                        "best_move_uci": move.uci(),
                        "mate_n": current_mate_n,
                        "mate_n_window": window_group_key,
                        "rating": puzzle_rating,
                        "ply_idx": ply_idx,
                        "game_id": game_id,
                        "source": self.config.source_tag,   # era "puzzle"
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

        self._log_summary(
            processed, accepted_puzzles, enqueued_positions, mate_n_counts,
            source_mate_n_counts, quality_filtered_positions, deduped_positions,
        )

        return {
            "processed_puzzles": processed,
            "accepted_puzzles": accepted_puzzles,
            "enqueued_positions": enqueued_positions,
            "mate_n_counts": dict(mate_n_counts),
            "source_mate_n_counts": dict(source_mate_n_counts),
            "quality_filtered_positions": quality_filtered_positions,
            "deduped_positions": deduped_positions,
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

    def _log_summary(self, processed, accepted, enqueued, mate_n_counts,
                      source_mate_n_counts, quality_filtered_positions, deduped_positions):
        logger.info("=" * 60)
        logger.info("PUZZLE BUILDER — RIEPILOGO (allineato a GamesBuilder)")
        logger.info("=" * 60)
        logger.info(f"Puzzle processati: {processed:,}")
        logger.info(f"Puzzle accettati (almeno una posizione): {accepted:,}")
        logger.info(f"Posizioni accodate: {enqueued:,}")
        logger.info(f"Posizioni scartate da filtri di compatibilita': {quality_filtered_positions:,}")
        logger.info(f"Posizioni scartate da dedupe_positions: {deduped_positions:,}")
        if source_mate_n_counts:
            logger.info("Puzzle SORGENTE per tema mateInN dichiarato (prima della generazione posizioni):")
            for n in sorted(source_mate_n_counts.keys()):
                logger.info(f"  mateIn{n}: {source_mate_n_counts[n]:,} puzzle")
        if mate_n_counts:
            logger.info("Posizioni GENERATE per profondità mate REALE (current_mate_n, N mosse intere):")
            for n in sorted(mate_n_counts.keys()):
                logger.info(f"  n={n}: {mate_n_counts[n]:,}")
        logger.info("=" * 60)