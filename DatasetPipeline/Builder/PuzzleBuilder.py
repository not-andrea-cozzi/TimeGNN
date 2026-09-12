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
    mate_range: Tuple[int, int] = (1, 5)
    max_puzzles: Optional[int] = None
    max_puzzles_per_theme: Optional[int] = None
    avg_time_by_rating: Dict[int, float] = field(default_factory=dict)
    chunksize: int = 50_000

    queue_state_path: Optional[str] = None
    shard_size: int = 500

    save_debug_jsonl: bool = True
    debug_jsonl_dir: Optional[str] = None

    split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2)
    split_seed: int = 42

    max_positions_per_puzzle: Optional[int] = None
    source_tag: str = "puzzle"

    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    max_piece_count: Optional[int] = None
    min_material_for_mate_attempt: int = 0
    min_material_diff_for_mate_attempt: int = 0
    require_heavy_piece: bool = False
    skip_trivial_endgame: bool = False
    dedupe_positions: bool = True

from DatasetPipeline.Model.ChessConstants import PIECE_VALUES as _CLASS_PIECE_VALUES
class PuzzleBuilder:

    _PIECE_VALUES: Dict[int, int] = _CLASS_PIECE_VALUES

    def __init__(self, config: PuzzleBuilderConfig):
        self.config = config
        self._validate_config()

        self._registry = PositionQueueRegistry.instance(
            state_path=config.queue_state_path,
            shard_size=config.shard_size
        )

        self._debug_records: List[Dict] = []
        self._debug_jsonl_path = None
        self._debug_records_raw_path = None
        if config.save_debug_jsonl:
            if config.debug_jsonl_dir:
                os.makedirs(config.debug_jsonl_dir, exist_ok=True)
                debug_dir = config.debug_jsonl_dir
            else:
                debug_dir = os.path.dirname(config.queue_state_path) if config.queue_state_path else "."
                os.makedirs(debug_dir, exist_ok=True)
            self._debug_jsonl_path = os.path.join(debug_dir, "puzzle_debug.jsonl")
            self._debug_records_raw_path = os.path.join(debug_dir, "puzzle_debug_records.pending.jsonl")

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

    def _load_filtered_rows(self) -> List[Dict]:
        lo, hi = self.config.mate_range
        themes_wanted = [f"mateIn{n}" for n in range(lo, hi + 1)]
        theme_pattern = "|".join(themes_wanted)

        reader = pd.read_csv(self.config.csv_path, chunksize=self.config.chunksize)

        if self.config.max_puzzles_per_theme is not None:
            return self._load_filtered_rows_stratified(reader, themes_wanted, theme_pattern)
        return self._load_filtered_rows_flat(reader, theme_pattern)

    def _row_passes_rating_filter(self, row: Dict) -> bool:
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
        if self.config.avg_time_by_rating:
            bucket = round(rating / 100) * 100
            return self.config.avg_time_by_rating.get(bucket, 15.0)
        return 5.0 + (rating / 3000.0) * 55.0

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

    def run(self) -> Dict[str, Any]:
        all_rows = self._load_filtered_rows()
        processed = 0
        accepted_puzzles = 0
        enqueued_positions = 0
        mate_n_counts: Dict[int, int] = defaultdict(int)
        source_mate_n_counts: Dict[int, int] = defaultdict(int)
        quality_filtered_positions = 0
        deduped_positions = 0

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

            first_move = chess.Move.from_uci(uci_moves[0])
            if first_move not in board.legal_moves:
                continue
            board.push(first_move)

            game_id = f"{self.config.source_tag}_{puzzle_id_raw}"

            window_group_key = mate_n_iniziale

            puzzle_enqueued = 0
            seen_positions: set = set()
            for ply_idx, uci in enumerate(uci_moves[1:], start=1):
                move = chess.Move.from_uci(uci)

                if ply_idx % 2 == 0:
                    if move in board.legal_moves:
                        board.push(move)
                    continue

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

                import random as _random

                current_mate_n = max(1, mate_n_iniziale - (ply_idx // 2))
                base_scaled = clock_base * (1 + 0.1 * ply_idx)
                noise_factor = _random.Random(f"{game_id}:{ply_idx}").gauss(1.0, 0.15)
                clock_seconds = max(0.5, base_scaled * max(0.3, noise_factor))

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
                    board.push(move)
                    continue

                self._registry.enqueue(
                    source_tag=self.config.source_tag,
                    data=data,
                    group_key=window_group_key,
                )
                puzzle_enqueued += 1
                mate_n_counts[current_mate_n] += 1

                if self.config.save_debug_jsonl:
                    self._debug_records.append({
                        "puzzle_id": row.get("PuzzleId"),
                        "fen": board.fen(),
                        "best_move_uci": move.uci(),
                        "mate_n": current_mate_n,
                        "mate_n_window": window_group_key,
                        "rating": puzzle_rating,
                        "ply_idx": ply_idx,
                        "game_id": game_id,
                        "source": self.config.source_tag,
                    })

                board.push(move)

            if puzzle_enqueued > 0:
                accepted_puzzles += 1
                enqueued_positions += puzzle_enqueued

            if (self.config.max_positions_per_puzzle is not None and
                enqueued_positions >= self.config.max_positions_per_puzzle):
                break

        self._registry.flush()

        if self.config.save_debug_jsonl and self._debug_records_raw_path:
            self._persist_pending_debug_records()

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

    def _persist_pending_debug_records(self) -> None:
        tmp_path = self._debug_records_raw_path + ".tmp"
        with open(tmp_path, "a", encoding="utf-8") as f:
            for rec in self._debug_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if os.path.exists(self._debug_records_raw_path):
            with open(self._debug_records_raw_path, "r", encoding="utf-8") as existing, \
                 open(tmp_path, "r", encoding="utf-8") as new_part:
                merged_lines = existing.readlines() + new_part.readlines()
            with open(tmp_path, "w", encoding="utf-8") as merged:
                merged.writelines(merged_lines)
        os.replace(tmp_path, self._debug_records_raw_path)

    @staticmethod
    def write_debug_jsonl_from_pending(
        pending_path: str,
        output_path: str,
        split_assignment: Dict[str, str],
    ) -> Optional[str]:
        if not os.path.exists(pending_path):
            return None

        missing_game_ids = set()
        tmp_path = output_path + ".tmp"
        wrote_any = False
        with open(pending_path, "r", encoding="utf-8") as src, \
             open(tmp_path, "w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                game_id = rec["game_id"]
                split_name = split_assignment.get(game_id)
                if split_name is None:
                    missing_game_ids.add(game_id)
                    continue
                rec["split"] = split_name
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote_any = True

        if not wrote_any:
            os.remove(tmp_path)
            return None

        os.replace(tmp_path, output_path)
        os.remove(pending_path)

        if missing_game_ids:
            logger.warning(
                "[PuzzleBuilder] %d game_id presenti nel debug JSONL ma assenti "
                "dallo split_assignment reale (probabile game_id scartato dal "
                "registry prima di build_splits): esclusi dal file scritto.",
                len(missing_game_ids),
            )

        return output_path

    def write_debug_jsonl(self, split_assignment: Dict[str, str]) -> Optional[str]:
        if not self.config.save_debug_jsonl or not self._debug_jsonl_path:
            return None
        if not self._debug_records:
            return None

        missing_game_ids = set()
        tmp_path = self._debug_jsonl_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in self._debug_records:
                game_id = rec["game_id"]
                split_name = split_assignment.get(game_id)
                if split_name is None:
                    missing_game_ids.add(game_id)
                    continue
                rec_out = dict(rec)
                rec_out["split"] = split_name
                f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self._debug_jsonl_path)

        if missing_game_ids:
            logger.warning(
                "[PuzzleBuilder] %d game_id presenti nel debug JSONL ma assenti "
                "dallo split_assignment reale (probabile game_id scartato dal "
                "registry prima di build_splits): esclusi dal file scritto.",
                len(missing_game_ids),
            )

        return self._debug_jsonl_path

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