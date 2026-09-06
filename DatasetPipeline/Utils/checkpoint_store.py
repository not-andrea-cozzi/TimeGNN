from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

logger = logging.getLogger("games_checkpoint_store")

DEFAULT_BUFFER_STATE_FILENAME = "games_buffer_state.pt"


class CheckpointStoreError(RuntimeError):
    pass


@dataclass
class _Window:
    game_id: str
    group_key: int
    positions: List[Data] = field(default_factory=list)


def _human_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return -1.0


class CheckpointStore:
    """Accumula finestre (liste di Data per una stessa partita/finestra di
    matto) in memoria, persiste lo stato per resume, e ad ogni checkpoint
    ricalcola lo split stratificato e riscrive i 3 file di output finali.

    NON e' un singleton (a differenza di PositionQueueRegistry): un
    GamesBuilder ne possiede una istanza propria, passata esplicitamente.
    """

    def __init__(
        self,
        output_dir: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int = 42,
        state_filename: str = DEFAULT_BUFFER_STATE_FILENAME,
        train_filename: str = "train_games.pt",
        val_filename: str = "val_games.pt",
        test_filename: str = "test_games.pt",
    ) -> None:
        if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
            raise CheckpointStoreError("split_ratios deve contenere 3 valori che sommano a 1.0.")

        self.output_dir = output_dir
        self.split_ratios = split_ratios
        self.seed = seed
        os.makedirs(output_dir, exist_ok=True)

        self._state_path = os.path.join(output_dir, state_filename)
        self.final_paths = {
            "train": os.path.join(output_dir, train_filename),
            "val": os.path.join(output_dir, val_filename),
            "test": os.path.join(output_dir, test_filename),
        }

        self._lock = threading.Lock()
        self._windows: Dict[str, _Window] = {}
        self._seen_game_ids: set = set()   # invariato

        # Contatori diagnostici cumulativi (vivono per tutta la durata del
        # processo, utili nei log per capire il trend tra un checkpoint e
        # l'altro senza dover fare la differenza a mano).
        self._checkpoints_written = 0
        self._total_windows_added = 0
        self._total_positions_added = 0

        logger.info(
            f"[CheckpointStore] Inizializzazione: output_dir='{output_dir}', "
            f"split_ratios={split_ratios} (train/val/test), seed={seed}."
        )
        logger.info(
            f"[CheckpointStore] File finali: train='{self.final_paths['train']}', "
            f"val='{self.final_paths['val']}', test='{self.final_paths['test']}'."
        )
        logger.info(f"[CheckpointStore] File di stato per resume: '{self._state_path}'.")

        self._load()

    # ------------------------------------------------------------------
    # RESUME
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self._state_path):
            logger.info(
                f"[CheckpointStore] Nessun file di stato preesistente in '{self._state_path}': "
                f"parto da buffer vuoto."
            )
            return

        logger.info(f"[CheckpointStore] Trovato file di stato '{self._state_path}', tentativo di ricarico...")
        t0 = time.monotonic()
        try:
            raw = torch.load(self._state_path, weights_only=False)
        except Exception as e:
            logger.warning(
                f"[CheckpointStore] Stato buffer in '{self._state_path}' illeggibile ({type(e).__name__}: {e}), "
                f"riparto da vuoto."
            )
            return

        loaded_windows = raw.get("windows", [])
        logger.debug(f"[CheckpointStore] {len(loaded_windows)} record di finestra letti dal file di stato.")

        skipped_empty = 0
        for rec in loaded_windows:
            window = _Window(
                game_id=rec["game_id"],
                group_key=rec["group_key"],
                positions=rec["positions"],
            )
            if not window.positions:
                skipped_empty += 1
                continue
            self._windows[window.game_id] = window
            self._seen_game_ids.add(window.game_id)

        elapsed = time.monotonic() - t0

        if skipped_empty:
            logger.warning(
                f"[CheckpointStore] {skipped_empty} finestre nel file di stato erano vuote "
                f"(nessuna posizione): scartate al resume."
            )

        if self._windows:
            by_bucket: Dict[int, int] = defaultdict(int)
            for w in self._windows.values():
                by_bucket[w.group_key] += 1
            total_positions = sum(len(w.positions) for w in self._windows.values())

            logger.info(
                f"[CheckpointStore] Resume completato in {elapsed:.2f}s: "
                f"{len(self._windows):,} finestre, {total_positions:,} posizioni ricaricate da checkpoint precedente."
            )
            for bucket in sorted(by_bucket.keys()):
                logger.debug(f"[CheckpointStore]   resume bucket mate_n={bucket}: {by_bucket[bucket]:,} finestre.")
        else:
            logger.info("[CheckpointStore] File di stato presente ma non conteneva finestre valide: buffer vuoto.")

    def _persist_state_locked(self) -> None:
        t0 = time.monotonic()
        tmp_path = self._state_path + ".tmp"
        payload = {
            "windows": [
                {"game_id": w.game_id, "group_key": w.group_key, "positions": w.positions}
                for w in self._windows.values()
            ]
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, self._state_path)
        elapsed = time.monotonic() - t0
        size_mb = _human_mb(self._state_path)
        logger.debug(
            f"[CheckpointStore] Stato di resume persistito in {elapsed:.2f}s "
            f"({size_mb:.2f} MB) -> '{self._state_path}'."
        )

    # ------------------------------------------------------------------
    # ACCUMULO
    # ------------------------------------------------------------------
    def add_window(self, game_id: str, group_key: int, positions: List[Data]) -> None:
        """Registra una finestra completa (tutte le posizioni di una
        partita/finestra di matto accettata). Non scrive su disco: la
        persistenza avviene solo su checkpoint() esplicito."""
        if not positions:
            logger.debug(f"[CheckpointStore] add_window ignorato: game_id={game_id} senza posizioni.")
            return
        with self._lock:
            if game_id in self._seen_game_ids:
                raise CheckpointStoreError(
                    f"add_window: game_id={game_id} gia' presente nel buffer (collisione o doppio enqueue)."
                )
            self._windows[game_id] = _Window(game_id=game_id, group_key=int(group_key), positions=positions)
            self._seen_game_ids.add(game_id)

            self._total_windows_added += 1
            self._total_positions_added += len(positions)

            logger.debug(
                f"[CheckpointStore] Finestra aggiunta: game_id={game_id}, mate_n={group_key}, "
                f"{len(positions)} posizioni (buffer ora: {len(self._windows):,} finestre totali)."
            )

            # Ogni 1000 finestre nuove, un log INFO di avanzamento (senza
            # bisogno di aspettare il prossimo checkpoint per sapere che
            # il buffer sta crescendo).
            if self._total_windows_added % 1000 == 0:
                logger.info(
                    f"[CheckpointStore] Avanzamento buffer: {self._total_windows_added:,} finestre accumulate "
                    f"da avvio processo ({self._total_positions_added:,} posizioni), "
                    f"{len(self._windows):,} attualmente in memoria."
                )

    def pending_windows(self) -> int:
        return len(self._windows)

    def pending_positions(self) -> int:
        return sum(len(w.positions) for w in self._windows.values())

    # ------------------------------------------------------------------
    # SPLIT STRATIFICATO (stessa logica di PositionQueueRegistry.build_splits)
    # ------------------------------------------------------------------
    def _compute_stratified_split(self) -> Tuple[Dict[str, List[Data]], Dict[str, int]]:
        t0 = time.monotonic()
        groups_of_windows: Dict[int, List[int]] = defaultdict(list)
        for gid, w in self._windows.items():
            groups_of_windows[w.group_key].append(gid)

        logger.debug(
            f"[CheckpointStore] Calcolo split stratificato su {len(groups_of_windows)} bucket "
            f"(mate_n distinti), {len(self._windows):,} finestre totali."
        )

        generator = torch.Generator().manual_seed(self.seed)
        train_ratio, val_ratio, _test_ratio = self.split_ratios
        result: Dict[str, List[Data]] = {"train": [], "val": [], "test": []}
        window_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

        for key in sorted(groups_of_windows.keys()):
            game_ids_in_group = sorted(groups_of_windows[key])
            n = len(game_ids_in_group)

            n_train = min(int(train_ratio * n), n)
            n_val = min(int(val_ratio * n), n - n_train)
            n_test = n - n_train - n_val

            perm = torch.randperm(n, generator=generator)
            shuffled_game_ids = [game_ids_in_group[i] for i in perm.tolist()]

            split_assignment = (
                [("train", gid) for gid in shuffled_game_ids[:n_train]]
                + [("val", gid) for gid in shuffled_game_ids[n_train:n_train + n_val]]
                + [("test", gid) for gid in shuffled_game_ids[n_train + n_val:]]
            )

            for split_name, gid in split_assignment:
                result[split_name].extend(self._windows[gid].positions)
                window_counts[split_name] += 1

            logger.debug(
                f"[CheckpointStore]   bucket mate_n={key}: {n:,} finestre totali -> "
                f"train={n_train:,}, val={n_val:,}, test={n_test:,}."
            )

        for split_name, data_list in result.items():
            if not data_list:
                continue
            perm = torch.randperm(len(data_list), generator=generator)
            result[split_name] = [data_list[i] for i in perm.tolist()]

        elapsed = time.monotonic() - t0
        logger.debug(f"[CheckpointStore] Split stratificato calcolato in {elapsed:.2f}s.")

        return result, window_counts

    # ------------------------------------------------------------------
    # CHECKPOINT: persiste stato di resume + riscrive i 3 file finali
    # ------------------------------------------------------------------
    def checkpoint(self) -> Dict[str, int]:
        """Ricalcola lo split stratificato su TUTTE le finestre accumulate
        finora e riscrive atomicamente train_games.pt/val_games.pt/
        test_games.pt. Persiste anche lo stato di resume.

        Returns:
            Dict con il conteggio di finestre per split (diagnostico).
        """
        checkpoint_t0 = time.monotonic()
        with self._lock:
            if not self._windows:
                logger.info("[CheckpointStore] checkpoint() chiamato con buffer vuoto: nessun file scritto.")
                return {"train": 0, "val": 0, "test": 0}

            self._checkpoints_written += 1
            checkpoint_index = self._checkpoints_written

            logger.info(
                f"[CheckpointStore] --- Avvio checkpoint #{checkpoint_index}: "
                f"{len(self._windows):,} finestre, {self.pending_positions():,} posizioni nel buffer ---"
            )

            self._persist_state_locked()
            splits, window_counts = self._compute_stratified_split()

            for split_name, data_list in splits.items():
                write_t0 = time.monotonic()
                out_path = self.final_paths[split_name]
                tmp_path = out_path + ".tmp"
                torch.save(data_list, tmp_path)
                os.replace(tmp_path, out_path)
                write_elapsed = time.monotonic() - write_t0
                size_mb = _human_mb(out_path)
                logger.info(
                    f"[CheckpointStore] Scritto {split_name}: {window_counts[split_name]:,} finestre, "
                    f"{len(data_list):,} posizioni, {size_mb:.2f} MB, {write_elapsed:.2f}s -> '{out_path}'."
                )

            total_positions = sum(len(v) for v in splits.values())
            total_windows = sum(window_counts.values())
            total_elapsed = time.monotonic() - checkpoint_t0

            logger.info(
                f"[CheckpointStore] --- Checkpoint #{checkpoint_index} completato in {total_elapsed:.2f}s: "
                f"{total_windows:,} finestre, {total_positions:,} posizioni totali su disco ---"
            )
            logger.info(
                f"[CheckpointStore] Riepilogo split: train={window_counts['train']:,} finestre, "
                f"val={window_counts['val']:,} finestre, test={window_counts['test']:,} finestre "
                f"(ratio target={self.split_ratios})."
            )

            return window_counts

    def finalize(self) -> Dict[str, int]:
        """Alias esplicito per il checkpoint finale a fine pipeline (stessa
        logica di checkpoint(), nome separato solo per leggibilita' del
        chiamante)."""
        logger.info("[CheckpointStore] finalize(): eseguo l'ultimo checkpoint di fine pipeline.")
        result = self.checkpoint()
        logger.info(
            f"[CheckpointStore] Pipeline terminata: {self._checkpoints_written} checkpoint scritti in totale "
            f"durante questa esecuzione, {self._total_windows_added:,} finestre aggiunte al buffer "
            f"({self._total_positions_added:,} posizioni)."
        )
        return result