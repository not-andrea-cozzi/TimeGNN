from __future__ import annotations

import argparse
import atexit
import functools
import glob
import json
import logging
import multiprocessing as mp
import os
import re
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
import yaml

from DatasetPipeline.Model.PositionGraphSchema import decode_move
from TrainPipeline.Shard.ShardDataset import ShardedGraphDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_llm")


# ============================================================================
# Scrittura JSON atomica (evita file troncati a 0 byte).
# ============================================================================

def _atomic_json_dump(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ============================================================================
# Prompt / estrazione mosse (invariato)
# ============================================================================

_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
_SAN_RE = re.compile(
    r"\b(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b"
)


def _build_llm_prompt(fen: str) -> str:
    return (
        "You are a chess engine. Analyze the position given below (in FEN "
        "notation) and find the best move for the side to move. Respond "
        "with ONLY the move in UCI notation: the origin square followed by "
        "the destination square (both as file-letter + rank-number, "
        "lowercase), optionally followed by a promotion piece letter if "
        "promoting a pawn. Do NOT use algebraic/SAN notation (no piece "
        "letters like 'R' or 'N', no '+', no '#', no 'x'). Do not repeat "
        "this instruction or give an example: compute the move for THIS "
        "exact position and output only that move, nothing else.\n\n"
        f"FEN: {fen}\n"
        "Best move (UCI):"
    )


def _extract_uci(text: str) -> Optional[str]:
    if not text:
        return None
    match = _UCI_RE.search(text.strip())
    return match.group(1).lower() if match else None


def _try_parse_move(text: str, board: "chess.Board") -> str:
    if not text:
        return ""
    uci_candidate = _extract_uci(text)
    if uci_candidate:
        try:
            move = chess.Move.from_uci(uci_candidate)
            if move in board.legal_moves:
                return move.uci()
        except ValueError:
            pass
    for san_candidate in _SAN_RE.findall(text):
        try:
            move = board.parse_san(san_candidate)
            return move.uci()
        except ValueError:
            continue
    return ""


# ============================================================================
# Solver LLM (Groq-compatibile)
# ============================================================================

class GroqLLMSolver:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 32,
        reasoning_effort: Optional[str] = None,
        reasoning_format: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        request_delay_seconds: float = 0.0,
    ):
        import requests  # import locale, come nell'originale

        self._requests = requests
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.reasoning_format = reasoning_format
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.request_delay_seconds = request_delay_seconds

    def solve(self, fen: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": _build_llm_prompt(fen)}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_format:
            payload["reasoning_format"] = self.reasoning_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._requests.post(
                    self.base_url, headers=headers, json=payload,
                    timeout=self.timeout_seconds,
                )
                if resp.status_code == 429:
                    raise RuntimeError(f"rate limited (429): {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                raw_text = message.get("content", "") or ""
                pred = _extract_uci(raw_text)
                if not pred:
                    reasoning_text = message.get("reasoning", "") or ""
                    if reasoning_text:
                        raw_text = reasoning_text
                        pred = _extract_uci(reasoning_text)
                if self.request_delay_seconds > 0:
                    time.sleep(self.request_delay_seconds)
                return {
                    "raw_text": raw_text,
                    "pred_move_uci": pred or "",
                    "error": None,
                }
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.debug(
                    f"[solve] attempt {attempt}/{self.max_retries} "
                    f"model={self.model} fallita: {last_err}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        return {"raw_text": "", "pred_move_uci": "", "error": last_err or "unknown error"}


# ============================================================================
# Worker multiprocessing
# ============================================================================

_W_SOLVER: Optional[GroqLLMSolver] = None
_W_CACHE: Dict[str, Dict[str, Any]] = {}
_W_CACHE_PATH: Optional[str] = None
_W_CACHE_DIRTY = 0

# [FIX] Soglia piu' bassa: con --limit piccoli i worker non raggiungevano
# mai 20 item e la loro cache locale andava persa.
_FLUSH_EVERY = 5


def _flush_worker_cache() -> None:
    global _W_CACHE_DIRTY
    if not _W_CACHE_PATH or _W_CACHE_DIRTY == 0:
        return
    try:
        path = f"{_W_CACHE_PATH}.{os.getpid()}"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_W_CACHE, f, indent=2)
        os.replace(tmp, path)
        _W_CACHE_DIRTY = 0
        logger.debug(f"[worker {os.getpid()}] flush cache -> {path} ({len(_W_CACHE)} voci)")
    except Exception:
        logger.warning(f"[worker {os.getpid()}] flush cache fallito", exc_info=True)


def _init_llm_worker(solver_factory: Optional[Callable[[], GroqLLMSolver]],
                     cache_path: Optional[str]) -> None:
    global _W_SOLVER, _W_CACHE, _W_CACHE_PATH, _W_CACHE_DIRTY

    # Ignora SIGINT nei worker (il padre gestisce l'interruzione).
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # [FIX] Registra il flush all'uscita NORMALE del worker. Con close()+join()
    # l'atexit viene eseguito; con terminate()/SIGKILL no.
    atexit.register(_flush_worker_cache)

    _W_CACHE = {}
    _W_CACHE_PATH = cache_path
    _W_CACHE_DIRTY = 0

    # [FIX] Inizializzazione solver con try/except + log esplicito.
    if solver_factory is not None:
        try:
            _W_SOLVER = solver_factory()
            logger.info(
                f"[worker {os.getpid()}] solver pronto "
                f"(model={getattr(_W_SOLVER, 'model', '?')})"
            )
        except Exception:
            logger.exception(
                f"[worker {os.getpid()}] inizializzazione solver FALLITA"
            )
            _W_SOLVER = None
    else:
        _W_SOLVER = None
        logger.warning(f"[worker {os.getpid()}] solver_factory=None passata al worker")

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                _W_CACHE = json.load(f)
        except Exception as e:
            logger.debug(f"[worker {os.getpid()}] cache base illeggibile ({e}), la ignoro.")
            try:
                os.replace(cache_path, cache_path + ".corrupt")
            except OSError:
                pass
            _W_CACHE = {}


def _worker_evaluate_one(args: Tuple[str, str]) -> Tuple[str, str]:
    global _W_CACHE_DIRTY

    pid, fen = args

    # [FIX] Anche se il solver non e' inizializzato, salviamo SEMPRE una voce
    # in cache (con error popolato). Prima si faceva `return pid, ""` che
    # perdeva completamente l'informazione e la cache finale non conteneva
    # ne' raw_text ne' error.
    if _W_SOLVER is None:
        result: Dict[str, Any] = {
            "raw_text": "",
            "pred_move_uci": "",
            "error": "solver_not_initialized_in_worker",
        }
        _W_CACHE[pid] = result
        _W_CACHE_DIRTY += 1
        if _W_CACHE_DIRTY >= _FLUSH_EVERY:
            _flush_worker_cache()
        return pid, ""

    result = _W_CACHE.get(pid)
    if not isinstance(result, dict):
        try:
            result = _W_SOLVER.solve(fen)
        except Exception as e:
            result = {"raw_text": "", "pred_move_uci": "", "error": f"{type(e).__name__}: {e}"}
        _W_CACHE[pid] = result
        _W_CACHE_DIRTY += 1
        if _W_CACHE_DIRTY >= _FLUSH_EVERY:
            _flush_worker_cache()

    pred = (result.get("pred_move_uci") or "") if isinstance(result, dict) else ""
    if not pred and isinstance(result, dict):
        raw = result.get("raw_text", "") or ""
        if raw:
            try:
                board = chess.Board(fen)
                pred = _try_parse_move(raw, board)
            except Exception:
                pred = ""

    # [FIX] Log quando la predizione e' vuota: cosi' si vede subito il perche'.
    if not pred:
        err = result.get("error") if isinstance(result, dict) else None
        raw = (result.get("raw_text") or "") if isinstance(result, dict) else ""
        logger.warning(
            f"[worker {os.getpid()}] pid={pid} pred vuota | "
            f"error={err!r} | raw_text[:100]={raw[:100]!r}"
        )

    return pid, pred


def _merge_worker_caches(cache_path: str, base: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    files = [p for p in glob.glob(f"{cache_path}.*")
             if not p.endswith(".tmp") and not p.endswith(".corrupt")]
    if files:
        logger.info(f"[LLM] Merge cache: trovati {len(files)} file worker: {files}")
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged.update(data)
            try:
                os.remove(path)
            except OSError:
                pass
        except Exception as e:
            logger.warning(f"[LLM] Impossibile leggere {path}: {e}")
            continue
    return merged


# ============================================================================
# Config / valutazione
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File di configurazione {config_path} non trovato.")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_llm_on_holdout(
    holdout_dir: str,
    solver_factory: Optional[Callable[[], GroqLLMSolver]],
    cache_path: Optional[str] = None,
    max_n: int = 10,
    limit: Optional[int] = None,
    max_workers: int = 1,
    pool_join_timeout: float = 60.0,
) -> Dict[str, Any]:
    if not os.path.exists(os.path.join(holdout_dir, "manifest.json")):
        raise FileNotFoundError(
            f"manifest.json non trovato in '{holdout_dir}'. "
            f"Esegui prima Buildexternalholdout.py."
        )

    ds = ShardedGraphDataset(holdout_dir, shuffle=False)

    base_cache: Dict[str, Any] = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                base_cache = json.load(f)
            logger.info(
                f"Cache LLM caricata da {cache_path}: "
                f"{len(base_cache)} risposte gia' disponibili."
            )
        except Exception as e:
            logger.warning(f"Cache LLM illeggibile ({e}), la sposto in .corrupt.")
            try:
                os.replace(cache_path, cache_path + ".corrupt")
            except OSError:
                pass
            base_cache = {}

    # --- Raccolta item ---
    items: List[Tuple[str, str, str, int]] = []
    missing_fen = 0
    for i, (data, label) in enumerate(ds):
        if limit is not None and i >= limit:
            logger.info(f"[LLM] Limite di {limit} posizioni raggiunto.")
            break
        pid = str(i)
        fen = getattr(data, "fen", None)
        if fen is None:
            missing_fen += 1
            continue
        from_sq, to_sq, promo = decode_move(int(label))
        true_uci = chess.Move(from_sq, to_sq, promotion=promo).uci()
        mate_n = int(getattr(data, "position_mate_n", 0))
        items.append((pid, fen, true_uci, mate_n))

    if missing_fen:
        logger.warning(
            f"[LLM] {missing_fen} posizioni senza campo 'fen' saltate."
        )

    results_by_pid: Dict[str, str] = {}

    # ==================================================================
    # Branch A: sequenziale
    # ==================================================================
    if solver_factory is None or max_workers <= 1:
        local_solver = solver_factory() if solver_factory is not None else None
        cache = dict(base_cache)
        total = len(items)
        for n_done, (pid, fen, _, _) in enumerate(items, 1):
            if local_solver is None:
                pred = ""
            else:
                result = cache.get(pid)
                if not isinstance(result, dict):
                    try:
                        result = local_solver.solve(fen)
                    except Exception as e:
                        result = {"raw_text": "", "pred_move_uci": "",
                                  "error": f"{type(e).__name__}: {e}"}
                    cache[pid] = result
                    if cache_path and (n_done % 10 == 0):
                        _atomic_json_dump(cache_path, cache)
                pred = result.get("pred_move_uci") or ""
                if not pred:
                    try:
                        board = chess.Board(fen)
                        pred = _try_parse_move(result.get("raw_text", "") or "", board)
                    except Exception:
                        pred = ""
            results_by_pid[pid] = pred
            if n_done % 5 == 0 or n_done == total:
                logger.info(f"[LLM] valutati {n_done}/{total} problemi.")

        if cache_path:
            _atomic_json_dump(cache_path, cache)

    # ==================================================================
    # Branch B: multiprocessing
    # ==================================================================
    else:
        cpu_count = os.cpu_count() or 2
        workers = min(max_workers, max(1, cpu_count))
        logger.info(f"[LLM] Avvio Pool con {workers} worker.")

        pool = mp.Pool(
            processes=workers,
            initializer=_init_llm_worker,
            initargs=(solver_factory, cache_path),
        )

        shutdown = threading.Event()

        def _sigint_handler(signum, frame):
            if shutdown.is_set():
                logger.warning("[LLM] Secondo SIGINT: termino i worker forzatamente.")
                try:
                    pool.terminate()
                except Exception:
                    pass
                raise KeyboardInterrupt
            shutdown.set()
            raise KeyboardInterrupt

        previous_sigint = signal.signal(signal.SIGINT, _sigint_handler)

        tasks = ((pid, fen) for pid, fen, _, _ in items)
        it = pool.imap_unordered(_worker_evaluate_one, tasks, chunksize=1)

        done = 0
        total = len(items)
        graceful = False
        try:
            for pid, pred in it:
                results_by_pid[pid] = pred
                done += 1
                if done % 5 == 0 or done == total:
                    logger.info(f"[LLM] valutati {done}/{total} problemi.")
            graceful = True
        except KeyboardInterrupt:
            logger.warning(
                f"[LLM] Interruzione ({done}/{total}). Salvo risultati parziali..."
            )
        except Exception:
            logger.exception("[LLM] Errore fatale durante la valutazione.")
            raise
        finally:
            signal.signal(signal.SIGINT, previous_sigint)

            # [FIX] Shutdown: se tutto e' andato bene usiamo close()+join() per
            # permettere ai worker di uscire normalmente e far girare i loro
            # atexit (flush cache). Solo in caso di errore/SIGINT terminiamo.
            try:
                if graceful:
                    pool.close()
                    pool.join()
                else:
                    pool.terminate()
                    deadline = time.monotonic() + 5.0
                    for p in pool._pool:
                        p.join(timeout=max(deadline - time.monotonic(), 0.1))
                    for p in pool._pool:
                        if p.is_alive():
                            try:
                                os.kill(p.pid, signal.SIGKILL)
                            except Exception:
                                pass
            except Exception:
                logger.exception("[LLM] Errore nello shutdown del pool.")
                for p in getattr(pool, "_pool", []):
                    try:
                        if p.is_alive():
                            os.kill(p.pid, signal.SIGKILL)
                    except Exception:
                        pass

        # Merge finale: base + file <cache_path>.<pid> dei worker.
        if cache_path:
            before = len(base_cache)
            base_cache = _merge_worker_caches(cache_path, base_cache)
            logger.info(
                f"[LLM] Merge cache: {before} -> {len(base_cache)} voci."
            )
            # Fallback: se un pid non e' nella cache, salviamo almeno la pred.
            for pid, pred in results_by_pid.items():
                if pid not in base_cache:
                    base_cache[pid] = {"pred_move_uci": pred}
            _atomic_json_dump(cache_path, base_cache)

    # ==================================================================
    # Metriche finali
    # ==================================================================
    problem_ids = [pid for pid, _, _, _ in items]
    mate_true = [mate_n for _, _, _, mate_n in items]
    best_uci = [true_uci for _, _, true_uci, _ in items]
    pred_uci = [results_by_pid.get(pid, "") for pid, _, _, _ in items]
    move_correct = [pred == true for pred, true in zip(pred_uci, best_uci)]

    # [FIX] Report diagnostico: quante predizioni sono vuote.
    n_empty = sum(1 for p in pred_uci if not p)
    if n_empty:
        logger.error(
            f"[LLM] ATTENZIONE: {n_empty}/{len(pred_uci)} predizioni sono VUOTE. "
            f"Controlla la cache per i campi 'error' e 'raw_text'."
        )

    move_correct_arr = np.array(move_correct, dtype=bool)
    mate_true_arr = np.array(mate_true, dtype=np.int64)

    per_n_rows = []
    for n in range(1, max_n + 1):
        mask = mate_true_arr == n
        count = int(mask.sum())
        acc = float(move_correct_arr[mask].mean()) if count > 0 else None
        per_n_rows.append({"mate_n": n, "count": count, "move_accuracy": acc})

    return {
        "problem_id": np.array(problem_ids, dtype=object),
        "mate_n": mate_true_arr,
        "best_move_uci": np.array(best_uci, dtype=object),
        "pred_move_uci": np.array(pred_uci, dtype=object),
        "move_correct": move_correct_arr,
        "per_n": per_n_rows,
    }


def save_per_n_csv(per_n_rows: List[Dict[str, Any]], path: str) -> None:
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mate_n", "count", "move_accuracy"])
        writer.writeheader()
        for row in per_n_rows:
            writer.writerow(row)
    logger.info(f"Salvato {path}")


def save_llm_results_pkl(res: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(res, tmp_path)
    os.replace(tmp_path, path)
    logger.info(f"Salvato {path}")


# ============================================================================
# Factory del solver
# ============================================================================

def _build_solver_factory_from_cfg(
    cfg: Dict[str, Any], api_key: str
) -> Callable[[], GroqLLMSolver]:
    return functools.partial(
        GroqLLMSolver,
        base_url=cfg["llm_base_url"],
        model=cfg["llm_model"],
        api_key=api_key,
        temperature=cfg.get("llm_temperature", 0.0),
        max_tokens=cfg.get("llm_max_tokens", 32),
        reasoning_effort=cfg.get("llm_reasoning_effort"),
        reasoning_format=cfg.get("llm_reasoning_format"),
        timeout_seconds=cfg.get("llm_timeout_seconds", 30),
        max_retries=cfg.get("llm_max_retries", 3),
        retry_backoff_seconds=cfg.get("llm_retry_backoff_seconds", 2.0),
        request_delay_seconds=cfg.get("llm_request_delay_seconds", 0.5),
    )


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Yaml/evaluate_llm.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    holdout_dir = cfg["holdout_dir"]
    out_dir = cfg.get("out_dir", "Dataset/Test_LLM/metrics")
    cache_path = os.path.join(out_dir, "llm_responses_cache.json")
    max_n = cfg.get("max_n", 10)

    # [FIX] Warning se la config sembra "reasoning + max_tokens basso".
    max_tokens = cfg.get("llm_max_tokens", 32)
    reasoning_effort = cfg.get("llm_reasoning_effort")
    if reasoning_effort and max_tokens < 128:
        logger.warning(
            f"[LLM] reasoning_effort='{reasoning_effort}' con "
            f"llm_max_tokens={max_tokens}: il budget potrebbe essere consumato "
            f"dal reasoning prima della risposta finale -> content vuoto. "
            f"Considera llm_max_tokens >= 512."
        )

    # Workers: CLI > config > (cpu-1)
    if args.workers is not None:
        max_workers = max(1, args.workers)
    elif "llm_max_workers" in cfg:
        max_workers = max(1, int(cfg["llm_max_workers"]))
    else:
        max_workers = max(1, (os.cpu_count() or 2) - 1)
    max_workers = 1
    logger.info(f"Worker LLM richiesti: {max_workers}")

    solver_factory: Optional[Callable[[], GroqLLMSolver]] = None
    if cfg.get("llm_enabled", True):
        api_key_env = cfg["llm_api_key_env"]
        api_key = os.environ.get(api_key_env, None)
        if not api_key:
            logger.error(
                f"Variabile d'ambiente {api_key_env} non impostata. "
                f"Esegui: export {api_key_env}=\"...\""
            )
            sys.exit(1)
        solver_factory = _build_solver_factory_from_cfg(cfg, api_key)
        logger.info(f"LLM solver pronto: model={cfg['llm_model']}")
    else:
        logger.info("llm_enabled=false: valutazione LLM saltata.")

    logger.info(f"Caricamento holdout esterno da {holdout_dir}...")
    res = evaluate_llm_on_holdout(
        holdout_dir=holdout_dir,
        solver_factory=solver_factory,
        cache_path=cache_path,
        max_n=max_n,
        limit=args.limit,
        max_workers=max_workers,
    )

    if len(res["move_correct"]) == 0:
        logger.error("Nessuna posizione valutata (holdout vuoto o tutte senza FEN).")
        sys.exit(1)

    logger.info(
        f"[LLM] Move Accuracy = {res['move_correct'].mean() * 100:.2f}% | "
        f"N={len(res['problem_id'])}"
    )
    save_per_n_csv(res["per_n"], os.path.join(out_dir, "llm_metrics_per_n.csv"))
    save_llm_results_pkl(res, os.path.join(out_dir, "llm_results.pkl"))

    logger.info(f"Valutazione LLM completata. Output salvati in {out_dir}.")


if __name__ == "__main__":
    main()