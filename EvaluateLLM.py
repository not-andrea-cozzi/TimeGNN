"""
EvaluateLLM.py

Valuta un LLM (via Groq) sull'holdout esterno costruito da
Buildexternalholdout.py, per completare il confronto GNN vs LLM
richiesto dalla traccia del progetto (obiettivo di ricerca 1).

Adattato da Valheldout.py (vecchio progetto ProgettoSADI): la classe
GroqLLMSolver e le funzioni di parsing (_build_llm_prompt, _extract_uci,
_try_parse_move) sono riprese cosi' come sono, generiche e indipendenti
dal formato dati. Cio' che cambia rispetto al vecchio holdout:

  - Il vecchio formato salvava data.fen, data.best_move_uci, data.mate_n,
    data.problem_id direttamente come stringhe/interi.
  - Il nuovo formato (PositionGraphSchema.build_position_data) salva la
    mossa giusta come intero codificato in data.y (vocabolario fisso
    MOVE_VOCAB_SIZE, vedi encode_move/decode_move) e la profondita' di
    matto in data.position_mate_n. Il FEN non veniva salvato affatto:
    e' stato aggiunto in Buildexternalholdout.py (data.fen = board.fen())
    e in TrainPipeline/CleanDataset.py (KEEP_FIELDS) appositamente per
    questo script. Se l'holdout con cui stai lavorando e' stato
    costruito PRIMA di queste due modifiche, va rigenerato.

Uso:
    export GROQ_API_KEY="..."
    python EvaluateLLM.py --config Yaml/evaluate_llm.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import chess
import numpy as np
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
# Sezione ripresa da Valheldout.py (ProgettoSADI) senza modifiche: e' generica,
# lavora solo su FEN/testo, non dipende dal formato dati del progetto.
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
    """Prova prima a interpretare 'text' come UCI, poi come SAN, validando
    contro le mosse legali di 'board'. Ritorna '' se non trova nulla di valido."""
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
        import requests

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

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._requests.post(
                    self.base_url, headers=headers, json=payload, timeout=self.timeout_seconds
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
                return {"raw_text": raw_text, "pred_move_uci": pred or "", "error": None}
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        return {"raw_text": "", "pred_move_uci": "", "error": str(last_err)}


# ============================================================================
# Sezione nuova: adattamento al formato dati di TimeGNN (fen + y codificato +
# position_mate_n al posto di fen/best_move_uci/mate_n/problem_id espliciti).
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File di configurazione {config_path} non trovato.")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_llm_on_holdout(
    holdout_dir: str,
    solver: Optional[GroqLLMSolver],
    cache_path: Optional[str] = None,
    max_n: int = 10,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    if not os.path.exists(os.path.join(holdout_dir, "manifest.json")):
        raise FileNotFoundError(
            f"manifest.json non trovato in '{holdout_dir}'. "
            f"Esegui prima Buildexternalholdout.py (versione aggiornata, con salvataggio del FEN)."
        )

    ds = ShardedGraphDataset(holdout_dir, shuffle=False)

    cache: Dict[str, Dict[str, str]] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"Cache LLM caricata da {cache_path}: {len(cache)} risposte gia' disponibili.")

    problem_ids: List[str] = []
    mate_true: List[int] = []
    best_uci: List[str] = []
    pred_uci: List[str] = []
    move_correct: List[bool] = []
    missing_fen = 0

    for i, (data, label) in enumerate(ds):
        if limit is not None and i >= limit:
            logger.info(f"[LLM] Limite di {limit} posizioni raggiunto, interrompo qui (test).")
            break
        pid = str(i)
        fen = getattr(data, "fen", None)
        if fen is None:
            missing_fen += 1
            continue

        from_sq, to_sq, promo = decode_move(int(label))
        true_uci = chess.Move(from_sq, to_sq, promotion=promo).uci()
        mate_n = int(getattr(data, "position_mate_n", 0))

        if solver is None:
            pred = ""
        else:
            if pid in cache:
                result = cache[pid]
            else:
                result = solver.solve(fen)
                cache[pid] = result
                if cache_path and (i % 10 == 0):
                    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache, f, indent=2)

            pred = result.get("pred_move_uci", "") or ""
            if not pred:
                board = chess.Board(fen)
                pred = _try_parse_move(result.get("raw_text", "") or "", board)

        problem_ids.append(pid)
        mate_true.append(mate_n)
        best_uci.append(true_uci)
        pred_uci.append(pred)
        move_correct.append(bool(pred == true_uci))

        if (i + 1) % 20 == 0:
            logger.info(f"[LLM] valutati {i + 1} problemi.")

    if missing_fen:
        logger.warning(
            f"[LLM] {missing_fen} posizioni senza campo 'fen' saltate: "
            f"l'holdout probabilmente e' stato costruito prima del fix, va rigenerato."
        )

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Yaml/evaluate_llm.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limita il numero di posizioni (per test veloci)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    holdout_dir = cfg["holdout_dir"]
    out_dir = cfg.get("out_dir", "Dataset/Test_LLM/metrics")
    cache_path = os.path.join(out_dir, "llm_responses_cache.json")
    max_n = cfg.get("max_n", 10)

    solver = None
    if cfg.get("llm_enabled", True):
        api_key = os.environ.get(cfg["llm_api_key_env"], None)
        if not api_key:
            logger.error(
                f"Variabile d'ambiente {cfg['llm_api_key_env']} non impostata: "
                f"impossibile interrogare l'LLM. Esegui: export {cfg['llm_api_key_env']}=\"...\""
            )
            sys.exit(1)
        solver = GroqLLMSolver(
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
        logger.info(f"LLM solver pronto: model={cfg['llm_model']}")
    else:
        logger.info("llm_enabled=false: valutazione LLM saltata (solo struttura dati verificata).")

    logger.info(f"Caricamento holdout esterno da {holdout_dir}...")
    res = evaluate_llm_on_holdout(holdout_dir, solver, cache_path=cache_path, max_n=max_n, limit=args.limit)

    if len(res["move_correct"]) == 0:
        logger.error("Nessuna posizione valutata (holdout vuoto o tutte senza FEN).")
        sys.exit(1)

    logger.info(
        f"[LLM] Move Accuracy = {res['move_correct'].mean() * 100:.2f}% | N={len(res['problem_id'])}"
    )
    save_per_n_csv(res["per_n"], os.path.join(out_dir, "llm_metrics_per_n.csv"))

    logger.info(f"Valutazione LLM completata. Output salvati in {out_dir}.")


if __name__ == "__main__":
    main()
