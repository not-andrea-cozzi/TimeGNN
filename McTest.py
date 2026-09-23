from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import binomtest, chi2


class AlignmentError(RuntimeError):
    pass


@dataclass
class AlignedResults:
    correct_a: np.ndarray
    correct_b: np.ndarray
    mate_n: np.ndarray
    n_positions: int

def align_model_results(
    res_a: Dict[str, Any],
    res_b: Dict[str, Any],
    *,
    strict: bool = True,
    name_a: str = "Modello A",
    name_b: str = "Modello B",
) -> AlignedResults:
    for key in ("move_correct", "mate_n"):
        if key not in res_a:
            raise KeyError(f"Il dizionario di {name_a} non contiene '{key}'")
        if key not in res_b:
            raise KeyError(f"Il dizionario di {name_b} non contiene '{key}'")

    correct_a = np.asarray(res_a["move_correct"])
    correct_b = np.asarray(res_b["move_correct"])
    mate_n_a = np.asarray(res_a["mate_n"])
    mate_n_b = np.asarray(res_b["mate_n"])

    len_a = len(correct_a)
    len_b = len(correct_b)

    if len_a != len_b:
        msg = f"Lunghezze diverse tra {name_a} ({len_a}) e {name_b} ({len_b})"
        if strict:
            raise AlignmentError(msg)
        print(f"[align_model_results] AVVISO (strict=False): {msg} — troncamento alla lunghezza minima.")

    n = min(len_a, len_b)
    correct_a = correct_a[:n]
    correct_b = correct_b[:n]
    mate_n_a = mate_n_a[:n]
    mate_n_b = mate_n_b[:n]

    if len(mate_n_a) != len(mate_n_b):
        raise AlignmentError(f"Gli array mate_n di {name_a} e {name_b} hanno lunghezze diverse dopo il troncamento")

    mismatch_mask = mate_n_a != mate_n_b
    n_mismatch = int(np.sum(mismatch_mask))

    if n_mismatch > 0:
        mismatch_idx = np.where(mismatch_mask)[0]
        preview = mismatch_idx[:10].tolist()
        msg = f"{n_mismatch}/{n} posizioni hanno mate_n diverso tra {name_a} e {name_b} allo stesso indice (prime: {preview})"
        if strict:
            raise AlignmentError(msg)
        print(f"[align_model_results] AVVISO (strict=False): {msg}")

    return AlignedResults(
        correct_a=correct_a,
        correct_b=correct_b,
        mate_n=mate_n_a,
        n_positions=n,
    )

def run_mcnemar(
    res_a: Dict[str, Any],
    res_b: Dict[str, Any],
    *,
    strict: bool = True,
    exact_threshold: int = 25,
    include_pooled: bool = True,
    name_a: str = "Modello A",
    name_b: str = "Modello B"
) -> List[McNemarResult]:
    aligned = align_model_results(res_a, res_b, strict=strict, name_a=name_a, name_b=name_b)
    return mcnemar_stratified_by_mate_n(
        aligned.correct_a, aligned.correct_b, aligned.mate_n,
        exact_threshold=exact_threshold, include_pooled=include_pooled,
    )


@dataclass
class McNemarResult:
    label: str
    n_total: int
    n00: int
    n01: int
    n10: int
    n11: int
    accuracy_a: float
    accuracy_b: float
    method: str
    statistic: Optional[float]
    p_value: Optional[float]
    note: str = ""

    @property
    def n_discordant(self) -> int:
        return self.n01 + self.n10

    def summary_line(self) -> str:
        if self.method == "undefined":
            return f"[{self.label}] n={self.n_total} nessuna discordanza (n01=n10=0): McNemar non definito. {self.note}"
        return (
            f"[{self.label}] n={self.n_total} n01={self.n01} n10={self.n10} "
            f"acc_A={self.accuracy_a:.3f} acc_B={self.accuracy_b:.3f} "
            f"method={self.method} stat={self.statistic:.4f} p={self.p_value:.4g}"
        )


def _build_contingency(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, int]:
    if len(correct_a) != len(correct_b):
        raise ValueError(f"correct_a e correct_b hanno lunghezze diverse ({len(correct_a)} vs {len(correct_b)})")
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)

    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n00 = int(np.sum(~a & ~b))
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11}


def mcnemar_single(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    label: str = "all",
    exact_threshold: int = 25,
    force_method: Optional[str] = None,
) -> McNemarResult:
    counts = _build_contingency(correct_a, correct_b)
    n00, n01, n10, n11 = counts["n00"], counts["n01"], counts["n10"], counts["n11"]
    n_total = n00 + n01 + n10 + n11

    acc_a = (n10 + n11) / n_total if n_total else float("nan")
    acc_b = (n01 + n11) / n_total if n_total else float("nan")

    n_discordant = n01 + n10

    if n_discordant == 0:
        return McNemarResult(
            label=label, n_total=n_total, n00=n00, n01=n01, n10=n10, n11=n11,
            accuracy_a=acc_a, accuracy_b=acc_b,
            method="undefined", statistic=None, p_value=None,
            note="n01=n10=0, nessun caso discordante in questo strato.",
        )

    use_exact = (force_method == "exact") or (
        force_method is None and n_discordant < exact_threshold
    )

    if use_exact:
        result = binomtest(n10, n_discordant, p=0.5, alternative="two-sided")
        return McNemarResult(
            label=label, n_total=n_total, n00=n00, n01=n01, n10=n10, n11=n11,
            accuracy_a=acc_a, accuracy_b=acc_b,
            method="exact", statistic=float(n10), p_value=float(result.pvalue),
            note=f"binomiale esatto su n_discordant={n_discordant} (soglia={exact_threshold}).",
        )

    statistic = ((abs(n01 - n10) - 1) ** 2) / n_discordant
    p_value = float(chi2.sf(statistic, df=1))
    return McNemarResult(
        label=label, n_total=n_total, n00=n00, n01=n01, n10=n10, n11=n11,
        accuracy_a=acc_a, accuracy_b=acc_b,
        method="chi2", statistic=float(statistic), p_value=p_value,
        note=f"chi2 con correzione di Yates, n_discordant={n_discordant} (soglia={exact_threshold}).",
    )


def mcnemar_stratified_by_mate_n(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    mate_n: Sequence[int],
    exact_threshold: int = 25,
    include_pooled: bool = True,
) -> List[McNemarResult]:
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    mate_n_arr = np.asarray(mate_n)

    if not (len(correct_a) == len(correct_b) == len(mate_n_arr)):
        raise ValueError(
            f"Lunghezze disallineate: correct_a={len(correct_a)}, "
            f"correct_b={len(correct_b)}, mate_n={len(mate_n_arr)}"
        )

    results: List[McNemarResult] = []
    for n in sorted(np.unique(mate_n_arr).tolist()):
        mask = mate_n_arr == n
        results.append(
            mcnemar_single(
                correct_a[mask], correct_b[mask],
                label=f"n={n}", exact_threshold=exact_threshold,
            )
        )

    if include_pooled:
        results.append(
            mcnemar_single(
                correct_a, correct_b, label="all", exact_threshold=exact_threshold,
            )
        )

    return results


def print_report(results: List[McNemarResult]) -> None:
    for r in results:
        print(r.summary_line())


def results_to_rows(results: List[McNemarResult]) -> List[Dict]:
    rows = []
    for r in results:
        rows.append({
            "label": r.label,
            "n_total": r.n_total,
            "n00": r.n00,
            "n01": r.n01,
            "n10": r.n10,
            "n11": r.n11,
            "n_discordant": r.n_discordant,
            "accuracy_a": r.accuracy_a,
            "accuracy_b": r.accuracy_b,
            "method": r.method,
            "statistic": r.statistic,
            "p_value": r.p_value,
            "note": r.note,
        })
    return rows


def run_mcnemar_gnn_vs_llm(
    res_gnn: Dict[str, Any],
    res_llm: Dict[str, Any],
    *,
    strict: bool = True,
    exact_threshold: int = 25,
    include_pooled: bool = True,
) -> List[McNemarResult]:
    aligned = align_gnn_llm_results(res_gnn, res_llm, strict=strict)
    return mcnemar_stratified_by_mate_n(
        aligned.correct_gnn, aligned.correct_llm, aligned.mate_n,
        exact_threshold=exact_threshold, include_pooled=include_pooled,
    )


HOLDOUT_DIR = "Dataset/ExternalHoldout/holdout_clean"


if __name__ == "__main__":
    print(f"[mcnemar_eval] holdout di riferimento: {HOLDOUT_DIR}")

    rng = np.random.default_rng(1)
    n = 200
    mate_n_demo = rng.integers(1, 6, size=n)
    
    # Simulazione dei risultati estratti dai due modelli
    res_time_aware = {"move_correct": rng.random(n) < (0.9 - 0.1 * mate_n_demo), "mate_n": mate_n_demo}
    res_basic = {"move_correct": rng.random(n) < (0.6 + 0.05 * mate_n_demo), "mate_n": mate_n_demo}

    print("\nEsecuzione test McNemar: TimeAware vs Basic")
    results_demo = run_mcnemar(
        res_time_aware, 
        res_basic, 
        name_a="TimeAware", 
        name_b="Basic"
    )
    print_report(results_demo)