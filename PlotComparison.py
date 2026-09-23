"""
PlotComparison.py

Genera il grafico a barre richiesto dalla traccia del progetto:
"Bar chart of accuracy by n" per il confronto GNN vs LLM, stratificato
per profondita' di matto (n).

Combina due file gia' generati dalle valutazioni:
  - Dataset/Test_330/metrics/metrics_per_n.csv  (GNN basic + time_aware)
  - Dataset/Test_LLM/metrics/llm_metrics_per_n.csv  (LLM)

Due tipi di barre "assenti", distinte visivamente:
  - GNN non allenata su questo n (grigio tratteggiato): Andrea non ha
    ancora allenato/valutato oltre n=5 per limiti di risorse.
  - Dati insufficienti (giallo tratteggiato): la barra esiste ma il
    conteggio campioni e' sotto MIN_SAMPLES, quindi la percentuale non
    e' statisticamente affidabile e viene nascosta invece che mostrata
    come se fosse un dato solido.

Uso:
    python PlotComparison.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

GNN_CSV = "Result/TestSet/metrics/metrics_per_n.csv"
LLM_CSV = "Dataset/Test_LLM/metrics/llm_metrics_per_n.csv"
OUT_DIR = "Result/Comparison"
OUT_PATH = os.path.join(OUT_DIR, "accuracy_by_n_comparison.png")

COLOR_BASIC = "#1E3A8A"
COLOR_TIME_AWARE = "#7DD3FC"
COLOR_LLM = "#86EFAC"
COLOR_NOT_TRAINED_BG = "#E5E5EA"
COLOR_LOW_N_BG = "#FFD23F"
TEXT_DARK = "#2B2D33"
TEXT_MUTED = "#8A8D94"
GRID_COLOR = "#DCDCE2"

GNN_MAX_N_TRAINED = 5
MIN_SAMPLES = 5  # sotto questa soglia, la barra viene nascosta per inaffidabilita' statistica


def load_gnn_per_n(path: str) -> dict:
    rows = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n"])
            rows[n] = {
                "basic": float(row["acc_untimed"]) if row["acc_untimed"] else None,
                "time_aware": float(row["acc_timed"]) if row["acc_timed"] else None,
                "count": int(row["totale"]),
            }
    return rows


def load_llm_per_n(path: str) -> dict:
    rows = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["mate_n"])
            acc = row["move_accuracy"]
            rows[n] = {
                "llm": float(acc) if acc not in (None, "", "None") else None,
                "count": int(row["count"]),
            }
    return rows


def main():
    if not os.path.exists(GNN_CSV):
        raise FileNotFoundError(f"Non trovato: {GNN_CSV}")
    if not os.path.exists(LLM_CSV):
        raise FileNotFoundError(f"Non trovato: {LLM_CSV}")

    gnn = load_gnn_per_n(GNN_CSV)
    llm = load_llm_per_n(LLM_CSV)
    all_n = sorted(set(gnn.keys()) | set(llm.keys()))

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = True
    plt.rcParams["text.color"] = TEXT_DARK
    plt.rcParams["axes.labelcolor"] = TEXT_DARK
    plt.rcParams["xtick.color"] = TEXT_DARK
    plt.rcParams["ytick.color"] = TEXT_DARK

    x = np.arange(len(all_n))
    width = 0.30

    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    saw_not_trained = False
    saw_low_n = False

    def draw_bar(xpos, value, color, first, label, show_n=None):
        ax.bar(xpos, value, width, color=color, edgecolor="white",
               linewidth=1.2, label=label if first else None, zorder=3)
        ax.text(xpos, value + 2.2, f"{value:.1f}", ha="center", va="bottom",
                fontsize=9, color=TEXT_DARK, fontweight="bold")
        if show_n is not None:
            ax.text(xpos, -2.6, f"N={show_n}", ha="center", va="top",
                    fontsize=7, color=TEXT_MUTED)

    def draw_not_trained(xpos):
        nonlocal saw_not_trained
        saw_not_trained = True
        ax.bar(xpos, 2.4, width, color=COLOR_NOT_TRAINED_BG, edgecolor="#D0D0D6",
               linewidth=0.8, hatch="////", zorder=2)

    def draw_low_n(xpos, count):
        nonlocal saw_low_n
        saw_low_n = True
        ax.bar(xpos, 2.4, width, color=COLOR_LOW_N_BG, edgecolor="#D9C067",
               linewidth=0.8, hatch="....", zorder=2)
        ax.text(xpos, -2.6, f"N={count}", ha="center", va="top",
                fontsize=7, color=TEXT_MUTED)

    for i, n in enumerate(all_n):
        gnn_row = gnn.get(n, {})
        llm_row = llm.get(n, {})
        gnn_trained = n <= GNN_MAX_N_TRAINED

        # basic
        if not gnn_trained:
            draw_not_trained(x[i] - width)
        elif gnn_row.get("basic") is not None and gnn_row["count"] < MIN_SAMPLES:
            draw_low_n(x[i] - width, gnn_row["count"])
        elif gnn_row.get("basic") is not None:
            draw_bar(x[i] - width, gnn_row["basic"] * 100, COLOR_BASIC,
                      i == 0, "GNN basic", show_n=gnn_row["count"])
        else:
            draw_not_trained(x[i] - width)

        # time_aware
        if not gnn_trained:
            draw_not_trained(x[i])
        elif gnn_row.get("time_aware") is not None and gnn_row["count"] < MIN_SAMPLES:
            draw_low_n(x[i], gnn_row["count"])
        elif gnn_row.get("time_aware") is not None:
            draw_bar(x[i], gnn_row["time_aware"] * 100, COLOR_TIME_AWARE,
                      i == 0, "GNN time-aware")
        else:
            draw_not_trained(x[i])

        # llm
        if llm_row.get("llm") is not None and llm_row["count"] < MIN_SAMPLES:
            draw_low_n(x[i] + width, llm_row["count"])
        elif llm_row.get("llm") is not None:
            draw_bar(x[i] + width, llm_row["llm"] * 100, COLOR_LLM,
                      i == 0, "LLM", show_n=llm_row["count"])

    handles, labels = ax.get_legend_handles_labels()
    if saw_not_trained:
        handles.append(Patch(facecolor=COLOR_NOT_TRAINED_BG, edgecolor="#D0D0D6", hatch="////"))
        labels.append("GNN non allenata su questo n")
    if saw_low_n:
        handles.append(Patch(facecolor=COLOR_LOW_N_BG, edgecolor="#D9C067", hatch="...."))
        labels.append(f"Dati insufficienti (N<{MIN_SAMPLES})")

    ax.set_xticks(x)
    ax.set_xticklabels([f"n = {n}" for n in all_n], fontsize=10.5)
    ax.set_xlabel("Profondità di matto (n)", fontsize=12, labelpad=26, color=TEXT_DARK)
    ax.set_ylabel("Move Accuracy (%)", fontsize=12, color=TEXT_DARK)
    fig.suptitle("Confronto accuratezza per profondità di matto — GNN vs LLM",
                 fontsize=15, fontweight="bold", color=TEXT_DARK, x=0.02, y=0.995, ha="left")
    fig.text(0.02, 0.965, "Basic • Time-aware • LLM (Gemma), holdout esterno condiviso",
             fontsize=10, color=TEXT_MUTED, ha="left")

    legend = ax.legend(handles=handles, labels=labels,
                        loc="lower center", bbox_to_anchor=(0.5, 1.06),
                        ncol=len(handles), frameon=False, fontsize=10)

    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=-4.5)
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(axis="both", length=0)

    fig.text(0.5, 0.005,
              f"N = numero di posizioni valutate. Barre con N<{MIN_SAMPLES} nascoste per inaffidabilità statistica (vedi legenda).",
              ha="center", fontsize=8.5, color=TEXT_MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.035, 1, 0.90))

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Salvato: {OUT_PATH}")

    print("\nRiepilogo:")
    print(f"{'n':>4} | {'basic':>20} | {'time_aware':>20} | {'llm':>20}")
    for n in all_n:
        gnn_row = gnn.get(n, {})
        llm_row = llm.get(n, {})
        gnn_trained = n <= GNN_MAX_N_TRAINED

        def fmt_gnn(key):
            if not gnn_trained or gnn_row.get(key) is None:
                return "non allenata"
            if gnn_row["count"] < MIN_SAMPLES:
                return f"N={gnn_row['count']} (nascosto)"
            return f"{gnn_row[key]*100:.1f}% (N={gnn_row['count']})"

        def fmt_llm():
            if llm_row.get("llm") is None:
                return "n/d"
            if llm_row["count"] < MIN_SAMPLES:
                return f"N={llm_row['count']} (nascosto)"
            return f"{llm_row['llm']*100:.1f}% (N={llm_row['count']})"

        print(f"{n:>4} | {fmt_gnn('basic'):>20} | {fmt_gnn('time_aware'):>20} | {fmt_llm():>20}")


if __name__ == "__main__":
    main()
