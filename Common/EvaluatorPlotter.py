import csv
import logging
import os
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("evaluator_plotter")

# Palette uniforme con PlotComparison.py: basic = blu scuro, time_aware = azzurrino.
COLOR_UNTIMED = "#1E3A8A"   # basic, blu scuro
COLOR_TIMED = "#7DD3FC"     # time_aware, azzurrino
TEXT_DARK = "#2B2D33"
TEXT_MUTED = "#8A8D94"
GRID_COLOR = "#DCDCE2"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["text.color"] = TEXT_DARK
plt.rcParams["axes.labelcolor"] = TEXT_DARK
plt.rcParams["xtick.color"] = TEXT_DARK
plt.rcParams["ytick.color"] = TEXT_DARK


def _style_axes(ax):
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(axis="both", length=0)


class EvaluatorPlotter:
    """
    Helper per calcolare metriche e generare grafici/CSV per la valutazione dei modelli.
    Accetta i risultati nel formato flat prodotto da evaluate() in TestModels.py:
        {
            "move_correct": np.ndarray[bool],
            "mate_correct": np.ndarray[bool],
            "mate_true":    np.ndarray[int],
            "mate_pred":    np.ndarray[int],
            "mate_n":       np.ndarray[int],
            ...
        }
    """

    def __init__(self, plots_dir: str, out_dir: str):
        self.plots_dir = plots_dir
        self.out_dir = out_dir
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def _extract_depth_arrays(
        results: dict, max_n: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Stratifica i risultati flat per profondità di matto (n).
        Depths coprono l'intervallo [1, max_n].
        Ritorna: (depths, move_accs, mate_accs, counts, corrects_move)
        """
        depths = np.arange(1, max_n + 1)
        move_accs = np.full(max_n, np.nan)
        mate_accs = np.full(max_n, np.nan)
        counts = np.zeros(max_n, dtype=np.int64)
        corrects_move = np.zeros(max_n, dtype=np.int64)

        mate_n = results.get("mate_n")
        move_correct = results.get("move_correct")
        mate_correct = results.get("mate_correct")

        if mate_n is None or len(mate_n) == 0:
            return depths, move_accs, mate_accs, counts, corrects_move

        for i, d in enumerate(depths):
            mask = (mate_n == d)
            c = int(mask.sum())
            counts[i] = c
            if c > 0:
                mc = int(move_correct[mask].sum())
                corrects_move[i] = mc
                move_accs[i] = mc / c
                mate_accs[i] = float(mate_correct[mask].mean())

        return depths, move_accs, mate_accs, counts, corrects_move

    def plot_depth_bars(self, res_t: dict, res_u: dict, max_n: int = 10, filename: str = "bars_per_n.png"):
        """Plot 1: Barre verticali per vedere risposte giuste / totale per ogni n (da 1 a max_n)."""
        depths, acc_t, _, counts, corrects_t = self._extract_depth_arrays(res_t, max_n)
        _, acc_u, _, _, corrects_u = self._extract_depth_arrays(res_u, max_n)

        width = 0.36
        x = np.arange(len(depths))
        fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        acc_t_plot = np.nan_to_num(acc_t, nan=0.0)
        acc_u_plot = np.nan_to_num(acc_u, nan=0.0)

        rects1 = ax.bar(x - width / 2, acc_t_plot * 100, width, label="Time-aware",
                         color=COLOR_TIMED, edgecolor="white", linewidth=1.2, zorder=3)
        rects2 = ax.bar(x + width / 2, acc_u_plot * 100, width, label="Basic",
                         color=COLOR_UNTIMED, edgecolor="white", linewidth=1.2, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"n = {d}" for d in depths], fontsize=10.5)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_xlabel("Profondità di matto (n)", fontsize=12, labelpad=14)
        ax.set_title("Risposte giuste / totale per profondità di matto",
                     fontsize=15, fontweight="bold", pad=16, loc="left")
        ax.set_ylim(0, 118)
        legend = ax.legend(loc="upper right", frameon=False, fontsize=10.5)
        _style_axes(ax)

        for i, rect in enumerate(rects1):
            if counts[i] > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 3,
                        f"{corrects_t[i]}/{counts[i]}", ha="center", va="bottom",
                        fontsize=8, rotation=90, color=TEXT_DARK)

        for i, rect in enumerate(rects2):
            if counts[i] > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 3,
                        f"{corrects_u[i]}/{counts[i]}", ha="center", va="bottom",
                        fontsize=8, rotation=90, color=TEXT_DARK)

        fig.tight_layout()
        out_path = os.path.join(self.plots_dir, filename)
        fig.savefig(out_path, dpi=220, facecolor="white")
        plt.close(fig)
        logger.info(f"Salvato {out_path}")

    def plot_depth_curves(self, res_t: dict, res_u: dict, max_n: int = 10, filename: str = "curves_per_n.png"):
        """Plot 2: Curve di accuracy al variare di n (da 1 a max_n)."""
        depths, acc_t, _, _, _ = self._extract_depth_arrays(res_t, max_n)
        _, acc_u, _, _, _ = self._extract_depth_arrays(res_u, max_n)

        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.plot(depths, acc_t * 100, marker="o", markersize=7, linewidth=2.4,
                label="Time-aware", color=COLOR_TIMED, zorder=3)
        ax.plot(depths, acc_u * 100, marker="s", markersize=7, linewidth=2.4,
                label="Basic", color=COLOR_UNTIMED, zorder=3)
        ax.set_xticks(depths)
        ax.set_xlabel("Profondità di matto (n)", fontsize=12, labelpad=14)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Andamento dell'accuracy per profondità di matto",
                     fontsize=15, fontweight="bold", pad=16, loc="left")
        ax.legend(loc="upper right", frameon=False, fontsize=10.5)
        _style_axes(ax)

        fig.tight_layout()
        out_path = os.path.join(self.plots_dir, filename)
        fig.savefig(out_path, dpi=220, facecolor="white")
        plt.close(fig)
        logger.info(f"Salvato {out_path}")

    def save_depth_metrics(self, res_t: dict, res_u: dict, max_n: int = 10, filename: str = "metrics_per_n.csv"):
        """Salvataggio su file CSV dei dati esatti divisi per n."""
        depths, acc_t, mate_acc_t, counts, corrects_t = self._extract_depth_arrays(res_t, max_n)
        _, acc_u, mate_acc_u, _, corrects_u = self._extract_depth_arrays(res_u, max_n)

        out_path = os.path.join(self.out_dir, filename)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "totale", "corrette_timed", "acc_timed", "corrette_untimed", "acc_untimed"])
            for i in range(len(depths)):
                w.writerow([
                    depths[i], counts[i],
                    corrects_t[i], f"{acc_t[i]:.6f}" if not np.isnan(acc_t[i]) else "",
                    corrects_u[i], f"{acc_u[i]:.6f}" if not np.isnan(acc_u[i]) else ""
                ])
        logger.info(f"Salvato {out_path}")

    def plot_aggregate_bars(self, res_t: dict, res_u: dict, filename: str = "aggregate_bars.png"):
        """Barre aggregate move_acc e mate_acc globali, calcolate come media pesata sui conteggi."""
        _, move_t, mate_t, counts_t, _ = self._extract_depth_arrays(res_t)
        _, move_u, mate_u, counts_u, _ = self._extract_depth_arrays(res_u)

        total = counts_t.sum()
        if total == 0:
            logger.warning("Nessun campione per il plot aggregato.")
            return

        valid_t = ~np.isnan(move_t)
        valid_u = ~np.isnan(move_u)

        global_move_t = np.average(move_t[valid_t], weights=counts_t[valid_t]) if valid_t.any() else 0.0
        global_mate_t = np.average(mate_t[valid_t], weights=counts_t[valid_t]) if valid_t.any() else 0.0
        global_move_u = np.average(move_u[valid_u], weights=counts_u[valid_u]) if valid_u.any() else 0.0
        global_mate_u = np.average(mate_u[valid_u], weights=counts_u[valid_u]) if valid_u.any() else 0.0

        metrics = ["Move Accuracy", "Mate Accuracy"]
        x = np.arange(len(metrics))
        width = 0.34
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        vals_t = [global_move_t, global_mate_t]
        vals_u = [global_move_u, global_mate_u]

        ax.bar(x - width / 2, vals_t, width, label="Time-aware", color=COLOR_TIMED,
               edgecolor="white", linewidth=1.2, zorder=3)
        ax.bar(x + width / 2, vals_u, width, label="Basic", color=COLOR_UNTIMED,
               edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Test set — Time-aware vs Basic (N={total})",
                     fontsize=15, fontweight="bold", pad=16, loc="left")
        for i, (vt, vu) in enumerate(zip(vals_t, vals_u)):
            ax.text(i - width / 2, vt + 0.015, f"{vt:.3f}", ha="center", fontsize=10,
                    color=TEXT_DARK, fontweight="bold")
            ax.text(i + width / 2, vu + 0.015, f"{vu:.3f}", ha="center", fontsize=10,
                    color=TEXT_DARK, fontweight="bold")
        ax.legend(loc="upper right", frameon=False, fontsize=10.5)
        _style_axes(ax)
        fig.tight_layout()
        out_path = os.path.join(self.plots_dir, filename)
        fig.savefig(out_path, dpi=220, facecolor="white")
        plt.close(fig)
        logger.info(f"Salvato {out_path}")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, title: str, filename: str):
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if int(t) < num_classes and int(p) < num_classes:
                cm[int(t), int(p)] += 1
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
        fig.patch.set_facecolor("white")
        im = ax.imshow(cm_norm, cmap="PuBu", vmin=0, vmax=1)
        ax.set_xlabel("Mate-in-N predetto", fontsize=12, labelpad=12)
        ax.set_ylabel("Mate-in-N reale", fontsize=12)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=16, loc="left")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                if cm[i, j] > 0:
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                            ha="center", va="center",
                            color="white" if cm_norm[i, j] > 0.5 else TEXT_DARK,
                            fontsize=8, fontweight="medium")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Frazione (row-normalized)")
        cbar.outline.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        out_path = os.path.join(self.plots_dir, filename)
        fig.savefig(out_path, dpi=220, facecolor="white")
        plt.close(fig)
        logger.info(f"Salvato {out_path}")
