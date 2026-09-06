from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from scipy import stats
import matplotlib.pyplot as plt


def _ensure_1d_attention(values):
    if isinstance(values, torch.Tensor):
        if values.ndim > 1:
            return values.mean(dim=1)
        return values
    if isinstance(values, np.ndarray):
        if values.ndim > 1:
            return values.mean(axis=1)
        return values
    return values


def _safe_decay_attention(decay, alpha):
    decay = _ensure_1d_attention(decay)
    alpha = _ensure_1d_attention(alpha)
    if decay is None:
        if isinstance(alpha, torch.Tensor):
            decay = torch.ones_like(alpha)
        elif isinstance(alpha, np.ndarray):
            decay = np.ones_like(alpha)
    return decay, alpha


def get_length_bins_from_attention(attention_data, bin_size: int = 30, max_len=None, save_path=None):
    """Compute length bins and counts for attention samples."""
    lengths = [attn["edge_index"].shape[1] + 1 for attn in attention_data]
    if max_len is None:
        max_len = max(lengths) if lengths else 0

    bin_edges = list(range(0, ((max_len // bin_size) + 1) * bin_size + 1, bin_size))
    length_ranges = []
    bin_counts = defaultdict(int)

    for i in range(len(bin_edges) - 1):
        min_len = bin_edges[i]
        max_len_bin = bin_edges[i + 1]
        range_name = f"{min_len}-{max_len_bin - 1}"
        length_ranges.append((min_len, max_len_bin, range_name))

    for length in lengths:
        for min_len, max_len_bin, range_name in length_ranges:
            if min_len <= length < max_len_bin:
                bin_counts[range_name] += 1
                break

    bin_df = pd.DataFrame(
        [
            {
                "range_name": rn,
                "min_len": mn,
                "max_len": mx - 1,
                "count": bin_counts.get(rn, 0),
            }
            for (mn, mx, rn) in length_ranges
        ]
    )

    if save_path:
        bin_df.to_csv(save_path, index=False)
        print(f"✅ Bin count saved to: {save_path}")

    return length_ranges, bin_df


def get_quantile_bins_from_attention(attention_data, num_bins: int = 25):
    """Compute length bins using quantiles of sequence lengths."""
    lengths = [attn["edge_index"].shape[1] + 1 for attn in attention_data]
    quantiles = np.quantile(lengths, np.linspace(0, 1, num_bins + 1))

    length_ranges = []
    for i in range(num_bins):
        min_len = int(np.floor(quantiles[i]))
        max_len = int(np.ceil(quantiles[i + 1])) + 1
        range_name = f"{min_len}-{max_len-1}"
        length_ranges.append((min_len, max_len, range_name))
    return length_ranges


def compute_importance_stats(attention_data, length_ranges, save_path=None):
    """Aggregate importance statistics per length range and rank."""
    range_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    range_importances = defaultdict(lambda: defaultdict(list))
    df_rows = []

    for attn_dict in attention_data:
        seq_length = attn_dict["edge_index"].shape[1]
        seq_len_adjusted = seq_length + 1

        for min_len, max_len, range_name in length_ranges:
            if min_len <= seq_len_adjusted < max_len:
                decay, alpha = _safe_decay_attention(
                    attn_dict.get("decay_final"), attn_dict.get("alpha_final")
                )
                importance = decay * alpha
                if isinstance(importance, torch.Tensor) and importance.ndim > 1:
                    importance = importance.mean(dim=1)
                ranked_nodes = torch.argsort(importance, descending=True).tolist()

                for rank, node in enumerate(ranked_nodes[:3], start=1):
                    range_stats[range_name][rank][node] += 1
                    range_importances[range_name][rank].append(importance[node].item())
                break

    def get_min_len(bin_name):
        return int(bin_name.split("-")[0].replace("<", "").replace(">", ""))

    for range_name in sorted(range_stats.keys(), key=get_min_len):
        row_data = {"Length Range": range_name}
        for rank in [1, 2, 3]:
            node_counts = range_stats[range_name][rank]
            if not node_counts:
                row_data[f"Rank{rank}_Node"] = None
                row_data[f"Rank{rank}_Dominance"] = 0.0
                row_data[f"Rank{rank}_MeanImp"] = 0.0
                continue

            top_node, count = max(node_counts.items(), key=lambda x: x[1])
            dominance = (count / sum(node_counts.values())) * 100
            mean_imp = torch.tensor(range_importances[range_name][rank]).mean().item()

            row_data[f"Rank{rank}_Node"] = top_node
            row_data[f"Rank{rank}_Dominance"] = round(dominance, 2)
            row_data[f"Rank{rank}_MeanImp"] = round(mean_imp, 4)

        df_rows.append(row_data)

    df_summary = pd.DataFrame(df_rows)

    if save_path:
        df_summary.to_csv(save_path, index=False)
        print(f"📄 Saved importance summary to: {save_path}")

    return range_stats, range_importances, df_summary


def plot_heatmap_from_stats(range_stats, range_importances, save_path=None):
    """Plot a heatmap of node importance by length range."""
    heatmap_data = defaultdict(dict)
    all_nodes = set()

    for range_name in range_stats:
        for rank in range_stats[range_name]:
            all_nodes.update(range_stats[range_name][rank].keys())

    max_node = max(all_nodes) if all_nodes else 0

    for range_name in sorted(range_stats.keys(), key=lambda x: int(x.split("-")[0].replace("<", "").replace(">", ""))):
        for node in range(max_node + 1):
            importances = []
            for rank in [1, 2, 3]:
                if node in range_stats[range_name][rank]:
                    importances.extend(range_importances[range_name][rank])
            heatmap_data[range_name][node] = np.mean(importances) if importances else np.nan

    df_heatmap = pd.DataFrame(heatmap_data).T

    colors = [
        "#f7fbff",
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#08306b",
        "#fff5eb",
        "#fee6ce",
        "#fdd0a2",
        "#fdae6b",
        "#fd8d3c",
        "#f16913",
        "#d94801",
        "#8c2d04",
    ]
    cmap = LinearSegmentedColormap.from_list("blue_orange", colors, N=256)

    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42

    hm = sns.heatmap(
        df_heatmap,
        cmap=cmap,
        cbar_kws={"label": "Mean Importance Score", "shrink": 0.75, "pad": 0.02, "aspect": 10},
        vmin=0,
        vmax=1,
        square=False,
        linewidths=0.5,
        linecolor="white",
        annot=False,
        norm=PowerNorm(gamma=0.4),
        ax=ax,
    )

    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, width=0.5)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.set_ylabel("Mean Importance Score", fontsize=12, rotation=270, labelpad=20, fontweight="normal")

    if len(df_heatmap.columns) <= 40:
        threshold = np.nanpercentile(df_heatmap.values, 85)
        for y in range(df_heatmap.shape[0]):
            for x in range(df_heatmap.shape[1]):
                val = df_heatmap.iloc[y, x]
                if not np.isnan(val) and val >= threshold:
                    ax.text(
                        x + 0.5,
                        y + 0.5,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if val > 0.5 else "#333333",
                        bbox=dict(
                            boxstyle="round",
                            facecolor="white" if val <= 0.5 else "#444444",
                            alpha=0.7,
                            edgecolor="none",
                            pad=0.1,
                        ),
                    )

    ax.set_xlabel("Node Position", fontsize=12, labelpad=10)
    ax.set_ylabel("Sequence Length Range", fontsize=12, labelpad=10)
    ax.tick_params(axis="both", which="both", labelsize=10, length=3, width=0.5, pad=2)
    plt.xticks(rotation=45, ha="right")

    plt.title("Node Importance Heatmap by Position and Sequence Length", fontsize=14, pad=16, fontweight="semibold")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#555555")
        spine.set_linewidth(0.8)

    ax.text(
        0.98,
        0.98,
        f"Range: {df_heatmap.min().min():.2f}-{df_heatmap.max().max():.2f}\n"
        f"Mean: {np.nanmean(df_heatmap.values):.2f} ± {np.nanstd(df_heatmap.values):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}.tiff", format="tiff", bbox_inches="tight", dpi=600)
        plt.savefig(f"{save_path}.png", format="png", bbox_inches="tight", dpi=300)

    plt.show()


def plot_rank_dominance(range_stats, title="Top Node Dominance by Rank and Length Range", save_path=None):
    """Plot rank dominance bars for top nodes per length range."""
    plt.figure(figsize=(12, 7), dpi=300)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42

    palette = {"Rank 1": "#7A5195", "Rank 2": "#009FDF", "Rank 3": "#FF6E54"}

    rank_dominance = []
    for range_name in sorted(range_stats.keys(), key=lambda x: int(x.split("-")[0].replace("<", "").replace(">", ""))):
        for rank in [1, 2, 3]:
            node_counts = range_stats[range_name][rank]
            if not node_counts:
                continue

            total = sum(node_counts.values())
            top_node, count = max(node_counts.items(), key=lambda x: x[1])
            rank_dominance.append(
                {
                    "Range": range_name,
                    "Rank": f"Rank {rank}",
                    "Dominance (%)": (count / total) * 100,
                    "Top Node": top_node,
                }
            )

    df_rank = pd.DataFrame(rank_dominance)

    ax = sns.barplot(
        data=df_rank,
        x="Range",
        y="Dominance (%)",
        hue="Rank",
        palette=palette,
        order=sorted(df_rank["Range"].unique(), key=lambda x: int(x.split("-")[0].replace("<", "").replace(">", ""))),
        edgecolor="white",
        linewidth=1.0,
        saturation=0.95,
        err_kws={"linewidth": 1.0},
    )

    for i, (_, row) in enumerate(df_rank.iterrows()):
        n_ranks = len(df_rank["Rank"].unique())
        bar_width = 0.8 / n_ranks
        x_pos = i // n_ranks - 0.4 + bar_width * (i % n_ranks) + bar_width / 2

        text_y_pos = max(5, row["Dominance (%)"] / 2)

        ax.text(
            x_pos,
            text_y_pos,
            f"N{row['Top Node']}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
            color="white",
            fontweight="bold",
            path_effects=[patheffects.withStroke(linewidth=2, foreground="#333333")],
        )

    plt.title(title, fontsize=14, pad=20, fontweight="semibold")
    plt.ylim(0, 110)
    plt.xlabel("Sequence Length Range", fontsize=12, labelpad=10)
    plt.ylabel("Dominance (%)", fontsize=12, labelpad=10)

    legend = plt.legend(
        title="Rank",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        framealpha=1,
        edgecolor="#333333",
    )
    legend.get_title().set_fontweight("semibold")

    ax.text(
        0.98,
        0.98,
        f"Max: {df_rank['Dominance (%)'].max():.1f}%\n"
        f"Min: {df_rank['Dominance (%)'].min():.1f}%\n"
        f"Avg: {df_rank['Dominance (%)'].mean():.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}.eps", format="eps", bbox_inches="tight")
        plt.savefig(f"{save_path}.tiff", format="tiff", bbox_inches="tight", dpi=600)
        plt.savefig(f"{save_path}.png", format="png", bbox_inches="tight", dpi=300)

    plt.show()


def plot_importance_timeline(attention_data, min_length=100, max_length=None, figsize=(20, 8), save_path=None):
    """Plot attention components over time for a long sequence."""
    def length_check(attn_dict):
        length = attn_dict["edge_index"].max().item() + 1
        meets_min = length >= min_length
        meets_max = (max_length is None) or (length <= max_length)
        return meets_min and meets_max

    long_seq = next((attn_dict for attn_dict in attention_data if length_check(attn_dict)), None)

    if long_seq is None:
        length_range = f">= {min_length}" if max_length is None else f"between {min_length}-{max_length}"
        print(f"No sequences found with length {length_range}")
        return

    if long_seq.get("decay_final") is None or long_seq.get("alpha_final") is None:
        print("Missing decay/attention values; skip plot_importance_timeline_edge.")
        return

    time = long_seq["time"].cpu().numpy()
    decay, attention = _safe_decay_attention(
        long_seq.get("decay_final").cpu().numpy() if long_seq.get("decay_final") is not None else None,
        long_seq.get("alpha_final").cpu().numpy() if long_seq.get("alpha_final") is not None else None,
    )
    importance = decay * attention
    node_numbers = np.arange(1, len(time) + 1)

    alpha_embed = _ensure_1d_attention(long_seq["alpha_embed"].cpu().numpy())
    alpha_event = _ensure_1d_attention(long_seq["alpha_event"].cpu().numpy())

    plt.figure(figsize=figsize, dpi=300)
    ax = plt.gca()

    time_diffs = np.diff(time)
    avg_time_diff = np.mean(time_diffs) if len(time_diffs) > 0 else 0.02
    bar_width = avg_time_diff * 0.25

    for i, t in enumerate(time):
        alpha = 0.3 if i % max(1, len(time) // 20) == 0 else 0.1
        ax.axvline(x=t, color="#888888", linestyle="-", alpha=alpha, linewidth=0.8)

    for i, t in enumerate(time):
        ax.bar(
            t,
            alpha_embed[i],
            width=bar_width,
            color="#17becf",
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            label="Embed Attention" if i == 0 else "",
        )
        ax.bar(
            t,
            alpha_event[i],
            width=bar_width,
            color="#e377c2",
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            label="Event Attention" if i == 0 else "",
        )

    plt.plot(time, decay, label="Decay", color="#4e79a7", linestyle="--", alpha=0.7, linewidth=2.5)
    plt.plot(time, attention, label="Final Attention", color="#f28e2b", linestyle=":", alpha=0.9, linewidth=2.5)
    plt.plot(time, importance, label="Importance (decay × attention)", color="#59a14f", linewidth=3.5)

    def get_non_overlapping_positions(times, y_max, num_labels=20):
        positions = []
        time_range = max(times) - min(times)
        min_x_spacing = time_range * 0.05
        min_y_spacing = y_max * 0.08

        key_indices = [0, len(times) - 1, *np.argsort(importance)[-3:][::-1]]

        for i in sorted(set(key_indices + list(np.linspace(0, len(times) - 1, num_labels, dtype=int)))):
            t = times[i]
            y_pos = y_max * 0.98

            while any(abs(t - pos[0]) < min_x_spacing and abs(y_pos - pos[1]) < min_y_spacing for pos in positions):
                y_pos -= min_y_spacing
                if y_pos < y_max * 0.2:
                    y_pos = y_max * 0.98
                    break

            positions.append((t, y_pos, i))
        return positions

    y_max = ax.get_ylim()[1]
    label_positions = get_non_overlapping_positions(time, y_max)

    for t, y_pos, i in label_positions:
        ax.text(
            t,
            y_pos,
            f"N{node_numbers[i]}",
            ha="center",
            va="top",
            rotation=45,
            fontsize=10,
            color="#333333",
            alpha=0.9,
            fontweight="semibold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

    top_indices = np.argsort(importance)[-3:][::-1]
    label_offsets = [0.95, 0.75, 0.55]

    for idx, pos in zip(top_indices, label_offsets):
        t = time[idx]
        node = node_numbers[idx]

        existing_labels = [(pos[0], pos[1]) for pos in label_positions]
        y_pos = y_max * pos
        while any(abs(t - x) < bar_width * 3 and abs(y_pos - y) < y_max * 0.1 for (x, y) in existing_labels):
            y_pos -= y_max * 0.05

        ax.axvline(x=t, color="#e15759", linestyle="-", alpha=0.6, linewidth=2)
        ax.text(
            t,
            y_pos,
            f"Top{top_indices.tolist().index(idx) + 1} (N{node})",
            ha="center",
            va="center",
            rotation=0,
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#e15759", boxstyle="round,pad=0.3", linewidth=1.5),
        )

    plt.gca().invert_xaxis()
    plt.xlabel("Normalized Time (1.0 = start, 0.0 = end)", fontsize=13, labelpad=10)
    plt.ylabel("Normalized Attention Score", fontsize=13, labelpad=10)

    title = (
        f"Attention Component Dynamics\n"
        f"Sequence Length: {len(time)} nodes | "
        f"Max Importance: {importance.max():.2f} at N{node_numbers[np.argmax(importance)]}"
    )
    plt.title(title, fontsize=15, pad=20, fontweight="semibold")

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        plt.Line2D([0], [0], color="#e15759", linewidth=2, alpha=0.6),
        plt.Line2D([0], [0], color="#888888", linewidth=0.8, alpha=0.3),
    ])
    labels.extend(["Top Events", "All Nodes"])

    legend = plt.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=1,
        borderaxespad=0.0,
        title="Components",
        title_fontsize=12,
        fontsize=11,
    )
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor("#cccccc")

    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}.eps", format="eps", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}.tiff", format="tiff", bbox_inches="tight", dpi=600)
        plt.savefig(f"{save_path}.png", format="png", bbox_inches="tight", dpi=300)

    plt.show()
    return plt.gcf()


def plot_critical_windows(attention_data, length_ranges, figsize=(12, 6), compare_ranges=None, alpha=0.05, save_path=None):
    """Identify critical windows across length ranges using attention scores."""
    plt.figure(figsize=figsize, dpi=300)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42

    window_stats = []
    for min_len, max_len, range_name in length_ranges:
        group_seqs = [attn for attn in attention_data if min_len <= (attn["edge_index"].shape[1] + 1) < max_len]

        if not group_seqs:
            continue

        pos_importance = defaultdict(list)
        for seq in group_seqs:
            decay, alpha = _safe_decay_attention(seq.get("decay_final"), seq.get("alpha_final"))
            importance = decay * alpha
            ranked_pos = torch.argsort(importance, descending=True)[:3]
            for pos in ranked_pos:
                pos_importance[pos.item()].append(importance[pos].item())

        for pos, imp_values in pos_importance.items():
            window_stats.append(
                {
                    "Range": range_name,
                    "Position": pos,
                    "Mean Importance": np.mean(imp_values),
                    "Frequency": len(imp_values) / len(group_seqs),
                }
            )

    df = pd.DataFrame(window_stats)

    ax = sns.scatterplot(
        data=df,
        x="Position",
        y="Mean Importance",
        size="Frequency",
        hue="Range",
        sizes=(50, 300),
        palette="viridis",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    if compare_ranges:
        sig_results = []
        for i, (range1, range2) in enumerate(compare_ranges):
            group1 = df[df["Range"] == range1]["Mean Importance"]
            group2 = df[df["Range"] == range2]["Mean Importance"]

            if len(group1) > 1 and len(group2) > 1:
                _, p = stats.ttest_ind(group1, group2)
                stars = "*" * sum(p < alpha / (2**i) for i in range(1, 3))
                sig_results.append(f"{range1} vs {range2}: p={p:.2e}{stars}")

        if sig_results:
            ax.text(
                0.98,
                0.98,
                "Statistical Comparisons:\n" + "\n".join(sig_results),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0, edgecolor="#cccccc", pad=1, boxstyle="round"),
            )

    plt.title("Critical Window Identification Across Length Ranges", fontsize=14, pad=20, fontweight="semibold")
    plt.xlabel("Node Position in Sequence", fontsize=12, labelpad=10)
    plt.ylabel("Mean Importance Score", fontsize=12, labelpad=10)

    ax.grid(True, alpha=0.2, linestyle=":")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(
        handles[: len(length_ranges) + 1],
        labels[: len(length_ranges) + 1],
        title="Length Range",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        framealpha=1,
        edgecolor="#333333",
    )
    legend.get_title().set_fontweight("semibold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}.eps", format="eps", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}.tiff", format="tiff", bbox_inches="tight", dpi=600)
        plt.savefig(f"{save_path}.png", format="png", bbox_inches="tight", dpi=300)

    plt.show()
    return df


def generate_pairwise_ttest_table(stats_df, length_ranges, alpha=0.05):
    """Generate pairwise t-test table for critical window comparisons."""
    range_names = [name for _, _, name in length_ranges]
    results = []

    for i, range1 in enumerate(range_names):
        for range2 in range_names[i + 1 :]:
            group1 = stats_df[stats_df["Range"] == range1]["Mean Importance"]
            group2 = stats_df[stats_df["Range"] == range2]["Mean Importance"]

            if len(group1) > 1 and len(group2) > 1:
                t_stat, p_value = stats.ttest_ind(group1, group2)

                pooled_std = np.sqrt(
                    ((len(group1) - 1) * group1.std() ** 2 + (len(group2) - 1) * group2.std() ** 2)
                    / (len(group1) + len(group2) - 2)
                )
                cohens_d = (group1.mean() - group2.mean()) / pooled_std

                stars = "*" * sum(p_value < alpha / (2**i) for i in range(1, 3))

                results.append(
                    {
                        "Comparison": f"{range1} vs {range2}",
                        "Mean Diff": group1.mean() - group2.mean(),
                        "t-statistic": t_stat,
                        "p-value": p_value,
                        "Cohen's d": cohens_d,
                        "Significance": stars,
                        "n1": len(group1),
                        "n2": len(group2),
                    }
                )

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df["p-value"] = results_df["p-value"].apply(lambda x: f"{x:.3e}")
        results_df["Mean Diff"] = results_df["Mean Diff"].apply(lambda x: f"{x:.3f}")
        results_df["Cohen's d"] = results_df["Cohen's d"].apply(lambda x: f"{x:.2f}")
        results_df["t-statistic"] = results_df["t-statistic"].apply(lambda x: f"{x:.2f}")

        results_df = results_df[
            [
                "Comparison",
                "Mean Diff",
                "t-statistic",
                "p-value",
                "Cohen's d",
                "Significance",
                "n1",
                "n2",
            ]
        ]

    return results_df


def get_top_significant_ranges(pairwise_table, num_top: int = 10):
    """Extract most significant range comparisons from pairwise table."""
    sig_comparisons = pairwise_table.copy()
    sig_comparisons = sig_comparisons[sig_comparisons["Significance"].str.len() > 0]

    if sig_comparisons.empty:
        return pd.DataFrame(columns=pairwise_table.columns), []

    sig_comparisons = sig_comparisons.assign(**{
        "p-value": sig_comparisons["p-value"].astype(float),
        "star_count": sig_comparisons["Significance"].str.len(),
    })
    sig_comparisons = sig_comparisons.sort_values(
        by=["star_count", "p-value", "Cohen's d"], ascending=[False, True, False]
    )

    top_comparisons = sig_comparisons.head(min(num_top, len(sig_comparisons)))

    comparison_tuples = []
    for comp in top_comparisons["Comparison"]:
        range1, range2 = comp.split(" vs ")
        comparison_tuples.append((range1, range2))

    formatted_comparisons = top_comparisons.copy()
    formatted_comparisons = formatted_comparisons.assign(**{
        "p-value": formatted_comparisons["p-value"].apply(lambda x: f"{x:.3e}"),
    })
    formatted_comparisons = formatted_comparisons.drop(columns="star_count")

    return formatted_comparisons, comparison_tuples


def calculate_window_metrics(attention_data, length_ranges):
    """Compute attention window metrics for downstream visualization."""
    metrics = []

    for attn in attention_data:
        seq_len = attn["edge_index"].shape[1] + 1

        for min_len, max_len, range_name in length_ranges:
            if min_len <= seq_len < max_len:
                decay, alpha = _safe_decay_attention(attn.get("decay_final"), attn.get("alpha_final"))
                importance = alpha * decay
                peak_attn = importance.max().item()
                threshold = 0.5 * peak_attn
                attention_span = (importance > threshold).sum().item() / seq_len
                peak_pos = torch.argmax(importance).item() / seq_len

                metrics.append(
                    {
                        "Length Range": range_name,
                        "Sort Key": min_len,
                        "Peak Attention": peak_attn,
                        "Attention Span": attention_span,
                        "Peak Position": peak_pos,
                        "Sequence Length": seq_len,
                    }
                )
                break

    df = pd.DataFrame(metrics)
    return df.sort_values("Sort Key")


def aggregate_metrics(df_metrics):
    """Aggregate window metrics for summary tables."""
    ordered_ranges = df_metrics["Length Range"].unique()
    df_metrics["Length Range"] = pd.Categorical(
        df_metrics["Length Range"], categories=ordered_ranges, ordered=True
    )

    agg_stats = df_metrics.groupby("Length Range", observed=True).agg(
        {
            "Peak Attention": ["mean", "std", "count"],
            "Attention Span": ["mean", "std"],
            "Peak Position": ["mean", "std"],
        }
    )

    return agg_stats.round(3).rename(columns={"mean": "Mean", "std": "SD", "count": "N"})


def plot_window_metrics(df_metrics, figsize=(20, 7), save_path=None):
    """Plot window metrics across length ranges."""
    fig = plt.figure(figsize=figsize, dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.25)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]

    palettes = {
        "peak": sns.light_palette("#4e79a7", n_colors=len(df_metrics["Length Range"].unique())),
        "span": sns.light_palette("#59a14f", n_colors=len(df_metrics["Length Range"].unique())),
        "pos": sns.light_palette("#e15759", n_colors=len(df_metrics["Length Range"].unique())),
    }

    sns.boxplot(
        data=df_metrics,
        x="Length Range",
        y="Peak Attention",
        hue="Length Range",
        palette=palettes["peak"],
        ax=axes[0],
        width=0.7,
        linewidth=1.5,
        fliersize=4,
        legend=False,
        dodge=False,
    )
    axes[0].set_title("Peak Attention Intensity", pad=12, fontsize=13, fontweight="semibold")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Attention Score", labelpad=10)

    medians = df_metrics.groupby("Length Range", observed=False)["Peak Attention"].median()
    for i, (_, m) in enumerate(medians.items()):
        axes[0].text(
            i,
            m + 0.03,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, pad=2, edgecolor="none"),
        )

    sns.violinplot(
        data=df_metrics,
        x="Length Range",
        y="Attention Span",
        hue="Length Range",
        palette=palettes["span"],
        ax=axes[1],
        cut=0,
        inner="quartile",
        linewidth=1.5,
        saturation=0.8,
        legend=False,
        dodge=False,
    )
    axes[1].set_title("Proportion of Sequence Attended", pad=12, fontsize=13, fontweight="semibold")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Proportion Attended", labelpad=10)

    for i in range(1, 10):
        axes[1].axhline(i * 0.1, color="white", alpha=0.1, linewidth=0.5, zorder=0)

    sns.stripplot(
        data=df_metrics,
        x="Length Range",
        y="Peak Position",
        hue="Length Range",
        palette=palettes["pos"],
        ax=axes[2],
        alpha=0.7,
        jitter=0.25,
        size=5,
        linewidth=0.5,
        edgecolor="white",
        legend=False,
        dodge=False,
    )

    sns.pointplot(
        data=df_metrics,
        x="Length Range",
        y="Peak Position",
        errorbar=("ci", 95),
        color="black",
        ax=axes[2],
        markersize=10,
        linestyle="none",
        capsize=0.2,
        err_kws={"linewidth": 1.5},
    )
    axes[2].set_title("Normalized Position of Peak Attention", pad=12, fontsize=13, fontweight="semibold")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Normalized Position", labelpad=10)

    for ax in axes:
        ax.set_xlabel("Sequence Length Range", labelpad=10)
        ax.tick_params(axis="x", rotation=45, labelsize=10)

        ax.grid(axis="y", alpha=0.2, linestyle=":", linewidth=0.8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        ax.set_facecolor("#f9f9f9")

    fig.suptitle(
        "Dynamic Window Attention Metrics by Sequence Length Group\n"
        "Comparative Analysis of Attention Patterns Across Input Lengths",
        y=1.1,
        fontsize=14,
        fontweight="bold",
    )

    plt.figtext(
        0.5,
        -0.03,
        f"Analysis based on {len(df_metrics)} samples | "
        f"{len(df_metrics['Length Range'].unique())} length groups | "
        f"Median values annotated",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}.eps", format="eps", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}.tiff", format="tiff", bbox_inches="tight", dpi=600)
        plt.savefig(f"{save_path}.png", format="png", bbox_inches="tight", dpi=300)

    plt.show()


def plot_edge_attention_correlation(attention_data, length_ranges, figsize=(16, 12), save_path=None):
    """Plot edge-type scores versus attention by length range."""
    records = []
    for attn in attention_data:
        if attn.get("edge_type_score_final") is None or attn.get("alpha_final") is None:
            continue
        seq_len = attn["edge_index"].shape[1]
        length_group = next((lr[2] for lr in length_ranges if lr[0] <= seq_len < lr[1]), None)
        if not length_group:
            continue

        for i in range(seq_len):
            records.append(
                {
                    "length_group": length_group,
                    "seq_length": seq_len,
                    "edge_score": attn["edge_type_score_final"][i].item(),
                    "attention": attn["alpha_final"][i].item(),
                    "edge_type": attn["edge_type"][i].item(),
                }
            )
    if not records:
        print("Missing edge-type scores; skip plot_edge_attention_correlation.")
        return
    df = pd.DataFrame(records)

    n_groups = len(length_ranges)
    palette = sns.color_palette("viridis", n_colors=n_groups)
    length_order = [lr[2] for lr in length_ranges]

    fig = plt.figure(figsize=figsize, dpi=300)
    gs = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)

    ax = plt.subplot(gs[1, 0])

    ax.hexbin(df["edge_score"], df["attention"], gridsize=30, cmap="Greys", mincnt=1, alpha=0.2, linewidths=0)

    sns.scatterplot(
        data=df.sample(frac=0.3, random_state=42),
        x="edge_score",
        y="attention",
        hue="length_group",
        hue_order=length_order,
        palette=palette,
        alpha=0.7,
        edgecolor="none",
        size="seq_length",
        sizes=(20, 100),
        ax=ax,
    )

    for i, group in enumerate(length_order):
        group_df = df[df["length_group"] == group]
        if len(group_df) > 10:
            sns.regplot(
                data=group_df,
                x="edge_score",
                y="attention",
                scatter=False,
                color=palette[i],
                line_kws={"lw": 2, "ls": "--"},
                ci=95,
                ax=ax,
            )

    ax_histx = plt.subplot(gs[0, :], sharex=ax)
    sns.kdeplot(
        data=df,
        x="edge_score",
        hue="length_group",
        hue_order=length_order,
        palette=palette,
        fill=True,
        alpha=0.2,
        linewidth=0.5,
        ax=ax_histx,
        legend=False,
    )
    ax_histx.set_xlabel("")
    ax_histx.set_ylabel("Density")

    ax_histy = plt.subplot(gs[1, 1], sharey=ax)
    sns.kdeplot(
        data=df,
        y="attention",
        hue="length_group",
        hue_order=length_order,
        palette=palette,
        fill=True,
        alpha=0.2,
        linewidth=0.5,
        ax=ax_histy,
    )
    ax_histy.set_ylabel("")
    ax_histy.set_xlabel("Density")

    r, p = stats.pearsonr(df["edge_score"], df["attention"])
    ax.text(0.05, 0.95, f"Overall: r = {r:.2f}, p = {p:.1e}", transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.8))

    top_edge_types = df["edge_type"].value_counts().head(3).index
    for i, edge_type in enumerate(top_edge_types):
        edge_df = df[df["edge_type"] == edge_type]
        if len(edge_df) > 10:
            r_edge, _ = stats.pearsonr(edge_df["edge_score"], edge_df["attention"])
            ax.text(
                0.05,
                0.85 - 0.05 * i,
                f"Type {edge_type}: r = {r_edge:.2f}",
                transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.6),
            )

    ax.set_xlabel("Edge Type Score (Final)", fontsize=12)
    ax.set_ylabel("Attention Score (Final)", fontsize=12)

    handles = [Line2D([], [], marker="o", linestyle="", color=palette[i], label=group) for i, group in enumerate(length_order)]
    ax.legend(
        handles=handles,
        title="Length Range",
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        frameon=True,
        framealpha=1,
        fontsize=8,
        title_fontsize=9,
        borderaxespad=0.0,
    )

    fig.suptitle("Edge-Attention Relationship by Sequence Length", fontsize=14, y=0.92, ha="center")
    fig.text(
        0.5,
        0.89,
        "(Top) Edge score distribution by length range; (Middle) Scatter plot with length-stratified trendlines; (Right) Attention score distribution",
        fontsize=12,
        ha="center",
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}.eps", format="eps", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}.tiff", format="tiff", bbox_inches="tight", dpi=600)
        plt.savefig(f"{save_path}.png", format="png", bbox_inches="tight", dpi=300)

    plt.show()
    return df


def plot_importance_timeline_edge(attention_data, min_length=100, max_length=None, figsize=(20, 8), save_path=None):
    """Plot timeline with attention components and edge-type scores."""
    def length_check(attn_dict):
        length = attn_dict["edge_index"].max().item() + 1
        meets_min = length >= min_length
        meets_max = (max_length is None) or (length <= max_length)
        return meets_min and meets_max

    long_seq = next((attn_dict for attn_dict in attention_data if length_check(attn_dict)), None)

    if long_seq is None:
        length_range = f">= {min_length}" if max_length is None else f"between {min_length}-{max_length}"
        print(f"No sequences found with length {length_range}")
        return

    time = long_seq["time"].cpu().numpy()
    node_numbers = np.arange(1, len(time) + 1)

    decay, attention = _safe_decay_attention(
        long_seq.get("decay_final").cpu().numpy() if long_seq.get("decay_final") is not None else None,
        long_seq.get("alpha_final").cpu().numpy() if long_seq.get("alpha_final") is not None else None,
    )
    importance = decay * attention
    alpha_embed = _ensure_1d_attention(long_seq["alpha_embed"].cpu().numpy())
    alpha_event = _ensure_1d_attention(long_seq["alpha_event"].cpu().numpy())

    if long_seq.get("edge_type_score_embed") is None:
        print("Missing edge-type scores; skip plot_importance_timeline_edge.")
        return

    edge_embed = _ensure_1d_attention(long_seq["edge_type_score_embed"].cpu().numpy())
    edge_event = _ensure_1d_attention(long_seq["edge_type_score_event"].cpu().numpy())
    edge_final = _ensure_1d_attention(long_seq["edge_type_score_final"].cpu().numpy())

    plt.figure(figsize=figsize, dpi=300)
    ax = plt.gca()

    time_diffs = np.diff(time)
    avg_time_diff = np.mean(time_diffs) if len(time_diffs) > 0 else 0.02
    bar_width = avg_time_diff * 0.8
    edge_offset = -bar_width * 0.6
    alpha_offset = bar_width * 0.6

    for i, t in enumerate(time):
        alpha = 0.3 if i % max(1, len(time) // 20) == 0 else 0.1
        ax.axvline(x=t, color="#888888", linestyle="-", alpha=alpha, linewidth=0.8)

    for i, t in enumerate(time):
        ax.bar(
            t + edge_offset,
            edge_embed[i],
            width=bar_width * 0.9,
            color="#aec7e8",
            alpha=0.9,
            label="Edge Embed Score" if i == 0 else "",
        )
        ax.bar(
            t + edge_offset,
            edge_event[i],
            width=bar_width * 0.9,
            color="#ffbb78",
            alpha=0.9,
            bottom=edge_embed[i],
            label="Edge Event Score" if i == 0 else "",
        )

    for i, t in enumerate(time):
        ax.bar(
            t + alpha_offset,
            alpha_embed[i],
            width=bar_width * 0.9,
            color="#17becf",
            alpha=0.7,
            label="Embed Attention" if i == 0 else "",
        )
        ax.bar(
            t + alpha_offset,
            alpha_event[i],
            width=bar_width * 0.9,
            color="#e377c2",
            alpha=0.7,
            label="Event Attention" if i == 0 else "",
        )

    plt.plot(time, decay, label="Decay", color="#4e79a7", linestyle="--", linewidth=2)
    plt.plot(time, attention, label="Final Attention", color="#f28e2b", linestyle=":", linewidth=2)
    plt.plot(time, edge_final, label="Edge Final Score", color="#8c564b", linestyle="-.", linewidth=2)
    plt.plot(time, importance, label="Importance", color="#59a14f", linewidth=3)

    def get_non_overlapping_positions(times, y_max, num_labels=20):
        positions = []
        time_range = max(times) - min(times)
        min_x_spacing = time_range * 0.05
        min_y_spacing = y_max * 0.08

        key_indices = [0, len(times) - 1, *np.argsort(importance)[-3:][::-1]]

        for i in sorted(set(key_indices + list(np.linspace(0, len(times) - 1, num_labels, dtype=int)))):
            t = times[i]
            y_pos = y_max * 0.98

            while any(abs(t - pos[0]) < min_x_spacing and abs(y_pos - pos[1]) < min_y_spacing for pos in positions):
                y_pos -= min_y_spacing
                if y_pos < y_max * 0.2:
                    y_pos = y_max * 0.98
                    break

            positions.append((t, y_pos, i))
        return positions

    y_max = ax.get_ylim()[1]
    label_positions = get_non_overlapping_positions(time, y_max)

    for t, y_pos, i in label_positions:
        ax.text(
            t,
            y_pos,
            f"N{node_numbers[i]}",
            ha="center",
            va="top",
            rotation=45,
            fontsize=10,
            color="#333333",
            alpha=0.9,
            fontweight="semibold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

    top_indices = np.argsort(importance)[-3:][::-1]
    label_offsets = [0.95, 0.75, 0.55]

    for idx, pos in zip(top_indices, label_offsets):
        t = time[idx]
        node = node_numbers[idx]

        existing_labels = [(pos[0], pos[1]) for pos in label_positions]
        y_pos = y_max * pos
        while any(abs(t - x) < bar_width * 3 and abs(y_pos - y) < y_max * 0.1 for (x, y) in existing_labels):
            y_pos -= y_max * 0.05

        ax.axvline(x=t, color="#e15759", linestyle="-", alpha=0.6, linewidth=2)
        ax.text(
            t,
            y_pos,
            f"Top{top_indices.tolist().index(idx) + 1} (N{node})",
            ha="center",
            va="center",
            rotation=0,
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#e15759", boxstyle="round,pad=0.3", linewidth=1.5),
        )

    plt.gca().invert_xaxis()
    plt.xlabel("Normalized Time (1.0 = start, 0.0 = end)", fontsize=13, labelpad=10)
    plt.ylabel("Normalized Attention Score", fontsize=13, labelpad=10)

    title = (
        f"Attention Component Dynamics\n"
        f"Sequence Length: {len(time)} nodes | "
        f"Max Importance: {importance.max():.2f} at N{node_numbers[np.argmax(importance)]}"
    )
    plt.title(title, fontsize=15, pad=20, fontweight="semibold")

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        plt.Line2D([0], [0], color="#e15759", linewidth=2, alpha=0.6),
        plt.Line2D([0], [0], color="#888888", linewidth=0.8, alpha=0.3),
    ])
    labels.extend(["Top Events", "All Nodes"])

    legend = plt.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=1,
        borderaxespad=0.0,
        title="Components",
        title_fontsize=12,
        fontsize=11,
    )
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor("#cccccc")

    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}.eps", format="eps", bbox_inches="tight", dpi=300)
        plt.savefig(f"{save_path}.tiff", format="tiff", bbox_inches="tight", dpi=600)
        plt.savefig(f"{save_path}.png", format="png", bbox_inches="tight", dpi=300)

    plt.show()
    return plt.gcf()
