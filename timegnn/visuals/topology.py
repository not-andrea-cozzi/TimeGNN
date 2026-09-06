from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch.nn as nn


@dataclass
class _TopologyNode:
    node_id: int
    parent_id: Optional[int]
    depth: int
    label: str


def _count_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _module_signature(module) -> str:
    parts: List[str] = [module.__class__.__name__]

    # Add common shape hints when available.
    attrs = [
        ("in_features", "in"),
        ("out_features", "out"),
        ("in_channels", "in"),
        ("out_channels", "out"),
        ("heads", "heads"),
    ]

    hints = []
    for attr, name in attrs:
        if hasattr(module, attr):
            hints.append(f"{name}={getattr(module, attr)}")

    if hints:
        parts.append("[" + ", ".join(hints) + "]")

    return " ".join(parts)


def _build_topology(module, max_depth: int) -> List[_TopologyNode]:
    nodes: List[_TopologyNode] = []
    next_id = 0

    def visit(mod, *, parent_id: Optional[int], depth: int, name: str) -> None:
        nonlocal next_id
        if depth > max_depth:
            return

        sig = _module_signature(mod)
        n_params = _count_params(mod)
        label = f"{name}\n{sig}\nparams={n_params:,}"

        node_id = next_id
        next_id += 1
        nodes.append(_TopologyNode(node_id=node_id, parent_id=parent_id, depth=depth, label=label))

        for child_name, child_mod in mod.named_children():
            visit(child_mod, parent_id=node_id, depth=depth + 1, name=child_name)

    visit(module, parent_id=None, depth=0, name="model")
    return nodes


def _detect_layer_depths(model) -> Dict[str, int]:
    """Detect stack depth for repeated blocks (e.g., ModuleList paths)."""
    depths: Dict[str, int] = {}

    if hasattr(model, "num_layers"):
        try:
            depths["num_layers"] = int(getattr(model, "num_layers"))
        except Exception:
            pass

    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            depths[name] = len(module)

    return depths


def plot_model_topology(
    model,
    *,
    max_depth: int = 4,
    figsize: Tuple[float, float] = (14.0, 8.0),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_depth_summary: bool = True,
    ax=None,
):
    """Plot a simplified topology graph for a PyTorch model.

    The function traverses ``named_children()`` and draws a hierarchical
    architecture graph (module tree) with parameter counts.

    Args:
        model: PyTorch ``nn.Module`` instance.
        max_depth: Maximum depth to render from the root model.
        figsize: Figure size used when ``ax`` is not provided.
        title: Optional custom title.
        save_path: Optional output path (e.g. ``"topology.png"``).
        show_depth_summary: If True, draw a small box with detected
            depth information (e.g., ModuleList lengths).
        ax: Optional Matplotlib axis to draw into.

    Returns:
        Tuple ``(fig, ax)``.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Matplotlib is required for plot_model_topology. "
            "Install optional dependency: pip install 'timegnn[analysis]'"
        ) from exc

    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    nodes = _build_topology(model, max_depth=max_depth)
    if not nodes:
        raise ValueError("Could not derive model topology from the provided model.")

    by_depth: Dict[int, List[_TopologyNode]] = {}
    for node in nodes:
        by_depth.setdefault(node.depth, []).append(node)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_axis_off()

    # Compute node positions by depth (y) and order inside each depth (x).
    max_depth_found = max(by_depth.keys())
    positions: Dict[int, Tuple[float, float]] = {}

    for depth in sorted(by_depth.keys()):
        level_nodes = by_depth[depth]
        count = len(level_nodes)
        for i, node in enumerate(level_nodes):
            x = (i + 1) / (count + 1)
            y = 1.0 - (depth / max(max_depth_found, 1)) * 0.9
            positions[node.node_id] = (x, y)

    # Draw edges first so boxes stay on top.
    for node in nodes:
        if node.parent_id is None:
            continue
        x0, y0 = positions[node.parent_id]
        x1, y1 = positions[node.node_id]
        ax.annotate(
            "",
            xy=(x1, y1 + 0.04),
            xytext=(x0, y0 - 0.04),
            arrowprops={"arrowstyle": "->", "color": "#5B6C7B", "lw": 1.2, "alpha": 0.8},
            xycoords="axes fraction",
            textcoords="axes fraction",
        )

    # Draw nodes.
    box_w = 0.22
    box_h = 0.09
    for node in nodes:
        x, y = positions[node.node_id]
        rect = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#2F3B4A",
            facecolor="#EEF4FA" if node.depth == 0 else "#F8FBFE",
            transform=ax.transAxes,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            node.label,
            ha="center",
            va="center",
            fontsize=8,
            color="#1F2A36",
            transform=ax.transAxes,
            zorder=3,
        )

    total_params = _count_params(model)
    plot_title = title or f"Model Topology ({model.__class__.__name__})"
    ax.set_title(f"{plot_title}\nTrainable parameters: {total_params:,}", fontsize=12, pad=12)

    if show_depth_summary:
        depth_info = _detect_layer_depths(model)
        if depth_info:
            preferred = [
                "gat_embed",
                "gat_event",
                "gat_concat",
                "gcn_embed",
                "gcn_event",
                "gcn_concat",
                "num_layers",
            ]
            ordered_keys = [k for k in preferred if k in depth_info]
            ordered_keys += [k for k in depth_info.keys() if k not in ordered_keys]

            lines = ["Detected depth"]
            for key in ordered_keys:
                lines.append(f"{key}: {depth_info[key]}")

            ax.text(
                0.01,
                0.99,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#1F2A36",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFFFFF", "edgecolor": "#A8B5C2", "alpha": 0.95},
                zorder=4,
            )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    return fig, ax
