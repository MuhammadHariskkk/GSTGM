"""Matplotlib helpers for trajectories, graphs, and training curves (GitHub-friendly)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from torch import Tensor

from gstgm.graph.graph_builder import SceneGraphBatch, build_from_collated_batch

# Neutral, light-on-dark safe palette; works in README / GitHub light theme
STYLE = {
    "obs": "#1f77b4",
    "future": "#2ca02c",
    "focal": "#d62728",
    "neighbor": "#9467bd",
    "edge": "#7f7f7f",
    "grid": "#e0e0e0",
    "text": "#333333",
}


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": STYLE["text"],
            "axes.labelcolor": STYLE["text"],
            "axes.titlecolor": STYLE["text"],
            "xtick.color": STYLE["text"],
            "ytick.color": STYLE["text"],
            "text.color": STYLE["text"],
            "legend.framealpha": 0.95,
            "legend.edgecolor": STYLE["grid"],
            "font.size": 10,
            "axes.grid": True,
            "grid.color": STYLE["grid"],
            "grid.linestyle": "-",
            "grid.alpha": 0.6,
        }
    )


def _to_xy2d(t: Tensor) -> np.ndarray:
    x = t.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def save_figure(path: Path, fig: plt.Figure | None = None, *, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = fig if fig is not None else plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return path.resolve()


def plot_trajectories_multimodal(
    obs_xy: Tensor,
    future_xy: Tensor,
    pred_positions_per_mode: Tensor,
    *,
    mode_probs: Tensor | None = None,
    mode_indices_to_plot: Sequence[int] | None = None,
    title: str | None = None,
    equal_aspect: bool = True,
) -> plt.Figure:
    """
    Plot observed trajectory, ground-truth future, and predicted futures (one line per mode).

    Parameters
    ----------
    obs_xy :
        ``[T_obs, 2]``
    future_xy :
        ``[T_pred, 2]``
    pred_positions_per_mode :
        ``[T_pred, M, 2]`` integrated positions (already in same coordinate frame as ``obs``/``future``).
    mode_probs :
        Optional ``[M]`` mean weights for legend labels.
    mode_indices_to_plot :
        Subset of mode indices; default all ``M``.
    """
    apply_plot_style()
    o = _to_xy2d(obs_xy)
    f = _to_xy2d(future_xy)
    pr = _to_xy2d(pred_positions_per_mode)
    _, m, _ = pr.shape
    if mode_indices_to_plot is None:
        idxs = list(range(m))
    else:
        idxs = list(mode_indices_to_plot)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot(o[:, 0], o[:, 1], color=STYLE["obs"], linewidth=2.0, marker="o", markersize=4, label="Observed")
    ax.plot(f[:, 0], f[:, 1], color=STYLE["future"], linewidth=2.0, marker="s", markersize=3, label="Ground-truth future")
    ax.scatter([o[-1, 0]], [o[-1, 1]], c=STYLE["focal"], s=80, zorder=5, marker="*", label="Last obs")

    cmap = plt.get_cmap("tab10")
    for mode_i in idxs:
        if mode_i < 0 or mode_i >= m:
            continue
        seg = pr[:, mode_i, :]
        prob_str = ""
        if mode_probs is not None:
            if mode_probs.dim() != 1 or mode_probs.shape[0] != m:
                raise ValueError(f"mode_probs must be [M] with M={m}, got {tuple(mode_probs.shape)}")
            prob_str = f" (p={float(mode_probs[mode_i]):.2f})"
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            color=cmap(mode_i % 10),
            linewidth=1.5,
            linestyle="--",
            alpha=0.9,
            label=f"Pred mode {mode_i}{prob_str}",
        )

    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    return fig


def plot_graph_connectivity(
    graph: SceneGraphBatch,
    b: int,
    t_frame: int,
    *,
    threshold: float = 1e-3,
    title: str | None = None,
) -> plt.Figure:
    """
    Draw nodes at stacked positions and edges where weighted adjacency exceeds ``threshold``.

    Focal agent is index ``0``; neighbors follow Phase 2 layout.
    """
    apply_plot_style()
    pos = graph.positions[b, t_frame].detach().float().cpu().numpy()
    mask = graph.node_mask[b, t_frame].detach().bool().cpu().numpy()
    adj = graph.adjacency_weighted[b, t_frame].detach().float().cpu().numpy()
    valid = np.where(mask)[0]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    # Edges
    for i in valid:
        for j in valid:
            if i == j:
                continue
            w = float(adj[i, j])
            if w >= threshold:
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    color=STYLE["edge"],
                    linewidth=max(0.4, min(3.0, w * 0.5)),
                    alpha=0.55,
                    zorder=1,
                )

    # Nodes
    for idx in valid:
        c = STYLE["focal"] if idx == 0 else STYLE["neighbor"]
        s = 120 if idx == 0 else 55
        ax.scatter(pos[idx, 0], pos[idx, 1], c=c, s=s, zorder=3, edgecolors="white", linewidths=0.8)
        ax.annotate(str(idx), (pos[idx, 0], pos[idx, 1]), fontsize=8, ha="center", va="bottom", color=STYLE["text"])

    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    return fig


def build_graph_for_batch(batch: Mapping[str, Any], cfg: Mapping[str, Any]) -> SceneGraphBatch:
    """Scene graph from a collated batch (same device as ``batch['obs']``)."""
    return build_from_collated_batch(batch, cfg)


def plot_training_curves(
    csv_path: Path | str,
    *,
    metrics: Sequence[str] = ("train_loss_epoch", "val_ade", "val_fde"),
) -> plt.Figure:
    """
    Parse trainer ``metrics.csv`` (wide rows with ``split``, ``epoch``, and metric columns).

    Plots one subplot per requested metric vs epoch (only rows where the metric is present).
    """
    apply_plot_style()
    csv_path = Path(csv_path)
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    by_split_metric: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        split = row.get("split", "")
        epoch_raw = row.get("epoch")
        if epoch_raw is None or epoch_raw == "":
            continue
        try:
            epoch = int(epoch_raw)
        except ValueError:
            continue
        for m in metrics:
            key = (split, m)
            cell = row.get(m, "")
            if cell is None or cell == "":
                continue
            try:
                val = float(cell)
            except ValueError:
                continue
            by_split_metric[key].append((epoch, val))

    nplots = len(metrics)
    fig, axes = plt.subplots(nplots, 1, figsize=(7, 3.2 * nplots), sharex=False)
    if nplots == 1:
        axes = [axes]

    colors = list(mcolors.TABLEAU_COLORS.values())
    for ax, metric in zip(axes, metrics):
        handles = []
        for i, split in enumerate(sorted({s for (s, m) in by_split_metric if m == metric})):
            pts = sorted(by_split_metric.get((split, metric), []), key=lambda p: p[0])
            if not pts:
                continue
            ee, vv = zip(*pts)
            line, = ax.plot(ee, vv, "o-", color=colors[i % len(colors)], linewidth=1.2, markersize=3, label=split)
            handles.append(line)
        ax.set_ylabel(metric)
        ax.set_xlabel("epoch")
        ax.legend(handles=[h for h in handles], labels=[h.get_label() for h in handles], fontsize=8)
    fig.suptitle(f"Training curves — {csv_path.name}", fontsize=11)
    fig.tight_layout()
    return fig
