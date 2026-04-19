"""
Scene-centric graph construction from trajectory batches (GSTGM, Khel et al. 2024).

**paper-specified:** Weighted adjacency entries follow Eq. (2) using **velocity** vectors
``v_i^t``, ``v_j^t`` — not positions. Node velocities are built from the batch:

* ``coordinate_mode`` in ``{absolute, relative_disp}``: temporal first difference of stacked
  focal + neighbour **positions** (same coordinate frame as ``obs`` / ``neighbor_pos``).
* ``coordinate_mode == velocity``: focal rows use ``obs`` as per-step velocity; neighbours use
  temporal first difference of ``neighbor_pos`` (Phase 2 stores neighbour **positions** in
  that frame; see dataset docstring).

**Tensor layout**
-----------------
* ``obs``: ``[B, T_obs, 2]``; ``neighbor_pos``: ``[B, T_obs, K_max, 2]``; ``N = 1 + K_max``.
* Index ``0`` = focal; ``1 … K_max`` = neighbours (padding masked).

References: Khel et al., *Image and Vision Computing* 151 (2024) 105245, §4.1–4.2, Eq. (2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from gstgm.graph.adjacency import build_normalized_adjacency


def graph_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Read ``cfg['graph']`` with defaults aligned to the GSTGM paper where stated."""
    g = cfg.get("graph", {}) or {}
    sim = g.get("similarity", {}) or {}
    return {
        "kernel": str(sim.get("kernel", "inverse_sq_euclidean")),
        "eps": float(sim.get("eps", 1e-6)),
        # Eq. (2) yields zero diagonal; self-loop is an optional propagation extension.
        "self_loop": bool(g.get("self_loop", False)),
        "normalize_adjacency": bool(g.get("normalize_adjacency", True)),
        "self_loop_weight": float(g.get("self_loop_weight", 1.0)),
        "degree_eps": float(g.get("degree_eps", sim.get("eps", 1e-6))),
    }


def stack_scene_nodes(
    obs: Tensor,
    neighbor_pos: Tensor,
    neighbor_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Stack focal + neighbour positions; ``positions[b,t,0]`` is focal, ``[...,1:]`` neighbours."""
    if obs.dim() != 3 or obs.shape[-1] != 2:
        raise ValueError(f"obs must be [B, T, 2], got {tuple(obs.shape)}")
    if neighbor_pos.dim() != 4 or neighbor_pos.shape[-1] != 2:
        raise ValueError(f"neighbor_pos must be [B, T, K, 2], got {tuple(neighbor_pos.shape)}")
    if neighbor_pos.shape[:-2] != obs.shape[:-1]:
        raise ValueError(
            f"obs [B,T,2] and neighbor_pos [B,T,K,2] must share B,T: obs {tuple(obs.shape)}, "
            f"neighbor_pos {tuple(neighbor_pos.shape)}"
        )
    if neighbor_mask.shape != neighbor_pos.shape[:-1]:
        raise ValueError("neighbor_mask must be [B, T, K] matching neighbor_pos")
    focal = obs.unsqueeze(-2)
    pos = torch.cat([focal, neighbor_pos], dim=-2)
    focal_mask = torch.ones(*obs.shape[:-1], 1, dtype=torch.bool, device=obs.device)
    nbr_m = neighbor_mask.bool() if neighbor_mask.dtype != torch.bool else neighbor_mask
    mask = torch.cat([focal_mask, nbr_m], dim=-1)
    return pos, mask


def stacked_node_velocities(
    obs: Tensor,
    neighbor_pos: Tensor,
    neighbor_mask: Tensor,
    coordinate_mode: str,
) -> Tensor:
    """
    Per-node planar velocity ``[B, T, N, 2]`` for Eq. (2).

    First time index uses zero velocity (no earlier sample in window).
    """
    mode = str(coordinate_mode).lower()
    if mode == "velocity":
        # Same layout checks as ``stack_scene_nodes`` (avoid building unused stacked positions).
        if obs.dim() != 3 or obs.shape[-1] != 2:
            raise ValueError(f"obs must be [B, T, 2], got {tuple(obs.shape)}")
        if neighbor_pos.dim() != 4 or neighbor_pos.shape[-1] != 2:
            raise ValueError(f"neighbor_pos must be [B, T, K, 2], got {tuple(neighbor_pos.shape)}")
        if neighbor_pos.shape[:-2] != obs.shape[:-1]:
            raise ValueError(
                f"obs [B,T,2] and neighbor_pos [B,T,K,2] must share B,T: obs {tuple(obs.shape)}, "
                f"neighbor_pos {tuple(neighbor_pos.shape)}"
            )
        if neighbor_mask.shape != neighbor_pos.shape[:-1]:
            raise ValueError("neighbor_mask must be [B, T, K] matching neighbor_pos")
        focal_v = obs
        nbr_v = torch.zeros_like(neighbor_pos)
        nbr_v[:, 1:] = neighbor_pos[:, 1:] - neighbor_pos[:, :-1]
        return torch.cat([focal_v.unsqueeze(-2), nbr_v], dim=-2)
    pos, _ = stack_scene_nodes(obs, neighbor_pos, neighbor_mask)
    vel = torch.zeros_like(pos)
    vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
    return vel


@dataclass
class SceneGraphBatch:
    """One batch of scene graphs over the observation horizon."""

    positions: Tensor
    """``[B, T, N, 2]`` focal + neighbour channels as stacked by Phase 2 (see ``coordinate_mode``)."""

    velocities: Tensor
    """``[B, T, N, 2]`` planar velocity per node — **used for Eq. (2) adjacency**."""

    node_mask: Tensor
    """``[B, T, N]`` valid node mask."""

    adjacency_weighted: Tensor
    """``[B, T, N, N]`` before symmetric normalization (masked; optional self-loop)."""

    adjacency_norm: Tensor
    """``[B, T, N, N]`` matrix for ``GraphConv`` / aggregation."""


def build_scene_graph_batch(
    obs: Tensor,
    neighbor_pos: Tensor,
    neighbor_mask: Tensor,
    *,
    coordinate_mode: str = "relative_disp",
    kernel: str = "inverse_sq_euclidean",
    eps: float = 1e-6,
    self_loop: bool = False,
    normalize_adjacency: bool = True,
    self_loop_weight: float = 1.0,
    degree_eps: float | None = None,
) -> SceneGraphBatch:
    """
    Build adjacency from **velocity** (Eq. (2)); keep stacked **positions** for downstream GCN features.

    ``eps`` in signature is used only for ``degree_eps`` when ``degree_eps`` is ``None``, or as
    ``kernel_eps`` if ``kernel`` is an epsilon ablation name.
    """
    pos, mask = stack_scene_nodes(obs, neighbor_pos, neighbor_mask)
    vel = stacked_node_velocities(obs, neighbor_pos, neighbor_mask, coordinate_mode)
    dew = float(eps) if degree_eps is None else float(degree_eps)
    kernel_eps = float(eps) if str(kernel).lower().replace("-", "_") in (
        "inverse_sq_euclidean_eps",
        "inv_sq_eps",
    ) else None
    weighted, conv = build_normalized_adjacency(
        vel,
        mask,
        kernel=kernel,
        self_loop=self_loop,
        self_loop_weight=self_loop_weight,
        normalize=normalize_adjacency,
        degree_eps=dew,
        kernel_eps=kernel_eps,
    )
    return SceneGraphBatch(
        positions=pos,
        velocities=vel,
        node_mask=mask,
        adjacency_weighted=weighted,
        adjacency_norm=conv,
    )


def build_from_collated_batch(batch: Mapping[str, Any], cfg: Mapping[str, Any]) -> SceneGraphBatch:
    """
    Extract ``collate_eth_ucy`` keys and merged YAML config (``graph`` + ``data.coordinate_mode``).
    """
    gc = graph_config(cfg)
    coord = str(cfg.get("data", {}).get("coordinate_mode", "relative_disp"))
    return build_scene_graph_batch(
        batch["obs"],
        batch["neighbor_pos"],
        batch["neighbor_mask"],
        coordinate_mode=coord,
        kernel=gc["kernel"],
        eps=gc["eps"],
        self_loop=gc["self_loop"],
        normalize_adjacency=gc["normalize_adjacency"],
        self_loop_weight=gc["self_loop_weight"],
        degree_eps=gc["degree_eps"],
    )
