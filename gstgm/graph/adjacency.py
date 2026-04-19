"""
Weighted adjacency and symmetric normalization for GSTGM scene graphs.

**paper-specified (Khel et al., 2024, Eq. (2)):** off-diagonal entries of the unnormalized
weighted matrix use ``1 / ||v_i^t - v_j^t||^2`` (velocity difference), and zero when that
squared norm is zero.

**engineering assumption (not in paper):** symmetric normalized propagation matrix
``D^{-1/2} \\tilde{W} D^{-1/2}`` with ``D_i = \\sum_j \\tilde{W}_{ij}`` and ``degree_eps``
floor before ``rsqrt`` for numerical stability (Kipf & Welling–style propagation common
to GCN implementations). Invalid padded nodes are masked to zero after normalization.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from gstgm.graph.kernels import apply_similarity_kernel, pairwise_squared_euclidean


def masked_outer_node_mask(node_mask: Tensor) -> Tensor:
    """``mask_ij = node_mask_i & node_mask_j``; shapes ``[..., N]`` -> ``[..., N, N]``."""
    return node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)


def build_weight_matrix(
    node_velocities: Tensor,
    node_mask: Tensor,
    *,
    kernel: str,
    self_loop: bool,
    self_loop_weight: float = 1.0,
    kernel_eps: float | None = None,
) -> Tensor:
    """
    Non-negative edge weights ``[..., N, N]`` from **velocity** differences (Eq. (2)),
    node masking, and optional self-loops.

    Parameters
    ----------
    node_velocities :
        ``[..., N, 2]`` planar velocity per node (GSTGM ``v_i^t`` stack).
    node_mask :
        ``[..., N]`` bool; invalid nodes have all incident edges zeroed.
    kernel :
        Passed to ``apply_similarity_kernel`` (default YAML uses paper Eq. (2)).
    self_loop :
        Eq. (2) yields zero on the diagonal when ``\\|v_i-v_i\\|^2=0``; set ``True`` to add
        ``self_loop_weight`` on ``(i,i)`` for valid nodes (**engineering extension** if not
        stated in the paper).
    self_loop_weight :
        Diagonal strength when ``self_loop`` is enabled.
    kernel_eps :
        Only used if ``kernel`` selects an epsilon ablation (e.g. ``inverse_sq_euclidean_eps``).
    """
    if node_mask.dtype != torch.bool:
        node_mask = node_mask.bool()
    sq = pairwise_squared_euclidean(node_velocities.to(torch.float32))
    w = apply_similarity_kernel(kernel, sq, kernel_eps)
    pair_valid = masked_outer_node_mask(node_mask).to(w.dtype)
    w = w * pair_valid
    if self_loop:
        diag_vec = node_mask.to(w.dtype) * float(self_loop_weight)
        w = w + torch.diag_embed(diag_vec, dim1=-2, dim2=-1)
    return w


def symmetric_normalized_adjacency(
    w: Tensor,
    *,
    degree_eps: float,
    node_mask: Tensor | None = None,
) -> Tensor:
    """``D^{-1/2} W D^{-1/2}`` with ``D_i = \\sum_j W_{ij}``."""
    w = w.to(torch.float32)
    deg = w.sum(dim=-1)
    inv_sqrt = deg.clamp_min(float(degree_eps)).rsqrt()
    a_norm = w * inv_sqrt.unsqueeze(-2) * inv_sqrt.unsqueeze(-1)
    if node_mask is not None:
        if node_mask.dtype != torch.bool:
            node_mask = node_mask.bool()
        mask2 = masked_outer_node_mask(node_mask).to(a_norm.dtype)
        a_norm = a_norm * mask2
    return a_norm


def build_normalized_adjacency(
    node_velocities: Tensor,
    node_mask: Tensor,
    *,
    kernel: str,
    self_loop: bool,
    self_loop_weight: float = 1.0,
    normalize: bool,
    degree_eps: float,
    kernel_eps: float | None = None,
) -> Tuple[Tensor, Tensor]:
    """
    Weighted matrix (pre-/post- self-loop per config) and matrix for GCN aggregation.

    Returns
    -------
    weighted, conv_adj
        ``conv_adj`` is symmetric-normalized if ``normalize`` is True, else ``weighted``.
    """
    weighted = build_weight_matrix(
        node_velocities,
        node_mask,
        kernel=kernel,
        self_loop=self_loop,
        self_loop_weight=self_loop_weight,
        kernel_eps=kernel_eps,
    )
    if normalize:
        conv_adj = symmetric_normalized_adjacency(
            weighted, degree_eps=float(degree_eps), node_mask=node_mask
        )
    else:
        conv_adj = weighted
    return weighted, conv_adj
