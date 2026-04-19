"""
Graph similarity for GSTGM weighted adjacency (Khel et al., 2024).

Equation (2) in *Image and Vision Computing* 151 (2024) 105245 defines the similarity kernel
between nodes *i* and *j* at time *t* from their **velocity** vectors ``v_i^t``, ``v_j^t``:

.. math::

    a^{\\mathrm{sim}}_{ij,t} = \\begin{cases}
        1 / \\|v_i^t - v_j^t\\|^2 & \\text{if } \\|v_i^t - v_j^t\\|^2 \\neq 0 \\\\
        0 & \\text{otherwise}
    \\end{cases}

Inputs to ``pairwise_squared_euclidean`` for adjacency must therefore be **per-node planar
velocity** ``[..., N, 2]``, not position.
"""

from __future__ import annotations

import torch
from torch import Tensor


def pairwise_squared_euclidean(x: Tensor) -> Tensor:
    """
    All-pairs squared Euclidean distances.

    Parameters
    ----------
    x :
        Tensor of shape ``[..., N, D]`` (GSTGM uses ``D=2`` for ``(v_x, v_y)``).

    Returns
    -------
    Tensor
        Shape ``[..., N, N]`` with entry ``[..., i, j] = ||x_i - x_j||^2``.
    """
    diff = x.unsqueeze(-2) - x.unsqueeze(-3)
    return diff.pow(2).sum(dim=-1)


def gstgm_adjacency_similarity(sq_dist: Tensor) -> Tensor:
    """
    GSTGM Eq. (2): reciprocal squared **velocity** difference magnitude; zero when that magnitude is zero.

    Parameters
    ----------
    sq_dist :
        ``[..., N, N]`` with ``sq_dist[..., i, j] = ||v_i - v_j||^2`` (velocity space).
    """
    sq = sq_dist.to(torch.float32)
    w = torch.zeros_like(sq)
    pos = sq > 0
    w[pos] = 1.0 / sq[pos]
    return w


def _broadcast_identity_mask(leading_shape: tuple[int, ...], n: int, device: torch.device) -> Tensor:
    """``[..., N, N]`` identity (float) for multiplying off-diagonal kernels that need zero diagonal."""
    eye = torch.eye(n, device=device, dtype=torch.float32)
    for _ in range(len(leading_shape)):
        eye = eye.unsqueeze(0)
    return eye.expand(*leading_shape, n, n)


def inverse_sq_euclidean_weights(sq_dist: Tensor, eps: float) -> Tensor:
    """
    Legacy helper: ``1/(d^2+eps)`` with zero diagonal.

    **Note:** GSTGM adjacency follows Eq. (2) with **no** ``eps`` in the kernel; use
    ``gstgm_adjacency_similarity`` for paper-faithful weights. This function remains for
    optional ablations only.
    """
    w = 1.0 / (sq_dist.to(torch.float32) + float(eps))
    n = sq_dist.shape[-1]
    lead = sq_dist.shape[:-2]
    one_minus_eye = 1.0 - _broadcast_identity_mask(lead, n, sq_dist.device)
    return w * one_minus_eye


def apply_similarity_kernel(name: str, sq_dist: Tensor, eps: float | None = None) -> Tensor:
    """
    Dispatch by config ``graph.similarity.kernel``.

    * ``inverse_sq_euclidean``, ``gstgm_eq2``, ``paper_eq2`` — Eq. (2) on squared velocity distances
      (``eps`` ignored).
    * ``inverse_sq_euclidean_eps`` — off-diagonal ``1/(d^2+eps)`` if an epsilon ablation is needed.
    """
    key = str(name).lower().replace("-", "_")
    if key in (
        "inverse_sq_euclidean",
        "inverse_square_euclidean",
        "gstgm_eq2",
        "paper_eq2",
    ):
        return gstgm_adjacency_similarity(sq_dist)
    if key in ("inverse_sq_euclidean_eps", "inv_sq_eps"):
        if eps is None:
            raise ValueError(f"Kernel {name!r} requires similarity.eps in config")
        return inverse_sq_euclidean_weights(sq_dist, eps)
    raise ValueError(f"Unknown graph similarity kernel: {name!r}")
