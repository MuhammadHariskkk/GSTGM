"""
Batched graph convolution / message passing (linear transform after adjacency aggregation).

Decoupled from the training loop and attention (Phase 4+). Uses precomputed normalized
adjacency ``[B, T, N, N]`` and node features ``[B, T, N, Fin]``.

**engineering assumption:** one-hop aggregation
``H' = \\tilde{A} H W + b`` with ``\\tilde{A}`` row-normalized or symmetric-normalized as built in
``adjacency.py`` (no separate learnable attention on edges here).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def batched_adjacency_aggregate(adj: Tensor, x: Tensor) -> Tensor:
    """
    Multiply adjacency with node features over the last two dimensions.

    Parameters
    ----------
    adj :
        ``[..., N, N]`` (e.g. ``[B, T, N, N]``).
    x :
        ``[..., N, F]`` matching leading batch dims of ``adj``.

    Returns
    -------
    Tensor
        ``[..., N, F]`` where ``out[..., i, :] = sum_j adj[..., i, j] * x[..., j, :]``.
    """
    if adj.shape[:-2] != x.shape[:-2]:
        raise ValueError(f"Leading shape mismatch: adj {adj.shape}, x {x.shape}")
    *lead, n_a = adj.shape[:-1]
    n_x, f = x.shape[-2:]
    if n_a != n_x:
        raise ValueError(f"N mismatch: adj {n_a}, x {n_x}")
    flat_adj = adj.reshape(-1, n_a, n_a)
    flat_x = x.reshape(-1, n_x, f)
    out = torch.bmm(flat_adj, flat_x)
    return out.view(*lead, n_a, f)


class GraphConv(nn.Module):
    """
    Single graph convolution layer: linear( aggregated features ).

    Forward
    -------
    * ``x`` — ``[B, T, N, Fin]``
    * ``adj_norm`` — ``[B, T, N, N]`` (typically symmetric-normalized)

    Output — ``[B, T, N, Fout]``
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: Tensor, adj_norm: Tensor) -> Tensor:
        agg = batched_adjacency_aggregate(adj_norm, x)
        return self.linear(agg)
