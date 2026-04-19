"""Tensor helpers for GMM / mixture heads (visualization & analysis)."""

from __future__ import annotations

import torch
from torch import Tensor


def positions_from_velocities(last_pos: Tensor, vel: Tensor) -> Tensor:
    """
    Integrate planar velocities into positions (same convention as training metrics).

    Parameters
    ----------
    last_pos :
        ``[B, 2]`` last observed position.
    vel :
        ``[B, T', 2]`` or ``[B, T', M, 2]`` per-step velocities.

    Returns
    -------
    Tensor
        ``[B, T', 2]`` or ``[B, T', M, 2]`` cumulative positions.
    """
    if vel.dim() == 3:
        return last_pos.unsqueeze(1) + torch.cumsum(vel, dim=1)
    if vel.dim() == 4:
        return last_pos[:, None, None, :] + torch.cumsum(vel, dim=1)
    raise ValueError(f"vel must be [B,T,2] or [B,T,M,2], got {tuple(vel.shape)}")


def mixture_probs_from_logits(pi_logits: Tensor) -> Tensor:
    """``\\mathrm{softmax}(\\pi)`` over the mode dimension; same dtype/device as logits."""
    return torch.softmax(pi_logits, dim=-1)


def time_averaged_mode_probs(pi_logits: Tensor) -> Tensor:
    """
    Parameters
    ----------
    pi_logits :
        ``[B, T', M]`` mixture logits per predicted step.

    Returns
    -------
    Tensor
        ``[B, M]`` mean mixture probability per mode over time.
    """
    return mixture_probs_from_logits(pi_logits).mean(dim=1)


def topk_mode_indices(
    scores: Tensor,
    k: int,
    *,
    dim: int = -1,
    largest: bool = True,
) -> Tensor:
    """
    Indices of top-``k`` modes.

    Parameters
    ----------
    scores :
        Typically ``time_averaged_mode_probs`` ``[B, M]`` (largest = True).
    k :
        Clamped to ``<= M``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    k_eff = min(k, scores.size(dim))
    return scores.topk(k_eff, dim=dim, largest=largest).indices
