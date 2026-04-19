"""
Evaluation metrics ADE / FDE for multimodal velocity predictions (Khel et al. 2024, §5.2).

**engineering assumption**
    * Predicted **positions** integrate mean modal velocities from ``obs[:, -1]`` (same convention as
      :mod:`gstgm.training.losses`). Oracle **min over modes** reports ``min_m ADE`` / ``min_m FDE`` per
      trajectory, then mean over batch (common multimodal protocol; align with paper's ``ADEM`` minimum).
"""

from __future__ import annotations

import torch
from torch import Tensor

from gstgm.training.losses import ade_per_mode, positions_from_velocity_means


@torch.no_grad()
def per_trajectory_oracle_ade_fde(
    pred_mu: Tensor,
    last_obs: Tensor,
    future: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    §5.2-style **min over mixture modes** per trajectory (oracle ``m``).

    Returns
    -------
    ade_per_traj, fde_per_traj :
        ``[B]`` each — mean over time of displacement error, and final-step error.
    """
    ade_m = ade_per_mode(pred_mu, last_obs, future)
    b_idx = torch.arange(pred_mu.size(0), device=pred_mu.device, dtype=torch.long)
    m_best = ade_m.argmin(dim=1)
    mu_best = pred_mu[b_idx, :, m_best, :]
    pos = positions_from_velocity_means(last_obs, mu_best)
    disp = (pos - future).norm(dim=-1)
    ade_t = disp.mean(dim=1)
    fde_t = disp[:, -1]
    return ade_t, fde_t


@torch.no_grad()
def batch_min_ade_fde(
    pred_mu: Tensor,
    last_obs: Tensor,
    future: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Parameters
    ----------
    pred_mu :
        ``[B, T', M, 2]`` velocity means.
    last_obs :
        ``[B, 2]`` last observed position.
    future :
        ``[B, T', 2]`` ground-truth future positions.

    Returns
    -------
    ade, fde :
        Scalars (mean over batch): oracle min-mode ADE and FDE in the same units as coordinates.
    """
    ade_t, fde_t = per_trajectory_oracle_ade_fde(pred_mu, last_obs, future)
    return ade_t.mean(), fde_t.mean()


@torch.no_grad()
def dict_from_val_batch(out: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, float]:
    """Keys ``val_ade``, ``val_fde`` for :class:`gstgm.utils.checkpoint.CheckpointManager`."""
    obs = batch["obs"]
    future = batch["future"]
    last = obs[:, -1, :]
    ade, fde = batch_min_ade_fde(out["pred_mu"], last, future)
    return {"val_ade": float(ade.item()), "val_fde": float(fde.item())}
