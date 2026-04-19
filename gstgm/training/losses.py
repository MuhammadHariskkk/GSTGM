"""
Training losses for GSTGM (Khel et al. 2024, §4.6 — WTA, regression, classification, KL).

**paper-specified**
    * Winner-takes-all: pick mode :math:`m^\\star` minimizing trajectory error (Eq. 16).
    * Classification: cross-entropy for mixture weights (Eq. 18).
    * KL annealing term as in Eq. (20) (scalar :math:`\\lambda_{KL}` from config).

**engineering assumption**
    * Eq. (17) uses Laplace NLL on positions in the paper; our decoder predicts **Gaussian mixture**
      over **velocities** (§4.4). Regression options: ``gaussian_nll`` on **step-wise planar velocity**
      (default, aligned with the head), or ``laplace`` on velocity with scale from predicted
      :math:`\\sigma` (heuristic link to §4.6 wording).
    * WTA mode index is chosen with ``no_grad`` from position ADE (cumsum of predicted velocity means);
      gradients flow only through the winner's parameters.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


def _training_loss_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    t = cfg.get("training") or {}
    return {
        "kl_weight": float(t.get("kl_weight", 1.0)),
        "kl_anneal_epochs": int(t.get("kl_anneal_epochs", 0)),
        "cls_weight": float(t.get("cls_weight", 1.0)),
        "regression_loss": str(t.get("regression_loss", "gaussian_nll")).lower(),
    }


def kl_anneal_factor(epoch: int, anneal_epochs: int) -> float:
    """Linear ramp from 0→1 over ``anneal_epochs``; if ``anneal_epochs <= 0``, return 1."""
    if anneal_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch + 1) / float(anneal_epochs))


def future_velocity_targets(obs: Tensor, future: Tensor) -> Tensor:
    """
    Planar velocity targets matching consecutive **positions** (same frame as batch tensors).

    * Step 0: ``future[:,0] - obs[:,-1]``
    * Step ``t>0``: ``future[:,t] - future[:,t-1]``

    Parameters
    ----------
    obs :
        ``[B, T_obs, 2]``
    future :
        ``[B, T', 2]``

    Returns
    -------
    Tensor
        ``[B, T', 2]``
    """
    if obs.dim() != 3 or future.dim() != 3:
        raise ValueError(f"obs [B,T,2] and future [B,T',2] required; got {obs.shape}, {future.shape}")
    last = obs[:, -1, :]
    prev_pos = torch.cat([last.unsqueeze(1), future[:, :-1]], dim=1)
    return future - prev_pos


def positions_from_velocity_means(last_obs: Tensor, vel_mu: Tensor) -> Tensor:
    """Integrate mean velocities: ``p_t = p_{last} + \\sum_{\\tau \\le t} v_\\tau`` (``vel_mu`` ``[...,T',2]``)."""
    return last_obs.unsqueeze(1) + torch.cumsum(vel_mu, dim=1)


def ade_per_mode(pred_mu: Tensor, last_obs: Tensor, future: Tensor) -> Tensor:
    """
    Mean displacement error per mode in **position** space using predicted **velocity** means.

    Returns
    -------
    Tensor
        ``[B, M]``
    """
    cum = torch.cumsum(pred_mu, dim=1)
    pos = last_obs[:, None, None, :] + cum
    err = (pos - future[:, :, None, :]).norm(dim=-1)
    return err.mean(dim=1)


def winner_modes(ade_per_mode_: Tensor) -> Tensor:
    """``[B, M]`` → ``m_star`` ``[B]`` long (no grad)."""
    return ade_per_mode_.detach().argmin(dim=1)


def gather_winner(
    pred_mu: Tensor,
    pred_sigma: Tensor,
    m_star: Tensor,
) -> tuple[Tensor, Tensor]:
    """Index winner mode: ``pred_*`` ``[B,T',M,2]`` → ``[B,T',2]``."""
    b = torch.arange(pred_mu.size(0), device=pred_mu.device, dtype=torch.long)
    return pred_mu[b, :, m_star, :], pred_sigma[b, :, m_star, :]


def regression_velocity_loss(
    gt_vel: Tensor,
    mu_w: Tensor,
    sigma_w: Tensor,
    *,
    kind: str = "gaussian_nll",
    eps: float = 1e-8,
) -> Tensor:
    """Scalar loss over batch and time."""
    kind_l = str(kind).lower()
    if kind_l == "gaussian_nll":
        s = sigma_w.clamp_min(eps)
        var = s * s
        nll = 0.5 * (math.log(2 * math.pi) + torch.log(var) + (gt_vel - mu_w) ** 2 / var)
        return nll.sum(dim=-1).mean()
    if kind_l == "laplace":
        b = sigma_w.clamp_min(eps)
        nll = math.log(2.0) + torch.log(b) + (gt_vel - mu_w).abs() / b
        return nll.sum(dim=-1).mean()
    raise ValueError(f"Unknown regression_loss: {kind!r} (use 'gaussian_nll' or 'laplace')")


def classification_loss(pi_logits: Tensor, m_star: Tensor) -> Tensor:
    """
    Cross-entropy on time-averaged mixture logits vs WTA mode (Eq. 18 style).

    ``pi_logits`` — ``[B, T', M]``; averages over ``T'`` then ``F.cross_entropy``.
    """
    logits = pi_logits.mean(dim=1)
    return F.cross_entropy(logits, m_star)


def gstgm_batch_loss(
    out: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    epoch: int,
) -> tuple[Tensor, dict[str, float]]:
    """
    Sum regression + classification + annealed KL for one forward dict from :class:`gstgm.models.gstgm.GSTGM`.

    Returns
    -------
    total_loss, component_scalars (detached floats for logging).
    """
    lc = _training_loss_cfg(cfg)
    obs = batch["obs"]
    future = batch["future"]
    last = obs[:, -1, :]
    pred_mu = out["pred_mu"]
    pred_sigma = out["pred_sigma"]
    pi_logits = out["pi_logits"]

    ade_m = ade_per_mode(pred_mu, last, future)
    m_star = winner_modes(ade_m)
    mu_w, sig_w = gather_winner(pred_mu, pred_sigma, m_star)
    gt_vel = future_velocity_targets(obs, future)

    reg = regression_velocity_loss(gt_vel, mu_w, sig_w, kind=lc["regression_loss"])
    cls = classification_loss(pi_logits, m_star)
    kl = out["kl"].mean()
    ann = kl_anneal_factor(epoch, lc["kl_anneal_epochs"])
    kl_term = lc["kl_weight"] * ann * kl
    total = reg + lc["cls_weight"] * cls + kl_term

    with torch.no_grad():
        parts = {
            "loss": float(total.item()),
            "loss_reg": float(reg.item()),
            "loss_cls": float(cls.item()),
            "loss_kl": float(kl.item()),
            "loss_kl_weighted": float(kl_term.item()),
            "kl_anneal": float(ann),
        }
    return total, parts
