"""
Gaussian mixture head over future planar velocities (GSTGM, Khel et al. 2024, §4.4, Eq. 12).

**paper-specified**
    * :math:`M` modes (default ``gmm.num_modes = 3``).
    * Per mode :math:`m`, per time :math:`t`, bivariate Gaussian on velocity components
      :math:`(v_x, v_y)` with means :math:`\\mu^{(m)}_{x,t},\\mu^{(m)}_{y,t}` and scales
      :math:`\\sigma^{(m)}_{x,t},\\sigma^{(m)}_{y,t}` as in Eq. (12) (diagonal covariance).
    * Mode logits :math:`\\pi_m` per paper ("governed by the likelihood :math:`\\pi_m`"); we output
      unconstrained logits over modes at each time step, normalized with :math:`\\mathrm{softmax}`.

**engineering assumption**
    * Each mode uses the same architecture: ``Linear → ELU → Linear`` to 4 scalars (means + raw scales);
      scales are mapped with :func:`torch.nn.functional.softplus` plus epsilon (positivity), instead
      of ``exp``, for numerical stability.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def gmm_head_kwargs_from_cfg(cfg: Mapping[str, Any], lstm_hidden_dim: int) -> dict[str, Any]:
    m = cfg.get("model") or {}
    d = m.get("decoder") or {}
    gm = cfg.get("gmm") or {}
    return {
        "in_dim": int(lstm_hidden_dim),
        "num_modes": int(gm.get("num_modes", 3)),
        "hidden_dim": int(d.get("gmm_head_hidden_dim", 128)),
        "sigma_floor": float(d.get("gmm_sigma_floor", 1.0e-4)),
    }


class MixtureVelocityHead(nn.Module):
    """
    Parameters
    ----------
    in_dim :
        LSTM output size :math:`H` (``model.decoder.lstm_hidden_dim``).
    num_modes :
        Mixture size :math:`M` (``gmm.num_modes``).
    hidden_dim :
        ELU MLP hidden width per mode (``model.decoder.gmm_head_hidden_dim``).
    sigma_floor :
        Added after softplus on :math:`\\sigma` components.
    """

    def __init__(
        self,
        in_dim: int,
        num_modes: int,
        hidden_dim: int,
        *,
        sigma_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.sigma_floor = float(sigma_floor)
        hid = int(hidden_dim)
        self.mode_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.ELU(inplace=True),
                nn.Linear(hid, 4),
            )
            for _ in range(self.num_modes)
        )
        self.pi_head = nn.Linear(in_dim, self.num_modes)

    def forward(self, h_seq: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        h_seq :
            ``[B, T', H]`` decoder sequence.

        Returns
        -------
        mu :
            ``[B, T', M, 2]`` means :math:`(\\mu_x, \\mu_y)`.
        sigma :
            ``[B, T', M, 2]`` standard deviations (positive).
        pi_logits :
            ``[B, T', M]`` unnormed mixture logits (:math:`\\pi` after softmax).
        """
        if h_seq.dim() != 3:
            raise ValueError(f"h_seq must be [B, T, H], got {tuple(h_seq.shape)}")
        _, _, hin = h_seq.shape
        if hin != self.pi_head.in_features:
            raise ValueError(f"H={hin} does not match pi_head in_features={self.pi_head.in_features}")

        mus: list[Tensor] = []
        sigs: list[Tensor] = []
        for head in self.mode_heads:
            o = head(h_seq)
            mus.append(o[..., :2])
            sigs.append(F.softplus(o[..., 2:]) + self.sigma_floor)
        mu = torch.stack(mus, dim=2)
        sigma = torch.stack(sigs, dim=2)
        pi_logits = self.pi_head(h_seq)
        return mu, sigma, pi_logits


def mixture_velocity_head_from_cfg(
    cfg: Mapping[str, Any],
    lstm_hidden_dim: int,
) -> MixtureVelocityHead:
    kw = gmm_head_kwargs_from_cfg(dict(cfg), lstm_hidden_dim)
    return MixtureVelocityHead(
        in_dim=kw["in_dim"],
        num_modes=kw["num_modes"],
        hidden_dim=kw["hidden_dim"],
        sigma_floor=float(kw["sigma_floor"]),
    )
