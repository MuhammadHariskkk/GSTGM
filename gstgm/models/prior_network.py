"""
Prior network :math:`\\psi_{\\mathrm{prior}}` (GSTGM, Khel et al. 2024, §4.3, Eq. (8)–(9)).

Maps a fixed-dimensional condition (extracted scene / agent encoding) to diagonal Gaussian
parameters :math:`\\mu_{\\mathrm{prior}}, \\sigma_{\\mathrm{prior}}` over latent :math:`z_0`.

**paper-specified**
    * Eq. (8): first segment of output = :math:`\\mu_{\\mathrm{prior}}`.
    * Eq. (9): second segment passed through :math:`\\exp` for :math:`\\sigma_{\\mathrm{prior}}`
      (element-wise standard deviations).

**engineering assumption**
    * MLP: ``Linear → ReLU → Linear`` with widths from ``generative.prior_hidden_dim`` and
      ``generative.latent_dim``; the paper names a single FCL with ReLU—extra projection is for
      capacity parity with common implementations.
    * ``sigma_min`` floors standard deviations for numerical stability in KL / sampling.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor


def _generative_yaml(cfg: Mapping[str, Any]) -> dict[str, Any]:
    g = cfg.get("generative") or {}
    return {
        "latent_dim": int(g.get("latent_dim", 32)),
        "posterior_hidden_dim": int(g.get("posterior_hidden_dim", 128)),
        "prior_hidden_dim": int(g.get("prior_hidden_dim", 128)),
        "sigma_min": float(g.get("sigma_min", 1.0e-6)),
    }


def prior_network_kwargs_from_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    y = _generative_yaml(cfg)
    return {
        "latent_dim": y["latent_dim"],
        "hidden_dim": y["prior_hidden_dim"],
        "sigma_min": y["sigma_min"],
    }


class PriorNetwork(nn.Module):
    """
    Parameters
    ----------
    input_dim :
        Dimension of condition ``y`` (typically ``model.gcn.hidden_dim`` after attention).
    hidden_dim :
        Hidden width (``generative.prior_hidden_dim``).
    latent_dim :
        Size of :math:`z_0` (``generative.latent_dim``).
    sigma_min :
        Clamp for :math:`\\sigma_{\\mathrm{prior}}` after ``exp``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        *,
        sigma_min: float = 1e-6,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.sigma_min = float(sigma_min)
        in_d = int(input_dim)
        hid = int(hidden_dim)
        lz = self.latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_d, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, 2 * lz),
        )

    def forward(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        condition :
            ``[B, input_dim]``.

        Returns
        -------
        mu_p, sigma_p :
            Each ``[B, latent_dim]``; ``sigma_p > 0`` (Eq. (9) with floor).
        """
        if condition.dim() != 2 or condition.size(-1) != self.net[0].in_features:
            raise ValueError(
                f"condition must be [B, {self.net[0].in_features}], got {tuple(condition.shape)}"
            )
        out = self.net(condition)
        mu = out[:, : self.latent_dim]
        raw = out[:, self.latent_dim :]
        sigma = torch.exp(raw).clamp_min(self.sigma_min)
        return mu, sigma


def prior_network_from_cfg(cfg: Mapping[str, Any], input_dim: int) -> PriorNetwork:
    """Build :class:`PriorNetwork` with merged YAML; ``input_dim`` is the condition size ``d``."""
    kw = prior_network_kwargs_from_cfg(dict(cfg))
    return PriorNetwork(
        input_dim=int(input_dim),
        hidden_dim=int(kw["hidden_dim"]),
        latent_dim=int(kw["latent_dim"]),
        sigma_min=float(kw["sigma_min"]),
    )
