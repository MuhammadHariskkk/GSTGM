"""
Variational encoder :math:`\\psi_{\\mathrm{enc}}` (GSTGM, Khel et al. 2024, §4.3, Eq. (8)–(9)).

Produces diagonal Gaussian **approximate posterior** parameters :math:`\\mu_z, \\sigma_z` for
latent :math:`z_0` given the same condition used by the extraction stack.

Also provides ``scene_encoding_to_condition`` — maps ``[B,T,N,d]`` focal features to ``[B,d]``
(**engineering**: focal slot index ``0``, last or mean time pooling; see docstring).

**paper-specified**
    * Encoder is an FCL with ReLU (we use one hidden FCL + ReLU + output FCL—see ``PriorNetwork``
      docstring for rationale).
    * Eq. (8)–(9) splitting: first half :math:`\\mu_z`, :math:`\\exp` on second half for
      :math:`\\sigma_z`.

**engineering assumption**
    * ``sigma_min`` floor on standard deviations.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from gstgm.models.prior_network import _generative_yaml


def generative_encoder_kwargs_from_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    y = _generative_yaml(cfg)
    return {
        "latent_dim": y["latent_dim"],
        "hidden_dim": y["posterior_hidden_dim"],
        "sigma_min": y["sigma_min"],
    }


def scene_encoding_to_condition(
    h: Tensor,
    node_mask: Tensor | None = None,
    *,
    pool: str = "focal_last",
    focal_index: int = 0,
) -> Tensor:
    """
    Parameters
    ----------
    h :
        ``[B, T, N, d]`` (e.g. after spatial–temporal attention).
    node_mask :
        Optional ``[B, T, N]``.
    pool :
        ``focal_last`` | ``focal_mean_time`` (masked when ``node_mask`` provided).
    focal_index :
        Default ``0`` (focal pedestrian in Phase 2 layout).

    Returns
    -------
    Tensor
        ``[B, d]`` conditioning vector.
    """
    if h.dim() != 4:
        raise ValueError(f"h must be [B, T, N, d], got {tuple(h.shape)}")
    b, t, n, _ = h.shape
    if focal_index < 0 or focal_index >= n:
        raise ValueError(f"focal_index {focal_index} out of range for N={n}")
    mode = str(pool).lower()
    if mode == "focal_last":
        x = h[:, -1, focal_index, :].clone()
        if node_mask is not None:
            if node_mask.shape != (b, t, n):
                raise ValueError(f"node_mask must be [B,T,N], got {tuple(node_mask.shape)}")
            m = node_mask[:, -1, focal_index]
            if m.dtype == torch.bool:
                mk = m.float()
            else:
                mk = (m > 0).float()
            x = x * mk.unsqueeze(-1)
        return x
    if mode == "focal_mean_time":
        if node_mask is None:
            return h[:, :, focal_index, :].mean(dim=1)
        if node_mask.shape != (b, t, n):
            raise ValueError(f"node_mask must be [B,T,N], got {tuple(node_mask.shape)}")
        m = node_mask[:, :, focal_index]
        mf = m.float() if m.dtype == torch.bool else (m > 0).float()
        m = mf.to(dtype=h.dtype)
        sel = h[:, :, focal_index, :]
        num = (sel * m.unsqueeze(-1)).sum(dim=1)
        den = m.sum(dim=1).clamp_min(1.0).unsqueeze(-1).to(dtype=h.dtype)
        return num / den
    raise ValueError(f"Unknown pool: {pool!r}")


class GenerativeEncoder(nn.Module):
    """
    Parameters
    ----------
    input_dim :
        Condition dimension (``model.gcn.hidden_dim``).
    hidden_dim :
        ``generative.posterior_hidden_dim``.
    latent_dim :
        ``generative.latent_dim``.
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
        mu_q, sigma_q :
            Approximate posterior :math:`q(z_0 \\mid \\cdot)` parameters, each ``[B, latent_dim]``.
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


def generative_encoder_from_cfg(cfg: Mapping[str, Any], input_dim: int) -> GenerativeEncoder:
    kw = generative_encoder_kwargs_from_cfg(dict(cfg))
    return GenerativeEncoder(
        input_dim=int(input_dim),
        hidden_dim=int(kw["hidden_dim"]),
        latent_dim=int(kw["latent_dim"]),
        sigma_min=float(kw["sigma_min"]),
    )
