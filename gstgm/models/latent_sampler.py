"""
Latent sampling and KL terms for the GSTGM generative stage (Khel et al. 2024, §4.3, Eq. (20)).

**paper-specified**
    * Reparameterization trick :math:`z = \\mu + \\sigma \\odot \\epsilon`,
      :math:`\\epsilon \\sim \\mathcal{N}(0, I)` (see §4.6 text).

**engineering assumption**
    * Analytic KL for diagonal Gaussians between posterior :math:`q(z_0|x)` and prior :math:`p(z_0|x)`
      (coefficient :math:`\\lambda_{KL}` applied in the training loop, not here).

Tensor shapes: ``μ, σ`` are ``[B, L]`` with ``L = latent_dim``; returned ``z`` matches.
"""

from __future__ import annotations

import torch
from torch import Tensor


def reparameterize_gaussian(mu: Tensor, sigma: Tensor, eps: Tensor | None = None) -> Tensor:
    """
    Parameters
    ----------
    mu, sigma :
        ``[B, L]``; ``sigma > 0`` element-wise.
    eps :
        Optional standard normal ``[B, L]``; sampled if ``None``.
    """
    if mu.shape != sigma.shape or mu.dim() != 2:
        raise ValueError(f"mu, sigma must be [B,L] with same shape, got {mu.shape}, {sigma.shape}")
    if eps is None:
        eps = torch.randn_like(mu)
    elif eps.shape != mu.shape:
        raise ValueError(f"eps must match mu shape {tuple(mu.shape)}, got {tuple(eps.shape)}")
    return mu + sigma * eps


def kl_diagonal_normals(
    mu_q: Tensor,
    sigma_q: Tensor,
    mu_p: Tensor,
    sigma_p: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """
    Per-batch KL ``KL(q || p)`` for independent univariate normals.

    .. math::

        \\sum_i \\big(\\log \\frac{\\sigma_{p,i}}{\\sigma_{q,i}}
        + \\frac{\\sigma_{q,i}^2 + (\\mu_{q,i}-\\mu_{p,i})^2}{2\\sigma_{p,i}^2} - \\tfrac12\\big)

    Returns
    -------
    Tensor
        ``[B]`` KL per batch element.
    """
    if not (mu_q.shape == sigma_q.shape == mu_p.shape == sigma_p.shape):
        raise ValueError("mu/sigma tensors must share shape")
    sq = sigma_q.clamp_min(eps)
    sp = sigma_p.clamp_min(eps)
    term_log = torch.log(sp / sq)
    term_quad = (sq.square() + (mu_q - mu_p).square()) / (2.0 * sp.square())
    return (term_log + term_quad - 0.5).sum(dim=-1)


def variational_latent_forward(
    mu_q: Tensor,
    sigma_q: Tensor,
    mu_p: Tensor,
    sigma_p: Tensor,
    *,
    eps: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Sample :math:`z_0` and return per-example KL matching ``mu_q, sigma_q`` vs ``mu_p, sigma_p``.

    Returns
    -------
    z :
        ``[B, L]``.
    kl :
        ``[B]``.
    """
    z = reparameterize_gaussian(mu_q, sigma_q, eps=eps)
    kl = kl_diagonal_normals(mu_q, sigma_q, mu_p, sigma_p)
    return z, kl
