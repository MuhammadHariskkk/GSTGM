"""Latent KL and reparameterization."""

from __future__ import annotations

import torch

from gstgm.models.latent_sampler import kl_diagonal_normals, reparameterize_gaussian


def test_kl_identical_gaussians_zero() -> None:
    mu = torch.zeros(4, 8)
    sigma = torch.ones(4, 8)
    kl = kl_diagonal_normals(mu, sigma, mu, sigma)
    assert kl.shape == (4,)
    assert torch.allclose(kl, torch.zeros(4), atol=1e-5)


def test_reparameterize_deterministic_eps() -> None:
    mu = torch.tensor([[1.0, 2.0]])
    sigma = torch.tensor([[0.5, 1.0]])
    eps = torch.tensor([[1.0, -1.0]])
    z = reparameterize_gaussian(mu, sigma, eps=eps)
    assert torch.allclose(z, mu + sigma * eps)
