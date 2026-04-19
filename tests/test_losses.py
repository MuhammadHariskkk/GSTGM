"""Training loss helpers."""

from __future__ import annotations

import torch

from gstgm.training.losses import future_velocity_targets, kl_anneal_factor


def test_future_velocity_targets_shape() -> None:
    obs = torch.randn(3, 8, 2)
    fut = torch.randn(3, 12, 2)
    vel = future_velocity_targets(obs, fut)
    assert vel.shape == fut.shape


def test_kl_anneal_saturates() -> None:
    assert kl_anneal_factor(0, 0) == 1.0
    assert kl_anneal_factor(99, 100) == 1.0
