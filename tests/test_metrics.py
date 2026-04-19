"""Oracle ADE / FDE metrics."""

from __future__ import annotations

import torch

from gstgm.training.metrics import batch_min_ade_fde


def test_oracle_ade_fde_zero_stationary() -> None:
    b, t_pred, m = 4, 12, 3
    last = torch.randn(b, 2)
    future = last.unsqueeze(1).expand(-1, t_pred, -1).clone()
    pred_mu = torch.zeros(b, t_pred, m, 2)
    ade, fde = batch_min_ade_fde(pred_mu, last, future)
    assert ade.item() < 1e-5
    assert fde.item() < 1e-5
