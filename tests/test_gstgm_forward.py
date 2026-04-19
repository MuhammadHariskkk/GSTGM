"""End-to-end model smoke test (no dataset files)."""

from __future__ import annotations

from typing import Any

import torch

from gstgm.models import gstgm_from_cfg


def test_gstgm_forward_finite(merged_cfg_eth: dict[str, Any], tiny_batch: dict[str, Any]) -> None:
    model = gstgm_from_cfg(merged_cfg_eth)
    model.eval()
    with torch.no_grad():
        out = model(tiny_batch, sample_posterior=False)
    assert "pred_mu" in out
    b = tiny_batch["obs"].size(0)
    t_pred = int(merged_cfg_eth["data"]["pred_len"])
    m = int(merged_cfg_eth["gmm"]["num_modes"])
    assert out["pred_mu"].shape == (b, t_pred, m, 2)
    assert torch.isfinite(out["pred_mu"]).all()
    assert torch.isfinite(out["kl"]).all()


def test_gstgm_forward_training_mode_sample(merged_cfg_eth: dict[str, Any], tiny_batch: dict[str, Any]) -> None:
    model = gstgm_from_cfg(merged_cfg_eth)
    model.train()
    torch.manual_seed(0)
    out = model(tiny_batch, sample_posterior=True)
    assert torch.isfinite(out["z"]).all()
