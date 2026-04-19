"""YAML parity for attention (legacy keys)."""

from __future__ import annotations

from gstgm.models.spatial_temporal_attention import attention_hyperparams_from_merged_cfg


def test_legacy_lambda_gamma_aliases() -> None:
    cfg = {"attention": {"spatial": {"d_model": 128, "lambda": 2.0, "gamma": 0.25, "dropout": 0.0}}}
    h = attention_hyperparams_from_merged_cfg(cfg)
    assert h["lambda_spatial"] == 2.0
    assert h["gamma_adj"] == 0.25
    assert h["d_model"] == 128
