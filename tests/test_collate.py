"""Collation invariants."""

from __future__ import annotations

import pytest
import torch

from gstgm.datasets.collate import collate_eth_ucy


def test_collate_stacks_batch_dim() -> None:
    obs_len, pred_len, k = 8, 12, 4
    def one(idx: int) -> dict:
        return {
            "obs": torch.randn(obs_len, 2),
            "future": torch.randn(pred_len, 2),
            "neighbor_pos": torch.randn(obs_len, k, 2),
            "neighbor_ped_ids": torch.zeros(obs_len, k, dtype=torch.long),
            "neighbor_mask": torch.ones(obs_len, k, dtype=torch.bool),
            "obs_frame": torch.arange(obs_len),
            "future_frame": torch.arange(pred_len),
            "focal_ped_id": torch.tensor(idx),
            "window_index": torch.tensor(idx),
            "scene": "eth",
        }

    b = collate_eth_ucy([one(0), one(1)])
    assert b["obs"].shape == (2, obs_len, 2)
    assert b["neighbor_mask"].shape == (2, obs_len, k)


def test_empty_collate_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_eth_ucy([])
