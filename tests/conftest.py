"""Shared fixtures for GSTGM tests (Phase 10)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from gstgm.datasets.collate import collate_eth_ucy
from gstgm.utils.config import load_config


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def merged_cfg_eth(repo_root: Path) -> dict[str, Any]:
    return load_config(repo_root / "configs" / "eth.yaml")


@pytest.fixture
def tiny_batch(merged_cfg_eth: dict[str, Any]) -> dict[str, Any]:
    """Synthetic batch matching ``collate_eth_ucy`` / ``EthUcyDataset`` layout (no raw trajectory files)."""
    d = merged_cfg_eth["data"]
    bsz = 2
    t_obs = int(d["obs_len"])
    t_pred = int(d["pred_len"])
    k_max = int(d["max_neighbors"])
    samples: list[dict[str, Any]] = []
    for i in range(bsz):
        obs = torch.randn(t_obs, 2, dtype=torch.float32) * 0.1
        future = torch.randn(t_pred, 2, dtype=torch.float32) * 0.1
        nbr = torch.randn(t_obs, k_max, 2, dtype=torch.float32) * 0.1
        mask = torch.zeros(t_obs, k_max, dtype=torch.bool)
        mask[:, :3] = True
        samples.append(
            {
                "obs": obs,
                "future": future,
                "neighbor_pos": nbr,
                "neighbor_ped_ids": torch.zeros(t_obs, k_max, dtype=torch.long),
                "neighbor_mask": mask,
                "obs_frame": torch.arange(t_obs, dtype=torch.long),
                "future_frame": torch.arange(t_pred, dtype=torch.long),
                "focal_ped_id": torch.tensor(i, dtype=torch.long),
                "window_index": torch.tensor(i, dtype=torch.long),
                "scene": "eth",
            }
        )
    return collate_eth_ucy(samples)
