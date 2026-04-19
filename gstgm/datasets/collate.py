"""Batch collation with variable pedestrian counts already padded per sample."""

from __future__ import annotations

from typing import Any

import torch


def collate_eth_ucy(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Stack a list of ``EthUcyDataset`` samples into a batch.

    All tensor fields are padded to fixed ``obs_len`` and ``max_neighbors`` per sample;
    this function stacks the batch dimension ``B``.

    Output shapes
    -------------
    * ``obs`` — ``[B, obs_len, 2]``
    * ``future`` — ``[B, pred_len, 2]``
    * ``neighbor_pos`` — ``[B, obs_len, max_neighbors, 2]``
    * ``neighbor_ped_ids`` — ``[B, obs_len, max_neighbors]``
    * ``neighbor_mask`` — ``[B, obs_len, max_neighbors]``
    * ``obs_frame`` — ``[B, obs_len]``
    * ``future_frame`` — ``[B, pred_len]``
    * ``focal_ped_id`` — ``[B]``
    * ``window_index`` — ``[B]``
    * ``scene`` — ``list[str]`` length ``B``

    **GSTGM graph (Phase 3):** weighted adjacency Eq. (2) uses **stacked node velocities**
    derived from this batch and ``data.coordinate_mode`` (see ``gstgm.graph.graph_builder``).
    """
    if not batch:
        raise ValueError("empty batch")
    return {
        "obs": torch.stack([b["obs"] for b in batch], dim=0),
        "future": torch.stack([b["future"] for b in batch], dim=0),
        "neighbor_pos": torch.stack([b["neighbor_pos"] for b in batch], dim=0),
        "neighbor_ped_ids": torch.stack([b["neighbor_ped_ids"] for b in batch], dim=0),
        "neighbor_mask": torch.stack([b["neighbor_mask"] for b in batch], dim=0),
        "obs_frame": torch.stack([b["obs_frame"] for b in batch], dim=0),
        "future_frame": torch.stack([b["future_frame"] for b in batch], dim=0),
        "focal_ped_id": torch.stack([b["focal_ped_id"] for b in batch], dim=0),
        "window_index": torch.stack([b["window_index"] for b in batch], dim=0),
        "scene": [str(b["scene"]) for b in batch],
    }
