"""Scene graph construction shapes (Phase 3)."""

from __future__ import annotations

import torch

from gstgm.graph.graph_builder import build_scene_graph_batch


def test_scene_graph_batch_shapes() -> None:
    b, t, n = 2, 8, 5
    obs = torch.randn(b, t, 2)
    nbr = torch.randn(b, t, n - 1, 2)
    mask = torch.ones(b, t, n - 1, dtype=torch.bool)
    g = build_scene_graph_batch(
        obs,
        nbr,
        mask,
        coordinate_mode="relative_disp",
        self_loop=False,
        normalize_adjacency=True,
    )
    assert g.positions.shape == (b, t, n, 2)
    assert g.velocities.shape == (b, t, n, 2)
    assert g.adjacency_norm.shape == (b, t, n, n)
