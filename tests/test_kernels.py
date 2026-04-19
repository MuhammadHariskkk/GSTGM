"""Graph similarity (paper Eq. (2)) — velocity space."""

from __future__ import annotations

import pytest
import torch

from gstgm.graph.kernels import gstgm_adjacency_similarity, pairwise_squared_euclidean


def test_pairwise_sq_euclidean_shape() -> None:
    x = torch.randn(2, 5, 2)
    sq = pairwise_squared_euclidean(x)
    assert sq.shape == (2, 5, 5)


def test_gstgm_adjacency_inverse_sq() -> None:
    """Unit velocity difference -> weight 1; identical velocities -> 0."""
    v = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    sq = pairwise_squared_euclidean(v)
    w = gstgm_adjacency_similarity(sq)
    assert w[0, 0, 1].item() == pytest.approx(1.0)
    assert w[0, 0, 0].item() == pytest.approx(0.0)
    assert w[0, 1, 1].item() == pytest.approx(0.0)
