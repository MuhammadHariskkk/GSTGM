"""Train / validation / test index splits for ETH/UCY-style window datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    """Integer indices into a window list for each split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_indices_loo(
    windows: Sequence[Mapping[str, Any]],
    *,
    holdout_scene: str,
    val_fraction: float,
    seed: int,
) -> SplitIndices:
    """
    Leave-one-scene-out: test = ``holdout_scene``, train/val from all other scenes.

    **Engineering assumption:** validation is a random subset of **training** scenes only
    (fraction ``val_fraction`` of those windows), stable under ``seed``.
    """
    holdout_scene = str(holdout_scene)
    n = len(windows)
    train_pool = [i for i in range(n) if str(windows[i]["scene"]) != holdout_scene]
    test_idx = np.array([i for i in range(n) if str(windows[i]["scene"]) == holdout_scene], dtype=np.int64)
    if not train_pool:
        raise ValueError(f"No training windows after holding out scene={holdout_scene!r}")
    rng = np.random.default_rng(seed)
    train_pool = np.array(train_pool, dtype=np.int64)
    rng.shuffle(train_pool)
    n_val = int(round(len(train_pool) * float(val_fraction)))
    n_val = min(max(n_val, 0), max(0, len(train_pool) - 1))
    val_idx = train_pool[:n_val]
    train_idx = train_pool[n_val:]
    if train_idx.size == 0:
        raise ValueError("val_fraction too large: no training windows left")
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def split_indices_random(
    n_windows: int,
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int,
) -> SplitIndices:
    """
    Random permutation split (ignores scene boundaries).

    **Engineering assumption:** default ``val_fraction`` / ``test_fraction`` match common 80/10/10-style
    proportions, clamped so each split is non-empty when ``n_windows >= 3``.
    """
    if n_windows < 3:
        raise ValueError("random split needs at least 3 windows")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_windows).astype(np.int64)
    n_test = max(1, int(round(n_windows * float(test_fraction))))
    n_val = max(1, int(round(n_windows * float(val_fraction))))
    n_test = min(n_test, n_windows - 2)
    n_val = min(n_val, n_windows - n_test - 1)
    n_train = n_windows - n_val - n_test
    if n_train < 1:
        raise ValueError("n_windows too small for requested val/test fractions")
    return SplitIndices(train=perm[:n_train], val=perm[n_train : n_train + n_val], test=perm[n_train + n_val :])


def mask_split(split_name: str, split: SplitIndices) -> np.ndarray:
    """Return index array for ``split_name`` in {``train``, ``val``, ``test``}."""
    name = split_name.lower()
    if name == "train":
        return split.train
    if name == "val":
        return split.val
    if name == "test":
        return split.test
    raise ValueError(f"Unknown split: {split_name!r}")
