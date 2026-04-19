"""Reproducibility: seed Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_all(seed: int, *, deterministic_cuda: bool = False) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Integer seed (project configs use ``training.seed``).
    deterministic_cuda:
        If True, may reduce GPU non-determinism at a performance cost
        (PyTorch CUBLAS deterministic algorithms). **Engineering assumption:** off by default.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass


def seed_worker(worker_id: int, base_seed: Optional[int] = None) -> None:
    """
    Worker init_fn for PyTorch DataLoader reproducibility.

    **Engineering assumption:** derives per-worker seed from ``base_seed`` and ``worker_id``.
    """
    if base_seed is None:
        return
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        import torch

        torch.manual_seed(worker_seed)
    except ImportError:
        pass
