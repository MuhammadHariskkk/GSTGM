"""Checkpoint save/load helpers for PyTorch training."""

from __future__ import annotations

import inspect
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import torch

UnionPath = str | Path


def save_checkpoint(
    path: UnionPath,
    *,
    model_state: Mapping[str, Any],
    optimizer_state: Optional[Mapping[str, Any]] = None,
    scheduler_state: Optional[Mapping[str, Any]] = None,
    meta: Optional[MutableMapping[str, Any]] = None,
    is_best: bool = False,
    best_path: Optional[UnionPath] = None,
) -> Path:
    """
    Save a training checkpoint (state dicts + optional metadata).

    Parameters
    ----------
    path:
        Destination ``.pt`` or ``.pth`` file.
    model_state:
        Typically ``model.state_dict()`` (CPU tensors recommended for portability).
    optimizer_state:
        Optional ``optimizer.state_dict()``.
    scheduler_state:
        Optional LR scheduler state.
    meta:
        Dict merged into checkpoint under ``"meta"`` (epoch, metrics, config hash, etc.).
    is_best:
        If True and ``best_path`` is set, copy this file to ``best_path``.
    best_path:
        Path for best model copy (e.g. ``outputs/run/best.pt``).

    Returns
    -------
    Path
        Resolved checkpoint path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model_state": dict(model_state)}
    if optimizer_state is not None:
        payload["optimizer_state"] = dict(optimizer_state)
    if scheduler_state is not None:
        payload["scheduler_state"] = dict(scheduler_state)
    if meta is not None:
        payload["meta"] = dict(meta)
    torch.save(payload, path)
    if is_best and best_path is not None:
        shutil.copy2(path, Path(best_path))
    return path.resolve()


def load_checkpoint(
    path: UnionPath,
    map_location: Optional[str | torch.device] = None,
) -> dict[str, Any]:
    """
    Load checkpoint dict from disk.

    Returns
    -------
    dict
        At minimum ``model_state``; may include ``optimizer_state``, ``scheduler_state``, ``meta``.
    """
    path = Path(path)
    # PyTorch 2.6+ defaults weights_only=True; training checkpoints include dict/meta/optim state.
    load_kw: dict[str, Any] = {}
    if map_location is not None:
        load_kw["map_location"] = map_location
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        load_kw["weights_only"] = False
    return torch.load(path, **load_kw)


@dataclass
class CheckpointManager:
    """
    Thin wrapper for periodic and best checkpoints (used by trainer in Phase 7+).

    **Engineering assumption:** ``metric_key`` lower is better; default ``val_ade`` matches
    ADE-first reporting on ETH/UCY (trainer should emit the same key).
    """

    output_dir: Path
    filename_last: str = "checkpoint_last.pt"
    filename_best: str = "checkpoint_best.pt"
    metric_key: str = "val_ade"
    lower_is_better: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._best_value: Optional[float] = None

    @property
    def path_last(self) -> Path:
        return self.output_dir / self.filename_last

    @property
    def path_best(self) -> Path:
        return self.output_dir / self.filename_best

    def update_best(self, metrics: Mapping[str, float]) -> bool:
        """Return True if ``metrics[metric_key]`` improves the running best."""
        if self.metric_key not in metrics:
            return False
        value = float(metrics[self.metric_key])
        if self._best_value is None:
            self._best_value = value
            return True
        if self.lower_is_better and value < self._best_value:
            self._best_value = value
            return True
        if not self.lower_is_better and value > self._best_value:
            self._best_value = value
            return True
        return False
