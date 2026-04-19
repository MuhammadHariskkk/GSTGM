"""
ETH/UCY trajectory preprocessing: load raw text, build sliding windows, save a reusable bundle.

Raw file assumptions (engineering — verify against your download):
--------------------------------------------------------------
* One or more ``.txt`` files per scene, ASCII, one record per line.
* Each non-empty, non-comment line has **four whitespace-separated fields**::

    <frame_id> <pedestrian_id> <pos_x> <pos_y>

* ``frame_id`` is coerced to int64 (must be integral or whole-valued floats).
* ``pedestrian_id`` is coerced to int64.
* Positions are planar coordinates (meters in standard releases).
* Lines starting with ``#`` are ignored.

Some public mirrors use tab separation; any whitespace delimiter is accepted via ``numpy.loadtxt``.
Coordinates are stored in **absolute** scene coordinates in the saved bundle; ``coordinate_mode`` in
the dataset is applied at sample time.
"""

from __future__ import annotations

import inspect
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

# Canonical scene ids matching ``configs/*.yaml`` ``data.scene`` values.
ETH_UCY_SCENES: tuple[str, ...] = ("eth", "hotel", "univ", "zara1", "zara2")

# Folder / file aliases for common dataset layouts (engineering assumption).
SCENE_SEARCH_NAMES: Mapping[str, tuple[str, ...]] = {
    "eth": ("eth",),
    "hotel": ("hotel",),
    "univ": ("univ",),
    "zara1": ("zara1", "zara01"),
    "zara2": ("zara2", "zara02"),
}


def torch_load_compat(path: Path, *, map_location: str | None = "cpu") -> Any:
    """``torch.load`` compatible with PyTorch 2.6+ (``weights_only`` default)."""
    load_kw: dict[str, Any] = {}
    if map_location is not None:
        load_kw["map_location"] = map_location
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        load_kw["weights_only"] = False
    return torch.load(path, **load_kw)


def resolve_scene_raw_paths(root: Path, scene: str) -> list[Path]:
    """Return sorted list of raw ``.txt`` files for a scene."""
    scene = str(scene)
    candidates: list[Path] = []
    for name in SCENE_SEARCH_NAMES.get(scene, (scene,)):
        subdir = root / name
        if subdir.is_dir():
            candidates.extend(sorted(subdir.glob("*.txt")))
        single = root / f"{name}.txt"
        if single.is_file():
            candidates.append(single)
    # De-duplicate preserving order
    seen: set[str] = set()
    out: list[Path] = []
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    if not out:
        raise FileNotFoundError(
            f"No raw .txt found for scene {scene!r} under {root}. "
            f"Tried subfolders / files: {SCENE_SEARCH_NAMES.get(scene, (scene,))}"
        )
    return out


def read_eth_ucy_trajectory_file(path: Path) -> np.ndarray:
    """
    Load a single trajectory table as float64 array ``[N, 4]``: frame, ped, x, y.

    Comments (#) and blank lines are skipped via a simple pre-filter.
    """
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    if not lines:
        raise ValueError(f"No data rows in {path}")
    # File-like input avoids version-specific behavior for list-of-lines loaders.
    data = np.loadtxt(io.StringIO("\n".join(lines)), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns in {path}, got shape {data.shape}")
    return data[:, :4]


def build_ped_trajectories(table: np.ndarray) -> dict[int, np.ndarray]:
    """
    Group rows into per-pedestrian trajectories, sorted by frame.

    Returns
    -------
    dict
        ``ped_id -> array [T, 3]`` with columns ``[frame, x, y]`` (float64, frames integral).
    """
    frames = np.rint(table[:, 0]).astype(np.int64)
    peds = np.rint(table[:, 1]).astype(np.int64)
    xy = table[:, 2:4].astype(np.float64)
    out: dict[int, list[list[float]]] = {}
    for fr, ped, x, y in zip(frames.tolist(), peds.tolist(), xy[:, 0].tolist(), xy[:, 1].tolist()):
        out.setdefault(int(ped), []).append([float(fr), float(x), float(y)])
    trajectories: dict[int, np.ndarray] = {}
    for ped, rows in out.items():
        arr = np.array(rows, dtype=np.float64)
        order = np.argsort(arr[:, 0], kind="mergesort")
        arr = arr[order]
        trajectories[ped] = arr
    return trajectories


def sliding_windows_for_trajectory(
    traj: np.ndarray,
    obs_len: int,
    pred_len: int,
) -> list[tuple[int, int]]:
    """
    Index pairs ``(start, end_exclusive)`` into trajectory rows for valid windows.

    **Engineering assumption:** windows are **consecutive rows** after frame sort (indices
    ``start .. start + obs_len + pred_len - 1``). Gaps in frame ids do not break a window.
    """
    total = obs_len + pred_len
    n = traj.shape[0]
    if n < total:
        return []
    return [(s, s + total) for s in range(0, n - total + 1)]


@dataclass
class WindowRecord:
    """One training/test sample (focal pedestrian)."""

    scene: str
    focal_ped: int
    obs_frames: np.ndarray  # [obs_len] int64
    future_frames: np.ndarray  # [pred_len] int64
    obs_pos_abs: np.ndarray  # [obs_len, 2] float32
    future_pos_abs: np.ndarray  # [pred_len, 2] float32


def window_record_to_dict(w: WindowRecord) -> dict[str, Any]:
    return {
        "scene": w.scene,
        "focal_ped": int(w.focal_ped),
        "obs_frames": w.obs_frames.astype(np.int64),
        "future_frames": w.future_frames.astype(np.int64),
        "obs_pos_abs": w.obs_pos_abs.astype(np.float32),
        "future_pos_abs": w.future_pos_abs.astype(np.float32),
    }


def collect_scene_trajectories(raw_root: Path, scene: str) -> dict[int, np.ndarray]:
    """Load and merge all raw files for one scene into per-ped trajectories (frames may interleave)."""
    paths = resolve_scene_raw_paths(raw_root, scene)
    merged: dict[int, list[np.ndarray]] = {}
    for p in paths:
        tbl = read_eth_ucy_trajectory_file(p)
        trajs = build_ped_trajectories(tbl)
        for ped, arr in trajs.items():
            merged.setdefault(ped, []).append(arr)
    # Concatenate and re-sort each pedestrian
    final: dict[int, np.ndarray] = {}
    for ped, parts in merged.items():
        cat = np.concatenate(parts, axis=0)
        order = np.argsort(cat[:, 0], kind="mergesort")
        cat = cat[order]
        fr = cat[:, 0].astype(np.int64)
        if fr.size == 0:
            continue
        # Duplicate frame ids (same ped): keep the last row after sort.
        keep = np.concatenate([fr[:-1] != fr[1:], np.array([True])])
        cat = cat[keep]
        if cat.shape[0] == 0:
            continue
        final[ped] = cat.astype(np.float32)
    return final


def build_windows_for_scene(
    scene: str,
    trajectories: Mapping[int, np.ndarray],
    obs_len: int,
    pred_len: int,
) -> list[dict[str, Any]]:
    """Enumerate all focal / sliding windows for every pedestrian in a scene."""
    windows: list[dict[str, Any]] = []
    for ped, traj in trajectories.items():
        for s, e in sliding_windows_for_trajectory(traj, obs_len, pred_len):
            chunk = traj[s:e]
            obs = chunk[:obs_len]
            fut = chunk[obs_len:]
            wr = WindowRecord(
                scene=scene,
                focal_ped=int(ped),
                obs_frames=obs[:, 0].astype(np.int64),
                future_frames=fut[:, 0].astype(np.int64),
                obs_pos_abs=obs[:, 1:3].astype(np.float32),
                future_pos_abs=fut[:, 1:3].astype(np.float32),
            )
            windows.append(window_record_to_dict(wr))
    return windows


def trajectories_dict_to_serializable(
    per_scene: Mapping[str, Mapping[int, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Convert nested dicts to string keys for torch.save stability."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for scene, trajs in per_scene.items():
        out[str(scene)] = {str(k): np.asarray(v, dtype=np.float32) for k, v in trajs.items()}
    return out


def trajectories_from_bundle(
    raw: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = {}
    for scene, trajs in raw.items():
        out[str(scene)] = {int(k): np.asarray(v, dtype=np.float32) for k, v in trajs.items()}
    return out


def build_processed_bundle(
    data_root: Path,
    scenes: Sequence[str],
    obs_len: int,
    pred_len: int,
) -> dict[str, Any]:
    """
    Load raw scenes and build the in-memory bundle (windows + absolute trajectories).

    Parameters
    ----------
    data_root:
        ``Path(root) / raw_subdir`` where scene folders or files live.
    """
    per_scene_traj: dict[str, dict[int, np.ndarray]] = {}
    all_windows: list[dict[str, Any]] = []
    for scene in scenes:
        trajs = collect_scene_trajectories(data_root, scene)
        per_scene_traj[scene] = trajs
        all_windows.extend(build_windows_for_scene(scene, trajs, obs_len, pred_len))
    bundle = {
        "meta": {
            "obs_len": int(obs_len),
            "pred_len": int(pred_len),
            "scenes": list(scenes),
            "format_version": 1,
        },
        "windows": all_windows,
        "trajectories": trajectories_dict_to_serializable(per_scene_traj),
    }
    return bundle


def save_processed_bundle(bundle: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(bundle), path)
    return path.resolve()


def default_processed_path(cfg: Mapping[str, Any]) -> Path:
    """Resolve processed bundle path from merged experiment config dict."""
    data = cfg["data"]
    root = Path(data["root"])
    sub = str(data.get("processed_subdir", "processed"))
    base = str(data.get("processed_basename", "eth_ucy_windows"))
    obs = int(data["obs_len"])
    pred = int(data["pred_len"])
    return root / sub / f"{base}_obs{obs}_pred{pred}.pt"


def load_processed_bundle(path: Path) -> dict[str, Any]:
    return torch_load_compat(path, map_location="cpu")


def scenes_to_preprocess(cfg: Mapping[str, Any]) -> list[str]:
    """Scene list from config: ``preprocess_scenes`` or full ETH/UCY list."""
    data = cfg["data"]
    explicit = data.get("preprocess_scenes")
    if explicit is None:
        return list(ETH_UCY_SCENES)
    if isinstance(explicit, str):
        return [explicit]
    return [str(s) for s in explicit]
