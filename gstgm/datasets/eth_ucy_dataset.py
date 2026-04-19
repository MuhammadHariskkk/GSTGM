"""PyTorch ``Dataset`` for preprocessed ETH/UCY sliding windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from gstgm.datasets.preprocessing import default_processed_path, load_processed_bundle, trajectories_from_bundle
from gstgm.datasets.scene_split import mask_split, split_indices_loo, split_indices_random


class EthUcyDataset(Dataset):
    """
    Load cached windows produced by ``scripts/preprocess_data.py``.

    Each sample is **focal-agent-centric**: one primary pedestrian per index, with
    per-timestep neighbor positions for scene graph construction (Phase 3).

    **Neighbor selection (engineering assumption):** at each observation frame,
    all other pedestrians recorded in that frame are candidates; up to ``max_neighbors``
    closest to the focal (Euclidean, planar) are kept. Ties broken by pedestrian id.

    **Velocity mode:** the focal trajectory uses step-wise displacements; neighbor
    nodes use positions **relative to the focal's last observation position** so
    spatial graphs remain geometrically meaningful (override if the paper specifies otherwise).

    Tensor shapes (single sample)
    ----------------------------
    * ``obs`` — ``[obs_len, 2]`` float32
    * ``future`` — ``[pred_len, 2]`` float32
    * ``neighbor_pos`` — ``[obs_len, max_neighbors, 2]`` float32 (padded)
    * ``neighbor_ped_ids`` — ``[obs_len, max_neighbors]`` int64 (-1 padding)
    * ``neighbor_mask`` — ``[obs_len, max_neighbors]`` bool (True = valid)
    * ``obs_frame`` — ``[obs_len]`` int64 (global frame ids)
    * ``future_frame`` — ``[pred_len]`` int64

    Use ``torch.utils.data.DataLoader(..., collate_fn=gstgm.datasets.collate_eth_ucy)`` so
    string ``scene`` labels and tensor fields batch correctly.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        split: str,
        *,
        processed_path: Path | str | None = None,
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got {split!r}")
        self.config = config
        self.split = split
        data = config["data"]
        self.coord_mode = str(data["coordinate_mode"])
        self.obs_len = int(data["obs_len"])
        self.pred_len = int(data["pred_len"])
        self.max_neighbors = int(data.get("max_neighbors", 32))
        self.split_strategy = str(data.get("split_strategy", "loo")).lower()
        self.val_fraction = float(data.get("val_fraction", 0.1))
        self.test_fraction = float(data.get("test_fraction", 0.1))
        seed = int(config.get("training", {}).get("seed", 42))

        path = Path(processed_path) if processed_path is not None else default_processed_path(config)
        bundle = load_processed_bundle(path)
        if bundle["meta"]["obs_len"] != self.obs_len or bundle["meta"]["pred_len"] != self.pred_len:
            raise ValueError(
                f"Cache {path} built with obs_len={bundle['meta']['obs_len']}, "
                f"pred_len={bundle['meta']['pred_len']} but config has "
                f"obs_len={self.obs_len}, pred_len={self.pred_len}"
            )
        self._windows: list[dict[str, Any]] = bundle["windows"]
        self._trajectories = trajectories_from_bundle(bundle["trajectories"])
        self._build_frame_index()
        self._indices = self._resolve_split_indices(seed)

    def _build_frame_index(self) -> None:
        # Map (scene, frame) -> ped_id -> (x,y); last row wins if duplicates appear in merged sources.
        buckets: dict[tuple[str, int], dict[int, tuple[float, float]]] = {}
        for scene, ped_map in self._trajectories.items():
            for ped, arr in ped_map.items():
                for row in arr:
                    f = int(round(float(row[0])))
                    key = (scene, f)
                    buckets.setdefault(key, {})[int(ped)] = (float(row[1]), float(row[2]))
        self._frame_index = {
            key: [(pid, xy[0], xy[1]) for pid, xy in sorted(d.items())] for key, d in buckets.items()
        }

    def _resolve_split_indices(self, seed: int) -> np.ndarray:
        n = len(self._windows)
        if self.split_strategy == "loo":
            holdout = self.config["data"].get("scene")
            if holdout is None:
                raise ValueError("split_strategy=loo requires data.scene (holdout test scene)")
            split = split_indices_loo(
                self._windows,
                holdout_scene=str(holdout),
                val_fraction=self.val_fraction,
                seed=seed,
            )
        elif self.split_strategy == "random":
            split = split_indices_random(
                n, val_fraction=self.val_fraction, test_fraction=self.test_fraction, seed=seed
            )
        else:
            raise ValueError(f"Unknown split_strategy: {self.split_strategy!r}")
        idx = np.asarray(mask_split(self.split, split), dtype=np.int64)
        if idx.size == 0:
            raise ValueError(f"No samples for split={self.split!r} with strategy={self.split_strategy!r}")
        return idx

    def __len__(self) -> int:
        return int(self._indices.size)

    def _focal_prev_position(self, scene: str, focal: int, first_obs_frame: int) -> np.ndarray | None:
        traj = self._trajectories.get(scene, {}).get(focal)
        if traj is None or traj.shape[0] == 0:
            return None
        fr = traj[:, 0].astype(np.int64)
        pos = traj[:, 1:3].astype(np.float64)
        idx = int(np.searchsorted(fr, int(first_obs_frame), side="left"))
        if idx <= 0:
            return None
        return pos[idx - 1]

    def _neighbor_tensor(
        self,
        scene: str,
        focal_ped: int,
        obs_frames: np.ndarray,
        focal_obs_abs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        L = self.obs_len
        K = self.max_neighbors
        pos = np.zeros((L, K, 2), dtype=np.float32)
        pids = np.full((L, K), -1, dtype=np.int64)
        mask = np.zeros((L, K), dtype=bool)
        for t in range(L):
            fr = int(obs_frames[t])
            entries = list(self._frame_index.get((scene, fr), []))
            cands: list[tuple[float, int, float, float]] = []
            px, py = float(focal_obs_abs[t, 0]), float(focal_obs_abs[t, 1])
            for ped, x, y in entries:
                if ped == focal_ped:
                    continue
                d2 = (x - px) ** 2 + (y - py) ** 2
                cands.append((d2, ped, x, y))
            cands.sort(key=lambda u: (u[0], u[1]))
            for k, (_, ped, x, y) in enumerate(cands[:K]):
                pos[t, k, 0] = float(x)
                pos[t, k, 1] = float(y)
                pids[t, k] = int(ped)
                mask[t, k] = True
        return pos, pids, mask

    def _transform_focal_and_neighbors(
        self,
        obs_abs: np.ndarray,
        fut_abs: np.ndarray,
        neigh_abs: np.ndarray,
        neigh_mask: np.ndarray,
        scene: str,
        focal_ped: int,
        obs_frames: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_abs = np.asarray(obs_abs, dtype=np.float64)
        fut_abs = np.asarray(fut_abs, dtype=np.float64)
        anchor = obs_abs[-1].copy()

        if self.coord_mode == "absolute":
            obs_t = torch.from_numpy(obs_abs.astype(np.float32))
            fut_t = torch.from_numpy(fut_abs.astype(np.float32))
            return obs_t, fut_t, torch.from_numpy(neigh_abs.copy())

        if self.coord_mode == "relative_disp":
            obs_t = torch.from_numpy((obs_abs - anchor).astype(np.float32))
            fut_t = torch.from_numpy((fut_abs - anchor).astype(np.float32))
            neigh = neigh_abs.copy().astype(np.float64)
            for t in range(neigh.shape[0]):
                for k in range(neigh.shape[1]):
                    if neigh_mask[t, k]:
                        neigh[t, k] -= anchor
            return obs_t, fut_t, torch.from_numpy(neigh.astype(np.float32))

        if self.coord_mode == "velocity":
            prev = self._focal_prev_position(scene, focal_ped, int(obs_frames[0]))
            obs_v = np.zeros_like(obs_abs)
            if prev is not None:
                obs_v[0] = obs_abs[0] - prev
            else:
                obs_v[0] = 0.0
            if obs_abs.shape[0] > 1:
                obs_v[1:] = np.diff(obs_abs, axis=0)
            fut_stack = np.vstack([obs_abs[-1:], fut_abs])
            fut_v = np.diff(fut_stack, axis=0)
            obs_t = torch.from_numpy(obs_v.astype(np.float32))
            fut_t = torch.from_numpy(fut_v.astype(np.float32))
            neigh = neigh_abs.copy().astype(np.float64)
            for t in range(neigh.shape[0]):
                for k in range(neigh.shape[1]):
                    if neigh_mask[t, k]:
                        neigh[t, k] -= anchor
            return obs_t, fut_t, torch.from_numpy(neigh.astype(np.float32))

        raise ValueError(f"Unknown coordinate_mode: {self.coord_mode!r}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        j = int(self._indices[index])
        w = self._windows[j]
        scene = str(w["scene"])
        focal = int(w["focal_ped"])
        obs_frames = w["obs_frames"].astype(np.int64)
        fut_frames = w["future_frames"].astype(np.int64)
        obs_abs = w["obs_pos_abs"].astype(np.float32)
        fut_abs = w["future_pos_abs"].astype(np.float32)

        neigh_abs, neigh_pids, neigh_mask = self._neighbor_tensor(scene, focal, obs_frames, obs_abs)
        obs_t, fut_t, neigh_t = self._transform_focal_and_neighbors(
            obs_abs, fut_abs, neigh_abs, neigh_mask, scene, focal, obs_frames
        )
        return {
            "obs": obs_t,
            "future": fut_t,
            "neighbor_pos": neigh_t,
            "neighbor_ped_ids": torch.from_numpy(neigh_pids),
            "neighbor_mask": torch.from_numpy(neigh_mask),
            "obs_frame": torch.from_numpy(obs_frames.copy()),
            "future_frame": torch.from_numpy(fut_frames.copy()),
            "focal_ped_id": torch.tensor(focal, dtype=torch.long),
            "scene": scene,
            "window_index": torch.tensor(j, dtype=torch.long),
        }
