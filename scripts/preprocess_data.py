#!/usr/bin/env python3
"""Build cached ETH/UCY sliding-window bundles (config-driven)."""

from __future__ import annotations

import argparse
from pathlib import Path

from gstgm.datasets.preprocessing import (
    build_processed_bundle,
    default_processed_path,
    save_processed_bundle,
    scenes_to_preprocess,
)
from gstgm.utils import load_config, seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ETH/UCY raw trajectories into a .pt cache.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config (e.g. configs/default.yaml or a scene file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path (default: from config data.processed_* + obs/pred).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    seed_all(int(cfg.get("training", {}).get("seed", 42)))

    raw_root = Path(data["root"]) / str(data.get("raw_subdir", "raw"))
    obs_len = int(data["obs_len"])
    pred_len = int(data["pred_len"])
    scenes = scenes_to_preprocess(cfg)

    bundle = build_processed_bundle(raw_root, scenes, obs_len, pred_len)
    out = Path(args.output) if args.output is not None else default_processed_path(cfg)
    save_processed_bundle(bundle, out)
    print(f"Wrote {len(bundle['windows'])} windows from scenes {scenes} to {out}")


if __name__ == "__main__":
    main()
