#!/usr/bin/env python3
"""Train GSTGM from a merged YAML (e.g. ``configs/eth.yaml``). Phase 7 entrypoint."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch

from gstgm.datasets import collate_eth_ucy
from gstgm.models import gstgm_from_cfg
from gstgm.training.trainer import GSTGMTrainer, build_dataloader
from gstgm.utils.config import load_config, parse_dotted_overrides, save_config
from gstgm.utils.logger import setup_logging
from gstgm.utils.seed import seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GSTGM (Phase 7).")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Primary YAML (scene files usually ``extends: default.yaml``).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help='Nested overrides, e.g. --set training.epochs=1 --set data.batch_size=4 (JSON values).',
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory (default: experiment.output_dir / name / UTC timestamp).",
    )
    args = parser.parse_args()

    overrides = parse_dotted_overrides(args.overrides) if args.overrides else None
    cfg = load_config(args.config, overrides=overrides)

    exp = cfg.get("experiment") or {}
    base_out = Path(str(exp.get("output_dir", "outputs")))
    name = str(exp.get("name", "gstgm_run"))
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = (base_out / name / stamp).resolve()

    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config_resolved.yaml")
    setup_logging(log_file=run_dir / "train.log")
    log = logging.getLogger("train")

    tr = cfg.get("training") or {}
    seed_all(
        int(tr.get("seed", 42)),
        deterministic_cuda=bool(tr.get("deterministic_cuda", False)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s run_dir=%s", device, run_dir)

    train_loader = build_dataloader(cfg, "train", collate_eth_ucy, shuffle=True)
    val_loader = build_dataloader(cfg, "val", collate_eth_ucy, shuffle=False)
    model = gstgm_from_cfg(cfg)
    trainer = GSTGMTrainer(model, cfg, train_loader, val_loader, run_dir, device=device)
    trainer.fit()
    log.info("finished")


if __name__ == "__main__":
    main()
