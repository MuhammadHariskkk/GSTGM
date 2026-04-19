#!/usr/bin/env python3
"""Evaluate a GSTGM checkpoint on a dataset split (Phase 8, §5.2 ADE / FDE)."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch

from gstgm.evaluation import run_evaluation
from gstgm.utils.config import load_config, parse_dotted_overrides
from gstgm.utils.logger import setup_logging
from gstgm.utils.seed import seed_all


def _metrics_json_safe(m: dict[str, object]) -> dict[str, object]:
    """Strict JSON: replace inf/nan floats with null (RFC 8259)."""
    out: dict[str, object] = {}
    for k, v in m.items():
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            out[k] = None
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GSTGM (Phase 8).")
    parser.add_argument("--config", type=Path, required=True, help="Merged YAML (e.g. configs/eth.yaml).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint .pt (e.g. run_dir/checkpoint_best.pt).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split (default: test).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Best-of-K latent samples (K = evaluation.multimodal.num_samples); default is deterministic.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Optional config overrides (same as train).",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Write metrics JSON to this path.")
    args = parser.parse_args()

    overrides = parse_dotted_overrides(args.overrides) if args.overrides else None
    cfg = load_config(args.config, overrides=overrides)

    tr = cfg.get("training") or {}
    seed_all(
        int(tr.get("seed", 42)),
        deterministic_cuda=bool(tr.get("deterministic_cuda", False)),
    )

    log_dir = args.checkpoint.resolve().parent
    setup_logging(log_file=log_dir / "eval.log")
    log = logging.getLogger("evaluate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = run_evaluation(
        cfg,
        args.checkpoint,
        split=args.split,
        device=device,
        stochastic=args.stochastic,
    )
    safe = _metrics_json_safe(metrics)
    log.info("metrics %s", metrics)
    print(json.dumps(safe, indent=2, allow_nan=False))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(safe, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
