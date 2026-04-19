#!/usr/bin/env python3
"""Visualize GSTGM trajectories (multimodal GMM), optional graph, and training curves."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from gstgm.datasets import collate_eth_ucy
from gstgm.models import gstgm_from_cfg
from gstgm.training.trainer import build_dataloader, move_batch_to_device
from gstgm.utils import load_config, seed_all
from gstgm.utils.checkpoint import load_checkpoint
from gstgm.utils.distribution import positions_from_velocities, time_averaged_mode_probs, topk_mode_indices
from gstgm.utils.visualization import (
    build_graph_for_batch,
    plot_graph_connectivity,
    plot_training_curves,
    plot_trajectories_multimodal,
    save_figure,
)


def _load_model_and_batch(
    cfg: dict[str, Any],
    checkpoint: Path,
    split: str,
    batch_idx: int,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    model = gstgm_from_cfg(cfg)
    ckpt = load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    loader = build_dataloader(cfg, split, collate_eth_ucy, shuffle=False)
    batch: dict[str, Any] | None = None
    for i, b in enumerate(loader):
        if i == batch_idx:
            batch = b
            break
    if batch is None:
        raise IndexError(f"No batch at index {batch_idx} for split {split!r}.")
    return model, batch


def main() -> None:
    parser = argparse.ArgumentParser(description="GSTGM prediction and training visualizations.")
    parser.add_argument("--config", type=Path, required=True, help="Merged YAML (e.g. configs/eth.yaml).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Training checkpoint (e.g. run_dir/checkpoint_best.pt).",
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch-index", type=int, default=0, help="Which dataloader batch (after shuffle=False).")
    parser.add_argument("--sample-indices", type=str, default="0", help="Comma-separated row indices in batch (e.g. 0,1,2).")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/gstgm_vis"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--top-k-modes",
        type=int,
        default=0,
        help="Plot only top-K modes by mean mixture probability; 0 = all modes.",
    )
    parser.add_argument(
        "--graph-frame",
        type=int,
        default=-1,
        help="Observation time index t in [0, obs_len-1] for graph panel; -1 skips.",
    )
    parser.add_argument(
        "--training-curves",
        type=Path,
        default=None,
        help="Optional path to run_dir/metrics.csv for loss / val curves.",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    log = logging.getLogger("visualize")

    cfg = load_config(args.config)
    tr = cfg.get("training") or {}
    seed_all(int(tr.get("seed", 42)), deterministic_cuda=bool(tr.get("deterministic_cuda", False)))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, batch_cpu = _load_model_and_batch(cfg, args.checkpoint, args.split, args.batch_index, device)
    batch_d = move_batch_to_device(batch_cpu, device)

    with torch.no_grad():
        out = model(batch_d, sample_posterior=False)

    obs_len = batch_d["obs"].size(1)
    if args.graph_frame >= obs_len:
        raise ValueError(f"--graph-frame must be < obs_len={obs_len}")

    sample_ids = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]
    bsz = batch_d["obs"].size(0)
    for sid in sample_ids:
        if sid < 0 or sid >= bsz:
            raise IndexError(f"sample index {sid} out of range for batch size {bsz}")

    pi_logits = out["pi_logits"]
    pred_mu = out["pred_mu"]
    probs_b = time_averaged_mode_probs(pi_logits)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sid in sample_ids:
        obs = batch_d["obs"][sid]
        fut = batch_d["future"][sid]
        last = batch_d["obs"][sid, -1, :]
        mus = pred_mu[sid : sid + 1]
        pos_modes = positions_from_velocities(last.unsqueeze(0), mus)[0]

        idx_plot: list[int] | None = None
        title_modes = "all modes"
        if args.top_k_modes > 0:
            scores = probs_b[sid : sid + 1]
            idx_plot = topk_mode_indices(scores, args.top_k_modes, dim=-1)[0].tolist()
            title_modes = f"top-{len(idx_plot)} modes"

        probs_1d = probs_b[sid]
        title = f"{args.split} batch={args.batch_index} sample={sid} ({title_modes})"
        fig = plot_trajectories_multimodal(
            obs,
            fut,
            pos_modes,
            mode_probs=probs_1d,
            mode_indices_to_plot=idx_plot,
            title=title,
        )
        traj_path = out_dir / f"trajectories_{args.split}_b{args.batch_index}_s{sid}.png"
        save_figure(traj_path, fig=fig, dpi=args.dpi)
        log.info("saved %s", traj_path)

        if args.graph_frame >= 0:
            graph = build_graph_for_batch(batch_d, cfg)
            fig_g = plot_graph_connectivity(
                graph,
                sid,
                args.graph_frame,
                threshold=1e-3,
                title=f"Graph (weighted adj) t={args.graph_frame} — {title}",
            )
            gp = out_dir / f"graph_{args.split}_b{args.batch_index}_s{sid}_t{args.graph_frame}.png"
            save_figure(gp, fig=fig_g, dpi=args.dpi)
            log.info("saved %s", gp)

    if args.training_curves is not None and args.training_curves.is_file():
        fig_c = plot_training_curves(args.training_curves)
        cp = out_dir / "training_curves.png"
        save_figure(cp, fig=fig_c, dpi=args.dpi)
        log.info("saved %s", cp)
    elif args.training_curves is not None:
        log.warning("training curves file not found: %s", args.training_curves)


if __name__ == "__main__":
    main()
