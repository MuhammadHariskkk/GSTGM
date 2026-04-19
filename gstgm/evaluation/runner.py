"""
Load trained :class:`~gstgm.models.gstgm.GSTGM` weights and report ADE / FDE (Khel et al. 2024, §5.2).

**paper-specified (structure)**
    * ``ADE_M`` / ``FDE_M``: min over mixture modes :math:`m` at each time / final time (Eq. 21–22).

**engineering assumption**
    * Deterministic eval uses ``sample_posterior=False`` (mean latent). Optional ``--stochastic`` repeats
      forward with i.i.d. latent noise and keeps the **best** per-trajectory oracle (min over modes
      **and** samples), using ``evaluation.multimodal.num_samples``.

Tensor bookkeeping matches :func:`gstgm.training.metrics.per_trajectory_oracle_ade_fde`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from gstgm.datasets import collate_eth_ucy
from gstgm.models import gstgm_from_cfg
from gstgm.training.metrics import per_trajectory_oracle_ade_fde
from gstgm.training.trainer import build_dataloader, move_batch_to_device
from gstgm.utils.checkpoint import load_checkpoint


def _eval_loader_kwargs(cfg: Mapping[str, Any]) -> tuple[int, int]:
    ev = cfg.get("evaluation") or {}
    data = cfg.get("data") or {}
    bs_raw = ev.get("batch_size")
    nw_raw = ev.get("num_workers")
    bs = int(data.get("batch_size", 8)) if bs_raw is None else int(bs_raw)
    nw = int(data.get("num_workers", 0)) if nw_raw is None else int(nw_raw)
    return bs, nw


def _aggregate_stochastic_one_batch(
    model: torch.nn.Module,
    batch_d: dict[str, Any],
    *,
    num_samples: int,
    latent_dim: int,
) -> tuple[Tensor, Tensor]:
    device = batch_d["obs"].device
    bsz = batch_d["obs"].size(0)
    last = batch_d["obs"][:, -1, :]
    future = batch_d["future"]
    best_ade = torch.full((bsz,), float("inf"), device=device, dtype=future.dtype)
    best_fde = torch.full((bsz,), float("inf"), device=device, dtype=future.dtype)
    for _ in range(num_samples):
        eps = torch.randn(bsz, latent_dim, device=device, dtype=future.dtype)
        out = model(batch_d, latent_eps=eps, sample_posterior=True)
        a_t, f_t = per_trajectory_oracle_ade_fde(out["pred_mu"], last, future)
        best_ade = torch.minimum(best_ade, a_t)
        best_fde = torch.minimum(best_fde, f_t)
    return best_ade, best_fde


@torch.no_grad()
def run_evaluation(
    cfg: Mapping[str, Any],
    checkpoint_path: Path | str,
    *,
    split: str = "test",
    device: torch.device | None = None,
    stochastic: bool = False,
) -> dict[str, float]:
    """
    Parameters
    ----------
    cfg :
        Full merged config (same as training).
    checkpoint_path :
        ``checkpoint_best.pt`` (or ``checkpoint_last.pt``) from :mod:`gstgm.training.trainer`.
    split :
        ``train`` | ``val`` | ``test`` (default held-out / split policy from dataset).
    device :
        Torch device; default CUDA if available.
    stochastic :
        If True, draw ``evaluation.multimodal.num_samples`` latent draws per batch and keep best
        per-trajectory ADE/FDE (after oracle over modes). If False, one deterministic forward.
    """
    path = Path(checkpoint_path)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gstgm_from_cfg(cfg)
    ckpt = load_checkpoint(path, map_location=dev)
    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint {path} missing 'model_state'")
    model.load_state_dict(ckpt["model_state"])
    model.to(dev)
    model.eval()

    bs, nw = _eval_loader_kwargs(cfg)
    loader = build_dataloader(
        cfg,
        split,
        collate_eth_ucy,
        shuffle=False,
        batch_size=bs,
        num_workers=nw,
    )

    latent_dim = int((cfg.get("generative") or {}).get("latent_dim", 32))
    mm = (cfg.get("evaluation") or {}).get("multimodal") or {}
    num_samples = int(mm.get("num_samples", 20)) if stochastic else 0

    ade_sum, fde_sum, n_traj = 0.0, 0.0, 0
    for batch in loader:
        batch_d = move_batch_to_device(batch, dev)
        if num_samples > 0:
            ade_t, fde_t = _aggregate_stochastic_one_batch(
                model,
                batch_d,
                num_samples=num_samples,
                latent_dim=latent_dim,
            )
        else:
            out = model(batch_d, sample_posterior=False)
            ade_t, fde_t = per_trajectory_oracle_ade_fde(
                out["pred_mu"],
                batch_d["obs"][:, -1, :],
                batch_d["future"],
            )
        ade_sum += float(ade_t.sum().item())
        fde_sum += float(fde_t.sum().item())
        n_traj += int(ade_t.numel())

    if n_traj == 0:
        return {
            "ade": float("inf"),
            "fde": float("inf"),
            "n": 0.0,
            "stochastic_samples": float(num_samples),
            "split": split,
        }

    return {
        "ade": ade_sum / n_traj,
        "fde": fde_sum / n_traj,
        "n": float(n_traj),
        "stochastic_samples": float(num_samples),
        "split": split,
    }
