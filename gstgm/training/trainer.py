"""Single-GPU GSTGM training loop (Phase 7): optimization, validation, checkpoints, CSV metrics."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from gstgm.models.gstgm import GSTGM
from gstgm.training.losses import gstgm_batch_loss
from gstgm.training.metrics import dict_from_val_batch
from gstgm.utils.checkpoint import CheckpointManager, save_checkpoint
from gstgm.utils.logger import TrainingLogger
from gstgm.utils.seed import seed_worker


def _build_optimizer(model: Module, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    o = cfg.get("optimizer") or {}
    name = str(o.get("name", "adam")).lower()
    lr = float(o.get("lr", 1e-3))
    wd = float(o.get("weight_decay", 0.0))
    params = list(model.parameters())
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=float(o.get("momentum", 0.9)))
    raise ValueError(f"Unknown optimizer.name: {name!r}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Mapping[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    sch = cfg.get("scheduler") or {}
    name = str(sch.get("name", "none")).lower()
    if name in ("none", ""):
        return None
    epochs = int((cfg.get("training") or {}).get("epochs", 80))
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch.get("step_size", 30)),
            gamma=float(sch.get("gamma", 0.1)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sch.get("t_max", epochs)),
            eta_min=float(sch.get("eta_min", 0.0)),
        )
    raise ValueError(f"Unknown scheduler.name: {name!r}")


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a collated batch to ``device`` (strings / lists unchanged)."""
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


class GSTGMTrainer:
    """
    Parameters
    ----------
    model :
        :class:`~gstgm.models.gstgm.GSTGM`.
    cfg :
        Fully merged config (must include ``data``, ``training``, ``experiment``, etc.).
    train_loader, val_loader :
        ``DataLoader`` using :func:`gstgm.datasets.collate_eth_ucy`.
    run_dir :
        Writes ``metrics.csv``, ``config_resolved.yaml``, checkpoints inside ``run_dir``.
    device :
        Training device.
    """

    def __init__(
        self,
        model: GSTGM,
        cfg: Mapping[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_dir: Path | str,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.cfg = dict(cfg)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        t = self.cfg.get("training") or {}
        self.epochs = int(t.get("epochs", 80))
        self.grad_clip_norm = float(t.get("grad_clip_norm", 1.0))
        self.log_every = int(t.get("log_every", 50))
        self.val_every = int(t.get("val_every", 1))
        self.ckpt_every = int(t.get("checkpoint_every", 1))
        self.early_patience = int(t.get("early_stopping_patience", 0))

        self.optimizer = _build_optimizer(self.model, self.cfg)
        self.scheduler = _build_scheduler(self.optimizer, self.cfg)
        self.ckpt_mgr = CheckpointManager(output_dir=self.run_dir, metric_key="val_ade")

        self.metrics_log = TrainingLogger(self.run_dir)
        self._global_step = 0
        self._best_wait = 0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        tot, n = 0.0, 0
        for step, batch in enumerate(self.train_loader):
            batch_d = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(batch_d, sample_posterior=True)
            loss, parts = gstgm_batch_loss(out, batch_d, self.cfg, epoch=epoch)
            loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            tot += float(loss.item())
            n += 1
            self._global_step += 1
            if self._global_step % self.log_every == 0:
                row = {"lr": float(self.optimizer.param_groups[0]["lr"]), **parts}
                self.metrics_log.log_step("train", epoch, self._global_step, row)
        return tot / max(n, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        ade_w, fde_w = 0.0, 0.0
        n_samples = 0
        for batch in self.val_loader:
            batch_d = move_batch_to_device(batch, self.device)
            bsz = int(batch_d["obs"].size(0))
            out = self.model(batch_d, sample_posterior=False)
            m = dict_from_val_batch(out, batch_d)
            ade_w += m["val_ade"] * bsz
            fde_w += m["val_fde"] * bsz
            n_samples += bsz
        if n_samples == 0:
            return {"val_ade": float("inf"), "val_fde": float("inf")}
        return {
            "val_ade": ade_w / n_samples,
            "val_fde": fde_w / n_samples,
        }

    def fit(self) -> None:
        for epoch in range(self.epochs):
            tr_loss = self.train_epoch(epoch)
            self.metrics_log.log_step("train_epoch", epoch, self._global_step, {"train_loss_epoch": tr_loss})

            metrics: dict[str, float] = {}
            if self.val_every > 0 and (epoch + 1) % self.val_every == 0:
                metrics = self.validate(epoch)
                self.metrics_log.log_step("val", epoch, self._global_step, metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.ckpt_every > 0 and (epoch + 1) % self.ckpt_every == 0:
                meta = {"epoch": epoch + 1, "train_loss_epoch": tr_loss, **metrics}
                save_checkpoint(
                    self.ckpt_mgr.path_last,
                    model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
                    meta=meta,
                )
                if metrics:
                    improved = self.ckpt_mgr.update_best(metrics)
                    if improved:
                        shutil.copy2(self.ckpt_mgr.path_last, self.ckpt_mgr.path_best)
                        self._best_wait = 0
                    else:
                        self._best_wait += 1
                    if self.early_patience > 0 and self._best_wait >= self.early_patience:
                        break

        self.metrics_log.close()


def build_dataloader(
    cfg: Mapping[str, Any],
    split: str,
    collate_fn: Callable[..., Any],
    *,
    shuffle: bool,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    from gstgm.datasets.eth_ucy_dataset import EthUcyDataset

    ds = EthUcyDataset(cfg, split=split)
    data = cfg.get("data") or {}
    bs = int(data.get("batch_size", 8)) if batch_size is None else int(batch_size)
    nw = int(data.get("num_workers", 0)) if num_workers is None else int(num_workers)
    pin = bool(data.get("pin_memory", True))
    seed = int((cfg.get("training") or {}).get("seed", 42))
    gen = torch.Generator()
    gen.manual_seed(seed + (0 if split == "train" else 1))
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        collate_fn=collate_fn,
        pin_memory=pin and torch.cuda.is_available(),
        worker_init_fn=lambda wid: seed_worker(wid, seed),
        generator=gen,
    )
