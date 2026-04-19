"""
LSTM trajectory decoder :math:`\\psi_{\\mathrm{dec}}` (GSTGM, Khel et al. 2024, §4.3, Eq. 10–11).

**paper-specified**
    * Recurrent state :math:`h_{t-1}` with conditional generation
      :math:`v_t \\mid x_0, z_0 \\sim \\mathcal{N}(\\mu_{v,t}, \\mathrm{diag}(\\sigma_{v,t}^2))` (Eq. 10–11).
    * Outputs :math:`T'` future steps in one forward pass (paper: "single iteration").

**engineering assumption**
    * Per-step LSTM input = concat of :math:`\\psi_z(z_0)`, :math:`\\psi_x(y^i)` (linear projections) and an
      optional time index embedding when ``generative.time_dependent`` is enabled (Phase 6 uses the
      flag from merged YAML).
    * Two-headed MLP after LSTM (ELU + linear) lives in :mod:`gstgm.models.gmm_head` for mixture
      parameters; this module only produces the LSTM sequence ``[B, T', H]``.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor


def decoder_kwargs_from_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Defaults keep configs without ``model.decoder`` valid."""
    m = cfg.get("model") or {}
    gcn = m.get("gcn") or {}
    d = m.get("decoder") or {}
    gen = cfg.get("generative") or {}
    data = cfg.get("data") or {}
    time_dep = bool(gen.get("time_dependent", True))
    te = int(d.get("time_embed_dim", 32))
    return {
        "latent_dim": int(gen.get("latent_dim", 32)),
        "condition_dim": int(gcn.get("hidden_dim", 128)),
        "pred_len": int(data.get("pred_len", 12)),
        "lstm_hidden_dim": int(d.get("lstm_hidden_dim", 128)),
        "lstm_num_layers": int(d.get("lstm_num_layers", 1)),
        "time_dependent": time_dep,
        "time_embed_dim": te if time_dep else 0,
        "dropout": float(d.get("dropout", 0.0)),
    }


class TrajectoryDecoderLSTM(nn.Module):
    """
    Parameters
    ----------
    latent_dim, condition_dim :
        Sizes of :math:`z_0` and the extraction condition vector (``model.gcn.hidden_dim``).
    pred_len :
        :math:`T'` from ``data.pred_len``.
    lstm_hidden_dim, lstm_num_layers :
        LSTM hidden size and depth (``model.decoder.*``).
    time_dependent :
        If True and ``time_embed_dim > 0``, embed step indices ``1,\\ldots,T'``.
    time_embed_dim :
        Embedding width; if 0, no time embedding is concatenated.
    dropout :
        LSTM inter-layer dropout when ``lstm_num_layers > 1``.
    max_step_embeddings :
        ``nn.Embedding`` table size (clamp longer horizons).
    """

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        pred_len: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int = 1,
        *,
        time_dependent: bool = True,
        time_embed_dim: int = 32,
        dropout: float = 0.0,
        max_step_embeddings: int = 512,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.time_dependent = bool(time_dependent) and int(time_embed_dim) > 0
        te = int(time_embed_dim) if self.time_dependent else 0
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        h = int(lstm_hidden_dim)
        nl = int(lstm_num_layers)
        self.lstm_hidden_dim = h

        self.proj_z = nn.Linear(self.latent_dim, h)
        self.proj_y = nn.Linear(self.condition_dim, h)
        self.step_emb = nn.Embedding(max(1, max_step_embeddings), te) if te > 0 else None

        in_feats = 2 * h + te
        lstm_dropout = float(dropout) if nl > 1 else 0.0
        self.lstm = nn.LSTM(in_feats, h, num_layers=nl, batch_first=True, dropout=lstm_dropout)

    def forward(self, z: Tensor, condition: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z :
            Latent sample ``[B, latent_dim]``.
        condition :
            Encoding :math:`y^i` ``[B, condition_dim]`` (same layout as Phase 5 ``scene_encoding_to_condition``).

        Returns
        -------
        Tensor
            LSTM outputs (sequence features) ``[B, T', lstm_hidden_dim]``.
        """
        if z.dim() != 2 or z.size(-1) != self.latent_dim:
            raise ValueError(f"z must be [B, {self.latent_dim}], got {tuple(z.shape)}")
        if condition.dim() != 2 or condition.size(-1) != self.condition_dim:
            raise ValueError(
                f"condition must be [B, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        b = z.size(0)
        device = z.device
        pz = self.proj_z(z)
        py = self.proj_y(condition)
        base = torch.cat([pz, py], dim=-1)

        if self.step_emb is not None:
            tid = torch.arange(1, self.pred_len + 1, device=device, dtype=torch.long)
            tid = tid.clamp(max=self.step_emb.num_embeddings - 1)
            emb = self.step_emb(tid)
            emb_b = emb.unsqueeze(0).expand(b, -1, -1)
            base_b = base.unsqueeze(1).expand(-1, self.pred_len, -1)
            inp = torch.cat([base_b, emb_b], dim=-1)
        else:
            inp = base.unsqueeze(1).expand(-1, self.pred_len, -1)

        out, _ = self.lstm(inp)
        return out


def trajectory_decoder_from_cfg(cfg: Mapping[str, Any]) -> TrajectoryDecoderLSTM:
    kw = decoder_kwargs_from_cfg(dict(cfg))
    return TrajectoryDecoderLSTM(
        latent_dim=kw["latent_dim"],
        condition_dim=kw["condition_dim"],
        pred_len=kw["pred_len"],
        lstm_hidden_dim=kw["lstm_hidden_dim"],
        lstm_num_layers=kw["lstm_num_layers"],
        time_dependent=kw["time_dependent"],
        time_embed_dim=int(kw["time_embed_dim"]),
        dropout=kw["dropout"],
    )
