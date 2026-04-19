"""
Spatial–temporal attention (GSTGM, Khel et al. 2024, §4.2, Eq. (3)–(7)).

**paper-specified**
    * Eq. (3): :math:`S_t(i,j)=\\exp(-\\lambda\\|v_i^t-v_j^t\\|^2+\\gamma A_t(i,j))`.
    * Eq. (4): :math:`\\alpha_t(i)=\\mathrm{softmax}_j S_t(i,j)`  (over neighbour indices ``j``, padded entries masked).
    * Eq. (5–6): :math:`M(t,t')=h_t^\\top h_{t'}`, :math:`\\beta_t(t')=\\mathrm{softmax}_{t'} M(t,t')`.
    * Eq. (7) (resolved indexing): refine node features by adding a **spatial** neighbour mix at time ``t``
      and a **temporal** mix over observation steps for the **same** node ``i``:

      .. math::

          \\tilde{h}_{t,i} = h_{t,i} + \\sum_j \\alpha_t(i,j)\\,h_{t,j}
          + \\sum_\\tau \\beta_t(\\tau)\\,h_{\\tau,i}

**engineering assumption**
    * :math:`h_t` in Eq. (5) is a **masked mean** over valid nodes at time ``t`` (fixed ``d``),
      so :math:`M` is :math:`T\\times T` with :math:`T` observation length. A full concatenation
      :math:`N\\cdot d` would be paper-ambiguous and batch-variable under padding.

``num_heads`` in YAML is **not used** here; the paper uses scalar scores. Heads remain reserved for
future multi-head extensions.

Tensor shapes are documented on ``forward``.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor


def _attention_subconfig(attn: Mapping[str, Any] | None) -> dict[str, float | int]:
    if not attn:
        attn = {}
    spat = attn.get("spatial") or {}
    temp = attn.get("temporal") or {}
    lam = spat.get("lambda_spatial", spat.get("lambda", 1.0))
    gam = spat.get("gamma_adj", spat.get("gamma", 1.0))
    return {
        "d_model": int(spat.get("d_model", temp.get("d_model", 128))),
        "dropout": float(spat.get("dropout", temp.get("dropout", 0.0))),
        "lambda_spatial": float(lam),
        "gamma_adj": float(gam),
    }


def attention_hyperparams_from_merged_cfg(cfg: Mapping[str, Any]) -> dict[str, float | int]:
    """Read ``cfg['attention']`` with defaults; supports legacy keys ``lambda`` / ``gamma``."""
    return _attention_subconfig(cfg.get("attention") if cfg else None)


class SpatialTemporalAttention(nn.Module):
    """
    One block: masked spatial softmax attention over ``j`` and temporal softmax over past time.

    Parameters
    ----------
    feature_dim :
        Channel dimension ``d`` of ``h`` (typically ``model.gcn.hidden_dim``).
    d_model :
        Must match ``feature_dim`` unless a projection is added; kept for config parity with YAML.
    lambda_spatial, gamma_adj :
        Eq. (3) scalars :math:`\\lambda, \\gamma` (fixed buffers; not trained).
    dropout :
        Applied to the **residual output** (engineering).
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int | None = None,
        *,
        lambda_spatial: float = 1.0,
        gamma_adj: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_in = int(feature_dim)
        d_cfg = int(d_model) if d_model is not None else d_in
        if d_cfg != d_in:
            raise ValueError(
                f"d_model ({d_cfg}) must equal feature_dim ({d_in}) for the scalar attention path; "
                "align `attention.spatial.d_model` with `model.gcn.hidden_dim`."
            )
        self.d_model = d_in
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.register_buffer("lambda_spatial", torch.tensor(float(lambda_spatial)))
        self.register_buffer("gamma_adj", torch.tensor(float(gamma_adj)))

    def forward(
        self,
        h: Tensor,
        velocities: Tensor,
        adjacency_weighted: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        h :
            Encoded node features ``[B, T, N, d]``.
        velocities :
            Planar velocities ``[B, T, N, 2]`` (same as graph construction / Eq. (3)).
        adjacency_weighted :
            **Pre-normalized** weights :math:`A_t` (Eq. (2) kernel, masking) ``[B, T, N, N]``.
        node_mask :
            Boolean or float ``[B, T, N]`` — padded nodes zero / False.

        Returns
        -------
        Tensor
            ``[B, T, N, d]`` refined features (Eq. (7) structure with residual).
        """
        if h.dim() != 4:
            raise ValueError(f"h must be [B, T, N, d], got {tuple(h.shape)}")
        b, t_obs, n, d = h.shape
        if velocities.shape != (b, t_obs, n, 2):
            raise ValueError(f"velocities must be [B, T, N, 2], got {tuple(velocities.shape)}")
        if adjacency_weighted.shape != (b, t_obs, n, n):
            raise ValueError(
                f"adjacency_weighted must be [B, T, N, N], got {tuple(adjacency_weighted.shape)}"
            )
        if node_mask.shape != (b, t_obs, n):
            raise ValueError(f"node_mask must be [B, T, N], got {tuple(node_mask.shape)}")

        mask = node_mask.bool() if node_mask.dtype != torch.bool else node_mask
        # Pair validity: both endpoints active (paper: neighbours j; we mask invalid j).
        m_i = mask.unsqueeze(-1)  # [B,T,N,1]
        m_j = mask.unsqueeze(-2)  # [B,T,1,N]
        pair_m = m_i & m_j  # [B,T,N,N]

        dv = velocities.unsqueeze(-2) - velocities.unsqueeze(-3)  # [B,T,N,N,2]
        d2 = (dv * dv).sum(dim=-1).clamp_min(0.0)  # [B,T,N,N]

        logits_s = -self.lambda_spatial * d2 + self.gamma_adj * adjacency_weighted
        logits_s = logits_s.masked_fill(~pair_m, torch.finfo(logits_s.dtype).min)
        # Invalid query nodes: all columns were masked to -inf → softmax would produce NaN.
        # Neutral logits give a finite uniform distribution; outputs are zeroed by ``node_mask`` below.
        row_ok = mask.unsqueeze(-1)  # [B,T,N,1]
        logits_s = torch.where(row_ok, logits_s, torch.zeros_like(logits_s))
        alpha = torch.softmax(logits_s, dim=-1)  # Eq. (4); equivalent to normalizing S=exp(logit) from Eq. (3)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        h_spatial = torch.matmul(alpha, h)  # [B,T,N,d]

        # Temporal: masked mean graph embedding h_t ∈ R^d (engineering; see module docstring)
        mask_f = mask.to(dtype=h.dtype)
        denom = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [B,T,1]
        h_mean = (h * mask_f.unsqueeze(-1)).sum(dim=-2) / denom  # [B,T,d]

        m_b = torch.matmul(h_mean, h_mean.transpose(-1, -2))  # [B,T,T] Eq. (5)
        beta = torch.softmax(m_b, dim=-1)  # [B,T,T] Eq. (6)
        h_temporal = torch.einsum("btu,bund->btnd", beta, h)  # Eq. (7) third term pattern

        out = h + h_spatial + h_temporal
        out = self.dropout(out)
        out = out * mask_f.unsqueeze(-1)
        return out
