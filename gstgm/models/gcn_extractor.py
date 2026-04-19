"""
GCN-based feature extractor before generative modeling (GSTGM, Khel et al. 2024, §4.1).

The paper states the scene-centric GCN uses pedestrian **position**, **neighbours**, and
**environment** attributes. We encode node inputs as concatenated **planar position** and
**planar velocity** tensors from :class:`gstgm.graph.graph_builder.SceneGraphBatch`, plus
optional ``environment_channels`` (broadcast or per-node).

**paper-specified (scope):** message passing along :math:`\\tilde{A}_t` built from Eq. (2) weights
(see Phase 3 ``adjacency_norm``).

**engineering assumption:** multi-layer ``GraphConv`` (``model.gcn``) with chosen activation;
exact layer count and width follow YAML, not a closed-form in the paper body.

Tensor shapes are documented on ``forward``.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from gstgm.graph.graph_builder import SceneGraphBatch
from gstgm.graph.message_passing import GraphConv


def _gcn_yaml(cfg: Mapping[str, Any]) -> dict[str, Any]:
    m = cfg.get("model") or {}
    g = m.get("gcn") or {}
    return {
        "hidden_dim": int(g.get("hidden_dim", 128)),
        "num_layers": int(g.get("num_layers", 2)),
        "activation": str(g.get("activation", "relu")).lower(),
        "environment_channels": int(m.get("environment_channels", 0)),
    }


def gcn_extractor_kwargs_from_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """kwargs for :class:`GCNFeatureExtractor` from merged training config."""
    y = _gcn_yaml(cfg)
    c = int(y["environment_channels"])
    return {
        "hidden_dim": y["hidden_dim"],
        "num_layers": y["num_layers"],
        "activation": y["activation"],
        "environment_channels": c,
        "in_channels": 4 + c,
    }


class GCNFeatureExtractor(nn.Module):
    """
    Stack ``num_layers`` graph convolutions on ``[B, T, N, F_in]`` features.

    Parameters
    ----------
    in_channels :
        ``4 + environment_channels`` (xy position + xy velocity [+ env]).
    hidden_dim :
        Output channels per layer (last layer also ``hidden_dim``).
    num_layers :
        Number of ``GraphConv`` applications.
    activation :
        ``relu`` | ``gelu`` | ``tanh``.
    environment_channels :
        Expected trailing dim of ``environment`` when > 0.
    dropout :
        Dropout after **intermediate** activations (engineering).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 2,
        activation: str = "relu",
        environment_channels: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.environment_channels = int(environment_channels)
        self.hidden_dim = int(hidden_dim)
        if int(in_channels) != 4 + self.environment_channels:
            raise ValueError(
                f"in_channels ({in_channels}) must be 4 + environment_channels ({self.environment_channels})."
            )

        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        acts: dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        Act = acts.get(activation)
        if Act is None:
            raise ValueError(f"Unknown activation: {activation}")

        convs: list[GraphConv] = []
        for i in range(num_layers):
            fin = in_channels if i == 0 else hidden_dim
            fout = hidden_dim
            convs.append(GraphConv(fin, fout))
        self.convs = nn.ModuleList(convs)
        self.act = Act()

    def forward(
        self,
        graph: SceneGraphBatch,
        environment: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        graph :
            Phase-3 batch (positions, velocities, ``adjacency_norm``, masks, …).
        environment :
            If ``environment_channels > 0``, supply ``[B, T, N, C_env]`` or ``[B, T, C_env]``
            (the latter broadcasts across ``N``).

        Returns
        -------
        Tensor
            ``[B, T, N, hidden_dim]`` node features.
        """
        pos = graph.positions
        vel = graph.velocities
        adj = graph.adjacency_norm
        if pos.shape != vel.shape:
            raise ValueError(f"positions and velocities shape mismatch: {pos.shape} vs {vel.shape}")

        parts: list[Tensor] = [pos, vel]
        c = self.environment_channels
        if c > 0:
            if environment is None:
                raise ValueError(f"environment_channels={c} but environment is None")
            env = environment
            if env.dim() == 4:
                if env.shape[:-1] != pos.shape[:-1] or env.shape[-1] != c:
                    raise ValueError(
                        f"environment must be [B,T,N,{c}] matching positions; got {tuple(env.shape)}"
                    )
            elif env.dim() == 3:
                if env.shape[:2] != pos.shape[:2] or env.shape[-1] != c:
                    raise ValueError(
                        f"environment must be [B,T,{c}] for broadcast; got {tuple(env.shape)}"
                    )
                env = env.unsqueeze(-2).expand(-1, -1, pos.size(2), -1)
            else:
                raise ValueError(f"environment must be [B,T,N,C] or [B,T,C]; got {tuple(env.shape)}")
            parts.append(env)

        x = torch.cat(parts, dim=-1)
        if x.shape[-1] != 4 + c:
            raise ValueError(f"concat feature dim {x.shape[-1]} != expected {4 + c}")

        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = self.drop(x)

        return x
