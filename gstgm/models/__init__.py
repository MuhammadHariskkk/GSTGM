"""GSTGM learned modules: Phases 4–6 (encoder stack + variational latent + decoder/GMM)."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor

from gstgm.models.gcn_extractor import (
    GCNFeatureExtractor,
    gcn_extractor_kwargs_from_cfg,
)
from gstgm.models.decoder import (
    TrajectoryDecoderLSTM,
    decoder_kwargs_from_cfg,
    trajectory_decoder_from_cfg,
)
from gstgm.models.generative_encoder import (
    GenerativeEncoder,
    generative_encoder_from_cfg,
    generative_encoder_kwargs_from_cfg,
    scene_encoding_to_condition,
)
from gstgm.models.gmm_head import (
    MixtureVelocityHead,
    gmm_head_kwargs_from_cfg,
    mixture_velocity_head_from_cfg,
)
from gstgm.models.gstgm import GSTGM, gstgm_from_cfg
from gstgm.models.latent_sampler import (
    kl_diagonal_normals,
    reparameterize_gaussian,
    variational_latent_forward,
)
from gstgm.models.prior_network import (
    PriorNetwork,
    prior_network_from_cfg,
    prior_network_kwargs_from_cfg,
)
from gstgm.models.spatial_temporal_attention import (
    SpatialTemporalAttention,
    attention_hyperparams_from_merged_cfg,
)


def spatial_temporal_attention_from_cfg(cfg: Mapping[str, Any]) -> SpatialTemporalAttention:
    """
    Construct :class:`SpatialTemporalAttention` from merged YAML.

    ``attention.spatial.d_model`` must equal ``model.gcn.hidden_dim`` (default config satisfies this).
    """
    gh = int(gcn_extractor_kwargs_from_cfg(cfg)["hidden_dim"])
    ah = attention_hyperparams_from_merged_cfg(cfg)
    dm = int(ah["d_model"])
    return SpatialTemporalAttention(
        feature_dim=gh,
        d_model=dm,
        lambda_spatial=float(ah["lambda_spatial"]),
        gamma_adj=float(ah["gamma_adj"]),
        dropout=float(ah["dropout"]),
    )


def gcn_feature_extractor_from_cfg(cfg: Mapping[str, Any]) -> GCNFeatureExtractor:
    """Construct :class:`GCNFeatureExtractor` from merged YAML."""
    kw = gcn_extractor_kwargs_from_cfg(cfg)
    return GCNFeatureExtractor(
        in_channels=int(kw["in_channels"]),
        hidden_dim=int(kw["hidden_dim"]),
        num_layers=int(kw["num_layers"]),
        activation=str(kw["activation"]),
        environment_channels=int(kw["environment_channels"]),
    )


def latent_encoder_and_prior_from_cfg(
    cfg: Mapping[str, Any],
    input_dim: int | None = None,
) -> tuple[GenerativeEncoder, PriorNetwork]:
    """
    Build §4.3 encoder and prior with condition size ``input_dim`` (defaults to GCN ``hidden_dim``).
    """
    if input_dim is None:
        d = int(gcn_extractor_kwargs_from_cfg(cfg)["hidden_dim"])
    else:
        d = int(input_dim)
    return generative_encoder_from_cfg(cfg, d), prior_network_from_cfg(cfg, d)


def latent_step_training(
    encoder: GenerativeEncoder,
    prior: PriorNetwork,
    condition: Tensor,
    *,
    eps: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    One forward draw :math:`z_0 \\sim q` plus KL ``KL(q\\|p)`` for the same condition (§4.3 / §4.6).

    Returns
    -------
    z, kl, mu_q, sigma_q, mu_p, sigma_p
    """
    mu_q, sigma_q = encoder(condition)
    mu_p, sigma_p = prior(condition)
    z, kl = variational_latent_forward(mu_q, sigma_q, mu_p, sigma_p, eps=eps)
    return z, kl, mu_q, sigma_q, mu_p, sigma_p


__all__ = [
    "GCNFeatureExtractor",
    "GSTGM",
    "GenerativeEncoder",
    "MixtureVelocityHead",
    "PriorNetwork",
    "SpatialTemporalAttention",
    "TrajectoryDecoderLSTM",
    "attention_hyperparams_from_merged_cfg",
    "gcn_extractor_kwargs_from_cfg",
    "decoder_kwargs_from_cfg",
    "gcn_feature_extractor_from_cfg",
    "generative_encoder_from_cfg",
    "gmm_head_kwargs_from_cfg",
    "gstgm_from_cfg",
    "generative_encoder_kwargs_from_cfg",
    "kl_diagonal_normals",
    "mixture_velocity_head_from_cfg",
    "latent_encoder_and_prior_from_cfg",
    "latent_step_training",
    "prior_network_from_cfg",
    "prior_network_kwargs_from_cfg",
    "reparameterize_gaussian",
    "scene_encoding_to_condition",
    "spatial_temporal_attention_from_cfg",
    "trajectory_decoder_from_cfg",
    "variational_latent_forward",
]
