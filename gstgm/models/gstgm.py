"""
End-to-end GSTGM stack: graph extraction (Phase 3–4), variational latent (Phase 5), decoder + GMM (Phase 6).

Training forward returns predicted velocity mixture parameters plus variational outputs and KL
(:math:`\\mathrm{KL}(q\\|p)` per batch example) for use with the §4.6 ELBO-style objectives in a
trainer (Phase 7+).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn
from torch import Tensor

from gstgm.graph.graph_builder import build_from_collated_batch
from gstgm.models.decoder import TrajectoryDecoderLSTM, decoder_kwargs_from_cfg, trajectory_decoder_from_cfg
from gstgm.models.gcn_extractor import GCNFeatureExtractor, gcn_extractor_kwargs_from_cfg
from gstgm.models.generative_encoder import GenerativeEncoder, generative_encoder_from_cfg, scene_encoding_to_condition
from gstgm.models.gmm_head import MixtureVelocityHead, mixture_velocity_head_from_cfg
from gstgm.models.latent_sampler import kl_diagonal_normals, reparameterize_gaussian
from gstgm.models.prior_network import PriorNetwork, prior_network_from_cfg
from gstgm.models.spatial_temporal_attention import (
    SpatialTemporalAttention,
    attention_hyperparams_from_merged_cfg,
)


class GSTGM(nn.Module):
    """
    Phases 4–6 wired for ``collate_eth_ucy`` batches and merged YAML config dicts.

    Parameters stored in ``self.cfg_reference`` are the same mapping passed to ``build_from_collated_batch``.
    """

    def __init__(
        self,
        *,
        gcn: GCNFeatureExtractor,
        st_attn: SpatialTemporalAttention,
        encoder: GenerativeEncoder,
        prior: PriorNetwork,
        decoder: TrajectoryDecoderLSTM,
        gmm_head: MixtureVelocityHead,
        cfg: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.gcn = gcn
        self.st_attn = st_attn
        self.encoder = encoder
        self.prior = prior
        self.decoder = decoder
        self.gmm_head = gmm_head
        self.cfg_reference: Mapping[str, Any] = cfg

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        latent_eps: Tensor | None = None,
        sample_posterior: bool = True,
    ) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        batch :
            ``collate_eth_ucy`` dict with ``obs``, ``neighbor_pos``, ``neighbor_mask``, etc.
        latent_eps :
            Optional noise for reparameterization ``[B, latent_dim]``.
        sample_posterior :
            If True, draw :math:`z` with reparameterization; if False, use posterior mean :math:`\\mu_q`.

        Returns
        -------
        dict
            * ``pred_mu`` — ``[B, T', M, 2]``
            * ``pred_sigma`` — ``[B, T', M, 2]``
            * ``pi_logits`` — ``[B, T', M]``
            * ``posterior_mu``, ``posterior_sigma``, ``prior_mu``, ``prior_sigma``
            * ``z`` — latent used for decoding
            * ``kl`` — ``[B]`` :math:`\\mathrm{KL}(q\\|p)`
        """
        graph = build_from_collated_batch(batch, self.cfg_reference)
        x = self.gcn(graph)
        h = self.st_attn(x, graph.velocities, graph.adjacency_weighted, graph.node_mask)
        cond = scene_encoding_to_condition(h, graph.node_mask)

        mu_q, sigma_q = self.encoder(cond)
        mu_p, sigma_p = self.prior(cond)
        if sample_posterior:
            z = reparameterize_gaussian(mu_q, sigma_q, eps=latent_eps)
        else:
            z = mu_q

        dec_h = self.decoder(z, cond)
        pred_mu, pred_sigma, pi_logits = self.gmm_head(dec_h)
        kl = kl_diagonal_normals(mu_q, sigma_q, mu_p, sigma_p)

        return {
            "pred_mu": pred_mu,
            "pred_sigma": pred_sigma,
            "pi_logits": pi_logits,
            "posterior_mu": mu_q,
            "posterior_sigma": sigma_q,
            "prior_mu": mu_p,
            "prior_sigma": sigma_p,
            "z": z,
            "kl": kl,
        }


def gstgm_from_cfg(cfg: Mapping[str, Any]) -> GSTGM:
    """Construct :class:`GSTGM` from merged YAML (all Phase 6 sub-keys optional with code defaults)."""
    cfg_d = dict(cfg)
    dec_kw = decoder_kwargs_from_cfg(cfg_d)

    gkw = gcn_extractor_kwargs_from_cfg(cfg_d)
    gcn = GCNFeatureExtractor(
        in_channels=int(gkw["in_channels"]),
        hidden_dim=int(gkw["hidden_dim"]),
        num_layers=int(gkw["num_layers"]),
        activation=str(gkw["activation"]),
        environment_channels=int(gkw["environment_channels"]),
    )
    ghid = int(gkw["hidden_dim"])
    ah = attention_hyperparams_from_merged_cfg(cfg_d)
    st = SpatialTemporalAttention(
        feature_dim=ghid,
        d_model=int(ah["d_model"]),
        lambda_spatial=float(ah["lambda_spatial"]),
        gamma_adj=float(ah["gamma_adj"]),
        dropout=float(ah["dropout"]),
    )
    d_cond = int(dec_kw["condition_dim"])
    encoder = generative_encoder_from_cfg(cfg_d, d_cond)
    prior = prior_network_from_cfg(cfg_d, d_cond)
    decoder = trajectory_decoder_from_cfg(cfg_d)
    gmm = mixture_velocity_head_from_cfg(cfg_d, dec_kw["lstm_hidden_dim"])

    return GSTGM(
        gcn=gcn,
        st_attn=st,
        encoder=encoder,
        prior=prior,
        decoder=decoder,
        gmm_head=gmm,
        cfg=cfg_d,
    )
