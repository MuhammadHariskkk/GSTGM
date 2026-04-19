# Implementation notes (GSTGM)

This document tracks **scientific fidelity** vs **engineering choices** across the repository.

## Legend

- **paper-specified:** Explicitly stated or equation-defined in the GSTGM paper (section/table cited when known).
- **recommended extension:** Not required for the baseline but useful for research.

## Phase 1 (skeleton)

| Topic | Classification | Notes |
|------|----------------|-------|
| `obs_len: 8`, `pred_len:12` | **paper-specified** | Khel et al. (2024) §5.1: 8 observed steps, 12 predicted (ETH/UCY @ 2.5 Hz). |
| YAML `extends: default.yaml` |   | Keeps scene configs DRY; merge order: optional `base_path` → extended file → primary file → `overrides`. |
| Graph similarity Eq. (2) on **velocities** | **paper-specified** | Khel et al. (2024) Eq. (2): `a_ij = 1/\|v_i-v_j\|^2` if `\|v_i-v_j\|^2 \neq 0`, else 0. Implementations: `gstgm/graph/kernels.py`. |
| `normalize_adjacency` (symmetric normalized `D^{-1/2} W D^{-1/2}`) | Standard GCN-style propagation; formula not given in the paper text (we follow a Kipf–Welling-style normalisation for stability). |
| `gmm.num_modes: 3` | **paper-specified** | Khel et al. (2024) §4.4: `M = 3` GMM modes. |
| `environment_channels: 0`  | Placeholder for scene context when the paper under-specifies external context. |
| `evaluation.multimodal.num_samples`  | Used for stochastic / best-of-sample evaluations (Phase 8 `scripts/evaluate.py`). |
| Deterministic CUDA off by default | Toggle via `training.deterministic_cuda`. |
| `CheckpointManager.metric_key` default `val_ade` | Aligns checkpoint selection with ADE-first ETH/UCY practice; trainer must log this key (Phase 7). |

## Phase 2 (data)

| Topic | Classification | Notes |
|------|----------------|-------|
| Raw 4-column text rows |   | Matches common ETH/UCY mirrors; confirm delimiter/order for your download (`gstgm/datasets/preprocessing.py` header). |
| Sliding windows over consecutive rows |   | Frames may be non-consecutive in time; windows use consecutive *rows* after sorting, not a fixed real-time step. |
| LOO split via `data.scene` |   | Matches one standard ETH/UCY protocol; requires a processed bundle with **all** evaluated scenes. |
| Neighbor cap + distance sort |   | Scene graph uses up to `max_neighbors` nearest agents per observation frame. |

## Phase 3 (graph)

| Topic | Classification | Notes |
|------|----------------|-------|
| Adjacency Eq. (2) on velocity differences | **paper-specified** | Khel et al. (2024): `a_ij = 1/\|v_i^t-v_j^t\|^2` if nonzero squared norm, else 0. |
| ``self_loop: false`` default | **paper-specified** for Eq. (2) | Diagonal zero from kernel; `self_loop: true` is an optional propagation extension. |
| Symmetric normalized adjacency | ``D^{-1/2} W D^{-1/2}`` not explicit in paper; used for stable GCN propagation. |
| Focal + padded neighbors ``N=1+K_max``  | Matches Phase 2 batch layout; masked pairs zero weight. |
| ``GraphConv`` = linear( A · H ) | Single-hop layer per call; ``GCNFeatureExtractor`` stacks ``model.gcn.num_layers`` in Phase 4. |
| ``graph.degree_eps`` | Degree clamp before ``rsqrt``. |

## Phase 4 (GCN extractor + spatial–temporal attention)

| Topic | Classification | Notes |
|------|----------------|-------|
| Eq. (3)–(4) spatial scores on velocities + `A_t` | **paper-specified** | ``gstgm/models/spatial_temporal_attention.py``; `gamma` multiplies **weighted** `adjacency_weighted` (pre-normalisation), not `adjacency_norm`. |
| Eq. (5)–(6) temporal dot-product | **paper-specified** | Dot-product between time-step embeddings. |
| Masked mean pooling for `h_t` in Eq. (5) | Paper: concatenate node encodings across time (ambiguous dimension with padding); we use valid-node mean → fixed `d`. |
| Eq. (7) indexing | **paper-specified** (resolved) | Residual = spatial neighbour mix at `t` + temporal mix over τ for the **same** node `i` (see module docstring). |
| `num_heads` unused in attention | YAML reserved; scalar scores only (§4.2). |
| GCN node input: position ‖ velocity [‖ env] | **paper-specified** (attributes) + **engineering** (layout) | §4.1 cites position, neighbours, environment; we stack focal+neighbour `(x,y)` and `(v_x,v_y)` from `SceneGraphBatch`; `environment_channels` is optional. |
| Multi-layer `GraphConv` + activation  | Depth/width from `model.gcn`. |

## Phase 5 (variational latent — §4.3 encoder + prior + sampling)

| Topic | Classification | Notes |
|------|----------------|-------|
| Eq. (8)–(9) split + ``exp`` on std segment | **paper-specified** | ``PriorNetwork``, ``GenerativeEncoder`` output `2 * latent_dim`; second half → :math:`\\sigma` via ``exp`` (``sigma_min`` floor is engineering). |
| Reparameterization + analytic KL | **paper-specified** (reparam) / standard | ``latent_sampler.py``; :math:`\\lambda_{KL}` scaling belongs in the trainer (§4.6 Eq. (20)). |
| MLP depth (2 Linear layers + ReLU) | Paper cites one FCL + ReLU; we use hidden + readout for stability. |
| ``scene_encoding_to_condition`` focal-last / mean-time | ** ** | Maps ``[B,T,N,d]`` → ``[B,d]``; focal index ``0`` matches Phase 2 layout. |
| ``generative.time_dependent`` | reserved | Decoder/LSTM (Phase 6) time conditioning; Phase 5 MLPs do not consume it. |

## Phase 6 (decoder + GMM)

| Topic | Classification | Notes |
|------|----------------|-------|
| LSTM + per-step conditioning (§4.3) | **paper-specified** (recurrent) + **engineering** (input layout) | ``TrajectoryDecoderLSTM``: concat :math:`\\psi_z(z)`, :math:`\\psi_x(y)`, optional step embedding when ``generative.time_dependent``. |
| GMM :math:`M=3`, bivariate normal / mode (§4.4) | **paper-specified** | ``MixtureVelocityHead`` outputs ``[B,T',M,2]`` means/scales and ``[B,T',M]`` logits. |
| ELU + linear per mode after LSTM | **paper-specified** (activation choice) | Two FCLs per mode with ELU then linear → 4 scalars. |
| ``softplus`` / ``sigma_floor`` on σ | ** ** | Paper lists :math:`\\sigma_{x},\\sigma_{y}`; positivity via softplus + floor (not ``exp``). |
| ``GSTGM`` forward | ** ** | One class wires graph → GCN → ST → VAE → decoder → GMM; losses in Phase 7+. |

## Phase 7 (training — §4.6, §5.2 metrics)

| Topic | Classification | Notes |
|------|----------------|-------|
| WTA mode from min position ADE (Eq. 16 spirit) | **paper-specified** | Mode index chosen ``no_grad``; gradients on winner velocities only. |
| Classification CE on ``π`` (Eq. 18) | **paper-specified** | Time-averaged ``pi_logits`` vs one-hot winner. |
| KL + annealing (Eq. 20) | **paper-specified** (structure) | ``training.kl_weight``, ``training.kl_anneal_epochs``; analytic KL from Phase 5. |
| Regression Laplace NLL (Eq. 17) | **paper-specified** (paper) / optional in code | Default ``regression_loss: gaussian_nll`` matches §4.4 Gaussian velocity head; ``laplace`` uses velocity-space heuristic. |
| Val ``val_ade`` / ``val_fde`` oracle min-mode | ** ** | Integrate mean velocities; min over modes then mean batch (multimodal convention). |
| Single-GPU trainer, CSV logs, checkpoints | ** ** | ``gstgm/training/trainer.py``, ``scripts/train.py``; ``CheckpointManager.metric_key`` = ``val_ade``. |

## Phase 8 (evaluation — §5.2)

| Topic | Classification | Notes |
|------|----------------|-------|
| ``ADE_M`` / ``FDE_M`` min over modes (Eq. 21–22) | **paper-specified** | :func:`gstgm.training.metrics.per_trajectory_oracle_ade_fde`; velocity means integrated from ``obs[:,-1]``. |
| Deterministic checkpoint eval | ** ** | ``sample_posterior=False``; mean latent. |
| Optional best-of-K latent draws | ** ** | ``scripts/evaluate.py --stochastic`` uses ``evaluation.multimodal.num_samples``; per trajectory, min over samples after oracle over modes. |
| Loader ``evaluation.batch_size`` / ``num_workers`` | ** ** | :func:`gstgm.training.trainer.build_dataloader` optional overrides; fall back to ``data.*``. |

## Phase 9 (documentation & install parity)

| Topic | Classification | Notes |
|------|----------------|-------|
| Root ``README.md`` (phases 1–8, layout, CLI) | ** ** | Handoff doc only; does not change model or training code paths. |
| ``requirements.txt`` comments vs ``setup.py`` | ** ** | Lists the same core packages as ``install_requires``; dev tools (pytest) also under ``extras_require["dev"]``. |
| ``docs/architecture.md`` Phase 9 updates | ** ** | Checklist paths corrected; mermaid includes Phase 8 eval alongside training for parity with the Phase 8 status line. |

## Phase 10 (automated smoke tests)

| Topic | Classification | Notes |
|------|----------------|-------|
| ``pytest`` + ``tests/*.py`` | ** ** | Fast regression checks (config merge, Eq. (2) kernel, graph shapes, collate, latent KL, oracle metrics, ``GSTGM`` forward) without ETH/UCY **files**; still imports real modules and merged ``configs/eth.yaml``. |
| ``pytest.ini`` (`testpaths`, `pythonpath`) | ** ** | Allows ``pytest`` from repo root without extra ``PYTHONPATH`` when combined with ``pip install -e .``. |

