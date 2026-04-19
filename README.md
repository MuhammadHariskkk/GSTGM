# GSTGM (PyTorch)

PyTorch reimplementation of **GSTGM** (*GSTGM: Graph, spatial–temporal attention and generative based model for pedestrian multi-path prediction*; Khel et al., *Image and Vision Computing* 151 (2024) 105245). The repo tracks the paper with an explicit phase roadmap; .

## Status

| Phase | Scope |
|-------|--------|
| 1 | Package skeleton, configs, YAML inheritance |
| 2 | ETH/UCY I/O, sliding windows, `EthUcyDataset`, collate |
| 3 | Scene-centric graph, message passing, kernels |
| 4 | GCN extractor, spatial–temporal attention (Eq. 2–7) |
| 5 | Posterior, prior, latent sampling (VAE path) |
| 6 | Decoder (LSTM), GMM head, `GSTGM` model, `gstgm_from_cfg` |
| 7 | Losses (NLL / KL), trainer, `scripts/train.py` |
| 8 | Oracle ADE/FDE evaluation, `scripts/evaluate.py` |
| 9 | Repository handoff: README, docs, `requirements.txt` alignment |
| 10 | Pytest smoke suite (`tests/`), `pytest.ini` — no raw ETH files required |

Optional tooling (multi-scene sweeps, CI, benchmarking) may follow; see `docs/implementation_notes.md`.

## Install

Python 3.10+ recommended.

```bash
pip install -e .
```

Optional development extras (e.g. pytest):

```bash
pip install -e ".[dev]"
```

Alternatively, from the repo root:

```bash
pip install -r requirements.txt
pip install -e .
```

(`requirements.txt` mirrors core runtime deps; `setup.py` remains the source of `install_requires`.)

## Repository layout

```
configs/           # default.yaml + scene YAMLs (extends)
data/              # raw/processed layout — see data/README.md
docs/              # architecture, implementation notes
experiments/       # nested experiment dirs (optional)
gstgm/
  datasets/        # ETH/UCY dataset, collate, preprocessing
  graph/           # graph builder, adjacency, GCN, kernels
  models/          # attention, latent stack, decoder, GMM, GSTGM
  training/        # losses, metrics, trainer
  evaluation/      # checkpoint eval runner
scripts/           # preprocess_data.py, train.py, evaluate.py
tests/             # pytest (Phase 10); run from repo root
```

## Tests (Phase 10)

Install dev extras, then from the repository root:

```bash
pip install -e ".[dev]"
pytest
```

The default suite uses **synthetic batches** and merged YAML under `configs/`; it does not read `data/raw` or processed `.pt` caches.

## Data (ETH/UCY)

1. Place raw `.txt` trajectories under `data/raw/<scene>/` (see `data/README.md`).
2. Preprocess to a cached tensor file:

```bash
python scripts/preprocess_data.py --config configs/default.yaml
```

3. Train or evaluate with a **scene config** that sets `data.scene` and `extends: default.yaml` (e.g. `configs/eth.yaml`).

## Training

```bash
python scripts/train.py --config configs/eth.yaml
```

By default the run directory is `experiment.output_dir` / `experiment.name` / a UTC timestamp (see merged YAML). To pin the directory:

```bash
python scripts/train.py --config configs/eth.yaml --run-dir experiments/my_run
```

Checkpoints are written in that directory as `checkpoint_last.pt` and `checkpoint_best.pt` (`gstgm.utils.checkpoint.CheckpointManager`). Device is chosen automatically (`cuda` if available, else `cpu`). Seed and CUDA determinism come from `training.seed` and `training.deterministic_cuda` in the config, or override at runtime with `--set`, e.g. `--set training.seed=123`.

Hyperparameters and loss weights live under `training.*` and `generative.*` in `configs/default.yaml`.

## Evaluation

Load a training checkpoint and report **oracle** min-mode ADE/FDE (paper-style multimodal metric):

```bash
python scripts/evaluate.py --config configs/eth.yaml --checkpoint experiments/my_run/checkpoint_best.pt
```

Optional stochastic rollouts for reported metrics:

```bash
python scripts/evaluate.py --config configs/eth.yaml --checkpoint experiments/my_run/checkpoint_best.pt --stochastic
```

See `configs/default.yaml` → `evaluation.multimodal.num_samples` when `--stochastic` is set.

## Configuration

- **`configs/default.yaml`** — shared keys: `data`, `graph`, `attention`, `model` (including `decoder`), `generative`, `gmm`, `training`, `evaluation`.
- **Scene files** — at minimum `extends: default.yaml` and scene-specific `data.scene`; `experiment.*` and other keys are usually inherited from `default.yaml` unless overridden.

Backward-compatible YAML aliases are documented in `docs/implementation_notes.md` (e.g. `attention.spatial.lambda` ↔ `lambda_spatial`, `gamma` ↔ `gamma_adj`).

## Paper

Equation and symbol mapping, engineering naming, and phase notes are in **`docs/implementation_notes.md`**. **`docs/architecture.md`**.

## Citation

If you use this code, cite the original paper "Khel, M. H. K., Greaney, P., McAfee, M., Moffett, S., & Meehan, K. (2024). GSTGM: Graph, Spatial–Temporal Attention and Generative Based Model for Pedestrian Multi-Path Prediction".

## License

See `LICENSE` if present in the repository root.
