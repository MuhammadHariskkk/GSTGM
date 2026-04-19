# Data layout (ETH/UCY)

## Raw trajectories

Place files under ``data/raw/<scene>/`` (recommended) or ``data/raw/<scene>.txt``.

**Expected format (Phase 2):** ASCII ``.txt``, one row per line, **four whitespace-separated fields**:

```text
<frame_id> <pedestrian_id> <x> <y>
```

* Lines starting with ``#`` and blank lines are ignored.
* Scene ids match ``configs/*`` ``data.scene`` values: ``eth``, ``hotel``, ``univ``, ``zara1``, ``zara2``.
* Alternate folder names ``zara01`` / ``zara02`` are accepted when resolving paths.

## Preprocessed cache

Run from the repository root (after ``pip install -e .``):

```bash
python scripts/preprocess_data.py --config configs/default.yaml
```

This writes ``data/processed/eth_ucy_windows_obs8_pred12.pt`` (name derives from ``data.processed_basename`` and horizons). The cache is **gitignored** under ``data/processed/``.

* Coordinate modes (``absolute``, ``relative_disp``, ``velocity``) are applied in ``EthUcyDataset``, not in the preprocessor, so one cache supports all modes.
* Leave-one-scene-out training uses ``data.scene`` from a scene config (e.g. ``configs/eth.yaml``) as the held-out test scene; the cache must include **all** scenes you plan to evaluate.
