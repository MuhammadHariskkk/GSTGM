"""Config loading and CLI-style overrides."""

from __future__ import annotations

from pathlib import Path

from gstgm.utils.config import deep_merge, load_config, parse_dotted_overrides


def test_load_eth_extends_default(repo_root: Path) -> None:
    cfg = load_config(repo_root / "configs" / "eth.yaml")
    assert cfg["data"]["scene"] == "eth"
    assert cfg["model"]["gcn"]["hidden_dim"] == 128
    assert "experiment" in cfg


def test_parse_dotted_overrides_json() -> None:
    out = parse_dotted_overrides(["training.epochs=3", "training.seed=1"])
    assert out["training"]["epochs"] == 3
    assert out["training"]["seed"] == 1


def test_deep_merge_replaces_leaf() -> None:
    base = {"a": {"b": 1, "c": 2}, "x": 0}
    over = {"a": {"b": 9}}
    m = deep_merge(base, over)
    assert m["a"]["b"] == 9
    assert m["a"]["c"] == 2
