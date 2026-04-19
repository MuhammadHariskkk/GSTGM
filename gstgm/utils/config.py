"""YAML configuration loading with nested dictionary overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Union

import yaml

ConfigDict = MutableMapping[str, Any]


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> ConfigDict:
    """
    Recursively merge ``overlay`` into a deep copy of ``base``.

    Nested dicts are merged; non-dict values in ``overlay`` replace those in ``base``.
    """
    result: ConfigDict = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: Union[str, Path]) -> ConfigDict:
    """Load a YAML file into a mutable dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, MutableMapping):
        raise TypeError(f"Root of YAML must be a mapping, got {type(data)}")
    return dict(data)


def save_config(config: Mapping[str, Any], path: Union[str, Path]) -> None:
    """Write config dict to YAML (sorted keys for readable diffs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            dict(config),
            f,
            default_flow_style=False,
            sort_keys=True,
            allow_unicode=True,
        )


def load_config(
    path: Union[str, Path],
    *,
    base_path: Union[str, Path, None] = None,
    overrides: Mapping[str, Any] | None = None,
) -> ConfigDict:
    """
    Load configuration from YAML.

    Parameters
    ----------
    path:
        Primary config file (e.g. ``configs/eth.yaml``).
    base_path:
        Optional base config merged first (e.g. ``configs/default.yaml``).
        **Paper / project convention:** scene YAMLs set ``extends`` to a default;
        callers may resolve ``extends`` before calling this, or pass ``base_path``
        explicitly for clarity.
    overrides:
        Optional nested dict merged last (e.g. from CLI --set key.subkey=value).

    Returns
    -------
    dict
        Fully merged configuration.
    """
    merged: ConfigDict = {}
    if base_path is not None:
        merged = deep_merge(merged, load_yaml(base_path))
    primary = load_yaml(path)
    # Support optional ``extends`` key in YAML: path relative to primary file's parent
    extends = primary.pop("extends", None)
    if extends:
        base_rel = Path(extends)
        if not base_rel.is_absolute():
            base_rel = Path(path).resolve().parent / base_rel
        merged = deep_merge(merged, load_yaml(base_rel))
    merged = deep_merge(merged, primary)
    if overrides:
        merged = deep_merge(merged, overrides)
    return merged


def parse_dotted_overrides(pairs: list[str]) -> ConfigDict:
    """
    Parse ``key.subkey=value`` strings into a nested dict for ``load_config(..., overrides=)``.

    Values are interpreted as JSON when possible (booleans, numbers, null, lists);
    otherwise kept as strings.
    """
    import json

    out: ConfigDict = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Override must contain '=': {raw!r}")
        key_path, value_str = raw.split("=", 1)
        key_path = key_path.strip()
        keys = key_path.split(".")
        if not keys or any(not segment for segment in keys):
            raise ValueError(f"Invalid override key path (empty segment): {raw!r}")
        try:
            value: Any = json.loads(value_str)
        except json.JSONDecodeError:
            value = value_str
        cursor: ConfigDict = out
        for k in keys[:-1]:
            if k not in cursor or not isinstance(cursor[k], MutableMapping):
                cursor[k] = {}
            cursor = cursor[k]  # type: ignore[assignment]
        cursor[keys[-1]] = value
    return out
