"""Logging setup and lightweight training metrics logging (CSV/JSON later in trainer)."""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, TextIO


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
) -> None:
    """
    Configure root logging for console and optional file.

    Idempotent for handlers: clears existing handlers on root before adding new ones.
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(format_string)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (inherits root configuration from ``setup_logging``)."""
    return logging.getLogger(name)


@dataclass
class TrainingLogger:
    """
    Append training/validation metrics to CSV and JSON lines for downstream plotting.

    **Engineering assumption:** flat dict rows; nested values are JSON-serialized in CSV cells.
    """

    run_dir: Path
    csv_name: str = "metrics.csv"
    jsonl_name: str = "metrics.jsonl"
    _csv_fieldnames: list[str] = field(default_factory=list)
    _csv_file: Optional[TextIO] = None
    _csv_writer: Optional[Any] = None

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_csv(self, row: Mapping[str, Any]) -> None:
        path = self.run_dir / self.csv_name
        new_file = not path.exists()
        keys = sorted(row.keys())
        if new_file or self._csv_writer is None:
            self._csv_fieldnames = keys if new_file else sorted(set(self._csv_fieldnames) | set(keys))
            self._csv_file = open(path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fieldnames, extrasaction="ignore")
            if new_file:
                self._csv_writer.writeheader()

    def log_step(self, split: str, epoch: int, step: int, metrics: Mapping[str, Any]) -> None:
        """Log one row with split/epoch/step plus metrics."""
        row: MutableMapping[str, Any] = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "epoch": epoch,
            "step": step,
        }
        for k, v in metrics.items():
            if isinstance(v, (dict, list, tuple)):
                row[k] = json.dumps(v)
            else:
                row[k] = v
        self._ensure_csv(row)
        assert self._csv_writer is not None
        self._csv_writer.writerow(row)  # type: ignore[arg-type]
        if self._csv_file is not None:
            self._csv_file.flush()

        jsonl_path = self.run_dir / self.jsonl_name
        with jsonl_path.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps(dict(row), default=str) + "\n")

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
