"""Logging shared by every stage: readable on the console, durable on disk."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_CONSOLE_FMT = "%(asctime)s  %(levelname)-7s %(name)-22s %(message)s"
_FILE_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_dir: Path, level: int = logging.INFO) -> Path:
    """Configure root logging once. Returns the run's log file path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):  # make repeat calls idempotent
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FMT))
    root.addHandler(file_handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return log_file


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
