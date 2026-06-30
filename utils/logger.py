"""Centralized logging setup. All modules call `get_logger(__name__)`."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(log_dir: str | Path = "logs", level: str = "INFO") -> None:
    """Idempotently configure the root logger with console + rotating file handlers."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_path / "fire_detection.log", maxBytes=10 * 1024 * 1024, backupCount=10
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    alarm_handler = RotatingFileHandler(
        log_path / "alarm_events.log", maxBytes=5 * 1024 * 1024, backupCount=20
    )
    alarm_handler.setFormatter(formatter)
    alarm_handler.addFilter(lambda record: record.name.startswith("alarm"))
    root.addHandler(alarm_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
