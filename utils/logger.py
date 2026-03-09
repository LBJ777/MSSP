"""
utils/logger.py
---------------
Structured logging configuration for MSSP.
Copied from DRIFT_new for standalone operation.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "MSSP",
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = "mssp.log",
    use_console: bool = True,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Configure and return the root MSSP logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if use_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_dir is not None and log_filename is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Log file: %s", log_path)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve a child logger under the MSSP namespace."""
    return logging.getLogger(f"MSSP.{name}")
