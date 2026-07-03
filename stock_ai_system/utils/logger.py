"""Small logging helper."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config import LOG_PATH


def get_logger(name: str = "stock_ai_system") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.addHandler(handler)
    return logger
