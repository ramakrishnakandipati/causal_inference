"""ASCII-safe logger (the Windows console is often cp1252)."""
from __future__ import annotations

import logging
import sys

from src.utils.config import resolve

_FMT = "%(asctime)s | %(levelname)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "geo", log_file: str | None = "outputs/logs/pipeline.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    logger.addHandler(sh)

    if log_file:
        path = resolve(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        logger.addHandler(fh)
    return logger
