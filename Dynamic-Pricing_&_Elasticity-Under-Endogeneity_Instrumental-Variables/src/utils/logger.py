"""Central logging: ISO timestamps, colored console, rotating file in outputs/logs/."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "outputs" / "logs"

_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}
_RESET = "\033[0m"
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if sys.stderr.isatty():
            return f"{_COLORS.get(record.levelname, '')}{msg}{_RESET}"
        return msg


class _IsoFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):  # noqa: N802
        from datetime import datetime

        return datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds")


class _IsoColorFormatter(_ColorFormatter, _IsoFormatter):
    pass


def get_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to the console and (optionally) outputs/logs/<log_file>.

    Calling it again with the same name returns the same configured logger; a new
    log_file adds a further file handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(getattr(h, "_is_console", False) for h in logger.handlers):
        console = logging.StreamHandler(sys.stderr)
        console._is_console = True  # type: ignore[attr-defined]
        console.setFormatter(_IsoColorFormatter(_FMT))
        logger.addHandler(console)

    if log_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = str((LOG_DIR / log_file).resolve())
        if not any(getattr(h, "baseFilename", None) == path for h in logger.handlers):
            fh = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
            fh.setFormatter(_IsoFormatter(_FMT))
            logger.addHandler(fh)
    return logger
