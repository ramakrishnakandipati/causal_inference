"""Shared constants / helpers (paths, config, feature schema, logging)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES = [
    "hist_booking_freq", "days_since_last_stay", "loyalty_tier", "hist_price_sensitivity",
    "avg_lead_time", "avg_adr", "weekend_stay_ratio", "cancellation_rate",
    "email_open_rate_90d", "points_balance",
]
LOYALTY_LEVELS = ["Non-Member", "Silver", "Gold", "Platinum"]
PERSONAS = ["Persuadable", "Always-Taker", "Lost Cause", "Sleeping Dog", "Neutral"]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def get_logger(name: str = "uplift") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def save_json(obj, path: str | Path) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
