"""Config loading and project paths."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiment_config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # YAML int keys are preserved; make sure tier keys are ints everywhere
    for sec, key in (("economics", "ota_commission_by_tier"), ("economics", "branded_cpc"),
                     ("experiment", "tier_shares")):
        cfg[sec][key] = {int(k): float(v) for k, v in cfg[sec][key].items()}
    return cfg


def path(name: str) -> Path:
    p = ROOT / load_config()["paths"][name]
    p.mkdir(parents=True, exist_ok=True)
    return p
