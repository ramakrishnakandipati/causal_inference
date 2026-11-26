"""Config loader and path helpers."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def out_dir(sub: str = "") -> Path:
    base = Path(os.environ["CAUSAL_OUT"]) if os.environ.get("CAUSAL_OUT") else ROOT / load_config()["paths"]["outputs"]
    p = base / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def shot_dir() -> Path:
    p = ROOT / load_config()["paths"]["screenshots"]
    p.mkdir(parents=True, exist_ok=True)
    return p
