"""Load / save fitted estimators and metric files shared by the API, Streamlit app and pipelines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from src.utils.gpu_utils import ROOT, load_config

OUT = ROOT / "outputs"


def load_models() -> Dict[str, Any]:
    p = OUT / "models" / "causal_estimators.joblib"
    return joblib.load(p) if p.exists() else {}


def load_json(name: str) -> Dict[str, Any]:
    p = OUT / "metrics" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_benchmark() -> pd.DataFrame:
    p = OUT / "metrics" / "benchmark_summary.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_bookings() -> pd.DataFrame:
    return pd.read_parquet(OUT / "data" / "hospitality_bookings.parquet")
