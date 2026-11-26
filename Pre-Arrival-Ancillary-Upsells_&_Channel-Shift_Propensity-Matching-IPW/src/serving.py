"""Shared scoring / optimisation logic used by BOTH the FastAPI service and the Streamlit app."""
from __future__ import annotations

import json
import re
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.common.config import load_config, out_dir
from src.data import synthetic_generator as sg

PERSONAS = {
    "Expedia family, Suite, 60-day lead (iOS)": dict(channel="OTA_Expedia", room_tier="Suite", lead_time=60, adults=2, children=2,
        device="iOS", hist_spend=420.0, loyalty_tier=1, los=5, resort=1),
    "Booking.com couple, Deluxe, 35-day lead (Android)": dict(channel="OTA_Booking", room_tier="Deluxe", lead_time=35, adults=2, children=0,
        device="Android", hist_spend=150.0, loyalty_tier=0, los=3, resort=1),
    "Budget OTA family, Standard room, last-minute": dict(channel="OTA_Booking", room_tier="Standard", lead_time=4, adults=2, children=3,
        device="Desktop", hist_spend=0.0, loyalty_tier=0, los=2, resort=0),
    "Direct web Platinum member, Suite": dict(channel="Direct_Web", room_tier="Suite", lead_time=45, adults=2, children=0,
        device="iOS", hist_spend=1800.0, loyalty_tier=3, los=4, resort=1),
    "Corporate solo traveller, Standard, 6-day lead": dict(channel="Corporate", room_tier="Standard", lead_time=6, adults=1, children=0,
        device="Desktop", hist_spend=260.0, loyalty_tier=2, los=2, resort=0),
    "Expedia solo, Deluxe, 90-day lead, resort (Desktop)": dict(channel="OTA_Expedia", room_tier="Deluxe", lead_time=90, adults=1, children=0,
        device="Desktop", hist_spend=60.0, loyalty_tier=0, los=6, resort=1),
}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _read_json(name):
    p = out_dir("metrics") / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def campaign_summary():
    return _read_json("campaign_summary.json")


def best_model(outcome: str) -> str:
    return campaign_summary()["best_model"][outcome]


def model_names(outcome: str) -> list[str]:
    lb = pd.read_csv(out_dir("metrics") / "cate_leaderboard.csv")
    return [m for m in lb[(lb.outcome == outcome) & ~lb.model.str.startswith("Oracle")].model]


@lru_cache(maxsize=64)
def load_model(outcome: str, name: str):
    base = out_dir("models") / f"{outcome}__{slug(name)}"
    if base.with_suffix(".pt").exists():
        from src.models.deep_causal import load_deep
        return load_deep(base.with_suffix(".pt"))
    return joblib.load(base.with_suffix(".joblib"))


def predict_guests(guests: pd.DataFrame, model_anc: str | None = None, model_dir: str | None = None) -> pd.DataFrame:
    """CATE for raw guest rows -> DataFrame (tau_ancillary, tau_direct, intervals if available, value, segment)."""
    b = load_config()["business"]
    ma, md = model_anc or best_model("ancillary"), model_dir or best_model("direct")
    X = sg.build_features(guests)
    res = pd.DataFrame(index=guests.index)
    for tag, o, name in (("ancillary", "ancillary", ma), ("direct", "direct", md)):
        m = load_model(o, name)
        res[f"tau_{tag}"] = m.predict_cate(X)
        if name.startswith("Causal Forest"):
            lo, hi = m.predict_interval(X)
            res[f"tau_{tag}_lo"], res[f"tau_{tag}_hi"] = lo, hi
    res["incremental_value"] = b["ancillary_margin"] * res["tau_ancillary"] + b["direct_booking_margin"] * res["tau_direct"] - b["cost_per_offer"]
    res["segment_pred"] = [classify(v, d, a) for v, d, a in zip(res.incremental_value, res.tau_direct, res.tau_ancillary)]
    return res


def classify(value, tau_dir, tau_anc) -> str:
    """Model-based (not truth-based) action segment."""
    if tau_dir <= -0.015 or (value < 0 and tau_dir < 0.02 and tau_anc < 0):
        return "Sleeping Dog"
    if tau_dir >= 0.08 or value >= 12:
        return "Persuadable"
    if value > 0:
        return "Marginal responder"
    return "Low response (Always-Taker / Lost Cause)"


def optimize_campaign(budget_frac: float, cost_per_offer: float, ancillary_margin: float, direct_margin: float,
                      only_positive: bool = True, channels: list[str] | None = None) -> dict:
    sc = pd.read_parquet(out_dir("data") / "scored_guests.parquet")
    if channels:
        sc = sc[sc.channel.isin(channels)].reset_index(drop=True)
    n = len(sc)
    val = ancillary_margin * sc["tau_hat_ancillary"] + direct_margin * sc["tau_hat_direct"] - cost_per_offer
    true_val = ancillary_margin * sc["tau_anc"] + direct_margin * sc["tau_direct"] - cost_per_offer
    k = int(round(budget_frac * n))
    order = np.argsort(-val.values, kind="stable")
    sel = order[:k]
    if only_positive:
        sel = sel[val.values[sel] > 0]
    mask = np.zeros(n, bool); mask[sel] = True
    rng = np.random.default_rng(0)
    rand_sel = rng.choice(n, len(sel), replace=False) if len(sel) else np.array([], int)
    oracle_sel = np.argsort(-true_val.values, kind="stable")[: len(sel)]
    spend = cost_per_offer * len(sel)
    def tot(idx): return float(true_val.values[idx].sum() + cost_per_offer * len(idx))  # gross margin, before cost
    out = dict(n_guests=n, n_selected=int(len(sel)), share_selected=float(len(sel) / max(n, 1)), spend=float(spend),
               est_net_value=float(val.values[sel].sum()),
               true_net_value=float(true_val.values[sel].sum()),
               random_net_value=float(true_val.values[rand_sel].sum()),
               oracle_net_value=float(true_val.values[oracle_sel].sum()),
               treat_all_net_value=float(true_val.sum()),
               roi_true=float((true_val.values[sel].sum() + spend) / spend - 1) if spend > 0 else None,
               gross_true=tot(sel))
    s = sc[mask]
    out["by_channel"] = (s.groupby("channel").agg(n=("channel", "size"), est_value=("tau_hat_ancillary", "size")).drop(columns="est_value").assign(
        share_of_channel=lambda d: d["n"] / sc.groupby("channel").size().reindex(d.index)).reset_index().to_dict("records"))
    out["by_true_segment"] = s.groupby("segment").size().rename("n").reset_index().to_dict("records")
    out["mask"] = mask
    return out
