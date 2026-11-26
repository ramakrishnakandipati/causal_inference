"""Economics layer: cannibalization, iROAS vs last-click ROAS, OTA commission, and the bidding policy.

Sign convention: every ``tau`` is the effect of PAUSING branded paid search (T=1) versus keeping it.
Value of branded search = -ATT_holdout.

    NCM = Direct GBV - Branded SEM spend + OTA GBV x (1 - commission rate)
    d NCM (pause) = d DirectGBV + d OTA_GBV x (1 - rate) + spend saved
    Policy: PAUSE iff predicted d NCM (pause) > threshold, else KEEP.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.utils.config import load_config

NIGHTS = 2.4
OTA_ADR_FACTOR = 0.95
DEFAULT_CVR = 0.025


# ----------------------------------------------------------------------------- programme level
def programme_economics(panel: pd.DataFrame, dma: pd.DataFrame, att_direct: float, att_ota: float,
                        att_direct_ci=None, att_ota_ci=None, label: str = "") -> dict:
    """Roll DMA-day ATTs (bookings) up to the 4-week holdout programme in dollars."""
    tr_ids = dma.loc[dma["T"] == 1, "dma_id"].values
    n_tr, n_days = len(tr_ids), int(panel.post.sum() / panel.dma_id.nunique())
    pre_tr = panel[(panel.post == 0) & panel.dma_id.isin(tr_ids)]
    g_d = pre_tr.direct_gbv.sum() / pre_tr.direct_bookings.sum()
    g_o = pre_tr.ota_gbv.sum() / pre_tr.ota_bookings.sum()
    rate = (pre_tr.ota_commission.sum() / pre_tr.ota_gbv.sum())
    spend_day = pre_tr.groupby("dma_id").branded_spend.mean().sum()          # $/day across treated DMAs
    lastclick_day = pre_tr.groupby("dma_id").lastclick_bookings.mean().sum()
    scale = n_tr * n_days

    def block(ad, ao):
        direct_lost = -ad * scale                                              # incremental direct bookings from keeping
        ota_gain = ao * scale
        spend = spend_day * n_days
        lastclick = lastclick_day * n_days
        inc_direct_gbv = direct_lost * g_d
        ota_commission = ota_gain * g_o * rate
        d_ncm_pause = -inc_direct_gbv + ota_gain * g_o * (1 - rate) + spend
        return {
            "incremental_direct_bookings_from_keeping": direct_lost,
            "ota_leakage_bookings_when_paused": ota_gain,
            "ota_commission_leakage_usd": ota_commission,
            "incremental_direct_gbv_usd": inc_direct_gbv,
            "branded_spend_avoided_usd": spend,
            "lastclick_bookings": lastclick,
            "cannibalization_rate": 1 - direct_lost / lastclick if lastclick else np.nan,
            "last_click_roas": lastclick * g_d / spend if spend else np.nan,
            "iroas": inc_direct_gbv / spend if spend else np.nan,
            "iroas_incl_ota_diversion": (inc_direct_gbv - ota_gain * g_o * (1 - rate)) / spend if spend else np.nan,
            "net_margin_change_if_paused_usd": d_ncm_pause,
            "net_value_of_branded_search_usd": -d_ncm_pause,
            # commission "saved" by keeping branded search = commission not paid on diverted OTA bookings
            "net_commission_saved_by_keeping_usd": ota_commission,
        }
    out = {"label": label, "n_treated_dmas": n_tr, "n_days": n_days, "gbv_per_direct_booking": g_d,
           "gbv_per_ota_booking": g_o, "commission_rate": rate, **block(att_direct, att_ota)}
    if att_direct_ci is not None and att_ota_ci is not None:
        lo = block(att_direct_ci[0], att_ota_ci[1])      # worst case for the direction below
        hi = block(att_direct_ci[1], att_ota_ci[0])
        for k in ("cannibalization_rate", "iroas", "net_margin_change_if_paused_usd"):
            out[f"{k}_range"] = sorted([lo[k], hi[k]])
    return out


# ----------------------------------------------------------------------------- DMA policy
def policy_table(dma: pd.DataFrame, lift_direct: np.ndarray, lift_ota: np.ndarray, threshold: float | None = None,
                 horizon_days: int = 28, margin_day: np.ndarray | float = 0.0) -> pd.DataFrame:
    """Per-DMA decision from predicted *relative* lifts (pre-period baseline units).

    ``margin_day`` ($/day) is an evidence margin added to the threshold: pause only when the predicted
    gain exceeds it (e.g. k x the disagreement between CATE learners).
    """
    thr = load_config()["economics"]["policy_margin_threshold"] if threshold is None else threshold
    d = dma.copy()
    g_d, g_o = d.adr * NIGHTS, d.adr * OTA_ADR_FACTOR * NIGHTS
    d["pred_delta_direct_bookings_day"] = lift_direct * d.pre_direct_mean
    d["pred_delta_ota_bookings_day"] = lift_ota * d.pre_ota_mean
    d["pred_direct_gbv_change_day"] = d.pred_delta_direct_bookings_day * g_d
    d["pred_commission_leakage_day"] = d.pred_delta_ota_bookings_day * g_o * d.commission_rate
    d["spend_saved_day"] = d.pre_spend_mean
    d["pred_ncm_change_if_paused_day"] = (d.pred_direct_gbv_change_day + d.pred_delta_ota_bookings_day * g_o * (1 - d.commission_rate)
                                          + d.spend_saved_day)
    d["evidence_margin_day"] = margin_day
    d["decision"] = np.where(d.pred_ncm_change_if_paused_day > thr + d["evidence_margin_day"], "Pause Branded SEM", "Keep Branded SEM")
    d["pred_ncm_change_if_paused_period"] = d.pred_ncm_change_if_paused_day * horizon_days
    d["true_ncm_change_if_paused_day"] = d.true_post_tau_ncm
    d["oracle_decision"] = np.where(d.true_post_tau_ncm > thr, "Pause Branded SEM", "Keep Branded SEM")
    return d


def gain_per_day(dma: pd.DataFrame, lift_direct, lift_ota) -> np.ndarray:
    """Predicted $/day change in net contribution margin if paused, from relative lifts."""
    g_d, g_o = dma.adr * NIGHTS, dma.adr * OTA_ADR_FACTOR * NIGHTS
    return (lift_direct * dma.pre_direct_mean * g_d + lift_ota * dma.pre_ota_mean * g_o * (1 - dma.commission_rate)
            + dma.pre_spend_mean).values


def policy_value(d: pd.DataFrame, horizon_days: int = 28) -> dict:
    """Realised value (using ground-truth effects) of the model policy vs always-keep/always-pause/oracle."""
    t = d.true_post_tau_ncm.values * horizon_days
    pause = (d.decision == "Pause Branded SEM").values
    return {"model_policy_usd": float(t[pause].sum()), "always_keep_usd": 0.0, "always_pause_usd": float(t.sum()),
            "oracle_usd": float(t[t > 0].sum()), "n_pause_model": int(pause.sum()),
            "n_pause_oracle": int((t > 0).sum()),
            "decision_accuracy": float((pause == (t > 0)).mean())}


# ----------------------------------------------------------------------------- structural simulator
@dataclass
class SimInputs:
    baseline_direct_bookings_day: float = 25.0
    ota_to_direct_ratio: float = 0.6
    lastclick_share: float = 0.30           # share of direct bookings last-click credited to branded paid
    cannibalization_rate: float = 0.75      # share of those that would arrive via free organic anyway
    ota_capture_rate: float = 0.20          # share of organic-bound clickers an OTA hijacks if paused
    diversion_rate: float = 0.45            # share of truly lost bookings that re-book on an OTA
    adr: float = 205.0
    commission_rate: float = 0.175
    cpc: float = 2.2
    cvr: float = DEFAULT_CVR
    horizon_days: int = 28


def simulate_policy(p: SimInputs | dict) -> dict:
    p = SimInputs(**p) if isinstance(p, dict) else p
    g_d, g_o = p.adr * NIGHTS, p.adr * OTA_ADR_FACTOR * NIGHTS
    lc = p.baseline_direct_bookings_day * p.lastclick_share                       # last-click bookings/day
    clicks = lc / p.cvr
    spend = clicks * p.cpc
    c, q, v = p.cannibalization_rate, p.ota_capture_rate, p.diversion_rate
    d_direct = -lc * ((1 - c) + c * q)
    d_ota = lc * ((1 - c) * v + c * q)
    direct_gbv_change = d_direct * g_d
    ota_margin_gain = d_ota * g_o * (1 - p.commission_rate)
    d_ncm = direct_gbv_change + ota_margin_gain + spend
    H = p.horizon_days
    # break-even cannibalization c* where d_ncm = 0 (linear in c)
    a = lc * (1 - q) * g_d + lc * (q - v) * g_o * (1 - p.commission_rate)       # d_ncm slope in c (excl. spend)
    b0 = -lc * g_d + lc * v * g_o * (1 - p.commission_rate)
    c_star = (-spend - b0) / a if abs(a) > 1e-12 else float("nan")
    inc_gbv = -d_direct * g_d
    return {
        "inputs": asdict(p), "horizon_days": H,
        "lastclick_bookings_period": lc * H, "incremental_direct_bookings_period": -d_direct * H,
        "ota_leakage_bookings_period": d_ota * H,
        "ota_commission_leakage_usd": d_ota * g_o * p.commission_rate * H,
        "branded_spend_usd": spend * H,
        "last_click_roas": lc * g_d / spend if spend else float("nan"),
        "iroas": inc_gbv / spend if spend else float("nan"),
        "cannibalization_rate": c,
        "net_margin_change_if_paused_usd": d_ncm * H,
        "net_commission_saved_by_keeping_usd": d_ota * g_o * p.commission_rate * H,
        "decision": "Pause Branded SEM" if d_ncm > 0 else "Keep Branded SEM",
        "break_even_cannibalization": float(np.clip(c_star, -1, 2)),
    }


def archetype_inputs(dma: pd.DataFrame, tier: int) -> SimInputs:
    """Calibrate simulator inputs from the DMA table's tier averages (DGP-known covariates, not truth)."""
    d = dma[dma.tier == tier]
    return SimInputs(baseline_direct_bookings_day=float(d.pre_direct_mean.mean()),
                     ota_to_direct_ratio=float((d.pre_ota_mean / d.pre_direct_mean).mean()),
                     lastclick_share=float(d.lastclick_share.mean()), adr=float(d.adr.mean()),
                     commission_rate=float(d.commission_rate.mean()), cpc=float(d.cpc.mean()))
