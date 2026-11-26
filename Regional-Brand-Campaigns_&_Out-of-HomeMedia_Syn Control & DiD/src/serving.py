"""Serving layer shared by the FastAPI service and the Streamlit workbench (so both always agree).

* Panel ATT: served from the pipeline's cached results; fast estimators (TWFE, CS-DiD, SCM, ASCM, SDID, event
  study) can also be re-run *live* for any preset or custom donor pool. Bayesian SC is cache-only.
* Geo-CATE: CPU copies of the trained learners score market-day covariates.
* Campaign simulator: fitted adstock/Hill response curve x CATE market-condition multiplier -> pacing what-ifs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.evaluation import diagnostics as dg
from src.models import panel_att as pa
from src.models import response_curve as rc
from src.models.cate_learners import LEARNER_ORDER
from src.utils import device as dev
from src.utils.config import load_config, read_json, resolve

PRESET_KEYS = ["mia_flagship", "las_convention", "nyc_spring_break", "tpa_placebo"]
OUTCOME_LABELS = {"room_nights_booked": "Room nights booked", "direct_web_sessions": "Direct web sessions",
                  "branded_search_index": "Branded search index", "incremental_net_revenue_usd": "Net revenue (USD)"}
FAST = ["TWFE DiD", "Event Study", "CS-DiD (DR)", "SCM", "ASCM (Ridge)", "SDID"]
STRATEGIES = ["even", "front_loaded", "back_loaded", "pulsed_7_on_7_off", "holiday_weighted", "weekend_weighted",
              "cate_weighted", "optimized"]


def _fname(learner: str) -> str:
    return "cate_" + learner.lower().replace(" ", "_").replace("-", "_") + ".joblib"


class Engine:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        dev.set_gpu_enabled(False)                      # serve on CPU: GPU inference from server threads is not worth the risk
        self.mdir = resolve(self.cfg["paths"]["metrics"])
        self.odir = resolve(self.cfg["paths"]["models"])
        self._panels: dict[str, pd.DataFrame] = {}
        self._cf: pd.DataFrame | None = None
        self.summary = pd.read_csv(self.mdir / "panel_att_summary.csv") if (self.mdir / "panel_att_summary.csv").exists() else pd.DataFrame()
        self.cate_models: dict[str, Any] = {}
        for k in LEARNER_ORDER:
            f = self.odir / _fname(k)
            if f.exists():
                try:
                    self.cate_models[k] = joblib.load(f)
                except Exception:
                    pass
        self.meta = joblib.load(self.odir / "cate_meta.joblib") if (self.odir / "cate_meta.joblib").exists() else {}
        self.curve = rc.ResponseCurve(**read_json(self.mdir / "response_curve.json")["fitted"]) if (self.mdir / "response_curve.json").exists() else None
        self.window = pd.read_parquet(self.mdir / "cate_predictions.parquet") if (self.mdir / "cate_predictions.parquet").exists() else None
        self.started = time.time()

    # ------------------------------------------------------------------ metadata
    @property
    def best_learner(self) -> str | None:
        return self.meta.get("best")

    @property
    def feature_names(self) -> list[str]:
        return self.meta.get("feature_names", [])

    def health(self) -> dict:
        return {"status": "ok", "uptime_s": round(time.time() - self.started, 1), "cate_models": sorted(self.cate_models),
                "best_learner": self.best_learner, "response_curve": self.curve.to_dict() if self.curve else None,
                "cached_scenarios": sorted(self.summary["scenario"].unique()) if len(self.summary) else [],
                "device": dev.device_info()["torch_device"], "serving_device": "cpu"}

    def presets(self) -> list[dict]:
        sc = self.cfg["dgp"]["scenarios"]
        return [{"key": k, **{kk: sc[k][kk] for kk in ("label", "treated", "donors", "total_spend_usd", "pattern")}} for k in PRESET_KEYS]

    # ------------------------------------------------------------------ data access
    def panel(self, scenario: str) -> pd.DataFrame:
        src = "mia_flagship" if scenario == "mia_all24" else scenario
        if src not in self._panels:
            df = pd.read_parquet(resolve(self.cfg["panel"]["scenario_dir"]) / f"{src}.parquet")
            df["dma_code"] = df["dma_code"].astype(str)
            self._panels[src] = df
        return self._panels[src]

    def scenario_spec(self, scenario: str) -> dict:
        base = self.cfg["dgp"]["scenarios"]["mia_flagship" if scenario == "mia_all24" else scenario]
        spec = dict(base)
        spec.setdefault("start_day", self.cfg["panel"]["campaign_start_day"])
        if scenario == "mia_all24":
            df = self.panel(scenario)
            spec["donors"] = [c for c in sorted(df["dma_code"].unique()) if c != spec["treated"]]
            spec["label"] = "Miami - headline analysis (all 24 control DMAs)"
        return spec

    def cf(self) -> pd.DataFrame:
        if self._cf is None:
            self._cf = pd.read_parquet(self.mdir / "counterfactual_paths.parquet")
        return self._cf

    # ------------------------------------------------------------------ ATT
    def att(self, scenario: str = "mia_all24", estimator: str = "ASCM (Ridge)", outcome: str = "room_nights_booked",
            live: bool = False, treated: str | None = None, donors: list[str] | None = None) -> dict:
        spec = self.scenario_spec(scenario)
        custom = treated is not None or donors is not None
        if estimator not in pa.ESTIMATOR_ORDER + ["ASCM (ElasticNet)"]:
            raise KeyError(f"unknown estimator '{estimator}'")
        df = self.panel(scenario)
        T = int(df["day_index"].max())
        T0 = spec["start_day"] - 1
        treated = treated or spec["treated"]
        if not live and not custom:
            row = self.summary[(self.summary.scenario == scenario) & (self.summary.outcome == outcome) & (self.summary.estimator == estimator)]
            if len(row):
                path = self.cf()
                path = path[(path.scenario == scenario) & (path.outcome == outcome) & (path.estimator == estimator)].sort_values("day_index")
                out = self._pack_cached(row.iloc[0].to_dict(), path, scenario, outcome, estimator, spec, df)
                out["source"] = "cache"
                return out
        if estimator == "Bayesian SC":
            raise ValueError("Bayesian SC is served from the pipeline cache only (run the pipeline, or choose a preset)")
        pdm = pa.PanelData.from_df(df, treated, donors or spec["donors"], outcome=outcome, start_day=spec["start_day"])
        fn = pa.ESTIMATOR_FUNCS.get(estimator) or pa.estimator_ascm
        kw = {"kind": "elasticnet"} if estimator == "ASCM (ElasticNet)" else {}
        t0 = time.time()
        res = fn(pdm, self.cfg, inference=True, **kw)
        if estimator == "Event Study":
            pass                                           # placebo bands are cached; live run reports the point path
        else:
            ckw = {"lam": res.meta["lam"]} if estimator == "ASCM (Ridge)" else dict(kw)
            pa.attach_placebo(res, pdm, fn, self.cfg, **ckw)
        truth = pa.truth_series(df, treated, outcome, pdm.T) if treated == spec["treated"] else None
        out = self._pack_live(res, pdm, truth, df, spec, scenario)
        out["source"] = f"live ({time.time() - t0:.1f}s)"
        return out

    def _econ(self, df: pd.DataFrame, spec: dict, att_daily: np.ndarray, ci: tuple | None, T0: int, draws=None) -> dict:
        one = df[df["dma_code"] == spec["treated"]].sort_values("day_index")
        nr = one["net_revenue_per_room_night"].to_numpy(float)[T0:]
        spend = float(one["ooh_ctv_spend_usd"].sum())
        return dg.economics(att_daily, nr, spend, ci, draws)

    def _pack_cached(self, row: dict, path: pd.DataFrame, scenario: str, outcome: str, estimator: str, spec: dict, df: pd.DataFrame) -> dict:
        T0 = spec["start_day"] - 1
        post = path.iloc[T0:]
        w = pd.read_csv(self.mdir / "donor_weights.csv")
        w = w[(w.scenario == scenario) & (w.outcome == outcome) & (w.estimator == estimator)]
        ci = (row["ci_lo"], row["ci_hi"]) if pd.notna(row.get("ci_lo")) else None
        pl = pd.read_csv(self.mdir / "placebo_in_space.csv") if (self.mdir / "placebo_in_space.csv").exists() else pd.DataFrame()
        pl = pl[(pl.scenario == scenario) & (pl.outcome == outcome) & (pl.estimator == estimator)] if len(pl) else pl
        econ = self._econ(df, spec, post["att_hat"].to_numpy(float), ci, T0) if outcome == "room_nights_booked" and spec["total_spend_usd"] > 0 else None
        return {"scenario": scenario, "estimator": estimator, "outcome": outcome, "treated": spec["treated"],
                "n_donors": int(row.get("n_donors", len(spec["donors"]))), "att_mean": _f(row["att_mean"]), "se": _f(row["se"]),
                "ci_lo": _f(row["ci_lo"]), "ci_hi": _f(row["ci_hi"]), "att_total": _f(row["att_total"]),
                "pre_rmspe": _f(row["pre_rmspe"]), "post_rmspe": _f(row["post_rmspe"]), "p_placebo": _f(row.get("p_placebo")),
                "true_att_mean": _f(row.get("att_true_mean")), "att_bias": _f(row.get("att_bias")),
                "economics": econ, "weights": dict(zip(w["donor"], w["weight"])),
                "placebo": pl[["unit", "pre_rmspe", "post_rmspe", "ratio", "att_mean", "kept_pre_fit"]].to_dict("records") if len(pl) else None,
                "series": {"day_index": path["day_index"].tolist(), "y_obs": path["y_obs"].tolist(), "y0_hat": path["y0_hat"].tolist(),
                           "att_hat": _clean(path["att_hat"]), "att_lo": _clean(path["att_lo"]), "att_hi": _clean(path["att_hi"]),
                           "true_att": _clean(path["true_att"])}, "start_day": spec["start_day"]}

    def _pack_live(self, res: pa.PanelResult, pdm: pa.PanelData, truth: np.ndarray | None, df: pd.DataFrame, spec: dict, scenario: str) -> dict:
        T0 = pdm.T0
        cf = res.counterfactual_frame()
        ev = pa.evaluate_against_truth(res, truth) if truth is not None else {}
        econ = None
        if pdm.outcome == "room_nights_booked" and spec["total_spend_usd"] > 0 and pdm.treated == spec["treated"]:
            econ = self._econ(df, spec, res.att_daily, (res.ci_lo, res.ci_hi) if res.ci_lo is not None else None, T0)
        return {"scenario": scenario, "estimator": res.name, "outcome": pdm.outcome, "treated": pdm.treated, "n_donors": pdm.n,
                "att_mean": _f(res.att_mean), "se": _f(res.se), "ci_lo": _f(res.ci_lo), "ci_hi": _f(res.ci_hi), "att_total": _f(res.att_total),
                "pre_rmspe": _f(res.pre_rmspe), "post_rmspe": _f(res.post_rmspe), "p_placebo": _f(res.meta.get("p_placebo")),
                "true_att_mean": _f(ev.get("att_true_mean")), "att_bias": _f(ev.get("att_bias")), "economics": econ,
                "weights": res.weights or {}, "start_day": pdm.start_day,
                "placebo": dg.in_space_table(res)[["unit", "pre_rmspe", "post_rmspe", "ratio", "att_mean", "kept_pre_fit"]].to_dict("records") if "placebo" in res.meta else None,
                "series": {"day_index": cf["day_index"].tolist(), "y_obs": cf["y_obs"].tolist(), "y0_hat": cf["y0_hat"].tolist(),
                           "att_hat": _clean(cf["att_hat"]), "att_lo": _clean(cf["att_lo"]), "att_hi": _clean(cf["att_hi"]),
                           "true_att": _clean(pd.Series(np.concatenate([np.zeros(T0), truth[T0:]]))) if truth is not None else None}}

    # ------------------------------------------------------------------ CATE
    def cate(self, rows: list[dict], learner: str | None = None, baseline: float | None = None, alpha: float = 0.05) -> dict:
        learner = learner or self.best_learner
        if learner not in self.cate_models:
            raise KeyError(f"learner '{learner}' not available; have {sorted(self.cate_models)}")
        feats = self.feature_names
        X = np.array([[float(r.get(f, np.nan)) for f in feats] for r in rows], dtype=np.float32)
        if np.isnan(X).any():
            defaults = self.window[feats].median() if self.window is not None else pd.Series(0.0, index=feats)
            X = np.where(np.isnan(X), defaults.to_numpy(np.float32)[None, :], X)
        m = self.cate_models[learner]
        tau = np.asarray(m.predict(X), float)
        base = float(self.meta["base"])
        baseline = baseline or base
        out: dict[str, Any] = {"learner": learner, "n": len(rows), "lift_pct": (100 * tau).tolist(),
                               "incremental_room_nights": (tau * baseline).tolist(), "baseline_room_nights": baseline,
                               "mean_lift_pct": float(100 * tau.mean()), "ci": None}
        iv = m.predict_interval(X, alpha) if getattr(m, "native_ci", False) else None
        if iv is not None:
            out["ci"] = {"level": 1 - alpha, "lift_pct_lo": (100 * iv[0]).tolist(), "lift_pct_hi": (100 * iv[1]).tolist(),
                         "incremental_room_nights_lo": (iv[0] * baseline).tolist(), "incremental_room_nights_hi": (iv[1] * baseline).tolist()}
        return out

    def window_features(self) -> pd.DataFrame:
        return self.window[self.feature_names].copy()

    # ------------------------------------------------------------------ campaign simulator
    def simulate(self, budget_usd: float, strategy: str = "even", learner: str | None = None, baseline_room_nights: float | None = None,
                 overrides: dict | None = None, net_revenue_per_room_night: float | None = None, return_plans: bool = False) -> dict:
        if self.curve is None or self.window is None:
            raise RuntimeError("response curve / CATE window not available - run the pipeline first")
        learner = learner or self.best_learner
        X = self.window_features()
        X_hist = X.copy()
        for k, v in (overrides or {}).items():
            if k in X.columns:
                X[k] = float(v)
        days = len(X)
        tau_h = np.asarray(self.cate_models[learner].predict(X_hist.to_numpy(np.float32)), float)
        tau_s = np.asarray(self.cate_models[learner].predict(X.to_numpy(np.float32)), float)
        mod = np.clip(tau_s / max(tau_h.mean(), 1e-6), 0.2, 3.0)
        cfp = self.cf()
        cfp = cfp[(cfp.scenario == "mia_all24") & (cfp.outcome == "room_nights_booked") & (cfp.estimator == "ASCM (Ridge)")].sort_values("day_index")
        base = cfp["y0_hat"].to_numpy(float)[-days:]
        if baseline_room_nights:
            base = base * baseline_room_nights / base.mean()
        nrpr = float(net_revenue_per_room_night or self._mia_nrpr())
        hol = (self.window["is_holiday_window"].to_numpy() > 0.5).astype(float)
        wkd = (self.window["is_weekend"].to_numpy() > 0.5).astype(float)

        def plan(s: str) -> np.ndarray:
            if s == "optimized":
                return rc.optimise_pacing(self.curve, budget_usd, base, mod)
            return rc.pacing_plan(s, budget_usd, days, weights=mod, holiday=hol, weekend=wkd)

        rows = []
        plans = {}
        for s in STRATEGIES:
            sp = plan(s)
            inc = self.curve.predict(sp, base, mod)
            plans[s] = (sp, inc)
            tot = float(inc.sum())
            rows.append({"strategy": s, "incremental_room_nights": tot, "cpirn_usd": float(budget_usd / tot) if tot > 0 else None,
                         "iroas": float(tot * nrpr / budget_usd) if budget_usd > 0 else None,
                         "net_revenue_usd": float(tot * nrpr)})
        sel = plans[strategy]
        best = max(rows, key=lambda r: r["incremental_room_nights"])["strategy"]
        weak = self.curve.r2 < 0.5 or self.curve.K > 2900 or self.curve.theta > 0.89
        extra = {"plans": {k: v[0].tolist() for k, v in plans.items()}} if return_plans else {}
        return {**extra, "learner": learner, "strategy": strategy, "budget_usd": budget_usd, "days": days, "net_revenue_per_room_night": nrpr,
                "response_curve": self.curve.to_dict(), "best_strategy": best,
                "curve_reliability": {"r2": self.curve.r2, "weak": bool(weak),
                                      "note": ("The fitted response curve is weakly identified (low R2 or a parameter at its bound); treat pacing "
                                               "rankings as indicative and prefer the simple, robust strategies.") if weak else
                                              "Response curve fits the estimated daily lift reasonably well."},
                "daily": {"day": list(range(1, days + 1)), "spend_usd": sel[0].tolist(), "incremental_room_nights": sel[1].tolist(),
                          "market_multiplier": mod.tolist(), "baseline_room_nights": base.tolist()},
                "totals": next(r for r in rows if r["strategy"] == strategy), "strategy_comparison": rows,
                "saturation": float(rc.hill(rc.adstock(sel[0] / 1000.0, self.curve.theta), self.curve.K, self.curve.s).mean())}

    def _mia_nrpr(self) -> float:
        df = self.panel("mia_flagship")
        m = df[(df.dma_code == "MIA") & (df.post_period == 1)]
        return float(m["net_revenue_per_room_night"].mean())


def _f(x) -> float | None:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _clean(s: pd.Series) -> list:
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in s.tolist()]
