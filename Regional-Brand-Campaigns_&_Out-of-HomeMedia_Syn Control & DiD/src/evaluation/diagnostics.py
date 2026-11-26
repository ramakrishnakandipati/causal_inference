"""Diagnostics, ground-truth benchmarking, economics and artifact writing.

  * in-space placebo permutation (post/pre RMSPE ratio + exact p-value) over all donor DMAs
  * in-time placebo (campaign back-dated to day 120)
  * leave-one-donor-out (LODO) and exclude-adjacent-market (spillover) sensitivity
  * PEHE, ATT bias, AUUC / Qini (oracle and observable versions), CPIRN, iROAS
  * Monte-Carlo replication of the whole panel (bias / RMSE / CI coverage by estimator)
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.generate_panel import (OUTCOMES, assemble_panel, build_latent, generate_scenarios, scenario_specs)
from src.models import cate_learners as cl
from src.models import panel_att as pa
from src.utils.config import ensure_dir, resolve, save_json
from src.utils.logger import get_logger

PRIMARY = "room_nights_booked"


# ============================================================================ placebo tests
def in_space_table(res: pa.PanelResult, name: str | None = None) -> pd.DataFrame:
    """Post/pre RMSPE ratios for the treated unit and every placebo donor, with exact p-values."""
    pl = res.meta["placebo"]
    treated_ratio = res.post_rmspe / max(res.pre_rmspe, 1e-9)
    ratio = pl["post_rmspe"] / np.maximum(pl["pre_rmspe"], 1e-9)
    t = pd.DataFrame({"unit": pl["units"] + ["TREATED"], "pre_rmspe": np.append(pl["pre_rmspe"], res.pre_rmspe),
                      "post_rmspe": np.append(pl["post_rmspe"], res.post_rmspe),
                      "ratio": np.append(ratio, treated_ratio), "att_mean": np.append(pl["att_mean"], res.att_mean)})
    t["estimator"] = name or res.name
    t["rank"] = t["ratio"].rank(ascending=False, method="min").astype(int)
    t["kept_pre_fit"] = np.append(res.meta.get("placebo_keep", np.ones(len(ratio), bool)), True)
    return t


def exact_p(table: pd.DataFrame, filtered: bool = False) -> float:
    t = table[table["kept_pre_fit"]] if filtered else table
    tr = t.loc[t["unit"] == "TREATED", "ratio"].iloc[0]
    return float((t["ratio"] >= tr).sum() / len(t))


def truncate(pd_: pa.PanelData, t_end: int, new_start: int) -> pa.PanelData:
    return pa.PanelData(pd_.outcome, pd_.treated, pd_.donors, new_start, pd_.days[:t_end], pd_.y_tr[:t_end],
                        pd_.Yc[:, :t_end], pd_.cov_tr[:t_end], pd_.cov_c[:, :t_end], pd_.cov_names,
                        pd_.spend[:t_end], pd_.cal[:t_end])


def in_time_placebo(pd_: pa.PanelData, cfg: dict, fake_day: int | None = None) -> pd.DataFrame:
    """Pretend the campaign began on `fake_day` (inside the pre-period); a credible estimator finds ~0."""
    fake_day = fake_day or cfg["diagnostics"]["placebo_day"]
    q = truncate(pd_, pd_.T0, fake_day)
    rows = []
    for name in ["TWFE DiD", "CS-DiD (DR)", "SCM", "ASCM (Ridge)", "SDID"]:
        fn = pa.ESTIMATOR_FUNCS[name]
        r = fn(q, cfg, inference=True)
        kw = {"lam": r.meta["lam"]} if name == "ASCM (Ridge)" else {}
        pa.attach_placebo(r, q, fn, cfg, **kw)
        scale = float(q.y_tr[:q.T0].mean())
        rows.append({"estimator": name, "fake_start_day": fake_day, "att_mean": r.att_mean,
                     "att_pct_of_baseline": 100 * r.att_mean / scale, "se": r.se, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                     "significant_at_5pct": bool(r.ci_lo > 0 or r.ci_hi < 0),
                     "p_placebo": r.meta["p_placebo"]})
    return pd.DataFrame(rows)


def leave_one_donor_out(pd_: pa.PanelData, cfg: dict, res_scm: pa.PanelResult, exclude: list[str]) -> pd.DataFrame:
    """Refit SCM/ASCM/SDID dropping each influential donor, and dropping adjacent (spillover-prone) markets."""
    w = res_scm.meta["w"]
    influential = [d for d, wi in zip(pd_.donors, w) if wi > 0.01]
    rows = []
    base: dict[str, float] = {}
    for label, drop in [("none", [])] + [(d, [d]) for d in influential] + [("exclude " + "+".join(exclude), exclude)]:
        q = pd_.restrict_donors([d for d in pd_.donors if d not in drop])
        for name, fn in [("SCM", pa.estimator_scm), ("ASCM (Ridge)", pa.estimator_ascm), ("SDID", pa.estimator_sdid)]:
            r = fn(q, cfg, inference=False)
            if label == "none":
                base[name] = r.att_mean
            rows.append({"dropped": label, "estimator": name, "att_mean": r.att_mean,
                         "delta_vs_full": r.att_mean - base.get(name, r.att_mean), "pre_rmspe": r.pre_rmspe})
    return pd.DataFrame(rows)


# ============================================================================ uplift / ranking metrics
def uplift_curve(score: np.ndarray, gain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(gain[order])
    x = np.arange(1, len(score) + 1) / len(score)
    return np.concatenate([[0.0], x]), np.concatenate([[0.0], cum])


def auuc_norm(score: np.ndarray, gain: np.ndarray) -> dict:
    """AUUC and Qini coefficient. 0 = random targeting, 1 = oracle ordering (by the gain itself)."""
    x, c = uplift_curve(score, gain)
    xo, co = uplift_curve(gain, gain)
    total = gain.sum()
    a, a_or, a_rand = np.trapezoid(c, x), np.trapezoid(co, xo), 0.5 * total
    qini = (a - a_rand) / (a_or - a_rand) if abs(a_or - a_rand) > 1e-12 else float("nan")
    return {"auuc": float(a), "auuc_random": float(a_rand), "auuc_oracle": float(a_or), "qini": float(qini)}


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def evaluate_cate(data: cl.CATEData, tau_rel: np.ndarray, gap_obs: np.ndarray | None = None,
                  ci: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Scores on the exposed market's campaign days; effects are in room nights per day."""
    m = data.eval_mask
    t_hat = tau_rel[m] * data.base
    t_true = data.tau_true[m]
    out = {"pehe": pehe(t_hat, t_true), "att_bias": float(t_hat.mean() - t_true.mean()),
           "att_hat": float(t_hat.mean()), "att_true": float(t_true.mean()),
           "corr": float(np.corrcoef(t_hat, t_true)[0, 1]) if t_hat.std() > 1e-12 else float("nan"),
           "spearman": float(spearmanr(t_hat, t_true)[0]) if t_hat.std() > 1e-12 else float("nan"),
           "pehe_const_baseline": float(t_true.std())}
    o = auuc_norm(t_hat, t_true)
    out.update({"auuc_oracle_gain": o["auuc"], "qini_oracle": o["qini"]})
    if gap_obs is not None:
        out["qini_observed"] = auuc_norm(t_hat, gap_obs)["qini"]
    if ci is not None:
        lo, hi = ci[0][m] * data.base, ci[1][m] * data.base
        out["ci_coverage"] = float(np.mean((t_true >= lo) & (t_true <= hi)))
        out["ci_width"] = float(np.mean(hi - lo))
    return out


# ============================================================================ economics
def economics(att_daily: np.ndarray, nrpr_post: np.ndarray, spend_total: float, att_mean_ci: tuple[float, float] | None,
              draws_daily: np.ndarray | None = None) -> dict:
    """CPIRN = spend / incremental room nights; iROAS = incremental net revenue / spend."""
    inc_rn = float(att_daily.sum())
    inc_rev = float((att_daily * nrpr_post).sum())
    out: dict[str, Any] = {"incremental_room_nights": inc_rn, "incremental_net_revenue_usd": inc_rev,
                           "spend_usd": float(spend_total),
                           "cpirn_usd": float(spend_total / inc_rn) if inc_rn > 0 else float("inf"),
                           "iroas": float(inc_rev / spend_total) if spend_total > 0 else float("nan")}
    if draws_daily is not None and spend_total > 0:
        rn = draws_daily.sum(axis=1)
        rev = (draws_daily * nrpr_post[None, :]).sum(axis=1)
        out.update({"inc_rn_lo": float(np.quantile(rn, 0.025)), "inc_rn_hi": float(np.quantile(rn, 0.975)),
                    "iroas_lo": float(np.quantile(rev / spend_total, 0.025)), "iroas_hi": float(np.quantile(rev / spend_total, 0.975))})
    elif att_mean_ci is not None and spend_total > 0:
        T1 = len(att_daily)
        lo, hi = att_mean_ci[0] * T1, att_mean_ci[1] * T1
        k = inc_rev / inc_rn if inc_rn != 0 else float(nrpr_post.mean())
        out.update({"inc_rn_lo": float(lo), "inc_rn_hi": float(hi), "iroas_lo": float(lo * k / spend_total),
                    "iroas_hi": float(hi * k / spend_total)})
    if "inc_rn_lo" in out and spend_total > 0:
        out["cpirn_lo_usd"] = float(spend_total / out["inc_rn_hi"]) if out["inc_rn_hi"] > 0 else float("inf")
        out["cpirn_hi_usd"] = float(spend_total / out["inc_rn_lo"]) if out["inc_rn_lo"] > 0 else float("inf")
    return out


# ============================================================================ scenario runner
def run_scenario(cfg: dict, name: str, df: pd.DataFrame, outcomes: list[str], bayes: bool, seed: int = 0,
                 spec: dict | None = None) -> dict:
    """Fit the full estimator suite for one preset on each requested outcome."""
    spec = spec or scenario_specs(cfg)[name]
    out: dict[str, Any] = {"spec": spec, "results": {}, "pd": {}, "truth": {}}
    for o in outcomes:
        pdm = pa.PanelData.from_df(df, spec["treated"], spec["donors"], outcome=o, start_day=spec["start_day"])
        out["pd"][o] = pdm
        out["truth"][o] = pa.truth_series(df, spec["treated"], o, pdm.T)
        out["results"][o] = pa.run_suite(pdm, cfg, seed=seed, bayes=bayes, placebo=True)
    return out


def summary_rows(cfg: dict, name: str, df: pd.DataFrame, sc: dict) -> list[dict]:
    spec = sc["spec"]
    rows = []
    y0_cols = {PRIMARY: "y0_true"}
    for o, results in sc["results"].items():
        truth = sc["truth"][o]
        one = df[df["dma_code"].astype(str) == spec["treated"]].sort_values("day_index")
        y0 = one[y0_cols.get(o, f"y0_true_{o}")].to_numpy(float)
        for est, r in results.items():
            ev = pa.evaluate_against_truth(r, truth, y0)
            rows.append({"scenario": name, "treated": spec["treated"], "outcome": o, **r.summary(), **ev,
                         "n_donors": len(sc["pd"][o].donors)})
    return rows


def write_scenario_artifacts(cfg: dict, name: str, df: pd.DataFrame, sc: dict, store: dict[str, list]) -> None:
    """Accumulate tidy tables for later CSV/parquet writing."""
    spec = sc["spec"]
    for o, results in sc["results"].items():
        pdm = sc["pd"][o]
        for est, r in results.items():
            cf = r.counterfactual_frame()
            cf.insert(0, "estimator", est)
            cf.insert(0, "outcome", o)
            cf.insert(0, "scenario", name)
            cf["true_att"] = np.concatenate([np.zeros(pdm.T0), sc["truth"][o][pdm.T0:]])
            store["cf"].append(cf)
            if r.weights and est in ("SCM", "ASCM (Ridge)", "SDID", "Bayesian SC", "ASCM (ElasticNet)"):
                for d, w in r.weights.items():
                    store["weights"].append({"scenario": name, "outcome": o, "estimator": est, "donor": d, "weight": w,
                                             "weight_sd": r.meta.get("w_sd", {}).get(d)})
            if est == "SCM" and "balance" in r.meta:
                b = r.meta["balance"].copy()
                b.insert(0, "outcome", o)
                b.insert(0, "scenario", name)
                store["balance"].append(b)
            if est in ("SCM", "ASCM (Ridge)", "SDID") and "placebo" in r.meta:
                if o == PRIMARY and name == "mia_all24":
                    pl = r.meta["placebo"]
                    gf = pd.DataFrame(pl["gap_full"].T, columns=pl["units"])
                    gf.insert(0, "TREATED", r.y_obs - r.y0_hat)
                    gf.insert(0, "day_index", r.days)
                    gf.insert(0, "estimator", est)
                    store.setdefault("placebo_gap", []).append(gf)
                t = in_space_table(r)
                t.insert(0, "outcome", o)
                t.insert(0, "scenario", name)
                store["placebo"].append(t)
                store["placebo_p"].append({"scenario": name, "outcome": o, "estimator": est,
                                           "p_exact_ratio": exact_p(t), "p_exact_ratio_filtered": exact_p(t, True),
                                           "rank_of_treated": int(t.loc[t["unit"] == "TREATED", "rank"].iloc[0]),
                                           "n_units": len(t), "min_possible_p": 1 / len(t)})
        es = results.get("Event Study")
        if es is not None and "beta" in es.meta:
            k = es.meta["k"]
            store["es"].append(pd.DataFrame({"scenario": name, "outcome": o, "k": k, "beta": es.meta["beta"],
                                             "se_hc1": es.meta["se_hc1"], "se_placebo": es.meta.get("se_placebo_k"),
                                             "true_att": np.concatenate([np.zeros((k < 0).sum()),
                                                                         sc["truth"][o][pdm.T0 + k[k >= 0]]])}))
            store["es_meta"].append({"scenario": name, "outcome": o, "pretrend_rms": es.meta["pretrend_rms"],
                                     "pretrend_slope": es.meta["pretrend_slope"],
                                     "p_pretrend_perm": es.meta.get("p_pretrend_perm"),
                                     "p_pretrend_slope_perm": es.meta.get("p_pretrend_slope_perm"),
                                     "post_early": es.meta["post_early"], "post_late": es.meta["post_late"]})
        bs = results.get("Bayesian SC")
        if bs is not None:
            store["bayes"].append({"scenario": name, "outcome": o, **{k: bs.meta[k] for k in
                                   ("rhat_max", "ess_min", "divergences", "ppc_pre_coverage", "prob_positive",
                                    "rho_mean", "sigma_mean", "jax_backend", "chain_method")}})
        # economics on the room-night outcome only
        if o == PRIMARY and spec["total_spend_usd"] > 0:
            one = df[df["dma_code"].astype(str) == spec["treated"]].sort_values("day_index")
            nr = one["net_revenue_per_room_night"].to_numpy(float)[pdm.T0:]
            true_att = sc["truth"][o][pdm.T0:]
            spend = float(one["ooh_ctv_spend_usd"].sum())
            true_econ = economics(true_att, nr, spend, None)
            for est, r in results.items():
                if est == "Event Study":
                    continue
                draws = r.meta.get("att_draws") if est == "Bayesian SC" else None
                e = economics(r.att_daily, nr, spend, (r.ci_lo, r.ci_hi) if r.ci_lo is not None else None, draws)
                store["econ"].append({"scenario": name, "estimator": est, **e, "true_iroas": true_econ["iroas"],
                                      "true_cpirn_usd": true_econ["cpirn_usd"],
                                      "true_incremental_room_nights": true_econ["incremental_room_nights"]})


def run_panel_stage(cfg: dict, quick: bool = False) -> dict:
    """Panel ATT estimators for all four presets + diagnostics; writes outputs/metrics/*."""
    log = get_logger("panel")
    seed = cfg["seed"]
    lat = build_latent(cfg, seed)
    scen = generate_scenarios(cfg, seed, lat)
    store: dict[str, list] = {k: [] for k in ["cf", "weights", "balance", "placebo", "placebo_p", "es", "es_meta",
                                              "bayes", "econ"]}
    rows: list[dict] = []
    main_sc = None
    base_spec = scenario_specs(cfg)["mia_flagship"]
    all_donors = [c for c in sorted(scen["mia_flagship"]["dma_code"].astype(str).unique()) if c != base_spec["treated"]]
    headline = dict(base_spec, donors=all_donors, label="Miami - headline analysis (all 24 control DMAs)")
    jobs = [("mia_all24", "mia_flagship", headline)] + [(n, n, None) for n in scen]
    for name, src, spec_over in jobs:
        headline_job = name == "mia_all24"
        full = headline_job or name == "mia_flagship"
        outcomes = (cfg["estimators"]["bayes"]["outcomes_full"] + ["incremental_net_revenue_usd"]) if headline_job else [PRIMARY]
        if quick:
            outcomes = [PRIMARY]
        bayes = headline_job or name == "mia_flagship" or cfg["estimators"]["bayes"]["run_other_scenarios"]
        spec = spec_over or scenario_specs(cfg)[src]
        log.info("scenario %-16s treated=%s donors=%d outcomes=%d", name, spec["treated"], len(spec["donors"]), len(outcomes))
        sc = run_scenario(cfg, name, scen[src], outcomes, bayes, seed, spec=spec)
        rows += summary_rows(cfg, name, scen[src], sc)
        write_scenario_artifacts(cfg, name, scen[src], sc, store)
        if headline_job:
            main_sc = sc
    mdir = ensure_dir(cfg["paths"]["metrics"])
    pd.DataFrame(rows).to_csv(mdir / "panel_att_summary.csv", index=False)
    pd.concat(store["cf"], ignore_index=True).to_parquet(mdir / "counterfactual_paths.parquet", index=False)
    pd.DataFrame(store["weights"]).to_csv(mdir / "donor_weights.csv", index=False)
    pd.concat(store["balance"], ignore_index=True).to_csv(mdir / "covariate_balance.csv", index=False)
    pd.concat(store["placebo"], ignore_index=True).to_csv(mdir / "placebo_in_space.csv", index=False)
    pd.DataFrame(store["placebo_p"]).to_csv(mdir / "placebo_pvalues.csv", index=False)
    pd.concat(store["es"], ignore_index=True).to_csv(mdir / "event_study.csv", index=False)
    pd.DataFrame(store["es_meta"]).to_csv(mdir / "event_study_pretrend.csv", index=False)
    pd.DataFrame(store["bayes"]).to_csv(mdir / "bayes_diagnostics.csv", index=False)
    pd.DataFrame(store["econ"]).to_csv(mdir / "economics.csv", index=False)
    pd.concat(store["placebo_gap"], ignore_index=True).to_parquet(mdir / "placebo_gap_paths.parquet", index=False)

    # diagnostics on the flagship / primary outcome
    pdm = main_sc["pd"][PRIMARY]
    res = main_sc["results"][PRIMARY]
    it = in_time_placebo(pdm, cfg)
    it.to_csv(mdir / "in_time_placebo.csv", index=False)
    lodo = leave_one_donor_out(pdm, cfg, res["SCM"], cfg["diagnostics"]["exclude_adjacent"])
    lodo.to_csv(mdir / "lodo_sensitivity.csv", index=False)
    log.info("in-time placebo (day %d): %s", cfg["diagnostics"]["placebo_day"],
             "; ".join(f"{r.estimator} {r.att_pct_of_baseline:+.1f}%" for r in it.itertuples()))
    log.info("LODO ATT range (ASCM): %.1f .. %.1f", lodo[lodo.estimator == "ASCM (Ridge)"].att_mean.min(),
             lodo[lodo.estimator == "ASCM (Ridge)"].att_mean.max())
    return {"scenarios": scen, "main": main_sc}


# ============================================================================ Monte-Carlo
def panel_monte_carlo(cfg: dict, n: int | None = None) -> pd.DataFrame:
    """Re-draw the whole panel n times; summarise bias / RMSE / CI coverage of every estimator."""
    log = get_logger("mc")
    n = n or cfg["diagnostics"]["mc_seeds"]
    rows = []
    t0 = time.time()
    for s in range(n):
        seed = cfg["diagnostics"]["mc_base_seed"] + s
        lat = build_latent(cfg, seed)
        df = assemble_panel(lat, scenario_specs(cfg)["mia_flagship"], seed)
        pdm = pa.PanelData.from_df(df, "MIA")
        truth = pa.truth_series(df, "MIA", PRIMARY, pdm.T)
        y0t = df[df["dma_code"].astype(str) == "MIA"].sort_values("day_index")["y0_true"].to_numpy(float)
        res = pa.run_suite(pdm, cfg, seed=seed, bayes=True, placebo=True, elasticnet=True)
        for k, r in res.items():
            ev = pa.evaluate_against_truth(r, truth, y0t)
            rows.append({"seed": seed, "estimator": k, "att_hat": r.att_mean, **ev, "se": r.se,
                         "p_placebo": r.meta.get("p_placebo")})
        log.info("MC %d/%d done (%.0fs)", s + 1, n, time.time() - t0)
    d = pd.DataFrame(rows)
    g = d.groupby("estimator").agg(mean_att=("att_hat", "mean"), true_att=("att_true_mean", "mean"),
                                   mean_bias=("att_bias", "mean"), rmse=("att_bias", lambda x: float(np.sqrt((x ** 2).mean()))),
                                   ci_coverage=("covered", "mean"), cf_rmse_post=("cf_rmse_post", "mean"),
                                   mean_se=("se", "mean"), power_p05=("p_placebo", lambda x: float((x <= 0.05).mean()))).reset_index()
    g["bias_pct"] = 100 * g["mean_bias"] / g["true_att"]
    order = {k: i for i, k in enumerate(pa.ESTIMATOR_ORDER + ["ASCM (ElasticNet)"])}
    g = g.sort_values("estimator", key=lambda s: s.map(order))
    mdir = ensure_dir(cfg["paths"]["metrics"])
    d.to_csv(mdir / "panel_monte_carlo_runs.csv", index=False)
    g.to_csv(mdir / "panel_monte_carlo.csv", index=False)
    return g


# ============================================================================ CATE benchmark
def cate_benchmark(cfg: dict, quick: bool = False) -> dict:
    """10 learners x n_seeds noise draws; per-seed and aggregated metrics plus main-seed artifacts."""
    log = get_logger("cate")
    n_seeds = 1 if quick else cfg["cate"]["n_seeds"]
    mdir = ensure_dir(cfg["paths"]["metrics"])
    specs = scenario_specs(cfg)
    rows: list[dict] = []
    main: dict[str, Any] = {}
    for si in range(n_seeds):
        seed = cfg["seed"] if si == 0 else cfg["seed"] + 100 * si
        lat = build_latent(cfg, seed)
        df = assemble_panel(lat, specs["mia_flagship"], seed)
        data = cl.build_cate_dataset(df, cfg)
        gap_obs = data.Y[data.eval_mask] * data.base
        log.info("CATE seed %d (%d/%d): %d rows, %d treated", seed, si + 1, n_seeds, len(data.D), data.D.sum())
        fitted = cl.fit_all(data, cfg, seed=seed, keep_models=(si == 0))
        # reference rows: constant ATE and oracle X-regression
        m = data.eval_mask
        tt = data.tau_true[m]
        rows.append({"seed": seed, "learner": "Constant ATE (oracle)", "pehe": float(tt.std()), "att_bias": 0.0,
                     "corr": float("nan"), "spearman": float("nan"), "qini_oracle": 0.0, "seconds": 0.0, "gpu": False,
                     "backend": "-", "reference": True})
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_predict

        orc = cross_val_predict(GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                                          random_state=seed), data.X[m], tt, cv=5)
        rows.append({"seed": seed, "learner": "Oracle X-regression (CV)", "pehe": pehe(orc, tt),
                     "att_bias": float(orc.mean() - tt.mean()), "corr": float(np.corrcoef(orc, tt)[0, 1]),
                     "spearman": float(spearmanr(orc, tt)[0]), "qini_oracle": auuc_norm(orc, tt)["qini"], "seconds": 0.0,
                     "gpu": False, "backend": "-", "reference": True})
        for name, r in fitted.items():
            if r["tau_rel"] is None:
                continue
            ci = None
            if si == 0 and r["model"] is not None and r["model"].native_ci:
                lo, hi = r["model"].predict_interval(data.X)
                ci = (lo, hi)
            ev = evaluate_cate(data, r["tau_rel"], gap_obs, ci)
            rows.append({"seed": seed, "learner": name, **ev, "seconds": r["seconds"], "gpu": r["gpu"],
                         "backend": r["backend"], "reference": False})
        if si == 0:
            main = {"data": data, "fitted": fitted, "df": df}
    d = pd.DataFrame(rows)
    d.to_csv(mdir / "cate_benchmark_seeds.csv", index=False)
    agg = d.groupby("learner").agg(pehe=("pehe", "mean"), pehe_sd=("pehe", "std"), att_bias=("att_bias", "mean"),
                                   corr=("corr", "mean"), spearman=("spearman", "mean"), qini_oracle=("qini_oracle", "mean"),
                                   qini_observed=("qini_observed", "mean"), seconds=("seconds", "mean"),
                                   ci_coverage=("ci_coverage", "mean"), gpu=("gpu", "max"), backend=("backend", "first"),
                                   n_seeds=("seed", "nunique")).reset_index()
    order = {k: i for i, k in enumerate(["Constant ATE (oracle)", "Oracle X-regression (CV)"] + cl.LEARNER_ORDER)}
    agg = agg.sort_values("learner", key=lambda s: s.map(order))
    agg["reference"] = agg["learner"].isin(["Constant ATE (oracle)", "Oracle X-regression (CV)"])
    agg.to_csv(mdir / "cate_benchmark.csv", index=False)
    main["agg"] = agg
    return main


def cate_main_artifacts(cfg: dict, main: dict) -> None:
    """Persist the main-seed predictions, training histories, driver analysis and CPU models."""
    import joblib

    data: cl.CATEData = main["data"]
    fitted = main["fitted"]
    mdir = ensure_dir(cfg["paths"]["metrics"])
    odir = ensure_dir(cfg["paths"]["models"])
    m = data.eval_mask
    pred = pd.DataFrame({"day_index": data.day[m], "tau_true": data.tau_true[m]})
    for k, r in fitted.items():
        if r["tau_rel"] is not None:
            pred[k] = r["tau_rel"][m] * data.base
            mod = r["model"]
            if mod is not None and mod.native_ci:
                lo, hi = mod.predict_interval(data.X)
                pred[k + " lo"], pred[k + " hi"] = lo[m] * data.base, hi[m] * data.base
    feats = pd.DataFrame(data.X[m], columns=data.feature_names)
    pd.concat([pred.reset_index(drop=True), feats], axis=1).to_parquet(mdir / "cate_predictions.parquet", index=False)

    hist = {k: {"history": r["history"], "best_epoch": r["best_epoch"]} for k, r in fitted.items()
            if r.get("history")}
    save_json(hist, mdir / "nn_training_history.json")

    # drivers: Linear DML coefficients + partial dependence of the best learner
    drivers: dict[str, Any] = {"features": data.feature_names, "base": data.base}
    ldml = fitted.get("Linear DML", {}).get("model")
    if ldml is not None:
        coef = ldml.coef_ * data.base
        drivers["linear_dml"] = {"intercept": float(ldml.intercept_ * data.base),
                                 "coef_room_nights_per_sd": dict(zip(data.feature_names, coef.tolist()))}
    agg = main["agg"]
    cand = agg[(~agg["reference"]) & agg["pehe"].notna()].sort_values("pehe")
    best = cand["learner"].iloc[0]
    drivers["best_learner"] = best
    best_model = fitted[best]["model"]
    Xe = data.X[m]
    pdp = {}
    for j, f in enumerate(data.feature_names):
        grid = np.unique(np.quantile(Xe[:, j], np.linspace(0.05, 0.95, 9)))
        vals = []
        for g in grid:
            Xg = Xe.copy()
            Xg[:, j] = g
            vals.append(float(best_model.predict(Xg).mean() * data.base))
        pdp[f] = {"grid": grid.tolist(), "tau_hat": vals}
    drivers["partial_dependence"] = pdp
    # true vs estimated subgroup effects
    t_hat = fitted[best]["tau_rel"][m] * data.base
    sub = {}
    for f, thr in [("is_weekend", 0.5), ("is_holiday_window", 0.5), ("weather_severity_index", 5.0),
                   ("flight_search_index", None), ("competitor_adr_usd", None), ("ota_promo_pressure_index", None)]:
        col = Xe[:, data.feature_names.index(f)]
        cut = thr if thr is not None else float(np.median(col))
        hi = col > cut
        sub[f] = {"cut": cut, "est_high": float(t_hat[hi].mean()), "est_low": float(t_hat[~hi].mean()),
                  "true_high": float(data.tau_true[m][hi].mean()), "true_low": float(data.tau_true[m][~hi].mean()),
                  "n_high": int(hi.sum()), "n_low": int((~hi).sum())}
    drivers["subgroups"] = sub
    save_json(drivers, mdir / "cate_drivers.json")

    for k, r in fitted.items():
        if r["model"] is not None:
            r["model"].to_cpu()
            joblib.dump(r["model"], odir / ("cate_" + k.lower().replace(" ", "_").replace("-", "_") + ".joblib"))
    joblib.dump({"feature_names": data.feature_names, "base": data.base, "best": best,
                 "mu": None}, odir / "cate_meta.joblib")


# ============================================================================ response curve + pacing validation
def response_curve_and_pacing_validation(cfg: dict, main: dict) -> dict:
    """Fit adstock/Hill on the *estimated* daily lift (with the CATE market-condition multiplier), then check every
    pacing strategy against the simulation's true response function - the only place a recommendation can be graded."""
    from src.data import generate_panel as gp
    from src.models import response_curve as rc
    from src.serving import Engine
    from src.utils import device as dev

    log = get_logger("pacing")
    mdir = ensure_dir(cfg["paths"]["metrics"])
    cf_path = mdir / "counterfactual_paths.parquet"
    if not cf_path.exists():
        log.info("counterfactual_paths.parquet missing - run the panel stage first; skipping response curve")
        return {}
    cf = pd.read_parquet(cf_path)
    a = cf[(cf.scenario == "mia_all24") & (cf.outcome == PRIMARY) & (cf.estimator == "ASCM (Ridge)")].sort_values("day_index")
    df, data, fitted, agg = main["df"], main["data"], main["fitted"], main["agg"]
    start = int(df.loc[df["treatment_active"] == 1, "day_index"].min())
    T0 = start - 1
    mia = df[df["dma_code"].astype(str) == "MIA"].sort_values("day_index")
    spend = mia["ooh_ctv_spend_usd"].to_numpy(float)[T0:]
    att = a["att_hat"].to_numpy(float)[T0:]
    base = a["y0_hat"].to_numpy(float)[T0:]
    cand = agg[(~agg["reference"]) & agg["pehe"].notna()].sort_values("pehe")
    best = cand["learner"].iloc[0]
    tau = fitted[best]["tau_rel"][data.eval_mask]
    mod = np.clip(tau / max(tau.mean(), 1e-9), 0.2, 3.0)
    curve = rc.fit_response_curve(spend, att, base, mod, s_fixed=1.0)
    th_true = cfg["dgp"]["adstock_theta"][PRIMARY]
    hk = cfg["dgp"]["hill"]
    save_json({"fitted": curve.to_dict(), "truth_dgp": {"theta": th_true, "K": hk["K"], "s": hk["s"]},
               "source": "ASCM (Ridge) daily ATT vs spend path with the best CATE learner's market multiplier",
               "modifier_from": best, "theta_upper_bound": 0.9}, mdir / "response_curve.json")
    log.info("response curve: theta=%.2f K=%.0f c=%.2f R2=%.2f (multiplier from %s)", curve.theta, curve.K, curve.c, curve.r2, best)

    # --- grade the pacing strategies against the simulation truth
    prev = dev.gpu_enabled()
    try:
        eng = Engine(cfg)
        sim = eng.simulate(float(spend.sum()), "even", learner=best, return_plans=True)
    finally:
        dev.set_gpu_enabled(prev)
    lat = gp.build_latent(cfg, cfg["seed"])
    i = lat.codes.index("MIA")
    y0 = lat.y0[PRIMARY][i][T0:]
    M = gp.effect_modifier(lat, i)[T0:]
    S_act = gp._hill(gp.adstock(spend / 1000.0, th_true), hk["K"], hk["s"])
    target = cfg["dgp"]["scenarios"]["mia_flagship"]["target_mean_lift"]
    k = target * y0.sum() / float((S_act * M * y0).sum())

    def true_total(plan: np.ndarray) -> float:
        return float((k * gp._hill(gp.adstock(plan / 1000.0, th_true), hk["K"], hk["s"]) * M * y0).sum())

    comp = {r["strategy"]: r for r in sim["strategy_comparison"]}
    rows = []
    for s_name, plan in sim["plans"].items():
        rows.append({"strategy": s_name, "projected_incremental_rn": comp[s_name]["incremental_room_nights"],
                     "true_incremental_rn": true_total(np.asarray(plan))})
    rows.append({"strategy": "actual campaign (reference)", "projected_incremental_rn": float(curve.predict(spend, base, mod).sum()),
                 "true_incremental_rn": true_total(spend)})
    pv = pd.DataFrame(rows)
    even = pv.loc[pv.strategy == "even"].iloc[0]
    pv["projected_vs_even_pct"] = 100 * (pv["projected_incremental_rn"] / even["projected_incremental_rn"] - 1)
    pv["true_vs_even_pct"] = 100 * (pv["true_incremental_rn"] / even["true_incremental_rn"] - 1)
    pv.to_csv(mdir / "pacing_validation.csv", index=False)
    log.info("pacing validation (projected vs true, %% vs even):\n%s",
             pv[["strategy", "projected_vs_even_pct", "true_vs_even_pct"]].round(1).to_string(index=False))
    return {"curve": curve, "pacing": pv}
