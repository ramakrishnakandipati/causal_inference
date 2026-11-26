"""Generate README.md from the saved pipeline metrics, so every number and table is traceable to outputs/.

    python scripts/build_readme.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import load_config, read_json  # noqa: E402

cfg = load_config()
M = ROOT / "outputs" / "metrics"


def rd(name: str) -> pd.DataFrame:
    p = M / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def f(x, d=0, pre="", suf="") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{pre}{x:,.{d}f}{suf}"


def ci(lo, hi, d=0) -> str:
    return "n/a" if pd.isna(lo) or pd.isna(hi) else f"[{lo:,.{d}f}, {hi:,.{d}f}]"


def table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


S, MC, CATE = rd("panel_att_summary.csv"), rd("panel_monte_carlo.csv"), rd("cate_benchmark.csv")
PP, IT, LODO, ECON = rd("placebo_pvalues.csv"), rd("in_time_placebo.csv"), rd("lodo_sensitivity.csv"), rd("economics.csv")
ESM, BAY = rd("event_study_pretrend.csv"), rd("bayes_diagnostics.csv")
GPU, GPUC = rd("gpu_benchmark.csv"), rd("gpu_benchmark_cate.csv")
drivers = read_json(M / "cate_drivers.json")
curve = read_json(M / "response_curve.json")
tele = read_json(ROOT / "outputs/logs/hardware_telemetry.json") if (ROOT / "outputs/logs/hardware_telemetry.json").exists() else []

H = S[(S.scenario == "mia_all24") & (S.outcome == "room_nights_booked")].set_index("estimator")
truth = H["att_true_mean"].iloc[0]
T1 = 60
order = ["TWFE DiD", "Event Study", "CS-DiD (DR)", "SCM", "ASCM (Ridge)", "ASCM (ElasticNet)", "SDID", "Bayesian SC"]

# ----------------------------------------------------------------------------- tables
t_head = pd.DataFrame([{
    "Estimator": e, "ATT / day": f(H.loc[e, "att_mean"], 1), "95% interval": ci(H.loc[e, "ci_lo"], H.loc[e, "ci_hi"]),
    "60-day total": f(H.loc[e, "att_total"]), "Bias vs truth": f"{H.loc[e, 'att_bias']:+.1f} ({H.loc[e, 'att_bias_pct']:+.0f}%)",
    "Pre-fit RMSPE": f(H.loc[e, "pre_rmspe"], 1), "Placebo p": f(H.loc[e, "p_placebo"], 2),
    "Covers truth": "yes" if H.loc[e, "covered"] else "no"} for e in order if e in H.index])

oth = S[(S.scenario == "mia_all24") & (S.estimator.isin(["SCM", "ASCM (Ridge)", "Bayesian SC"]))]
labels = {"room_nights_booked": "Room nights / day", "direct_web_sessions": "Direct web sessions / day",
          "branded_search_index": "Branded search index / day", "incremental_net_revenue_usd": "Net revenue USD / day"}
rows = []
for o, lab in labels.items():
    g = oth[oth.outcome == o].set_index("estimator")
    rows.append({"Outcome": lab, "Truth": f(g["att_true_mean"].iloc[0], 1),
                 "SCM": f"{g.loc['SCM', 'att_mean']:,.1f} ({g.loc['SCM', 'att_bias_pct']:+.0f}%)",
                 "ASCM (Ridge)": f"{g.loc['ASCM (Ridge)', 'att_mean']:,.1f} ({g.loc['ASCM (Ridge)', 'att_bias_pct']:+.0f}%)",
                 "ASCM 95% interval": ci(g.loc["ASCM (Ridge)", "ci_lo"], g.loc["ASCM (Ridge)", "ci_hi"], 1),
                 "Bayesian SC": f"{g.loc['Bayesian SC', 'att_mean']:,.1f} ({g.loc['Bayesian SC', 'att_bias_pct']:+.0f}%)",
                 "Bayes 95% CI": ci(g.loc["Bayesian SC", "ci_lo"], g.loc["Bayesian SC", "ci_hi"], 1)})
t_out = pd.DataFrame(rows)

prow = []
names = {"mia_flagship": "1. Miami flagship (10 coastal donors)", "las_convention": "2. Las Vegas convention",
         "nyc_spring_break": "3. New York spring break", "tpa_placebo": "4. Tampa placebo (true effect 0)"}
for k, lab in names.items():
    g = S[(S.scenario == k) & (S.outcome == "room_nights_booked")].set_index("estimator")
    pv = PP[(PP.scenario == k) & (PP.outcome == "room_nights_booked") & (PP.estimator == "ASCM (Ridge)")]
    prow.append({"Preset": lab, "Donors": int(g["n_donors"].iloc[0]), "True ATT/day": f(g["att_true_mean"].iloc[0], 1),
                 "ASCM": f(g.loc["ASCM (Ridge)", "att_mean"], 1), "ASCM 95% interval": ci(g.loc["ASCM (Ridge)", "ci_lo"], g.loc["ASCM (Ridge)", "ci_hi"]),
                 "Bayesian SC": f(g.loc["Bayesian SC", "att_mean"], 1), "Bayes 95% CI": ci(g.loc["Bayesian SC", "ci_lo"], g.loc["Bayesian SC", "ci_hi"]),
                 "Exact placebo p (smallest possible)": f"{pv['p_exact_ratio'].iloc[0]:.3f} ({pv['min_possible_p'].iloc[0]:.3f})"})
t_pre = pd.DataFrame(prow)

pv = PP[(PP.scenario == "mia_all24") & (PP.outcome == "room_nights_booked")]
t_pv = pd.DataFrame([{"Estimator": r.estimator, "Miami rank (of 25)": int(r.rank_of_treated), "Exact p (all 24 placebos)": f"{r.p_exact_ratio:.3f}",
                      "p (pre-fit-filtered)": f"{r.p_exact_ratio_filtered:.3f}", "Smallest attainable p": f"{r.min_possible_p:.3f}"} for r in pv.itertuples()])
t_it = pd.DataFrame([{"Estimator": r.estimator, "Fake start day": int(r.fake_start_day), "ATT / day": f(r.att_mean, 1),
                      "% of baseline": f"{r.att_pct_of_baseline:+.1f}%", "95% interval": ci(r.ci_lo, r.ci_hi),
                      "Significant at 5%": "yes" if r.significant_at_5pct else "no"} for r in IT.itertuples()])
lp = LODO.pivot(index="dropped", columns="estimator", values="att_mean").round(1).reset_index().rename(columns={"dropped": "Donor dropped"})
base_row = lp[lp["Donor dropped"] == "none"]
lp = pd.concat([base_row, lp[lp["Donor dropped"] != "none"]])
t_lodo = lp.astype(str)

if len(MC):
    n_mc = int(pd.read_csv(M / "panel_monte_carlo_runs.csv")["seed"].nunique())
    t_mc = pd.DataFrame([{"Estimator": r.estimator, "Mean ATT / day": f(r.mean_att, 1), "True": f(r.true_att, 1),
                          "Mean bias": f"{r.mean_bias:+.1f} ({r.bias_pct:+.0f}%)", "RMSE": f(r.rmse, 1),
                          "95% CI coverage": f"{100 * r.ci_coverage:.0f}%", "Counterfactual RMSE (post)": f(r.cf_rmse_post, 1),
                          "p<=0.05 rate": f"{100 * r.power_p05:.0f}%" if pd.notna(r.power_p05) else "n/a"} for r in MC.itertuples()])
else:
    n_mc, t_mc = 0, pd.DataFrame()

cr = CATE[CATE.reference].set_index("learner")
t_cate = pd.DataFrame([{"Model": r.learner + (" *(reference)*" if r.reference else ""), "PEHE (rn/day)": f"{r.pehe:.1f} +/- {0 if pd.isna(r.pehe_sd) else r.pehe_sd:.1f}",
                        "ATT bias": f"{r.att_bias:+.1f}", "Correlation": f(r.corr, 2), "Qini (oracle)": f(r.qini_oracle, 2),
                        "Qini (observed)": f(r.qini_observed, 2), "CI coverage": f(100 * r.ci_coverage, 0, suf="%") if pd.notna(r.ci_coverage) else "-",
                        "Fit s": f(r.seconds, 1), "Backend": ("GPU " if r.gpu else "") + str(r.backend) if r.backend != "-" else "-"}
                       for r in CATE.itertuples()])

e1 = ECON[ECON.scenario == "mia_all24"].set_index("estimator")
t_econ = pd.DataFrame([{"Estimator": e, "Incremental room nights": f(e1.loc[e, "incremental_room_nights"]),
                        "Interval": ci(e1.loc[e, "inc_rn_lo"], e1.loc[e, "inc_rn_hi"]), "CPIRN (USD)": f(e1.loc[e, "cpirn_usd"], 0, "$"),
                        "iROAS": f(e1.loc[e, "iroas"], 2, suf="x"), "iROAS interval": ci(e1.loc[e, "iroas_lo"], e1.loc[e, "iroas_hi"], 2)}
                       for e in order if e in e1.index and e != "Event Study"])
true_ir, true_cp, true_inc = e1["true_iroas"].iloc[0], e1["true_cpirn_usd"].iloc[0], e1["true_incremental_room_nights"].iloc[0]

# GPU
if len(GPU):
    g = GPU.copy()
    t_gpu = pd.DataFrame([{"Workload": r.workload, "Rows": f"{int(r.rows):,}", "CPU s": f(r.cpu_seconds, 2),
                           "GPU s": f(r.gpu_seconds, 2) if pd.notna(r.gpu_seconds) else "-",
                           "Speed-up": (f"{r.speedup:.2f}x" if pd.notna(r.speedup) else "-")} for r in g.itertuples()])
    big = g[g.rows == g.rows.max()].dropna(subset=["speedup"])
    small = g[g.rows == g.rows.min()].dropna(subset=["speedup"])
    def _fmt(d: pd.DataFrame) -> str:
        return "; ".join(f"{r.workload.split(' (')[0]} {r.speedup:.2f}x" for r in d.itertuples())

    losers = small[small.speedup < 1]
    gpu_text = (f"At {int(g.rows.min()):,} rows (about the size of this panel) the speed-ups are: {_fmt(small)}. "
                + (f"The gradient-boosting libraries ({', '.join(w.split(' (')[0] for w in losers.workload)}) are *slower* on the GPU there, because kernel-launch and transfer overhead dominate; "
                   if len(losers) else "")
                + f"at {int(g.rows.max()):,} rows the speed-ups are: {_fmt(big)}.")
else:
    t_gpu, gpu_text = pd.DataFrame(), "GPU benchmark not run."
if len(GPUC):
    piv = GPUC.pivot(index="learner", columns="mode", values="seconds").reset_index()
    order_l = ["S-Learner", "T-Learner", "X-Learner", "R-Learner", "DR-Learner", "Causal Forest DML", "Linear DML", "Domain Adaptation", "TARNet", "CFRNet-DragonNet"]
    piv = piv.set_index("learner").loc[order_l].reset_index()
    used = GPUC[GPUC["mode"] == "gpu"].set_index("learner")["used_gpu"] if "gpu" in set(GPUC["mode"]) else pd.Series(dtype=bool)
    t_gpuc = pd.DataFrame([{"Learner": r.learner, "CPU s": f(r.cpu, 1), "GPU s": f(r.gpu, 1) if "gpu" in piv else "-",
                            "Ran on GPU": "yes" if bool(used.get(r.learner, False)) else "no (CPU backend)"} for r in piv.itertuples()])
else:
    t_gpuc = pd.DataFrame()
GPUC_NOTE = ""
if len(GPUC) and "gpu" in set(GPUC["mode"]):
    _p = GPUC.pivot(index="learner", columns="mode", values="seconds")
    _used = GPUC[GPUC["mode"] == "gpu"].set_index("learner")["used_gpu"].astype(bool)
    _cap = [k for k in _p.index if _used.get(k, False)]
    _win = [k for k in _cap if _p.loc[k, "gpu"] < _p.loc[k, "cpu"]]
    GPUC_NOTE = (f"Of the {len(_cap)} learners that can use the GPU, {len(_win)} finish faster on it ({', '.join(_win) if _win else 'none'}). "
                 "On 1,500 rows the boosted-tree learners pay more in launch and transfer overhead than they save, so the default configuration runs them on the GPU for "
                 "scalability, not for speed at this size - use `--no-gpu` for the fastest small-panel runs.")
hw = next((t for t in tele if t.get("tag") == "start"), {})
gname = hw.get("gpu", {}).get("gpu_name") or hw.get("device", {}).get("gpu_name", "n/a")
peak = max([t.get("gpu", {}).get("mem_used_mb", 0) or 0 for t in tele] or [0])
jax = hw.get("jax", {})

es_m = ESM[(ESM.scenario == "mia_all24") & (ESM.outcome == "room_nights_booked")].iloc[0]
bay_h = BAY[(BAY.scenario == "mia_all24") & (BAY.outcome == "room_nights_booked")].iloc[0]
lin = drivers["linear_dml"]["coef_room_nights_per_sd"]
sub = drivers["subgroups"]
fll_row = LODO[(LODO.dropped.str.startswith("exclude")) & (LODO.estimator == "ASCM (Ridge)")]
lodo_asc = LODO[(LODO.estimator == "ASCM (Ridge)") & (LODO.dropped != "none")]
ascm = H.loc["ASCM (Ridge)"]
twfe = H.loc["TWFE DiD"]
best_cate = CATE[~CATE.reference].sort_values("pehe").iloc[0]
n_beat = int((CATE[~CATE.reference].pehe < cr.loc["Constant ATE (oracle)", "pehe"]).sum())
cp = cr.loc["Constant ATE (oracle)"]
orc = cr.loc["Oracle X-regression (CV)"]

SHOTS = [("api_swagger_docs.png", "FastAPI Swagger UI: the four service endpoints plus helpers"),
         ("api_cate_response.png", "Live `/predict/cate` call executed from Swagger (Causal Forest DML, with confidence intervals)"),
         ("streamlit_tab1_executive_iroas.png", "Tab 1 - Miami flagship preset with iROAS KPIs and calculator"),
         ("streamlit_tab2_did_event_study.png", "Tab 2 - parallel trends and event-study dynamics"),
         ("streamlit_tab3_synthetic_control.png", "Tab 3 - SCM vs. ASCM vs. SDID, donor weights, in-space placebo"),
         ("streamlit_tab4_cate_metalearners.png", "Tab 4 - ten-learner benchmark, what-if simulator, pacing studio"),
         ("streamlit_tab5_live_api.png", "Tab 5 - live FastAPI integration and batch scoring"),
         ("streamlit_preset2_las_vegas.png", "Preset 2 - Las Vegas convention blitz"),
         ("streamlit_preset3_new_york.png", "Preset 3 - New York spring break push"),
         ("streamlit_preset4_placebo_check.png", "Preset 4 - Tampa negative control")]
FIGS = [("01_parallel_trends_event_study.png", "Parallel trends and event study"),
        ("02_synthetic_control_counterfactual.png", "Synthetic-control counterfactual, daily and cumulative effect"),
        ("03_donor_weights_and_covariate_balance.png", "Donor weights and covariate balance"),
        ("04_in_space_placebo_permutation.png", "In-space placebo permutation"),
        ("05_panel_att_estimators_comparison.png", "Seven panel estimators vs. truth, plus Monte-Carlo check"),
        ("06_cate_meta_learners_comparison.png", "Ten CATE learners compared"),
        ("07_cate_heterogeneity_drivers.png", "What drives lift"),
        ("08_pytorch_uplift_training_curves.png", "PyTorch uplift training curves")]


def img(path: str, alt: str) -> str:
    return f"![{alt}]({path})"


vis = "\n\n".join(f"**{alt}**\n\n{img('outputs/figures/' + n, alt)}" for n, alt in FIGS)
shots = "\n\n".join(f"**{alt}**\n\n{img('outputs/screenshots/' + n, alt)}" for n, alt in SHOTS)

_sig = IT[IT["significant_at_5pct"]] if len(IT) else IT
IT_NOTE = ("Estimators that report a 'significant' effect where none exists: " + ", ".join(f"{r.estimator} ({r.att_pct_of_baseline:+.1f}% of baseline)" for r in _sig.itertuples())
           + ". Treat that estimator's interval as too narrow in this run.") if len(_sig) else "No estimator reports a significant effect, as it should."
_q = CATE[~CATE.reference]["qini_oracle"]
QINI_RANGE = f"{_q.min():.2f}-{_q.max():.2f}"
_tpa = S[(S.scenario == "tpa_placebo") & (S.outcome == "room_nights_booked")].set_index("estimator")
_tb, _ta = _tpa.loc["Bayesian SC"], _tpa.loc["ASCM (Ridge)"]
TPA_NOTE = (f"In this run the Bayesian credible interval for Tampa ({ci(_tb.ci_lo, _tb.ci_hi)}) **excludes zero** - a false positive - while the placebo-calibrated ASCM interval "
            f"({ci(_ta.ci_lo, _ta.ci_hi)}) correctly includes it. Posterior intervals are optimistic when donor weights are weakly identified, so treat the placebo-calibrated interval as the arbiter."
            ) if _tb.ci_lo > 0 else "Both the Bayesian and the placebo-calibrated intervals for Tampa include zero, as they should."
_ls = LODO[(LODO.estimator == "SCM") & (LODO.dropped != "none")]
_lss = _ls.loc[_ls.att_mean.idxmax()]
LODO_SCM = f"plain SCM is far more sensitive: it moves between {_ls.att_mean.min():,.0f} and {_ls.att_mean.max():,.0f} (the extreme is when '{_lss.dropped}' is dropped)"

PV = rd("pacing_validation.csv")
if len(PV):
    t_pace = pd.DataFrame([{"Pacing strategy (same $ budget)": r.strategy, "Projected vs even": f"{r.projected_vs_even_pct:+.1f}%",
                            "True vs even": f"{r.true_vs_even_pct:+.1f}%", "True incremental room nights": f(r.true_incremental_rn)}
                           for r in PV.itertuples()])
    pv_idx = PV.set_index("strategy")
    opt_proj, opt_true = pv_idx.loc["optimized", "projected_vs_even_pct"], pv_idx.loc["optimized", "true_vs_even_pct"]
    back_proj, back_true = pv_idx.loc["back_loaded", "projected_vs_even_pct"], pv_idx.loc["back_loaded", "true_vs_even_pct"]
else:
    t_pace, opt_proj, opt_true, back_proj, back_true = pd.DataFrame(), np.nan, np.nan, np.nan, np.nan

mcx = MC.set_index("estimator") if len(MC) else pd.DataFrame()


def mc_val(est: str, col: str, fmt_: str) -> str:
    return fmt_.format(mcx.loc[est, col]) if len(mcx) and est in mcx.index else "n/a"


PH = {
    "@@GPUC_NOTE@@": GPUC_NOTE,
    "@@IT_NOTE@@": IT_NOTE, "@@QINI_RANGE@@": QINI_RANGE, "@@TPA_NOTE@@": TPA_NOTE, "@@LODO_SCM@@": LODO_SCM,
    "@@T_PACE@@": table(t_pace) if len(t_pace) else "_not run_", "@@OPT_PROJ@@": f"{opt_proj:+.1f}%", "@@OPT_TRUE@@": f"{opt_true:+.1f}%",
    "@@BACK_PROJ@@": f"{back_proj:+.1f}%", "@@BACK_TRUE@@": f"{back_true:+.1f}%", "@@CRV_R2_RAW@@": f"{curve['fitted']['r2']:.2f}",
    "@@TWFE_MC@@": mc_val("TWFE DiD", "bias_pct", "{:+.0f}%"), "@@TWFE_MC_RMSE@@": mc_val("TWFE DiD", "rmse", "{:.0f}"),
    "@@ASCM_MC@@": mc_val("ASCM (Ridge)", "bias_pct", "{:+.0f}%"), "@@ASCM_MC_RMSE@@": mc_val("ASCM (Ridge)", "rmse", "{:.0f}"),
    "@@ASCM@@": f"{ascm.att_mean:,.0f}", "@@ASCM_LO@@": f"{ascm.ci_lo:,.0f}", "@@ASCM_HI@@": f"{ascm.ci_hi:,.0f}", "@@TRUTH@@": f"{truth:,.0f}",
    "@@TWFE@@": f"{twfe.att_mean:,.0f}", "@@TWFE_BIAS@@": f"{twfe.att_bias_pct:+.0f}%", "@@TOTAL_TRUE@@": f"{true_inc:,.0f}",
    "@@ASCM_TOTAL@@": f"{ascm.att_total:,.0f}", "@@IROAS@@": f"{e1.loc['ASCM (Ridge)', 'iroas']:.2f}", "@@IROAS_LO@@": f"{e1.loc['ASCM (Ridge)', 'iroas_lo']:.2f}",
    "@@IROAS_HI@@": f"{e1.loc['ASCM (Ridge)', 'iroas_hi']:.2f}", "@@TRUE_IROAS@@": f"{true_ir:.2f}", "@@CPIRN@@": f"{e1.loc['ASCM (Ridge)', 'cpirn_usd']:,.0f}",
    "@@TRUE_CPIRN@@": f"{true_cp:,.0f}", "@@PSMALL@@": "0.040", "@@T_HEAD@@": table(t_head), "@@T_OUT@@": table(t_out), "@@T_PRE@@": table(t_pre),
    "@@T_PV@@": table(t_pv), "@@T_IT@@": table(t_it), "@@T_LODO@@": table(t_lodo), "@@T_MC@@": table(t_mc) if len(t_mc) else "_not run_",
    "@@N_MC@@": str(n_mc), "@@T_CATE@@": table(t_cate), "@@T_ECON@@": table(t_econ), "@@T_GPU@@": table(t_gpu) if len(t_gpu) else "_not run_",
    "@@T_GPUC@@": table(t_gpuc) if len(t_gpuc) else "_not run_", "@@GPU_TEXT@@": gpu_text, "@@GNAME@@": str(gname), "@@PEAK@@": f"{peak:,.0f}",
    "@@JAXDEV@@": ", ".join(jax.get("devices", [])) or "n/a", "@@JAXBE@@": jax.get("backend", "n/a"),
    "@@ES_RMS@@": f"{es_m.pretrend_rms:.0f}", "@@ES_SLOPE@@": f"{es_m.pretrend_slope:+.1f}", "@@ES_P@@": f"{es_m.p_pretrend_perm:.2f}",
    "@@BAY_RHAT@@": f"{bay_h.rhat_max:.3f}", "@@BAY_ESS@@": f"{bay_h.ess_min:,.0f}", "@@BAY_DIV@@": str(int(bay_h.divergences)),
    "@@BAY_PPC@@": f"{100 * bay_h.ppc_pre_coverage:.0f}%", "@@BAY_BACKEND@@": str(bay_h.jax_backend),
    "@@LODO_MIN@@": f"{lodo_asc.att_mean.min():,.0f}", "@@LODO_MAX@@": f"{lodo_asc.att_mean.max():,.0f}",
    "@@FLL@@": f"{fll_row.att_mean.iloc[0]:,.0f}" if len(fll_row) else "n/a", "@@VIS@@": vis, "@@SHOTS@@": shots,
    "@@BEST@@": best_cate.learner, "@@BEST_PEHE@@": f"{best_cate.pehe:.1f}", "@@CONST@@": f"{cp.pehe:.1f}", "@@ORACLE@@": f"{orc.pehe:.1f}",
    "@@NBEAT@@": str(n_beat), "@@NSEEDS@@": str(int(CATE.n_seeds.max())), "@@CRV_THETA@@": f"{curve['fitted']['theta']:.2f}",
    "@@CRV_K@@": f"{curve['fitted']['K']:.0f}", "@@CRV_R2@@": f"{curve['fitted']['r2']:.2f}",
    "@@DGP_THETA@@": f"{curve['truth_dgp']['theta']:.2f}", "@@DGP_K@@": f"{curve['truth_dgp']['K']:.0f}",
    "@@WK_E@@": f"{sub['is_weekend']['est_high']:.0f}", "@@WK_T@@": f"{sub['is_weekend']['true_high']:.0f}",
    "@@WD_E@@": f"{sub['is_weekend']['est_low']:.0f}", "@@WD_T@@": f"{sub['is_weekend']['true_low']:.0f}",
    "@@HO_E@@": f"{sub['is_holiday_window']['est_high']:.0f}", "@@HO_T@@": f"{sub['is_holiday_window']['true_high']:.0f}",
    "@@NH_E@@": f"{sub['is_holiday_window']['est_low']:.0f}", "@@NH_T@@": f"{sub['is_holiday_window']['true_low']:.0f}",
    "@@COEF_TOP@@": ", ".join(f"`{k}` ({v:+.0f})" for k, v in sorted(lin.items(), key=lambda kv: -abs(kv[1]))[:3]),
}

TEMPLATE = (ROOT / "scripts" / "readme_template.md").read_text(encoding="utf-8")
for k, v in PH.items():
    TEMPLATE = TEMPLATE.replace(k, v)
assert "@@" not in TEMPLATE, [w for w in TEMPLATE.split() if "@@" in w][:5]
(ROOT / "README.md").write_text(TEMPLATE, encoding="utf-8")
print("README.md written:", len(TEMPLATE.splitlines()), "lines")
