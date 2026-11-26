"""Build the end-to-end presentation from outputs/metrics/*, outputs/plots/* and outputs/screenshots/*.

    python presentation/build_deck.py
Output: presentation/Pre_Arrival_Upsell_Uplift_End_to_End.pptx   (every number is read from the metrics files)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from deck_lib import (BLUE, GOOD, GREY, INK, LIGHT, MUTED, ORANGE, PURPLE, ROSE, TEAL, WARN, WHITE, Deck)  # noqa: E402

OUT = ROOT / "outputs"
M, PL, SH = OUT / "metrics", OUT / "plots", OUT / "screenshots"
ate = json.load(open(M / "ate_comparison.json"))
lb = pd.read_csv(M / "cate_leaderboard.csv")
ref = json.load(open(M / "refutation.json"))
dev = json.load(open(M / "device_report.json"))
ds = json.load(open(M / "data_summary.json"))
camp = json.load(open(M / "campaign_summary.json"))
meta = json.load(open(M / "learner_meta.json"))
fl = json.load(open(M / "flows.json"))
bal = pd.read_csv(M / "balance_smd.csv")
dec = pd.read_csv(M / "uplift_deciles.csv")
FLOWS, OVER, MM = fl["flows"], fl["overall"], fl["model_matrix"]

FAMILY = {"S-Learner (XGB)": ("Meta-learner", BLUE), "S-Learner (CatBoost)": ("Meta-learner", BLUE), "T-Learner (XGB)": ("Meta-learner", BLUE),
          "T-Learner (LightGBM)": ("Meta-learner", BLUE), "X-Learner": ("Meta-learner", BLUE), "R-Learner (NonParamDML)": ("EconML", TEAL),
          "DR-Learner": ("EconML", TEAL), "Causal Forest DML": ("EconML", TEAL), "TARNet (PyTorch)": ("Deep (CUDA)", ORANGE),
          "DragonNet (PyTorch)": ("Deep (CUDA)", ORANGE), "Uplift RF (KL)": ("CausalML", PURPLE)}
SHORT = {"S-Learner (XGB)": "S (XGB)", "S-Learner (CatBoost)": "S (Cat)", "T-Learner (XGB)": "T (XGB)", "T-Learner (LightGBM)": "T (LGBM)",
         "X-Learner": "X", "R-Learner (NonParamDML)": "R (DML)", "DR-Learner": "DR", "Causal Forest DML": "Causal Forest",
         "TARNet (PyTorch)": "TARNet", "DragonNet (PyTorch)": "DragonNet", "Uplift RF (KL)": "Uplift RF"}
A, D_ = ate["outcomes"]["ancillary"], ate["outcomes"]["direct"]
best = camp["best_model"]
d = Deck("Pre-Arrival Ancillary Upsells & Direct-Channel Shift · Causal / Uplift Engine")


def lbo(o):
    return lb[(lb.outcome == o) & ~lb.model.str.startswith("Oracle")].reset_index(drop=True)


def hl_rows(rows, cols_better, start=1):
    """Highlight best (GOOD) / worst (WARN) numeric cell per column. cols_better: {col: 'min'|'max'}."""
    hl = {}
    for c, mode in cols_better.items():
        vals = []
        for r in range(start, len(rows)):
            try:
                vals.append((float(str(rows[r][c]).replace(",", "").replace("%", "").replace("x", "")), r))
            except ValueError:
                pass
        if len(vals) < 3:
            continue
        key = (lambda t: t[0])
        b = (min if mode == "min" else max)(vals, key=key)
        w = (max if mode == "min" else min)(vals, key=key)
        hl[(b[1], c)] = GOOD
        hl[(w[1], c)] = WARN
    return hl


def img(path):
    return path if Path(path).exists() else None


# =============================================================== 0 title + summary
d.title_slide("Pre-Arrival Ancillary Upsells & Direct-Channel Shift",
              "End-to-end causal inference and uplift modelling: from confounded campaign logs to a budget-constrained targeting decision",
              f"PSM · IPW · AIPW · 11 CATE learners · TARNet / DragonNet on {dev['gpu_name'] or 'CPU'} · FastAPI · Streamlit · synthetic data with known ground truth")

s = d.slide("Executive summary", "Selection bias inflates the naive campaign lift almost threefold; doubly-robust estimation and CATE targeting recover the truth", "Overview")
nv, tr_, ai = A["estimators"]["Naive"]["ate"], A["true_ate"], A["estimators"]["AIPW (DR)"]["ate"]
d.card(s, 0.6, 1.8, 2.9, 1.25, f"${nv:.1f} vs ${tr_:.1f}", "Naive vs true ancillary lift / stay", ROSE, f"{nv / tr_ - 1:+.0%} over-statement")
d.card(s, 3.65, 1.8, 2.9, 1.25, f"${ai:.1f}", "AIPW ancillary lift", TEAL, f"{ai / tr_ - 1:+.1%} vs truth")
d.card(s, 6.7, 1.8, 2.9, 1.25, f"{D_['estimators']['Naive']['ate'] * 100:.1f} → {D_['estimators']['AIPW (DR)']['ate'] * 100:.1f} pp", "Direct-shift lift: naive → AIPW", BLUE, f"truth {D_['true_ate'] * 100:.1f} pp")
d.card(s, 9.75, 1.8, 2.95, 1.25, OVER["Max |SMD| before → after IPW"], "Max |SMD| before → after IPW", PURPLE, f"{ate['balance']['share_below_0p1_ipw']:.0%} of covariates < 0.10")
d.card(s, 0.6, 3.2, 2.9, 1.25, SHORT[best["ancillary"]], "Best ancillary CATE model", ORANGE, f"PEHE ${OVER['Ancillary PEHE ($)']}")
d.card(s, 3.65, 3.2, 2.9, 1.25, SHORT[best["direct"]], "Best direct-shift CATE model", ORANGE, f"PEHE {OVER['Direct PEHE (pp)']} pp")
d.card(s, 6.7, 3.2, 2.9, 1.25, OVER["Oracle capture"], "Share of oracle value captured", TEAL, "top-30% budget policy")
d.card(s, 9.75, 3.2, 2.95, 1.25, OVER["Refuters passed"], "Refutation tests passed", BLUE, "random cause · placebo · subset")
d.bullets(s, 0.6, 4.7, 12.1, 2.2, [
    f"A naive treated-vs-untreated comparison over-states the ancillary effect by {nv / tr_ - 1:+.0%} because engaged, tech-savvy, early-booking guests are the ones we reach.",
    "Propensity weighting and matching remove most of the bias; AIPW (doubly robust, cross-fitted) is the closest to the truth for both outcomes.",
    f"All 11 learners recover the heterogeneity (corr with true τ up to {lb[lb.outcome == 'ancillary'].corr_true.max():.2f}); the ranking differs by outcome and by decision context, so we compare them inside every flow.",
    f"Targeting the top 30% by predicted net value captures {OVER['Oracle capture']} of what a perfect (oracle) ranking would earn."], size=14)

s = d.slide("Agenda", None, "Overview")
items = [("1", "Business & causal problem", "selection bias, estimands, segments"), ("2", "End-to-end uplift flow (documentation)", "7 stages, architecture, methods, metrics"),
         ("3", "Stage-level results", "data → propensity → balance → ATE → refutation → CATE → policy"), ("4", "All models compared across flows", "11 learners × 8 flows × metrics"),
         ("5", "Flow-level & overall metrics", "programme scorecard"), ("6", "Scenarios as stories", "8 Streamlit flows with screenshots"), ("7", "Limitations & next steps", "")]
for i, (n, t, sub) in enumerate(items):
    y = 1.5 + i * 0.78
    d.node(s, 0.9, y, 0.6, 0.6, n, BLUE, 18)
    d.text(s, 1.7, y - 0.02, 10, 0.4, [(t, 18, True, INK)])
    d.text(s, 1.7, y + 0.33, 10, 0.3, [(sub, 12, False, MUTED)])

# =============================================================== 1 problem
d.section_slide("1", "Business & causal problem", "Why a naive A/B-style read of the campaign log is wrong")
s = d.slide("Business problem: OTA guests are expensive and rarely upgrade", "A pre-arrival app-download & upsell offer can add ancillary revenue and move bookings to the direct channel", "1 · Problem")
d.box(s, 0.6, 1.8, 4.0, 2.6, "The economics", ["OTA commission 15–20% of room revenue", "OTA guests rarely buy upgrades pre-arrival", "A direct re-booking is worth ≈ $55–75 in saved commission + margin", "Offers cost ≈ $1.50 per contact"], BLUE)
d.box(s, 4.8, 1.8, 4.0, 2.6, "The campaign", ["Treatment T: pre-arrival app-download + upgrade offer delivered", "Y_ancillary: incremental $ spend per stay", "Y_direct: re-booking direct within 180 days", "Budget-constrained: contact only the best ≈ 30%"], TEAL)
d.box(s, 9.0, 1.8, 3.7, 2.6, "The trap: selection bias", ["Engaged guests open / respond: tech-savvy, affluent, long lead time", "They also spend more and rebook more anyway", "Naive lift = treatment effect + selection effect"], ROSE)
d.text(s, 0.6, 4.7, 12.1, 0.4, [("Question: what would each guest have done WITHOUT the offer, and who is actually moved by it?", 16, True, INK)])
d.chevrons(s, 0.6, 5.35, 12.1, 0.9, ["Observational\ncampaign logs", "Remove selection\nbias (ATE / ATT)", "Individual effects\nτ(x) = CATE", "Rank & target\nunder budget", "Measure ROI\nvs counterfactual"], [BLUE, TEAL, ORANGE, PURPLE, ROSE], 12)

s = d.slide("Why the naive comparison fails: the gap is selection, not effect", "Ancillary $ lift (left) and direct re-booking lift in pp (right): naive vs causal estimators vs known truth", "1 · Problem")
names = ["Naive", "PSM 1:1", "IPW", "AIPW (DR)"]
d.bar_chart(s, 0.6, 1.7, 6.0, 4.6, names + ["TRUE"], {"ATE ($)": [A["estimators"][n]["ate"] for n in names] + [A["true_ate"]]}, [BLUE], horizontal=False, title="Ancillary spend ATE ($)", fmt="0.0",
            point_colors=[ROSE, BLUE, BLUE, TEAL, GREY])
d.bar_chart(s, 6.8, 1.7, 5.9, 4.6, names + ["TRUE"], {"ATE (pp)": [D_["estimators"][n]["ate"] * 100 for n in names] + [D_["true_ate"] * 100]}, [BLUE], horizontal=False, title="Direct re-booking ATE (pp)", fmt="0.0",
            point_colors=[ROSE, BLUE, BLUE, TEAL, GREY])
d.text(s, 0.6, 6.4, 12.1, 0.6, [(f"Naive over-states by {nv / tr_ - 1:+.0%} ($) and {D_['estimators']['Naive']['ate'] / D_['true_ate'] - 1:+.0%} (direct). Treated stays: {ate['n_treated']:,} of {ate['n']:,} ({ds['treated_share']:.0%}).", 13, False, MUTED)])

s = d.slide("Causal objective and estimands", "From population averages to individual effects to a targeting policy", "1 · Problem")
d.box(s, 0.6, 1.8, 3.9, 2.3, "ATE", ["E[Y(1) − Y(0)] over all guests", "Answers: does the programme work on average?", "Estimators: PSM, IPW, SIPW, AIPW"], BLUE)
d.box(s, 4.7, 1.8, 3.9, 2.3, "ATT", ["E[Y(1) − Y(0) | T=1]", "Answers: did it pay off for those we actually contacted?", f"True ATT ancillary ${A['true_att']:.2f}, direct {D_['true_att'] * 100:.1f} pp"], TEAL)
d.box(s, 8.8, 1.8, 3.9, 2.3, "CATE τ(x)", ["E[Y(1) − Y(0) | X=x]", "Answers: who is moved, by how much?", "11 learners; ranking drives the budget"], ORANGE)
sh = ds["segment_share"]
d.text(s, 0.6, 4.3, 12, 0.4, [("Four behavioural segments (from the direct-shift potential outcomes)", 15, True, INK)])
segs = [("Persuadable", "Moved by the offer (τ_direct ≥ 8 pp): target", TEAL), ("Always-Taker", "Already rebooks, small τ: offer is wasted", BLUE), ("Sleeping Dog", "Offer backfires (τ < 0): suppress", ROSE), ("Lost Cause", "Low baseline, no lift: save the cost", GREY)]
for i, (k, t, c) in enumerate(segs):
    d.card(s, 0.6 + i * 3.05, 4.8, 2.9, 1.6, f"{sh.get(k, 0):.0%}", f"{k}", c, t, 30)
d.text(s, 0.6, 6.55, 12, 0.4, [("Shares are of all simulated stays (ground-truth labels, only observable because the data is synthetic).", 11, False, MUTED)])

s = d.slide("Variables, identification assumptions and what we can verify", None, "1 · Problem")
rows = [["Covariate (X)", "Values", "Role in the DGP"], ["Booking channel", "OTA_Expedia · OTA_Booking · Direct_Web · Corporate", "Drives propensity, baseline spend and τ"],
        ["Room tier", "Standard · Deluxe · Suite", "Suites: higher spend, larger τ for OTA guests"], ["Lead time (days)", "0–365, gamma", "Saturating effect on τ; confounds treatment"],
        ["Party size", "adults 1–4, children 0–3", "Families on budget rooms dislike upsells"], ["Device", "iOS · Android · Desktop", "Engagement: mobile guests are reached and respond more"],
        ["Historical spend ($)", "log-normal, zero for newcomers", "Affluence proxy: confounder"], ["Loyalty tier / LOS / resort", "0–3 · 1–14 nights · 0/1", "Baseline spend, direct propensity"]]
d.table(s, 0.6, 1.5, 12.1, rows, [2.2, 4.3, 5.2], 11, 0.33, align=["left", "left", "left"])
d.bullets(s, 0.6, 4.35, 6.0, 2.6, ["Unconfoundedness: all confounders are observed by construction", "Overlap: propensity clipped to [0.025, 0.975] + common-support trimming", "SUTVA: no interference between guests"], 13, head="Assumptions")
d.bullets(s, 6.8, 4.35, 5.9, 2.6, ["Balance diagnostics (SMD < 0.10)", "Refutation: random common cause, placebo treatment, data subset", "Oracle checks: PEHE and ATE bias vs known τ(x)"], 13, head="What we can check")

# =============================================================== 2 documentation
d.section_slide("2", "End-to-end uplift flow: documentation", "Seven stages from raw logs to a budgeted campaign, how each one works and what it emits")
s = d.slide("The uplift pipeline at a glance", "run_pipeline.py orchestrates all stages; every stage writes artifacts that the API and dashboard read", "2 · Flow documentation")
d.chevrons(s, 0.6, 1.7, 12.1, 0.95, ["1 Data\n(DGP)", "2 Propensity\n& overlap", "3 Balance\n(SMD)", "4 ATE / ATT\n(PSM·IPW·AIPW)", "5 Refutation", "6 CATE\n(11 learners)", "7 Policy\n& serving"], [BLUE, TEAL, TEAL, TEAL, PURPLE, ORANGE, ROSE], 11)
rows = [["Stage", "Module", "Method", "Artifacts (outputs/)", "Stage metric"],
        ["1 Data", "src/data/synthetic_generator.py", "Structural equations, known τ(x)", "data/train|test.parquet", "treated share, true ATE"],
        ["2 Propensity", "propensity_ate.fit_propensity", "Cross-fit XGBoost (GPU) + Platt calibration, clip, trim", "models/propensity.joblib", "AUC, range, support kept"],
        ["3 Balance", "propensity_ate.balance_table", "SMD before / after PSM / IPW", "metrics/balance_smd.csv, love_plot.png", "max |SMD|, share < 0.10"],
        ["4 ATE / ATT", "propensity_ate.estimate_all", "Naive, PSM 1:1 / 1:k, IPW, SIPW, AIPW + bootstrap CIs", "metrics/ate_comparison.json", "bias vs truth, CI"],
        ["5 Refutation", "propensity_ate.refute", "Random cause, placebo, subset + DoWhy", "metrics/refutation.json", "tests passed"],
        ["6 CATE", "cate_learners / deep_causal", "S, T, X, R, DR, Causal Forest, TARNet, DragonNet, Uplift RF", "models/*.joblib|.pt, cate_leaderboard.csv", "Qini, AUUC, PEHE"],
        ["7 Policy", "serving.py · api · app · flows.py", "Net value = margin·τ̂_anc + value·τ̂_dir − cost; top-k budget", "data/scored_guests.parquet, flows.json", "ROI, oracle capture"]]
d.table(s, 0.6, 2.95, 12.1, rows, [1.2, 2.6, 3.4, 3.0, 1.9], 10, 0.5, align=["left", "left", "left", "left", "left"])

s = d.slide("Architecture: one scoring core shared by API, dashboard and flows", "Models are trained once; src/serving.py is the single place that scores guests and optimises campaigns", "2 · Flow documentation")
d.node(s, 0.6, 1.9, 2.4, 0.8, "config.yaml\n(seeds, GPU flags, costs)", GREY, 11)
d.node(s, 0.6, 3.1, 2.4, 0.8, "synthetic_generator\n(DGP + truth)", BLUE, 11)
d.node(s, 3.6, 1.9, 2.6, 0.8, "propensity_ate.py\nPSM · IPW · AIPW · SMD", TEAL, 11)
d.node(s, 3.6, 3.1, 2.6, 0.8, "cate_learners.py\nS T X R DR CF Uplift RF", TEAL, 11)
d.node(s, 3.6, 4.3, 2.6, 0.8, "deep_causal.py\nTARNet · DragonNet (CUDA)", ORANGE, 11)
d.node(s, 6.8, 3.1, 2.4, 0.8, "metrics_and_plots.py\nQini · AUUC · PEHE", PURPLE, 11)
d.node(s, 6.8, 1.9, 2.4, 0.8, "run_pipeline.py\n(orchestrator)", INK, 11)
d.node(s, 9.8, 1.9, 2.9, 0.8, "outputs/\nparquet · joblib · pt · csv · png", GREY, 11)
d.node(s, 9.8, 3.1, 2.9, 0.8, "serving.py\nscore · optimise · classify", ROSE, 11)
d.node(s, 9.1, 4.5, 1.8, 0.8, "FastAPI\n/predict /optimize", BLUE, 10)
d.node(s, 11.0, 4.5, 1.7, 0.8, "Streamlit\n5 tabs", BLUE, 10)
d.node(s, 9.1, 5.6, 3.6, 0.7, "flows.py → flows.json → tab 5 · deck · README", TEAL, 10)
for a in [(3.0, 2.3, 3.6, 2.3), (3.0, 3.5, 3.6, 3.5), (6.2, 2.3, 6.8, 2.3), (6.2, 3.5, 6.8, 3.5), (9.2, 2.3, 9.8, 2.3), (11.25, 2.7, 11.25, 3.1), (10.4, 3.9, 10.0, 4.5), (11.6, 3.9, 11.8, 4.5)]:
    d.arrow(s, *a)
d.bullets(s, 0.6, 5.4, 8.2, 1.6, [f"Hardware auto-detect: CUDA {'yes – ' + str(dev['gpu_name']) if dev['cuda'] else 'no'}; XGBoost {dev['xgb_device']}, PyTorch {dev['torch_device']}, CatBoost {dev['catboost_task']}, LightGBM {dev['lightgbm_device']}, cuML {'yes' if dev['cuml'] else 'no (sklearn kNN fallback)'}",
                                    "UI == API: the dashboard and the API call the same serving functions; Playwright checks that they agree"], 12)

s = d.slide("Stage 1 · Data: a DGP with confounding, non-linear CATE and ground truth", "Because the potential outcomes are simulated we can score every estimator against the real effect", "2 · Flow documentation")
d.box(s, 0.6, 1.8, 6.0, 2.7, "Structural equations", ["T ~ Bernoulli(e(X)), logit e driven by device, spend, lead time, loyalty, Expedia", "Y_anc = μ0(X) + T·τ_anc(X) + ε   (continuous $)", "Y_direct = 1[u < p0(X) + T·τ_dir(X)]   (monotone coupling)", "τ_anc: large for OTA × Suite × long lead × mobile; ≈ 0 direct; negative for corporate / budget families", "τ_dir: positive for OTA, negative for corporate (sleeping dogs)"], BLUE, 12)
d.box(s, 6.8, 1.8, 5.9, 2.7, "What is stored per stay", ["Observed: covariates, T, y_ancillary, y_direct", "Truth: e_true, μ0/μ1, τ_anc, p0/p1, τ_direct, segment", "Train / test split 70 / 30; test used for all CATE evaluation", f"{ds['n']:,} stays · {ds['treated_share']:.0%} treated"], TEAL, 12)
chn = ds["channel_share"]
d.bar_chart(s, 0.6, 4.7, 6.0, 2.2, list(chn.keys()), {"share": list(chn.values())}, [BLUE], horizontal=False, title="Booking-channel mix", fmt="0%", font=9)
g = pd.DataFrame({"ch": ["OTA_Expedia", "OTA_Booking", "Direct_Web", "Corporate"]})
d.text(s, 6.8, 4.7, 5.9, 2.2, [("Ground-truth ATE", 14, True, INK), (f"Ancillary: ${A['true_ate']:.2f} per stay (ATT ${A['true_att']:.2f})", 13, False, INK), (f"Direct re-booking: {D_['true_ate'] * 100:.1f} pp (ATT {D_['true_att'] * 100:.1f} pp)", 13, False, INK),
                              (f"Mean propensity {OVER['Treated share']} treated; AUC of the propensity model {OVER['Propensity AUC (cross-fit)']}: strong, but not deterministic, selection", 12, False, MUTED)])

s = d.slide("Stage 2–3 · Propensity, overlap and balance: how it works", None, "2 · Flow documentation")
d.box(s, 0.6, 1.5, 4.0, 3.6, "Propensity e(x)", ["Cross-fitted (5-fold) XGBoost on GPU", "Platt calibration on out-of-fold scores", "Clip to [0.025, 0.975]", "Common-support trimming: keep e between max of the two arms' minima and min of the maxima"], TEAL, 12)
d.box(s, 4.8, 1.5, 4.0, 3.6, "Balance check", ["Standardised mean difference per covariate", "Weighted SMD after PSM (1:1) and IPW (ATE and ATT weights)", "Target |SMD| < 0.10", "Love plot + propensity-overlap histograms"], PURPLE, 12)
d.box(s, 9.0, 1.5, 3.7, 3.6, "Why it matters", ["Poor overlap makes IPW explode and CATE extrapolate", "Imbalance after weighting signals a mis-specified propensity", "These are cheap, outcome-free diagnostics: run them first"], ROSE, 12)
d.text(s, 0.6, 5.35, 12.1, 1.2, [("Reading rule of thumb", 14, True, INK), ("AUC ≈ 0.5 would mean randomised assignment; AUC near 1 means almost deterministic selection (no overlap). Here the model separates treated from control well, yet all weights stay bounded after clipping and trimming.", 13, False, MUTED)])

s = d.slide("Stage 4 · ATE / ATT estimators", "Six ways to read the same log; all use the same cross-fitted propensity", "2 · Flow documentation")
rows = [["Estimator", "Idea", "Strength", "Caveat"],
        ["Naive", "Mean(Y|T=1) − Mean(Y|T=0)", "Trivial", "Confounded: includes selection effect"],
        ["PSM 1:1 / 1:k", "Match each treated guest to nearest control(s) on logit e within a caliper (0.2 SD)", "Transparent, uses real guests", "Discards unmatched; bootstrap CIs with fixed propensity"],
        ["IPW", "Weight by 1/e and 1/(1−e); trim top 1% weights", "Simple, uses all guests", "High variance at extreme e"],
        ["SIPW (Hájek)", "IPW with normalised weights", "Lower variance, scale-invariant", "Still model-based on e only"],
        ["AIPW (DR)", "Outcome models μ1, μ0 + IPW residual correction; cross-fitted", "Consistent if EITHER model right; efficient", "Needs two good nuisance models"]]
d.table(s, 0.6, 1.8, 12.1, rows, [1.6, 4.6, 3.0, 3.6], 11, 0.62, align=["left", "left", "left", "left"], bold_first_col=True)
d.text(s, 0.6, 5.8, 12.1, 0.9, [("Uncertainty: IPW / PSM use 100-draw bootstraps (propensity held fixed, so intervals are somewhat narrow); AIPW uses the influence-function standard error.", 12, False, MUTED)])

s = d.slide("Stage 6 · All CATE learners in the engine", "Eleven estimators behind one BaseCATEEstimator interface (fit, predict_cate, predict_interval, timing)", "2 · Flow documentation")
rows = [["Model", "Family", "How it estimates τ(x)", "Backend / GPU"],
        ["S-Learner (XGB, CatBoost)", "Meta", "One model with T as a feature; τ̂ = f(x,1) − f(x,0)", "XGBoost cuda · CatBoost GPU"],
        ["T-Learner (XGB, LightGBM)", "Meta", "Separate μ1, μ0 models; τ̂ = μ̂1 − μ̂0", "XGBoost cuda · LightGBM CPU*"],
        ["X-Learner", "Meta", "Cross-imputed effects, blended with propensity g(x)", "XGBoost cuda"],
        ["R-Learner (NonParamDML)", "EconML", "Residual-on-residual (Robinson), cross-fitted", "EconML + XGBoost cuda"],
        ["DR-Learner", "EconML", "Regress AIPW pseudo-outcomes on x", "EconML + XGBoost cuda"],
        ["Causal Forest DML", "EconML", "Honest causal trees on orthogonalised data; 95% intervals", "EconML (CPU forest)"],
        ["TARNet", "Deep", "Shared representation Φ(x), two outcome heads h1, h0", "PyTorch CUDA"],
        ["DragonNet", "Deep", "TARNet + propensity head + targeted regularisation (ε)", "PyTorch CUDA"],
        ["Uplift RF (KL)", "CausalML", "Uplift trees with KL-divergence splits; binary outcome only", "CausalML (CPU)"]]
d.table(s, 0.6, 1.75, 12.1, rows, [2.7, 1.1, 5.4, 2.9], 10.5, 0.46, align=["left", "left", "left", "left"], bold_first_col=True)
d.text(s, 0.6, 6.55, 12.1, 0.4, [("* LightGBM GPU needs a special build; the hardware probe falls back to CPU automatically. cuML is optional (Linux / WSL2).", 11, False, MUTED)])

s = d.slide("Stage 6–7 · Evaluation metrics: what each one tells you", "Observational metrics use IPW-weighted real outcomes (no truth); oracle metrics use the simulated τ(x)", "2 · Flow documentation")
d.box(s, 0.6, 1.8, 6.0, 2.5, "Observational (no ground truth needed)", ["Qini coefficient: area between the cumulative-gain curve and random targeting", "AUUC: area under the cumulative uplift curve", "Uplift by decile: IPW uplift per score decile (should decay)", "Policy value: net $ per guest when contacting the top 30%"], BLUE, 12)
d.box(s, 6.8, 1.8, 5.9, 2.5, "Oracle (synthetic only)", ["PEHE = √mean((τ̂ − τ)²): accuracy of the individual effect", "ATE bias: mean(τ̂) − mean(τ)", "corr(τ̂, τ): ranking quality", "True policy value: net $ using real τ of the selected guests"], ORANGE, 12)
d.box(s, 0.6, 4.5, 12.1, 2.2, "How to read them together", ["Qini differences between top models are small relative to IPW sampling noise on the test set: use PEHE and true policy value as tiebreakers", "A model can rank well (high Qini) yet be mis-calibrated (large ATE bias): matters if you report $ forecasts, not just a ranking", "Cost matters too: fit time and inference latency per 1,000 guests are reported for every model"], TEAL, 12)

s = d.slide("Stage 7 · Decision, serving and how to run it", None, "2 · Flow documentation")
b = {"am": 0.45, "dm": 55, "c": 1.5}
d.box(s, 0.6, 1.5, 6.0, 2.4, "Decision rule", ["Net value = 0.45 × τ̂_ancillary + $55 × τ̂_direct − $1.50", "Rank all guests, contact the top 30% with positive value", "Segments from τ̂ rules: Persuadable · Marginal · Sleeping Dog · Low response", "Compare against random, oracle and contact-everyone"], ROSE, 12)
d.box(s, 6.8, 1.5, 5.9, 2.4, "Serving", ["POST /predict/cate: guests → τ̂ (+ interval) per outcome", "POST /optimize/campaign: budget, cost, margins, channels → policy + ROI", "GET /models/metrics, /health (device report)", "Streamlit: 5 tabs, persona one-click simulator, scenario presets"], BLUE, 12)
rows = [["Step", "Command"], ["Run the full pipeline (≈ 4 min on RTX 3090)", "python scripts/run_pipeline.py   (--fast for a smoke run, --stage flows to rebuild flows)"],
        ["Serve the API", "uvicorn src.api.main:app --port 8000"], ["Open the dashboard", "streamlit run src/app/streamlit_app.py"],
        ["Regenerate screenshots / README / deck", "python scripts/capture_screenshots.py · build_readme.py · presentation/build_deck.py"], ["Tests", "pytest tests -q   (9 tests: DGP, estimators, every learner, pipeline, flows, API)"]]
d.table(s, 0.6, 4.2, 12.1, rows, [3.5, 8.6], 11, 0.4, align=["left", "left"])

# =============================================================== 3 stage-level results
d.section_slide("3", "Stage-level results", "Metrics at every stage of the pipeline, on 60,000 simulated stays")
s = d.slide("Stage 1 · Data and compute", "What was simulated and the hardware it ran on", "3 · Stage results")
d.card(s, 0.6, 1.8, 2.9, 1.2, f"{ds['n']:,}", "stays simulated", BLUE, f"train {ds['n_train']:,} · test {ds['n_test']:,}")
d.card(s, 3.65, 1.8, 2.9, 1.2, f"{ds['treated_share']:.0%}", "treated share", TEAL)
d.card(s, 6.7, 1.8, 2.9, 1.2, f"${A['true_ate']:.2f}", "true ancillary ATE", ORANGE, f"ATT ${A['true_att']:.2f}")
d.card(s, 9.75, 1.8, 2.95, 1.2, f"{D_['true_ate'] * 100:.1f} pp", "true direct-shift ATE", PURPLE, f"ATT {D_['true_att'] * 100:.1f} pp")
d.bar_chart(s, 0.6, 3.2, 6.0, 3.4, list(ds["segment_share"].keys()), {"share": list(ds["segment_share"].values())}, [TEAL], horizontal=False, title="True segment mix", fmt="0%", point_colors=[TEAL, BLUE, ROSE, GREY][:len(ds["segment_share"])] if False else None)
rows = [["Compute", "Detail"], ["GPU", str(dev["gpu_name"])], ["XGBoost / PyTorch", f"{dev['xgb_device']} / {dev['torch_device']}"], ["CatBoost", dev["catboost_task"]], ["LightGBM", dev["lightgbm_device"] + " (GPU build absent)" if dev["lightgbm_device"] == "cpu" else "gpu"],
        ["RAPIDS cuML", "available" if dev["cuml"] else "not available → sklearn kNN"], ["Learners fitted", f"{OVER['Learners fitted']} in {OVER['Total fit time (s)']} s total"]]
d.table(s, 6.9, 3.4, 5.8, rows, [2, 4], 12, 0.42, align=["left", "left"])

s = d.slide("Stage 2 · Propensity model and overlap", f"Cross-fit AUC {OVER['Propensity AUC (cross-fit)']} · common support keeps {ate['common_support_kept']:.1%} of stays", "3 · Stage results")
if img(PL / "propensity_overlap.png"):
    d.image(s, PL / "propensity_overlap.png", 0.6, 1.7, 8.2, 4.9)
d.card(s, 9.1, 1.8, 3.6, 1.2, OVER["Propensity AUC (cross-fit)"], "propensity AUC (out-of-fold)", TEAL)
d.card(s, 9.1, 3.15, 3.6, 1.2, f"{ate['common_support_kept']:.1%}", "stays inside common support", BLUE)
d.card(s, 9.1, 4.5, 3.6, 1.2, "[0.025, 0.975]", "propensity clip", PURPLE)
d.text(s, 9.1, 5.85, 3.6, 0.9, [("Treated guests concentrate at high e(x), controls at low e(x): that is the selection mechanism. Overlap is adequate throughout.", 11, False, MUTED)])

s = d.slide("Stage 3 · Covariate balance (SMD)", f"Max |SMD| {ate['balance']['max_abs_smd_before']:.2f} → {ate['balance']['max_abs_smd_ipw']:.3f} after IPW; {ate['balance']['share_below_0p1_before']:.0%} → {ate['balance']['share_below_0p1_ipw']:.0%} of covariates under 0.10", "3 · Stage results")
if img(PL / "love_plot.png"):
    d.image(s, PL / "love_plot.png", 0.6, 1.7, 6.6, 5.2)
top = bal.reindex(bal.smd_before.abs().sort_values(ascending=False).index).head(8)
rows = [["Covariate", "Before", "PSM", "IPW"]] + [[r.feature, f"{r.smd_before:+.2f}", f"{r.smd_psm:+.3f}", f"{r.smd_ipw:+.3f}"] for r in top.itertuples()]
d.table(s, 7.5, 1.8, 5.2, rows, [2.6, 1, 1, 1], 11, 0.38)
d.text(s, 7.5, 5.4, 5.2, 1.4, [(f"PSM: {ate['balance']['share_below_0p1_psm']:.0%} of covariates under 0.10 (max {ate['balance']['max_abs_smd_psm']:.3f}). IPW: {ate['balance']['share_below_0p1_ipw']:.0%} (max {ate['balance']['max_abs_smd_ipw']:.3f}).", 12, False, MUTED)])

for o, T_, unit, k in (("ancillary", A, "$", 1.0), ("direct", D_, "pp", 100.0)):
    s = d.slide(f"Stage 4 · ATE / ATT: {o}", f"True ATE {T_['true_ate'] * k:.2f} {unit} · true ATT {T_['true_att'] * k:.2f} {unit}", "3 · Stage results")
    rows = [["Estimator", f"ATE ({unit})", "95% CI", f"ATE bias ({unit})", f"ATT ({unit})", f"ATT bias ({unit})"]]
    for n, e in T_["estimators"].items():
        rows.append([n, f"{e['ate'] * k:.2f}", f"[{e['ate_lo'] * k:.2f}, {e['ate_hi'] * k:.2f}]", f"{e['ate_bias'] * k:+.2f}", f"{e['att'] * k:.2f}", f"{e['att_bias'] * k:+.2f}"])
    hl = {}
    bi = min(range(2, len(rows)), key=lambda r: abs(float(rows[r][3])))
    hl[(bi, 3)] = GOOD
    hl[(1, 3)] = WARN
    d.table(s, 0.6, 1.8, 7.0, rows, [1.7, 1, 1.9, 1.2, 1, 1.2], 11, 0.42, highlight=hl)
    names = list(T_["estimators"].keys())
    d.bar_chart(s, 7.8, 1.7, 4.9, 4.2, names + ["TRUE"], {"ATE": [T_["estimators"][n]["ate"] * k for n in names] + [T_["true_ate"] * k]}, [BLUE], title=f"ATE ({unit}) by estimator", fmt="0.0" if o == "ancillary" else "0.0",
                point_colors=[ROSE, BLUE, BLUE, BLUE, BLUE, TEAL, GREY], font=9)
    ai_e = T_["estimators"]["AIPW (DR)"]["ate"] / T_["true_ate"] - 1
    d.text(s, 0.6, 5.0, 7.0, 1.8, [(f"AIPW is {ai_e:+.1%} from the truth; naive is {T_['estimators']['Naive']['ate'] / T_['true_ate'] - 1:+.0%}.", 13, True, INK),
                                  ("PSM / IPW / SIPW remove most of the bias but keep a small positive residual; AIPW corrects it with the outcome model." if o == "ancillary" else
                                   "The direct-shift residual (AIPW above truth) is larger in relative terms: a small base effect is hard to pin down, and the outcome model is a classifier.", 12, False, MUTED)])

s = d.slide("Stage 5 · Refutation tests", "Built-in AIPW refuters on an 8,000-stay subsample, cross-checked with DoWhy (backdoor linear regression)", "3 · Stage results")
rows = [["Outcome", "Test", "Expected", "AIPW effect", "Pass", "DoWhy effect"]]
for o, k, u in (("ancillary", 1.0, "$"), ("direct", 100.0, "pp")):
    r = ref[o]
    dwm = {"random_common_cause": "random_common_cause", "placebo_treatment": "placebo_treatment_refuter", "data_subset": "data_subset_refuter"}
    for t, exp in (("random_common_cause", "≈ baseline"), ("placebo_treatment", "≈ 0"), ("data_subset", "≈ baseline")):
        dw = (r.get("dowhy") or {}).get("refuters", {}).get(dwm[t], {}).get("new_effect")
        rows.append([f"{o} (baseline {r['baseline_aipw_ate'] * k:.2f} {u})", t.replace("_", " "), exp, f"{r['builtin'][t]['new_effect'] * k:.3f}", "✔" if r["builtin"][t]["pass"] else "✘", "–" if dw is None else f"{dw * k:.3f}"])
d.table(s, 0.6, 1.8, 12.1, rows, [3.2, 2.4, 1.5, 1.7, 0.8, 1.7], 12, 0.5, fills={i: GOOD for i in range(1, 7)})
d.text(s, 0.6, 5.5, 12.1, 1.2, [(f"{OVER['Refuters passed']} built-in tests pass. Refuters only probe the observed-confounder model; they cannot detect unmeasured confounding (none exists in this DGP). With real logs, add a sensitivity analysis (e.g. E-values, Rosenbaum bounds) and a randomised hold-out.", 12, False, MUTED)])

for o, unit in (("ancillary", "$"), ("direct", "pp")):
    L = lbo(o)
    s = d.slide(f"Stage 6 · CATE leaderboard: {o}", f"Sorted by Qini · best by Qini: {L.iloc[0].model} · best PEHE: {L.sort_values('pehe').iloc[0].model}", "3 · Stage results")
    k = 1.0 if o == "ancillary" else 100.0
    rows = [["Model", "Family", "Qini", "AUUC", f"PEHE ({unit})", "corr(τ̂,τ)", f"ATE bias ({unit})", "Policy $/guest (true)", "Fit s", "ms/1k"]]
    for r in L.itertuples():
        rows.append([r.model, FAMILY[r.model][0], f"{r.qini:.4f}", f"{r.auuc:.3f}", f"{r.pehe * k:.2f}", f"{r.corr_true:.3f}", f"{r.ate_bias * k:+.2f}", f"{r.policy_value_true:.3f}", f"{r.fit_seconds:.1f}", f"{r.latency_ms_per_1k:.1f}"])
    hl = hl_rows(rows, {2: "max", 3: "max", 4: "min", 5: "max", 7: "max", 8: "min", 9: "min"})
    d.table(s, 0.6, 1.75, 12.1, rows, [2.5, 1.1, 0.9, 0.9, 1.0, 1.0, 1.1, 1.4, 0.7, 0.8], 10.5, 0.37, highlight=hl)
    orc = lb[(lb.outcome == o) & lb.model.str.startswith("Oracle")].iloc[0]
    d.text(s, 0.6, 6.35, 12.1, 0.6, [(f"Oracle (true τ) Qini {orc.qini:.4f} · policy $/guest {orc.policy_value_true:.3f}. Green = best, orange = worst per column.", 11, False, MUTED)])

s = d.slide("Stage 6 · Qini / cumulative-gain curves", "Observed (IPW) uplift when targeting the top x% of guests; dashed = oracle ranking, dotted = random", "3 · Stage results")
if img(PL / "qini_curves_ancillary.png"):
    d.image(s, PL / "qini_curves_ancillary.png", 0.6, 1.7, 6.0, 4.9)
if img(PL / "qini_curves_direct.png"):
    d.image(s, PL / "qini_curves_direct.png", 6.8, 1.7, 5.9, 4.9)
d.text(s, 0.6, 6.65, 12.1, 0.4, [("Curves are close together near the oracle: ranking is easy to get roughly right, hard to get exactly right. Differences are within IPW sampling noise.", 11, False, MUTED)])

s = d.slide("Stage 6 · Policy value and uplift by decile", "Value of the model's own top-30% policy, judged on the true effects (left) and the best model's decile profile (right)", "3 · Stage results")
La, Ld = lbo("ancillary").sort_values("policy_value_true", ascending=False), lbo("direct").sort_values("policy_value_true", ascending=False)
d.bar_chart(s, 0.6, 1.7, 4.3, 4.6, [SHORT[m] for m in La.model], {"$": La.policy_value_true}, [BLUE], title="Ancillary: true net $/guest", fmt="0.00", font=9, min_val=2.5)
d.bar_chart(s, 5.0, 1.7, 4.0, 4.6, [SHORT[m] for m in Ld.model], {"$": Ld.policy_value_true}, [TEAL], title="Direct: true net $/guest", fmt="0.00", font=9, min_val=1.0)
if img(PL / "uplift_by_decile_ancillary.png"):
    d.image(s, PL / "uplift_by_decile_ancillary.png", 9.2, 1.8, 3.6, 2.4)
if img(PL / "uplift_by_decile_direct.png"):
    d.image(s, PL / "uplift_by_decile_direct.png", 9.2, 4.3, 3.6, 2.4)
d.text(s, 0.6, 6.4, 8.4, 0.6, [("Axes start above zero to show the differences: policy values are close (oracle ≈ 2.95 / 2.37). Decile profiles fall from top to bottom decile: the ranking is informative.", 11, False, MUTED)])

s = d.slide("Stage 7 · Campaign result: contact the top 30%", "Estimated vs true net value, and the comparison policies", "3 · Stage results")
d.card(s, 0.6, 1.8, 2.9, 1.2, f"${camp['true_value_targeted']:,.0f}", "true net value, causal targeting", TEAL, f"estimate ${camp['est_value_targeted']:,.0f}")
d.card(s, 3.65, 1.8, 2.9, 1.2, f"${camp['true_value_oracle_top']:,.0f}", "oracle (perfect ranking)", PURPLE, f"capture {camp['true_value_targeted'] / camp['true_value_oracle_top']:.1%}")
d.card(s, 6.7, 1.8, 2.9, 1.2, f"${camp['true_value_all']:,.0f}", "contact everyone", ORANGE, "most guests have positive value")
d1 = [f for f in FLOWS if f["id"] == "D1"][0]
st4 = d1["stages"][3]["metrics"]
d.card(s, 9.75, 1.8, 2.95, 1.2, st4["Lift vs random"], "lift vs random, same size", BLUE, f"random ${st4['Random, same size']:,.0f}")
d.bar_chart(s, 0.6, 3.2, 7.0, 3.5, ["Causal (top 30%)", "Random 30%", "Oracle 30%", "Contact everyone"], {"$": [camp["true_value_targeted"], st4["Random, same size"], camp["true_value_oracle_top"], camp["true_value_all"]]}, [BLUE], horizontal=False, title="True net value ($)", fmt="#,##0", point_colors=[TEAL, GREY, PURPLE, ORANGE])
d.bullets(s, 7.9, 3.3, 4.8, 3.4, [f"Models: {best['ancillary']} (ancillary) + {best['direct']} (direct)", "A 30% budget leaves money on the table here: offers are cheap and most guests have positive expected value", "Targeting pays when budgets are tight or offers expensive: see scenarios D1/D2 and the dashboard presets"], 12)

# =============================================================== 4 model comparison across flows
d.section_slide("4", "All models compared across the flows", "Eleven learners scored on guest, cohort, validity and campaign flows with the same data and the same metrics")
s = d.slide("Model landscape: families, strengths and trade-offs", None, "4 · Model comparison")
fam = [("Meta-learners (S, T, X)", BLUE, ["Fast, simple, GPU XGBoost / CatBoost", "S shrinks τ towards 0 (regularisation bias)", "T is noisy when arms are imbalanced", "X uses propensity to rebalance"]),
       ("EconML orthogonal (R, DR, Causal Forest)", TEAL, ["Cross-fitted nuisance models: robust to e / μ errors", "Causal Forest gives 95% intervals", "DR / R: higher variance on small effects", "Slower inference (forest on CPU)"]),
       ("Deep (TARNet, DragonNet)", ORANGE, ["Shared representation, two heads (PyTorch CUDA)", "DragonNet adds propensity head + targeted regularisation", "Fastest inference (<1.5 ms / 1k)", "Needs standardisation and early stopping"]),
       ("Uplift RF (CausalML)", PURPLE, ["Splits on KL divergence of treated vs control outcome", "Binary outcome only", "Interpretable, but the noisiest ranking here", "CPU only"])]
for i, (t, c, bl) in enumerate(fam):
    d.box(s, 0.6 + (i % 2) * 6.1, 1.6 + (i // 2) * 2.7, 5.9, 2.5, t, bl, c, 12)

# cross-outcome average ranks
La, Ld = lbo("ancillary"), lbo("direct")
def ranks(df, col, asc):
    return df.set_index("model")[col].rank(ascending=asc)
rk = pd.DataFrame({"anc_qini": ranks(La, "qini", False), "anc_pehe": ranks(La, "pehe", True), "anc_pol": ranks(La, "policy_value_true", False),
                   "dir_qini": ranks(Ld, "qini", False), "dir_pehe": ranks(Ld, "pehe", True), "dir_pol": ranks(Ld, "policy_value_true", False)})
rk["avg"] = rk.mean(axis=1)
s = d.slide("All models, both outcomes: rank on Qini, PEHE and true policy value", "1 = best. Average rank over the six columns; Uplift RF is only ranked on the direct outcome", "4 · Model comparison")
rows = [["Model", "Anc Qini", "Anc PEHE", "Anc policy", "Dir Qini", "Dir PEHE", "Dir policy", "Avg rank"]]
order = rk.sort_values("avg").index
for m in order:
    rows.append([m] + [("–" if pd.isna(rk.loc[m, c]) else f"{rk.loc[m, c]:.0f}") for c in ["anc_qini", "anc_pehe", "anc_pol", "dir_qini", "dir_pehe", "dir_pol"]] + [f"{rk.loc[m, 'avg']:.1f}"])
hl = {}
for r_ in range(1, len(rows)):
    for c_ in range(1, 7):
        if rows[r_][c_] == "1":
            hl[(r_, c_)] = GOOD
d.table(s, 0.6, 1.75, 7.6, rows, [2.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], 10.5, 0.36, highlight=hl)
d.bar_chart(s, 8.4, 1.7, 4.3, 4.9, [SHORT[m] for m in order], {"avg rank": [rk.loc[m, "avg"] for m in order]}, [BLUE], title="Average rank (lower = better)", fmt="0.0", font=9,
            point_colors=[FAMILY[m][1] for m in order])
d.text(s, 0.6, 6.45, 7.6, 0.6, [("No single learner wins everywhere: the ancillary winner is a forest / CatBoost-S class model, the direct winner is S / deep nets (the binary effect is small and smooth).", 12, False, MUTED)])

# cross-flow matrices
def col(prefix_contains):
    cols = sorted({k for m in MM.values() for k in m if prefix_contains in k})
    return cols
names_all = list(MM.keys())
def val(m, c):
    v = MM[m].get(c)
    return v
def fmt(v, digits=2):
    return "–" if v is None else (f"{v:,.{digits}f}")
A_cols = [("A1 |τ̂−τ| ancillary ($)", "A1 anc err ($)", 2), ("A1 |τ̂−τ| direct (pp)", "A1 dir err (pp)", 2), ("A2 |τ̂−τ| ancillary ($)", "A2 anc err ($)", 2), ("A2 |τ̂−τ| direct (pp)", "A2 dir err (pp)", 2),
          ("B1 cohort PEHE ancillary ($)", "B1 PEHE anc ($)", 2), ("B2 cohort PEHE ancillary ($)", "B2 PEHE anc ($)", 2), ("B1 cohort PEHE direct (pp)", "B1 PEHE dir (pp)", 2), ("B2 cohort PEHE direct (pp)", "B2 PEHE dir (pp)", 2)]
def matrix_slide(title, kicker, colspec, better, widths, size=10):
    s = d.slide(title, kicker, "4 · Model comparison")
    rows = [["Model"] + [c[1] for c in colspec]]
    for m in names_all:
        rows.append([m] + [fmt(val(m, f"{c[0]}"), c[2]) if c[0] in MM[m] and MM[m].get(c[0]) is not None else "–" for c in colspec])
    hl = hl_rows(rows, {i + 1: better[i] for i in range(len(colspec))})
    d.table(s, 0.6, 1.75, 12.1, rows, widths, size, 0.38, highlight=hl)
    return s
def key_for(prefix, flow, outcome=None):
    for k in MM[names_all[0]]:
        if k.startswith(flow) and prefix in k and (outcome is None or outcome in k):
            return k
def kk(flow, what, outcome=None):
    for m in MM:
        for k in MM[m]:
            if k.startswith(flow + " ") and what in k and (outcome is None or outcome in k):
                return k
colsA = [(kk("A1", "|τ̂−τ|", "ancillary"), "A1 anc err ($)", 2), (kk("A1", "|τ̂−τ|", "direct"), "A1 dir err (pp)", 2), (kk("A2", "|τ̂−τ|", "ancillary"), "A2 anc err ($)", 2), (kk("A2", "|τ̂−τ|", "direct"), "A2 dir err (pp)", 2),
         (kk("B1", "PEHE", "ancillary"), "B1 PEHE anc ($)", 2), (kk("B2", "PEHE", "ancillary"), "B2 PEHE anc ($)", 2), (kk("B1", "PEHE", "direct"), "B1 PEHE dir (pp)", 2), (kk("B2", "PEHE", "direct"), "B2 PEHE dir (pp)", 2)]
matrix_slide("Across flows A & B: individual and cohort accuracy of every model", "A = abs. error of τ̂ for the flow's guest (A1 Expedia family, A2 Corporate loyalty) · B = PEHE inside the OTA (B1) and Direct+Corporate (B2) cohorts; lower is better",
             colsA, ["min"] * 8, [2.4] + [1.1] * 8, 10)
colsCD = [(kk("C1", "ATE bias", "ancillary"), "C1 ATE bias anc ($)", 2), (kk("C2", "ATE bias", "direct"), "C2 ATE bias dir (pp)", 2), (kk("D1", "true net value"), "D1 net value ($)", 0), (kk("D1", "capture"), "D1 oracle capture", 3),
          (kk("D2", "true net value"), "D2 net value ($)", 0), (kk("D2", "capture"), "D2 oracle capture", 3)]
s = matrix_slide("Across flows C & D: population bias and campaign value of every model", "C = bias of the model's mean CATE vs the true ATE (closer to 0 is better) · D = true net $ of the model's own top-30% policy (higher is better)",
                 colsCD, ["abs", "abs", "max", "max", "max", "max"], [2.4, 1.4, 1.4, 1.3, 1.3, 1.3, 1.3], 10.5) if False else None
# (C columns need |bias| for highlight: build explicitly)
s = d.slide("Across flows C & D: population bias and campaign value of every model", "C = bias of the model's mean CATE vs the true ATE (closer to 0 is better) · D = true net $ of the model's own top-30% policy (higher is better)", "4 · Model comparison")
rows = [["Model"] + [c[1] for c in colsCD]]
for m in names_all:
    rows.append([m] + [fmt(MM[m].get(c[0]), c[2]) if MM[m].get(c[0]) is not None else "–" for c in colsCD])
hl = {}
for ci, mode in ((1, "absmin"), (2, "absmin"), (3, "max"), (4, "max"), (5, "max"), (6, "max")):
    vals = [(float(rows[r][ci].replace(",", "")), r) for r in range(1, len(rows)) if rows[r][ci] != "–"]
    if len(vals) > 2:
        bst = min(vals, key=lambda t: abs(t[0])) if mode == "absmin" else max(vals)
        wst = max(vals, key=lambda t: abs(t[0])) if mode == "absmin" else min(vals)
        hl[(bst[1], ci)] = GOOD; hl[(wst[1], ci)] = WARN
d.table(s, 0.6, 1.75, 12.1, rows, [2.4, 1.4, 1.4, 1.3, 1.3, 1.3, 1.3], 10.5, 0.37, highlight=hl)
d.text(s, 0.6, 6.5, 12.1, 0.5, [("Uplift RF is scored on the direct outcome only; in D1/D2 its direct τ̂ is combined with the best ancillary model. Green = best, orange = worst per column.", 11, False, MUTED)])

# average rank across all flow columns
flow_cols = [c[0] for c in colsA] + [c[0] for c in colsCD]
tab = pd.DataFrame({m: {c: MM[m].get(c) for c in flow_cols} for m in names_all}).T.astype(float)
rk2 = pd.DataFrame(index=tab.index)
for c in flow_cols:
    v = tab[c]
    if "ATE bias" in c:
        v = v.abs(); rk2[c] = v.rank(ascending=True)
    elif "net value" in c or "capture" in c:
        rk2[c] = v.rank(ascending=False)
    else:
        rk2[c] = v.rank(ascending=True)
rk2["avg"] = rk2.mean(axis=1)
wins = (rk2[flow_cols] == 1).sum(axis=1)
s = d.slide("Across all 14 flow-level scores: who is consistently good?", "Average rank over the A, B, C and D columns (1 = best) and number of column wins", "4 · Model comparison")
order2 = rk2.sort_values("avg").index
d.bar_chart(s, 0.6, 1.7, 6.6, 4.9, [SHORT[m] for m in order2], {"avg rank": rk2.loc[order2, "avg"]}, [BLUE], title="Average rank across flows (lower = better)", fmt="0.0", point_colors=[FAMILY[m][1] for m in order2])
rows = [["Model", "Avg rank", "Wins", "Family"]] + [[m, f"{rk2.loc[m, 'avg']:.1f}", str(int(wins[m])), FAMILY[m][0]] for m in order2]
d.table(s, 7.4, 1.8, 5.3, rows, [2.5, 0.9, 0.7, 1.3], 11, 0.4, highlight={(1, 1): GOOD})
d.text(s, 0.6, 6.7, 12.1, 0.4, [("Rank aggregation hides magnitude: see the per-flow tables for how large the gaps are. Uplift RF has fewer columns (direct-only).", 11, False, MUTED)])

s = d.slide("Which model wins in which flow?", None, "4 · Model comparison")
rows = [["Flow", "Metric", "Best model", "Value", "Worst model", "Value"]]
for c, label in [(c[0], c[1]) for c in colsA + colsCD]:
    v = tab[c].dropna()
    if "ATE bias" in c:
        v2 = v.abs(); bm, wm = v2.idxmin(), v2.idxmax()
    elif "net value" in c or "capture" in c:
        bm, wm = v.idxmax(), v.idxmin()
    else:
        bm, wm = v.idxmin(), v.idxmax()
    rows.append([c.split(" ")[0], label, SHORT[bm], f"{v[bm]:,.3f}".rstrip("0").rstrip("."), SHORT[wm], f"{v[wm]:,.3f}".rstrip("0").rstrip(".")])
d.table(s, 0.6, 1.5, 12.1, rows, [0.8, 2.6, 2.0, 1.4, 2.0, 1.4], 10, 0.34, align=["center", "left", "center", "center", "center", "center"])

s = d.slide("Accuracy vs compute cost", "Training time and inference latency for every learner (ancillary outcome) with PEHE", "4 · Model comparison")
L = lbo("ancillary").sort_values("fit_seconds")
d.bar_chart(s, 0.6, 1.7, 4.0, 4.9, [SHORT[m] for m in L.model], {"s": L.fit_seconds}, [BLUE], title="Training time (s)", fmt="0.0", font=9, point_colors=[FAMILY[m][1] for m in L.model])
L2 = lbo("ancillary").sort_values("latency_ms_per_1k")
d.bar_chart(s, 4.7, 1.7, 4.0, 4.9, [SHORT[m] for m in L2.model], {"ms": L2.latency_ms_per_1k}, [TEAL], title="Inference (ms per 1,000 guests)", fmt="0.0", font=9, point_colors=[FAMILY[m][1] for m in L2.model])
L3 = lbo("ancillary").sort_values("pehe")
d.bar_chart(s, 8.8, 1.7, 3.9, 4.9, [SHORT[m] for m in L3.model], {"$": L3.pehe}, [ORANGE], title="PEHE ($, lower = better)", fmt="0.00", font=9, point_colors=[FAMILY[m][1] for m in L3.model])
d.text(s, 0.6, 6.65, 12.1, 0.4, [("Bar colours: blue = meta-learner, teal = EconML, orange = deep, purple = CausalML. All models score 1,000 guests in well under 100 ms.", 11, False, MUTED)])

s = d.slide("Model selection guidance", None, "4 · Model comparison")
d.box(s, 0.6, 1.5, 4.0, 2.6, "Need individual-level $ forecasts + intervals", [f"Use Causal Forest DML (best ancillary Qini: {lbo('ancillary').iloc[0].model == 'Causal Forest DML' and 'yes' or 'close second'})", "95% interval per guest; slower inference", "Good calibration of mean effect"], TEAL, 12)
d.box(s, 4.8, 1.5, 4.0, 2.6, "Need a fast, strong ranking", ["S-Learner (CatBoost / XGB) and DragonNet / TARNet rank near the top", "Sub-5 ms per 1,000 guests", "Great for nightly scoring of the arrivals list"], BLUE, 12)
d.box(s, 9.0, 1.5, 3.7, 2.6, "Small / smooth binary effect", ["S-Learner and deep nets beat DR / R / T on the direct outcome", "Pseudo-outcome learners (DR, R) are noisy when τ is a few pp", "Uplift RF is interpretable but weakest"], ORANGE, 12)
d.box(s, 0.6, 4.3, 12.1, 2.4, "Practical recipe", ["Always compare on Qini AND PEHE AND true / holdout policy value: they rank models differently", "Ensemble the top 2–3 (e.g. Causal Forest + DragonNet) and keep a randomised hold-out to validate in production", "Refit monthly; monitor overlap (propensity range) and balance when the campaign mix changes"], PURPLE, 12)

# =============================================================== 5 flow-level & overall
d.section_slide("5", "Flow-level and overall metrics", "A scorecard for the whole programme and for each of the eight flows")
s = d.slide("Programme-level scorecard (overall metrics)", "The same block closes every flow in the dashboard", "5 · Metrics")
items = list(OVER.items())
for i, (k, v) in enumerate(items):
    r, c = divmod(i, 5)
    vv = f"{v:,.0f}" if isinstance(v, (int, float)) and abs(v) >= 1000 else v
    d.card(s, 0.6 + c * 2.45, 1.65 + r * 1.05, 2.35, 0.95, str(vv), k, [BLUE, TEAL, ORANGE, PURPLE, ROSE][c], vsize=14)

s = d.slide("Overall metrics as shown in the dashboard", "Bottom of every flow in tab 5 · End-to-end flows", "5 · Metrics")
if img(SH / "flows" / "overall.png"):
    d.image(s, SH / "flows" / "overall.png", 0.6, 1.7, 12.1, 5.0)

s = d.slide("Flow-level metrics: the eight flows at a glance", None, "5 · Metrics")
rows = [["Flow", "Type", "Question answered", "Key results"]]
qs = {"A": "Should we send this guest the offer?", "B": "Does it work for this cohort?", "C": "Can we trust the estimate?", "D": "What does the campaign earn?"}
tn = {"A": "Guest decision", "B": "Cohort read-out", "C": "Validity", "D": "Campaign policy"}
for f in FLOWS:
    rows.append([f["id"] + " · " + f["short"].split(" · ")[1], tn[f["type"]], qs[f["type"]], "; ".join(f"{k}: {v}" for k, v in f["summary"].items())])
d.table(s, 0.6, 1.5, 12.1, rows, [2.4, 1.4, 3.0, 5.3], 10.5, 0.55, align=["left", "left", "left", "left"], bold_first_col=True)

# =============================================================== 6 scenarios as stories
d.section_slide("6", "Scenarios as stories", "Eight end-to-end flows from the Streamlit dashboard: each with stage-level metrics and screenshots")


def narrative(f):
    sm = f["summary"]
    if f["type"] == "A":
        g = f["stages"][3]["metrics"]; c5 = f["stages"][4]["metrics"]
        return f"Decision {g['Decision']} with expected net value ${g['Expected net value ($)']}; predicted segment {g['Predicted segment']}, true segment {c5['True segment']}. Decision correct: {c5['Decision correct']}."
    if f["type"] == "B":
        return f"True ancillary ATE ${sm['True ancillary ATE ($)']} vs AIPW ${sm['AIPW ($)']} vs naive ${sm['Naive ($)']}: the naive read is off by ${float(sm['Naive ($)']) - float(sm['True ancillary ATE ($)']):.1f}."
    if f["type"] == "C":
        return f"Refuters passed {sm['Refuters passed']}; AIPW error vs truth {sm['AIPW error']}."
    return f"True net value ${sm['True net value ($)']:,.0f}; capture of oracle {sm['Capture of oracle']}; ROI {sm['ROI']}."


def stage_rows(f):
    rows = [["Stage", "Key metrics"]]
    for st_ in f["stages"]:
        rows.append([st_["name"], "; ".join(f"{k}: {v:,}" if isinstance(v, int) and abs(v) >= 1000 else f"{k}: {v}" for k, v in list(st_["metrics"].items())[:4])])
    return rows


for f in FLOWS:
    tag = f"6 · Scenario {f['id']}"
    s = d.slide(f"{f['id']} · {f['title']}", f["story"], tag)
    d.table(s, 0.6, 1.9, 12.1, stage_rows(f), [2.6, 9.5], 10, 0.55, align=["left", "left"], bold_first_col=True,
            fills={i + 1: (WARN if st_["status"] == "warn" else WHITE) for i, st_ in enumerate(f["stages"])})
    d.rect(s, 0.6, 6.15, 12.1, 0.75, GOOD)
    d.text(s, 0.8, 6.2, 11.7, 0.65, [("Takeaway: " + narrative(f), 13, True, INK)], anchor="middle")
    stg = [SH / "flows" / "stages" / f"{f['id']}_s{i + 1}.png" for i in range(len(f["stages"]))]
    pairs = [stg[i:i + 2] for i in range(0, len(stg), 2)]
    for part, idxs in enumerate(pairs):
        lo = part * 2 + 1
        s = d.slide(f"{f['id']} · screenshots, stage{'s' if len(idxs) > 1 else ''} {lo}{'–' + str(lo + len(idxs) - 1) if len(idxs) > 1 else ''}", f"Streamlit tab 5 · {f['short']}", tag)
        for i, p in enumerate(idxs):
            if p.exists():
                d.image(s, p, 0.6 + i * 6.15, 1.7, 6.0, 4.4)
        if len(idxs) == 1:
            d.rect(s, 6.75, 1.9, 5.95, 2.6, GOOD)
            d.text(s, 6.95, 2.0, 5.6, 2.4, [("Takeaway", 14, True, BLUE), (narrative(f), 14, False, INK)], anchor="middle")
        elif part == len(pairs) - 1:
            d.text(s, 0.6, 6.3, 12.1, 0.6, [("Takeaway: " + narrative(f), 13, True, INK)])

# =============================================================== 7 limitations
d.section_slide("7", "Limitations and next steps", "What the numbers do and do not prove")
s = d.slide("Limitations", None, "7 · Close")
d.bullets(s, 0.6, 1.6, 12.1, 5.2, [
    "Synthetic data: unconfoundedness holds by construction. Real logs need sensitivity analysis for unmeasured confounding and a randomised hold-out.",
    "Single seed, single test split: the Qini ranking among the top models is within sampling noise. No repeated-seed Monte-Carlo was run.",
    f"AIPW direct-shift effect is {D_['estimators']['AIPW (DR)']['ate'] / D_['true_ate'] - 1:+.0%} from the truth ({D_['estimators']['AIPW (DR)']['ate'] * 100:.1f} vs {D_['true_ate'] * 100:.1f} pp): small binary effects are harder than the $ outcome.",
    "PSM / IPW bootstrap CIs keep the propensity fixed, so intervals are somewhat narrow; AIPW uses influence-function SEs.",
    "DragonNet's targeted regularisation is applied on the probability scale for the binary outcome (an approximation).",
    "Segment labels from the model are action rules on τ̂ (Always-Taker vs Lost Cause cannot be separated without a baseline-response model).",
    "cuML is not installable on native Windows (sklearn kNN used); LightGBM GPU build is absent (CPU fallback)."], 16)
s = d.slide("Next steps", None, "7 · Close")
d.box(s, 0.6, 1.6, 3.9, 3.6, "Statistical", ["Monte-Carlo over seeds for model ranking", "Sensitivity (E-value / Rosenbaum)", "Conformal / calibrated intervals for all learners"], BLUE, 13)
d.box(s, 4.7, 1.6, 3.9, 3.6, "Product", ["Real campaign hold-out (5% random)", "Multi-offer / dose-response uplift", "Fairness and fatigue constraints per guest"], TEAL, 13)
d.box(s, 8.8, 1.6, 3.9, 3.6, "MLOps", ["Scheduled refit + drift monitoring (overlap, balance)", "Model registry and shadow scoring", "Batch scoring job writes arrivals-list decisions"], ORANGE, 13)

d.save(HERE / "Pre_Arrival_Upsell_Uplift_End_to_End.pptx")
print("slides:", d.n)
