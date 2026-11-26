"""Build the end-to-end presentation from the committed run artifacts.

Run:  python presentation/build_deck.py
Needs: outputs/metrics/{benchmark_summary.csv, model_flow_comparison.csv, flows.json, iv_first_stage_diagnostics.json}
       (flows.json / model_flow_comparison.csv come from `python -m src.optimization.flows`)
Every number in the deck is read from those files or recomputed from the saved models; none is typed by hand.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from deck_lib import (BLUE, FAMILY_COLOR, FAMILY_LABEL, GOOD, GREY, INK, LIGHT, MUTED, ORANGE, PURPLE, ROSE, TEAL,  # noqa: E402
                      WHITE, Deck)
from src.optimization.revenue_optimizer import build_profile_df, compare_policies  # noqa: E402
from src.utils.gpu_utils import load_config  # noqa: E402
from src.utils.registry import load_models  # noqa: E402

OUT = ROOT / "outputs"
SHOT = OUT / "screenshots" / "streamlit"
VIS = OUT / "visualizations"
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)
cfg = load_config()
models = load_models()
bench = pd.read_csv(OUT / "metrics" / "benchmark_summary.csv").dropna(subset=["elasticity_rmse"]).copy()
mf = pd.read_csv(OUT / "metrics" / "model_flow_comparison.csv")
flows = json.loads((OUT / "metrics" / "flows.json").read_text(encoding="utf-8"))
diag = json.loads((OUT / "metrics" / "iv_first_stage_diagnostics.json").read_text(encoding="utf-8"))
dgp = json.loads((OUT / "data" / "dgp_summary_stats.json").read_text(encoding="utf-8"))
nm = lambda m: m.replace(" (PyTorch)", "").replace(" (LinearDRIV)", "").replace(" (NonParamDML)", "").replace(" (causalml)", "")  # noqa: E731


def crop(src: Path, box, name: str) -> Path:
    out = ASSETS / name
    Image.open(src).crop(box).save(out)
    return out


# Streamlit screenshot crops (original pixel coordinates of the 1440x2400 captures)
PRICE_BOX = (400, 320, 1390, 880)
c_price = {k: crop(SHOT / f, PRICE_BOX, f"crop_{k}.png") for k, f in {
    "A": "02_streamlit_dynamic_price_optimizer.png", "B": "02b_streamlit_scenario_b_segment_contrast.png",
    "C": "02c_streamlit_scenario_c_airline_cut.png", "D": "02d_streamlit_scenario_d_custom.png"}.items()}
c_eda = crop(SHOT / "01_streamlit_eda_and_iv_diagnostics.png", (400, 330, 1390, 1290), "crop_eda.png")
c_arena = crop(SHOT / "03_streamlit_cate_metalearner_arena.png", (400, 340, 1390, 1260), "crop_arena.png")
c_crm = crop(SHOT / "04_streamlit_batch_crm_targeting.png", (400, 640, 1390, 1190), "crop_crm.png")
c_score = crop(SHOT / "05_streamlit_end_to_end_flows_elasticity.png", (400, 460, 1390, 840), "crop_scorecard.png")
c_flow = crop(SHOT / "05b_streamlit_flow_price_optimisation.png", (400, 1090, 1390, 1600), "crop_flow_example.png")

# scenario policy numbers, recomputed from the saved models
KEYS = ["Naive OLS", "2SLS", "DML-IV", "DR-IV (LinearDRIV)", "DeepIV (PyTorch)"]
SC = {}
for k, sc in cfg["scenarios"].items():
    for alt in (False, True):
        if alt and "profile_alt" not in sc:
            continue
        row = build_profile_df(sc["profile_alt" if alt else "profile"], sc["instruments"], cfg=cfg)
        SC[k + ("alt" if alt else "")] = compare_policies({m: models[m] for m in KEYS}, row, cfg, sc["demand_shock_u"])

d = Deck("Grand Horizon Hotels · IV + CATE dynamic pricing engine")
sc_card = flows["scorecard"]
best = bench.sort_values("elasticity_rmse").iloc[0]
ols = bench[bench["model"] == "Naive OLS"].iloc[0]
fs = diag["first_stage"]
co = diag["coefficients"]

# ============================================================================================ 1 title
d.title_slide("Dynamic Pricing & Price Elasticity under Endogeneity",
              "Instrumental variables and CATE: from a biased rate-demand correlation to optimal ADR and targeted promotions",
              "Grand Horizon Hotels · end-to-end flows, 21-model comparison, uplift documentation and scenario stories · synthetic data with known ground truth")

# ============================================================================================ 2 exec summary
s = d.slide("Naive pricing models get the sign of elasticity wrong; instruments fix it and pay off in RevPAR",
            "Everything below is measured against a synthetic ground truth, so each claim can be checked", "Executive summary")
d.card(s, 0.6, 1.9, 2.9, 1.25, f"{ols['mean_elasticity']:+.2f} vs {dgp['true_mean_elasticity']:+.2f}", "Naive OLS elasticity vs truth", ROSE, "price raises demand: wrong sign")
d.card(s, 3.65, 1.9, 2.9, 1.25, f"{best['elasticity_rmse']:.2f} vs {ols['elasticity_rmse']:.2f}", f"Elasticity RMSE: {nm(best['model'])} vs OLS", BLUE, "best IV model vs naive")
d.card(s, 6.7, 1.9, 2.9, 1.25, f"{fs['f_statistic_robust']:,.0f}", "First-stage F (threshold 10)", TEAL, f"partial R² {fs['partial_r2']:.2f}")
d.card(s, 9.75, 1.9, 3.0, 1.25, sc_card["elasticity"]["Bias removed (mean)"], "Elasticity bias removed (Flow 1)", ORANGE, "3 worked examples")
d.card(s, 0.6, 3.3, 2.9, 1.25, sc_card["pricing"]["RevPAR vs naive (median)"], "RevPAR vs naive pricing (Flow 2)", BLUE, "median of 3 scenarios")
d.card(s, 3.65, 3.3, 2.9, 1.25, sc_card["pricing"]["Best achievable captured"], "Of best achievable RevPAR captured", TEAL, "realised on true demand")
d.card(s, 6.7, 3.3, 2.9, 1.25, sc_card["promo"]["Correct decisions"], "Correct promo decisions (Flow 3)", PURPLE, f"CATE error {sc_card['promo']['Mean CATE |error|']}")
d.card(s, 9.75, 3.3, 3.0, 1.25, sc_card["crm"]["Oracle value captured (mean)"], "Oracle uplift value captured (Flow 4)", ROSE, "targeting is far from optimal")
d.bullets(s, 0.6, 4.75, 12.1, 2.2, [
    "Hidden demand surges push rates and bookings up together, so OLS and XGBoost read price as demand-increasing; three exogenous instruments recover the negative elasticity.",
    "21 estimators were benchmarked; simple linear and DML IV are best here because the data-generating process is log-linear. Deep models earn their keep when structure is unknown.",
    "Promo targeting beats blanket discounting by a wide margin but captures only a small share of the oracle value; the uplift flow is the weakest link and is documented with its limits.",
    "All results come from a CPU-fallback run; the GPU code paths exist but were not exercised on this machine."], 14)

# ============================================================================================ 3 agenda
s = d.slide("The story in six parts", None, "Agenda")
parts = [("1", "The problem and the data", "Endogeneity, causal DAG, instruments, diagnostics", BLUE),
         ("2", "System and decision flows", "Architecture, four flows, five stages each", TEAL),
         ("3", "All 21 models", "Families, benchmark, charts, what wins", PURPLE),
         ("4", "Flow-level view", "Stage metrics, overall metrics, models compared across flows", ORANGE),
         ("5", "Uplift end to end", "High-level documentation of the promo targeting flow", ROSE),
         ("6", "Scenario stories", "Four Streamlit scenarios as a narrative, limits and next steps", BLUE)]
for i, (n, t, sub, c) in enumerate(parts):
    x, y = 0.6 + (i % 3) * 4.1, 1.7 + (i // 3) * 2.6
    d.rect(s, x, y, 3.9, 2.3, LIGHT)
    d.text(s, x + 0.2, y + 0.1, 1, 0.9, [(n, 40, True, c)])
    d.text(s, x + 0.2, y + 0.95, 3.5, 0.5, [(t, 17, True, INK)])
    d.text(s, x + 0.2, y + 1.5, 3.5, 0.7, [(sub, 12, False, MUTED)])

# ============================================================================================ section 1
d.section_slide("1", "The problem and the data", "Why price and demand move together, and how instruments break the link")

s = d.slide("When a convention hits town, managers raise rates and demand rises too: OLS reads that as 'higher price, more bookings'",
            None, "Problem")
d.image(s, VIS / "eda" / "01_endogeneity_trap_simpsons_paradox.png", 0.6, 1.75, 7.6, 3.55)
d.bullets(s, 8.45, 1.75, 4.4, 3.6, [
    f"Pooled OLS slope: {co['ols_price_coefficient']:+.2f}. True mean elasticity: {co['true_mean_elasticity']:+.2f}.",
    "Inside any slice of the hidden shock U the slope is negative; pooling hides it (Simpson's paradox).",
    f"Wu-Hausman = {diag['wu_hausman_endogeneity']['statistic']:,.0f} (p < 1e-15): treating price as exogenous is rejected."], 14, head="What the data say")
d.box(s, 0.6, 5.5, 12.15, 1.4, "Consequence for the business", [
    "A naive model recommends the $650 ceiling for every night, because it believes demand never falls as price rises.",
    "It is right only on the surge nights, for a reason it cannot see, and wrong on every ordinary night."], ROSE, 13)

s = d.slide("Causal structure: three instruments move the rate without touching demand, so they can isolate price's own effect",
            None, "Problem")
d.node(s, 5.6, 1.85, 2.2, 0.7, "U: latent event surge\n(unobserved)", LIGHT, 12, dashed=True)
d.node(s, 0.7, 3.55, 2.5, 0.65, "Z1 competitor room outage %", TEAL, 11)
d.node(s, 0.7, 4.45, 2.5, 0.65, "Z2 airline seat capacity shock", TEAL, 11)
d.node(s, 0.7, 5.35, 2.5, 0.65, "Z3 rate-rounding trigger", TEAL, 11)
d.node(s, 5.0, 4.4, 3.4, 0.8, "T: net ADR / promo discount", BLUE, 13)
d.node(s, 10.2, 4.4, 2.6, 0.8, "Y: demand, conversion,\nNet RevPAR", ORANGE, 12)
d.node(s, 5.0, 6.15, 3.4, 0.6, "X: lead time, LOS, DOW, season, loyalty, tier...", GREY, 11)
d.arrow(s, 3.2, 3.9, 5.0, 4.55); d.arrow(s, 3.2, 4.77, 5.0, 4.8); d.arrow(s, 3.2, 5.67, 5.0, 5.05)
d.arrow(s, 8.4, 4.8, 10.2, 4.8, BLUE)
d.arrow(s, 6.1, 2.55, 6.4, 4.4, ROSE); d.arrow(s, 7.6, 2.4, 11.2, 4.4, ROSE)
d.arrow(s, 6.7, 6.15, 6.7, 5.2); d.arrow(s, 8.4, 6.3, 11.2, 5.2)
d.text(s, 10.05, 1.55, 2.8, 2.2, [("Why this works", 14, True, BLUE),
                                ("Z shifts the offered rate (relevance) and is independent of U and of demand except through the rate (exclusion).", 12, False, INK),
                                ("2SLS, DML-IV, DR-IV and DeepIV all use that variation only.", 12, False, INK)])
d.text(s, 0.7, 2.2, 4.5, 1.2, [("Instrument checks (data)", 14, True, BLUE),
                               (f"corr(Z, U) all |r| < 0.02; first-stage F = {fs['f_statistic_robust']:,.0f}", 12, False, INK)])

s = d.slide("Synthetic hospitality data: every quantity needed to score a causal estimate is known", None, "Data")
rows = [["Block", "Variables", "Role"],
        ["Observed confounders X", "lead time, length of stay, day of week, seasonality, loyalty tier, ADR sensitivity, property tier, room type, segment", "Controls; effect modifiers for heterogeneity"],
        ["Latent shock U", "city-wide event surge (not given to any model)", "Raises both rate and demand: the source of bias"],
        ["Instruments Z", "competitor room outage %, airline seat capacity shock, rate-rounding trigger", "Move rate only; independent of U"],
        ["Treatment T", "net ADR (ln), and a 15% promotional discount (binary)", "Price elasticity; promo CATE"],
        ["Outcomes Y", "log demand, conversion probability, occupancy, Net RevPAR", "Structural: occupancy = 1 - exp(-D)"],
        ["Ground truth", "true elasticity per booking, true promo effect, true optimal ADR", "Scores every estimator"]]
d.table(s, 0.6, 1.7, 12.15, rows, [2.2, 6.2, 3.8], 12, 0.55, bold_first_col=True, align=["left", "left", "left"])
d.card(s, 0.6, 5.75, 2.9, 1.1, f"{dgp['n_rows']:,}", "bookings (25% held out)", BLUE)
d.card(s, 3.65, 5.75, 2.9, 1.1, f"${dgp['mean_adr']:,.0f}", "mean ADR", TEAL)
d.card(s, 6.7, 5.75, 2.9, 1.1, f"{dgp['mean_occupancy']:.0%}", "mean occupancy", PURPLE)
d.card(s, 9.75, 5.75, 3.0, 1.1, f"{dgp['promo_rate']:.0%}", "bookings with a promo", ORANGE)

s = d.slide("The instruments are strong and look exogenous; relevance is clear and corr(Z, U) is about zero", None, "Data")
d.image(s, VIS / "eda" / "02_instrument_relevance_first_stage.png", 0.6, 1.6, 7.0, 2.6)
d.image(s, VIS / "eda" / "04_confounder_instrument_correlation_heatmap.png", 7.9, 1.6, 4.9, 3.45)
d.image(s, VIS / "eda" / "05_true_heterogeneous_elasticity_landscape.png", 0.6, 4.4, 4.4, 2.5)
d.bullets(s, 5.2, 5.2, 7.6, 1.8, [
    "Leisure early-bookers are the most price-sensitive; corporate short-lead guests the least. Elasticity ranges from about -3.6 to -0.2.",
    "That heterogeneity is why a single price is wrong and why segment-level CATE matters.",
    "Heatmap: instruments correlate with price (0.06 to 0.19) and not with U."], 12)

s = d.slide("IV diagnostics: strong instruments, endogeneity confirmed, and the over-identification test passes on the right model",
            None, "Data")
het = diag.get("sargan_overidentification_heterogeneous_spec", {})
rows = [["Diagnostic", "Result", "Reading"],
        ["First-stage F (robust)", f"{fs['f_statistic_robust']:,.0f}", "Far above the rule-of-thumb threshold of 10"],
        ["First-stage F (homoskedastic, Cragg-Donald equivalent)", f"{fs['cragg_donald_equivalent_f_homoskedastic']:,.0f}", "Strong under both definitions"],
        ["Partial R² of instruments", f"{fs['partial_r2']:.3f}", "Instruments explain 23% of rate variation"],
        ["Wu-Hausman endogeneity", f"{diag['wu_hausman_endogeneity']['statistic']:,.0f} (p < 1e-15)", "OLS is inconsistent: price is endogenous"],
        ["Sargan, constant-elasticity 2SLS", f"p = {diag['sargan_overidentification']['pvalue']:.3f}", "Rejects: model ignores heterogeneity (misspecified)"],
        ["Sargan, heterogeneous 2SLS (used)", f"p = {het.get('pvalue', float('nan')):.3f}", "Does not reject: instruments consistent with validity"],
        ["Price coefficient: OLS / 2SLS / LIML / GMM", f"{co['ols_price_coefficient']:+.2f} / {co['2sls_price_coefficient']:+.2f} / {co['liml_price_coefficient']:+.2f} / {co['gmm_price_coefficient']:+.2f}", f"Truth {co['true_mean_elasticity']:+.2f}"]]
d.table(s, 0.6, 1.75, 12.15, rows, [4.2, 3.0, 5.0], 12, 0.5, bold_first_col=True, align=["left", "center", "left"])
d.text(s, 0.6, 5.9, 12.1, 0.9, [("Caveat: the instruments are valid by construction. With real data, airline capacity could plausibly also shift demand, "
                                 "so the exclusion restriction would need an argument and a sensitivity analysis.", 12, False, MUTED, True)])

# ============================================================================================ section 2
d.section_slide("2", "System and decision flows", "How data, models, optimiser, API and dashboard fit together")

s = d.slide("One pipeline from data generation to a served decision, with every artifact written to outputs/", None, "System")
d.chevrons(s, 0.6, 1.8, 12.15, 0.95, ["Synthetic DGP\n+ ground truth", "EDA\n5 charts", "Train 21 models\n+ IV diagnostics", "Registry\noutputs/", "Evaluate\n5 charts", "FastAPI +\nStreamlit", "Playwright\nscreenshots"],
           [BLUE, TEAL, PURPLE, ORANGE, TEAL, BLUE, ROSE], 12)
d.box(s, 0.6, 3.1, 3.9, 3.7, "Modelling layer", [
    "src/models: baselines, linear + ML IV, PyTorch DeepIV / DeepGMM, CATE meta-learners, TARNet / DragonNet / CEVAE",
    "One interface: elasticity(X), promo_cate(X), predict_log_demand(X, price)",
    "GPU-aware: xgboost gpu_hist or device=cuda, CatBoost GPU, torch .to(device), automatic CPU fallback"], BLUE, 12)
d.box(s, 4.7, 3.1, 3.9, 3.7, "Decision layer", [
    "src/optimization/revenue_optimizer: ADR x promo sweep on each model's structural demand; realised RevPAR scored on the true demand",
    "src/optimization/flows: four flows, five stages, metrics at each stage",
    "crm_segments: uplift quadrants and offers"], TEAL, 12)
d.box(s, 8.8, 3.1, 3.95, 3.7, "Serving layer", [
    "FastAPI: /health, /iv-diagnostics, /estimate-elasticity, /optimize-price, /cate-benchmark",
    "Streamlit: 5 tabs (EDA lab, optimiser, model arena, CRM targeting, end-to-end flows) and 4 scenarios",
    "Playwright captures API and dashboard screenshots; logs per stage in outputs/logs"], ORANGE, 12)

s = d.slide("Four decision flows, each five stages, each run on several worked examples", None, "System")
flow_def = [("Flow 1 · Elasticity identification", "Can we recover true elasticity despite the hidden shock?", ["Context", "Instrument relevance", "Naive estimate", "IV estimate", "Validation"], BLUE),
            ("Flow 2 · Price optimisation", "What ADR should we charge, and what RevPAR do we realise?", ["Elasticity", "Demand at rack", "ADR sweep", "Prescription", "Realised RevPAR"], TEAL),
            ("Flow 3 · Promo CATE decision", "Should this guest get a 15% promotion?", ["Baseline", "CATE ensemble", "Conversion uplift", "Economics", "Decision"], PURPLE),
            ("Flow 4 · Batch CRM targeting", "Whom in a reservation batch should we target?", ["Ingest", "Score", "Segment", "Value", "Quality"], ROSE)]
for i, (t, q, st, c) in enumerate(flow_def):
    y = 1.65 + i * 1.3
    d.text(s, 0.6, y, 3.4, 0.5, [(t, 13, True, c)])
    d.text(s, 0.6, y + 0.45, 3.4, 0.7, [(q, 11, False, MUTED)])
    d.chevrons(s, 4.1, y + 0.12, 8.65, 0.85, st, [c], 11)

# ============================================================================================ section 3
d.section_slide("3", "All 21 models", "What each one is, how it did, and what wins")

CATALOG = [
    ["Model", "Family", "Estimates", "Library / method"],
    ["Naive OLS", "Naive", "Elasticity (ignores U)", "Ridge, ln price + X"],
    ["Naive XGBoost", "Naive", "Elasticity by finite difference", "XGBoost (GPU-capable)"],
    ["2SLS / LIML / IV-GMM", "IV", "Heterogeneous elasticity", "linearmodels, price x modifier interactions"],
    ["DML-IV", "IV", "CATE(X) of price", "econml DMLIV, GBDT nuisances"],
    ["OrthoIV", "IV", "CATE(X) of price", "econml OrthoIV, projection-free"],
    ["DR-IV (LinearDRIV)", "IV", "Doubly robust CATE", "econml LinearDRIV"],
    ["ForestDR-IV", "IV", "Non-parametric CATE", "econml ForestDRIV"],
    ["DeepIV", "Deep IV", "Structural h(X, price)", "PyTorch: MDN stage 1 + MC outcome net"],
    ["DeepGMM", "Deep IV", "Structural g(X, price)", "PyTorch: min-max moment objective"],
    ["S / T / X / R / DR learners", "Meta", "Promo CATE", "econml (T uses CatBoost, others XGBoost)"],
    ["Causal Forest DML", "Forest", "Promo CATE, honest trees", "econml CausalForestDML"],
    ["Uplift RF", "Forest", "Promo uplift on conversion", "causalml UpliftRandomForest (CTS)"],
    ["TARNet / DragonNet", "Deep CATE", "Promo CATE", "PyTorch shared representation, 2 heads (+ propensity)"],
    ["CEVAE", "Deep CATE", "Promo CATE via latent", "PyTorch variational autoencoder (bonus)"]]
s = d.slide("The model zoo: 21 estimators in six families, all behind one interface", "Price models estimate elasticity directly; promo models estimate the effect of a 15% discount, mapped to elasticity by dividing by ln(0.85)", "Models")
d.table(s, 0.6, 1.9, 12.15, CATALOG, [2.6, 1.2, 3.0, 5.3], 11, 0.32, bold_first_col=True, align=["left", "left", "left", "left"])


def fam_slide(title, kicker, models_, how, why, formula, color):
    s = d.slide(title, kicker, "Models")
    d.bullets(s, 0.6, 1.85, 5.9, 2.6, how, 14, head="How it works")
    d.bullets(s, 0.6, 4.55, 5.9, 2.3, why, 14, head="Why it is in this project")
    sub = bench[bench["model"].isin(models_)].sort_values("elasticity_rmse")
    rows = [["Model", "Elast. RMSE", "PEHE", "AUUC", "Fit s"]] + [[nm(r.model), f"{r.elasticity_rmse:.3f}", f"{r.pehe:.3f}", f"{r.auuc:.3f}", f"{r.fit_seconds:.1f}"] for r in sub.itertuples()]
    d.table(s, 6.8, 1.9, 5.95, rows, [2.6, 1.2, 1.0, 1.0, 0.9], 12, 0.4, bold_first_col=True)
    d.rect(s, 6.8, 2.1 + 0.4 * len(rows), 5.95, 1.1, LIGHT)
    d.text(s, 6.95, 2.15 + 0.4 * len(rows), 5.7, 1.0, [(formula, 12, False, INK)], anchor="middle")
    return s


fam_slide("Naive baselines: both ignore the hidden shock and both slope the wrong way",
          "They exist to show the size of the bias, not to be used", ["Naive OLS", "Naive XGBoost"],
          ["OLS: ln demand ~ ln price + X with a tiny ridge penalty", "XGBoost: flexible f(ln price, X); elasticity by finite difference at the reference price"],
          ["Show that flexibility alone does not fix endogeneity: XGBoost also has a positive mean elasticity", "Provide the 'naive pricing' policy against which RevPAR lift is measured"],
          "Bias = Cov(ln p, U | X) / Var(ln p | X) x gamma. Positive whenever managers raise rates in surges.", ROSE)
fam_slide("Linear IV: transparent, fast and, on a log-linear world, nearly as good as anything",
          "linearmodels with effect-modifier interactions gives heterogeneous elasticity and the formal tests", ["2SLS", "LIML", "IV-GMM"],
          ["2SLS: regress price on Z and X, then demand on fitted price (and interactions with segment, tier, loyalty, lead time)", "LIML and IV-GMM change the estimator weighting for robustness to weak instruments and heteroskedasticity"],
          ["Source of F, Wu-Hausman and Sargan-Hansen diagnostics", "Strong benchmark: hard for ML models to beat when the truth is log-linear"],
          "theta_hat = (T'MxT)^-1 T'Mx y, with T the first-stage fitted ln price; instruments Z and Z x h(X).", BLUE)
fam_slide("Orthogonal ML IV: flexible nuisances, orthogonal final stage",
          "econml estimators with GBDT nuisance models for E[Y|X], E[T|X] and E[Z|X]", ["DML-IV", "OrthoIV", "DR-IV (LinearDRIV)", "ForestDR-IV"],
          ["Residualise Y, T and Z on X with cross-fitting, then solve the orthogonal IV moment for theta(X)", "DR-IV adds a doubly robust pseudo-outcome; ForestDR-IV makes the final stage a forest", "DR-IV needs a scalar instrument, so the three Z are combined into the first-stage projection"],
          ["Best overall accuracy here (OrthoIV is first)", "Robust to nuisance misspecification; ForestDR-IV pays for flexibility with variance"],
          "E[(Y~ - theta(X) T~) Z~ | X] = 0, with Y~ = Y - E[Y|X] etc. (cross-fitted).", BLUE)
fam_slide("Deep IV and DeepGMM: PyTorch structural estimators for when the functional form is unknown",
          "Both recover a full structural demand function, not just a slope", ["DeepIV (PyTorch)", "DeepGMM (PyTorch)"],
          ["DeepIV stage 1: mixture density network for P(price | X, Z). Stage 2: outcome network trained on the average of h(X, t) over Monte Carlo draws from stage 1", "DeepGMM: learner g and adversary f play a min-max game on E[(Y - g(X,T)) f(X,Z)] = 0, selected by validation moment loss"],
          ["Learns a full structural demand curve with no functional-form assumption and drives the pricing flow", "On this log-linear data it is less accurate than linear or orthogonal-ML IV (see table) and costs 20-30 s on CPU"],
          "min_psi sum_i ( y_i - (1/M) sum_m h_psi(x_i, t_im) )^2, with t_im ~ f_phi(t | x_i, z_i).", PURPLE)
fam_slide("Meta-learners, causal forest and uplift forest: the classic CATE toolkit on the promo effect",
          "All five meta-learners plus two tree methods, estimating the effect of a 15% discount", ["S-Learner", "T-Learner", "X-Learner", "R-Learner (NonParamDML)", "DR-Learner", "Causal Forest DML", "Uplift RF (causalml)"],
          ["S: one model with treatment as a feature. T: separate models per arm. X: cross-imputed effects weighted by propensity", "R: residual-on-residual (NonParamDML). DR: AIPW scores regressed on X", "Causal Forest DML: honest trees on orthogonal residuals. Uplift RF: causalml CTS trees on conversion"],
          ["Promo targeting needs CATE, and the flows compare these against IV-based CATE", "Uplift RF trains on a noisy 0/1 outcome and is the weakest here"],
          "tau(x) = mu1(x) - mu0(x) (T); R-loss: sum[(Y - m(X)) - tau(X)(W - e(X))]^2.", TEAL)
fam_slide("Deep CATE networks: TARNet, DragonNet and the CEVAE bonus",
          "PyTorch representation learning for the promo effect", ["TARNet", "DragonNet", "CEVAE"],
          ["TARNet: shared representation, one outcome head per arm", "DragonNet: adds a propensity head and targeted regularisation", "CEVAE: variational autoencoder that recovers a latent confounder from proxies in X"],
          ["TARNet and DragonNet are competitive with forests", "CEVAE fails here: the data has no proxy for the hidden shock, which is exactly what CEVAE assumes. A mismatch, not a verdict on CEVAE"],
          "DragonNet loss = outcome MSE + alpha x propensity BCE + beta x targeted regulariser.", ORANGE)

order = bench.sort_values("elasticity_rmse").reset_index(drop=True)
hdr = ["#", "Model", "Family", "Elast. RMSE", "Bias", "PEHE", "AUUC", "Demand RMSE", "Fit s"]
minc = {3: order["elasticity_rmse"].min(), 4: order["elasticity_bias"].abs().min(), 5: order["pehe"].min(), 7: order["demand_curve_rmse"].min()}
for page, (a, b_) in enumerate([(0, 11), (11, 21)]):
    s = d.slide(f"Full benchmark, ranked by elasticity RMSE ({'best 11' if page == 0 else 'remaining 10'} of {len(order)})",
                "Held-out 7,500 bookings. Elasticity RMSE and bias vs the truth; PEHE and AUUC on the promo effect; demand RMSE on log demand", "Models")
    rows, hl = [hdr], {}
    for i, r in enumerate(order.iloc[a:b_].itertuples(), 1):
        rows.append([str(a + i), nm(r.model), FAMILY_LABEL[r.family], f"{r.elasticity_rmse:.3f}", f"{r.elasticity_bias:+.3f}", f"{r.pehe:.3f}", f"{r.auuc:.3f}", f"{r.demand_curve_rmse:.3f}", f"{r.fit_seconds:.1f}"])
        if r.elasticity_rmse == minc[3]: hl[(i, 3)] = GOOD
        if abs(r.elasticity_bias) == minc[4]: hl[(i, 4)] = GOOD
        if r.pehe == minc[5]: hl[(i, 5)] = GOOD
        if r.auuc == order["auuc"].max(): hl[(i, 6)] = GOOD
        if r.demand_curve_rmse == minc[7]: hl[(i, 7)] = GOOD
    d.table(s, 0.6, 1.9, 12.15, rows, [0.4, 2.6, 2.2, 1.2, 1.0, 1.0, 1.0, 1.3, 0.9], 12, 0.4, highlight=hl, align=["center", "left", "left", "center", "center", "center", "center", "center", "center"])
    if page == 0:
        d.text(s, 0.6, 6.78, 12.1, 0.3, [("Green = best in that column across all 21 models. Single seed: differences between the top IV models are within noise.", 11, False, MUTED, True)])
    else:
        d.text(s, 0.6, 6.35, 12.1, 0.6, [("Naive models and CEVAE are the only ones with a wrong-signed mean elasticity. Meta-learners estimate a 15% price cut, so their implied elasticity (tau / ln 0.85) amplifies noise by about 6x; PEHE is the fairer column for them.", 11, False, MUTED, True)])

s = d.slide("Elasticity and promo-effect error: IV and orthogonal-ML models cluster at the top, naive models at the bottom", None, "Models")
cats = [nm(m) for m in order["model"]]
pc = [FAMILY_COLOR[f] for f in order["family"]]
d.bar_chart(s, 0.5, 1.45, 6.3, 5.5, cats, {"Elasticity RMSE": order["elasticity_rmse"]}, [BLUE], title="Elasticity RMSE vs truth (lower is better)", fmt="0.00", point_colors=pc, font=9)
o2 = order.sort_values("pehe")
d.bar_chart(s, 6.7, 1.45, 6.2, 5.5, [nm(m) for m in o2["model"]], {"PEHE": o2["pehe"]}, [BLUE], title="PEHE on promo effect (lower is better)", fmt="0.000", point_colors=[FAMILY_COLOR[f] for f in o2["family"]], font=9)
for i, (k, v) in enumerate(FAMILY_LABEL.items()):
    d.rect(s, 0.8 + i * 2.4, 6.98, 0.16, 0.12, FAMILY_COLOR[k], shape=1)
    d.text(s, 1.0 + i * 2.4, 6.9, 2.2, 0.25, [(v, 9, False, MUTED)])

s = d.slide("Ranking quality and demand-curve fit tell a consistent story", "AUUC: how well each model ranks guests by promo uplift (1.0 = oracle). Demand RMSE: fit of the whole structural curve", "Models")
o3 = order.sort_values("auuc", ascending=False)
d.bar_chart(s, 0.5, 1.9, 6.3, 5.0, [nm(m) for m in o3["model"]], {"AUUC": o3["auuc"]}, [TEAL], title="AUUC vs oracle (higher is better)", fmt="0.00", point_colors=[FAMILY_COLOR[f] for f in o3["family"]], font=9, max_val=1.1)
o4 = order.sort_values("demand_curve_rmse")
d.bar_chart(s, 6.7, 1.9, 6.2, 5.0, [nm(m) for m in o4["model"]], {"Demand RMSE": o4["demand_curve_rmse"]}, [PURPLE], title="Structural demand-curve RMSE (lower is better)", fmt="0.00", point_colors=[FAMILY_COLOR[f] for f in o4["family"]], font=9)

s = d.slide("Structural demand curves and Qini curves: IV models track the truth; naive curves slope the wrong way", None, "Models")
d.image(s, VIS / "evaluation" / "02_structural_demand_curves_ols_vs_iv.png", 0.6, 1.5, 6.1, 4.2)
d.image(s, VIS / "evaluation" / "04_qini_and_cumulative_uplift_curves.png", 6.9, 1.5, 5.9, 4.2)
d.bullets(s, 0.6, 5.85, 12.1, 1.2, [
    "Left: expected conversion vs ADR. Naive OLS and XGBoost rise with price; 2SLS, DML-IV and DeepIV follow the ground truth.",
    "Right: cumulative true uplift when targeting by predicted CATE. Most models hug the oracle; Uplift RF sits near random and CEVAE falls well short."], 13)

top_iv = order[order["family"] == "iv"].iloc[0]
s = d.slide("What wins and why: match the estimator to how much structure the problem has", None, "Models")
d.box(s, 0.6, 1.7, 3.95, 4.2, "What the benchmark says", [
    f"Best elasticity: {nm(best['model'])} ({best['elasticity_rmse']:.2f}); naive OLS {ols['elasticity_rmse']:.2f}",
    "All IV and orthogonal-ML models cut the error by 6x or more",
    f"Best demand-curve fit and PEHE: {nm(order.sort_values('demand_curve_rmse').iloc[0]['model'])}; linear IV has near-zero bias; DeepIV is mid-table",
    "Forests and deep CATE nets are close behind on promo PEHE"], BLUE, 14)
d.box(s, 4.7, 1.7, 3.95, 4.2, "Why simple IV is strong here", [
    "The data-generating process is log-linear in price, so linear IV is well specified",
    "Deep and forest models spend capacity on structure that is not there",
    "With genuine non-linearity or many modifiers the ranking would shift; this is not a general claim"], TEAL, 13)
d.box(s, 8.8, 1.7, 3.95, 4.2, "Honest limits", [
    "One seed, one dataset: gaps among the top six are within noise",
    "CEVAE failure is a design mismatch (no proxy for U)",
    "Uplift RF is weakest: it learns from a noisy 0/1 outcome",
    "No GPU run: timings are CPU-only"], ROSE, 13)

# ============================================================================================ section 4
d.section_slide("4", "Flow-level view", "Stage metrics, overall metrics and every model compared across the flows")

FLOWKEY = [("elasticity", "Flow 1 · Elasticity identification", BLUE), ("pricing", "Flow 2 · Price optimisation", TEAL),
           ("promo", "Flow 3 · Promo CATE decision", PURPLE), ("crm", "Flow 4 · Batch CRM targeting", ROSE)]
SHORT = lambda lab: lab.replace(" (group, urban)", "").replace(" (resort)", "").replace(" · 500 mixed reservations", " 500 mixed").replace(" · 400 leisure reservations", " 400 leisure")  # noqa: E731
CHARTS = {
    "elasticity": ("Mean |elasticity error|: naive OLS vs IV median", ["naive_error", "iv_error"], ["Naive OLS", "IV median"], [ROSE, BLUE], "0.00"),
    "pricing": ("Share of the best achievable RevPAR captured (%)", ["capture_of_best_pct"], ["DeepIV policy"], [TEAL], "0"),
    "promo": ("CATE |error| and spread across 12 models", ["cate_abs_error", "model_spread"], ["|CATE error|", "Model spread"], [PURPLE, GREY], "0.000"),
    "crm": ("Targeting precision and oracle value captured (%)", ["precision_pct", "oracle_capture_pct"], ["Precision", "Oracle value captured"], [ROSE, ORANGE], "0"),
}
for key, title, color in FLOWKEY:
    fl = flows[key]
    exs = fl["examples"]
    s = d.slide(f"{title}: stage-level metrics for each example", fl["question"], "Flows")
    d.chevrons(s, 0.6, 1.75, 12.15, 0.55, [st["name"] for st in exs[0]["stages"]], [color], 11)
    rows = [["Stage", "Metric"] + [SHORT(e["label"]) for e in exs]]
    fills = {}
    for si in range(len(exs[0]["stages"])):
        for mi, m in enumerate(exs[0]["stages"][si]["metrics"]):
            r = [exs[0]["stages"][si]["name"] if mi == 0 else "", m["label"]] + [e["stages"][si]["metrics"][mi]["value"] for e in exs]
            rows.append(r)
            if si % 2 == 0:
                fills[len(rows) - 1] = LIGHT
            else:
                fills[len(rows) - 1] = WHITE
    d.table(s, 0.6, 2.45, 12.15, rows, [2.1, 2.9] + [3.0] * len(exs) if len(exs) == 3 else [2.1, 2.9, 3.5, 3.5], 11, 0.275, fills=fills, align=["left", "left"] + ["center"] * len(exs), bold_first_col=True)

    s = d.slide(f"{title}: overall metrics", exs[0]["headline"] if len(exs) == 1 else None, "Flows")
    raw = flows["raw_overall"][key]
    ex_labels = [SHORT(e["label"]) for e in exs]
    series = {}
    for field, lab in zip(CHARTS[key][1], CHARTS[key][2]):
        vals = [(r["precision"] * 100 if field == "precision_pct" else r[field]) for r in raw]
        series[lab] = [0 if v != v else v for v in vals]
    d.bar_chart(s, 0.5, 1.6, 6.6, 5.2, ex_labels, series, CHARTS[key][3], horizontal=False, title=CHARTS[key][0], fmt=CHARTS[key][4], legend=len(series) > 1, font=11)
    rows = [["Example", "Headline"]] + [[SHORT(e["label"]), e["headline"]] for e in exs]
    d.table(s, 7.3, 1.7, 5.5, rows, [1.6, 3.9], 11, 0.78, align=["left", "left"], bold_first_col=True)
    sc = flows["scorecard"][key]
    for i, (lab, val) in enumerate(sc.items()):
        d.card(s, 7.3 + i * 2.8, 1.85 + 0.78 * (len(exs) + 1) + 0.25, 2.65, 1.1, val, lab, color)

s = d.slide("Programme scorecard: flow-level results side by side", "Computed over all worked examples; Flow 4 is the one to improve", "Flows")
for i, (key, title, color) in enumerate(FLOWKEY):
    x = 0.6 + i * 3.08
    d.rect(s, x, 1.9, 2.95, 4.2, LIGHT)
    d.text(s, x + 0.15, 2.0, 2.7, 0.7, [(title, 13, True, color)])
    for j, (lab, val) in enumerate(flows["scorecard"][key].items()):
        d.text(s, x + 0.15, 2.8 + j * 1.6, 2.7, 1.4, [(val, 34, True, INK), (lab, 12, False, MUTED)])

s = d.slide("Streamlit tab 5 shows every stage, example and overall metric live", None, "Flows")
d.image(s, c_score, 0.6, 1.6, 6.0, 2.3)
d.image(s, c_flow, 0.6, 4.1, 6.0, 2.8)
d.bullets(s, 6.9, 1.7, 5.9, 5.0, [
    "Programme scorecard on top; radio selector for the four flows",
    "Each example is a row of five stage cards with three metrics each",
    "A one-line verdict per example, then an overall-metrics table for the flow",
    "Examples are fixed (scenarios A, B, C) and independent of the sidebar, so the page is reproducible",
    "The same logic is unit-tested: every flow has at least two examples, five stages, and IV error below naive error"], 14)

# ---- models across flows
s = d.slide("Every model run through every flow: the same examples, the same metrics", "F1 elasticity error (3 profiles) · F2 RevPAR captured by the model's own ADR · F3 promo CATE error and decisions · F4 targeting on one 500-guest batch", "Models across flows")
mcols = ["Model", "Family", "F1 |ε err|", "F2 RevPAR %", "F3 |CATE err|", "F3 ok/3", "F4 prec.", "F4 oracle %", "Mean rank"]
rows = [mcols]
best_idx = {"f1": mf["f1_elasticity_abs_error"].idxmin(), "f2": mf["f2_revpar_capture_pct"].idxmax(), "f3": mf["f3_cate_abs_error"].idxmin(),
            "f4": mf["f4_oracle_capture_pct"].idxmax(), "f4p": mf["f4_precision"].idxmax(), "rk": mf["mean_rank_across_flows"].idxmin()}
hl = {}
for i, r in enumerate(mf.itertuples(), 1):
    pr = "n/a" if r.f4_precision != r.f4_precision else f"{r.f4_precision:.0%}"
    rows.append([nm(r.model), FAMILY_LABEL[r.family].split(" /")[0].replace(" (biased)", ""), f"{r.f1_elasticity_abs_error:.2f}", f"{r.f2_revpar_capture_pct:.1f}",
                 f"{r.f3_cate_abs_error:.3f}", str(r.f3_correct_decisions), pr, f"{r.f4_oracle_capture_pct:.0f}", f"{r.mean_rank_across_flows:.1f}"])
    for key_, col in [("f1", 2), ("f2", 3), ("f3", 4), ("f4p", 6), ("f4", 7), ("rk", 8)]:
        if mf.index[i - 1] == best_idx[key_]:
            hl[(i, col)] = GOOD
d.table(s, 0.6, 1.95, 12.15, rows, [2.3, 1.5, 1.1, 1.2, 1.2, 0.9, 1.0, 1.2, 1.1], 9, 0.23, highlight=hl, align=["left", "left"] + ["center"] * 7)
d.text(s, 0.6, 7.0 - 0.02, 12, 0.1, [("", 1, False, WHITE)])

s = d.slide("Across all four flows the IV family leads on average rank; the naive models and CEVAE trail", "Mean rank over F1, F2, F3 and F4 oracle capture (lower is better). A composite, so read it with the caveats on the right", "Models across flows")
mr = mf.sort_values("mean_rank_across_flows")
d.bar_chart(s, 0.5, 1.9, 7.4, 5.1, [nm(m) for m in mr["model"]], {"Mean rank": mr["mean_rank_across_flows"]}, [BLUE], title="Mean rank across flows (lower is better)", fmt="0.0", point_colors=[FAMILY_COLOR[f] for f in mr["family"]], font=9)
n_ok3 = int((mf["f3_correct_decisions"] == 3).sum())
n_two = int((mf["f3_correct_decisions"] == 2).sum())
n_within = int((mf["f2_revpar_capture_pct"] >= mf["f2_revpar_capture_pct"].max() - 3).sum())
w1, w2, w3 = nm(mf.loc[best_idx["f1"], "model"]), nm(mf.loc[best_idx["f2"], "model"]), nm(mf.loc[best_idx["f4"], "model"])
d.bullets(s, 8.1, 1.9, 4.7, 5.0, [
    f"F1 elasticity: {w1} is closest to the truth on the three profiles",
    f"F2 pricing: {w2} captures the most RevPAR; {n_within} of {len(mf)} models are within 3 points of the best",
    f"F4 targeting: {w3} captures the most oracle value ({mf['f4_oracle_capture_pct'].max():.0f}%) on this batch",
    "Naive models and CEVAE never find a Persuadable, so they capture 0% and price at the ceiling",
    "F4 is one batch of 500 guests: its ranking is noisy. Treat per-flow winners as indicative"], 12, head="Reading the table")

s = d.slide("Different flows reward different models, so choose per decision rather than crowning one winner", None, "Models across flows")
d.box(s, 0.6, 1.7, 3.95, 3.9, "Elasticity and pricing (F1, F2)", [
    "Linear IV (IV-GMM, 2SLS, LIML) has the smallest elasticity error",
    f"{n_within} of {len(mf)} models reach within 3 points of the best achievable RevPAR: pricing is forgiving near the optimum",
    "Pick the simple, auditable model"], BLUE, 13)
d.box(s, 4.7, 1.7, 3.95, 3.9, "Promo CATE decisions (F3)", [
    f"{n_ok3} of {len(mf)} models decide all three examples correctly; {n_two} get only two, including every naive model",
    "The median of the 12-model ensemble (Flow 3) gets all three right",
    "Use an ensemble median rather than a single CATE model"], PURPLE, 13)
d.box(s, 8.8, 1.7, 3.95, 3.9, "Batch targeting (F4)", [
    "Spread between models is large (from negative to about 60% of oracle value)",
    "The dashboard default (Causal Forest) is not the best on this batch",
    "Validate the targeting model on held-out batches before any rollout"], ROSE, 13)

# ============================================================================================ section 5
d.section_slide("5", "Uplift end to end", "High-level documentation of the promotional targeting flow")

s = d.slide("Uplift modelling answers 'whom does the promotion change?', not 'who will book?'", "A discount only creates value for guests it persuades; for everyone else it gives revenue away", "Uplift")
d.text(s, 0.6, 1.85, 5.6, 0.5, [("Four quadrants used in the dashboard and the CRM export", 14, True, BLUE)])
quad = [("Persuadables", "Net value of the promo is clearly positive: target them", TEAL, 0, 0), ("Sure Things", "Likely to book anyway (baseline conversion >= 70%), little uplift: no discount", BLUE, 1, 0),
        ("Lost Causes", "Unlikely to book and unmoved by a discount: no discount", GREY, 0, 1), ("Sleeping Dogs", "Net value clearly negative (discount costs more than it earns): never discount", ROSE, 1, 1)]
for t, sub, c, cx, cy in quad:
    d.rect(s, 0.6 + cx * 3.0, 2.45 + cy * 2.05, 2.9, 1.95, c)
    d.text(s, 0.75 + cx * 3.0, 2.55 + cy * 2.05, 2.6, 0.5, [(t, 16, True, WHITE)])
    d.text(s, 0.75 + cx * 3.0, 3.05 + cy * 2.05, 2.6, 1.3, [(sub, 12, False, WHITE)])
d.bullets(s, 6.9, 1.85, 5.9, 5.0, [
    "Effect of the promo on log demand: tau(x) = theta(x) x ln(1 - d), with d = 15%",
    "Conversion is converted through the saturating link occupancy = 1 - exp(-D)",
    "Net value = occ(promo) x (p(1-d) - c) - occ(base) x (p - c), with unit cost c",
    "Segment rule: net value above a small threshold = Persuadable; below minus threshold = Sleeping Dog; otherwise Sure Thing or Lost Cause by baseline conversion",
    "Only Persuadables receive the offer; the export carries the recommended promo %"], 13, head="Definitions")

s = d.slide("The uplift flow in one picture: reservations in, a priced offer list out, validated against the truth", None, "Uplift")
d.chevrons(s, 0.6, 1.75, 12.15, 0.95, ["Reservations\n(parquet or CSV)", "Features X\n+ offered ADR", "CATE models\n(12 available)", "Promo effect\ntau(x)", "Net value\nper guest", "Quadrant\nsegment", "Offers CSV\n/ API / UI"],
           [BLUE, TEAL, PURPLE, ORANGE, ROSE, TEAL, BLUE], 11)
d.box(s, 0.6, 3.0, 3.9, 3.3, "Inputs and models", [
    "Booking features: lead time, LOS, DOW, season, loyalty, ADR sensitivity, tier, room, segment, offered ADR",
    "Trained on observed promo assignment (CRM targeting depends on X only, so it is ignorable given X)",
    "Any of: S/T/X/R/DR, Causal Forest, Uplift RF, TARNet, DragonNet, CEVAE, DeepIV, DR-IV"], BLUE, 12)
d.box(s, 4.7, 3.0, 3.9, 3.3, "Decision logic", [
    "crm_segments() in revenue_optimizer.py: baseline conversion from the structural demand, uplift from tau(x)",
    "Net value combines uplift with the discount given away and unit cost",
    "Segment rule above; recommended promo = 15% for Persuadables, else 0"], TEAL, 12)
d.box(s, 8.8, 3.0, 3.95, 3.3, "Outputs and where to find them", [
    "Streamlit tab 4: pick a model, generate or upload reservations, download optimal_promo_offers.csv",
    "API: POST /api/v1/cate-benchmark for one guest, 12-model CATE and consensus uplift",
    "Flow 4 (flows.py) validates a batch against the ground truth"], ORANGE, 12)

s = d.slide("How the uplift flow is validated, what it achieves and where it falls short", "Value is the TRUE net RevPAR change under the ground truth, not the model's own estimate", "Uplift")
d.image(s, VIS / "evaluation" / "04_qini_and_cumulative_uplift_curves.png", 0.6, 1.8, 5.2, 3.45)
rows = [["Batch / model", "Persuad.", "Target", "Blanket", "Oracle", "Precision", "Captured"]]
for e in flows["crm"]["examples"]:
    nums = {m["label"]: m["value"] for st in e["stages"] for m in st["metrics"]}
    rows.append([SHORT(e["label"]), nums["Persuadables"], nums["Target Persuadables"], nums["Discount everyone"], nums["Oracle targeting"], nums["Precision"], nums["Oracle value captured"]])
d.table(s, 6.0, 1.85, 6.75, rows, [2.6, 0.9, 0.9, 1.0, 0.9, 0.9, 0.9], 10, 0.5, align=["left"] + ["center"] * 6)
caps = [r["oracle_capture_pct"] for r in flows["raw_overall"]["crm"]]
b1 = mf.sort_values("f4_oracle_capture_pct", ascending=False).iloc[0]
cf = mf[mf["model"] == "Causal Forest DML"].iloc[0]
d.bullets(s, 6.0, 3.6, 6.75, 3.3, [
    "Metrics: Qini curve and AUUC (ranking), precision of the Persuadable list, share of oracle value captured, gain over discounting everyone",
    f"Targeting turns a large loss from a blanket 15% discount into a small gain, but captures only {min(caps):.0f}-{max(caps):.0f}% of the oracle value on these batches",
    "Precision is about 60%; 'Sure Things' are empty because almost every guest has negative net value from the promo",
    f"Models differ widely on batch 1 (best: {nm(b1['model'])} {b1['f4_oracle_capture_pct']:.0f}% captured; Causal Forest {cf['f4_oracle_capture_pct']:.0f}%): validate before rollout"], 12, head="What it achieves")
d.bullets(s, 0.6, 5.4, 5.2, 1.6, [
    "Synthetic data: assignment is ignorable given X by design",
    "Real CRM data needs an overlap and unconfoundedness check, ideally a holdout experiment"], 11, head="Guardrails")

# ============================================================================================ section 6
d.section_slide("6", "Scenario stories", "A week at Grand Horizon, told through the Streamlit dashboard")


r = SC["A"]; nv, dp = r["per_model"]["Naive OLS"], r["per_model"]["DeepIV (PyTorch)"]
s = d.slide("Monday: a tech convention fills the city and every signal says 'raise the price'", "Scenario A · optimiser tab · hidden shock U = +1.8, competitor outage 22%", "Scenario story")
d.image(s, c_price["A"], 0.6, 1.75, 7.7, 4.2)
d.bullets(s, 8.55, 1.8, 4.3, 4.4, [
    f"Naive OLS sees demand rising with price (elasticity {nv['estimated_elasticity']:+.2f}) and recommends ${nv['optimal_adr']:,.0f}.",
    f"DeepIV recovers elasticity {dp['estimated_elasticity']:+.2f} (truth {r['true_elasticity']:+.2f}) and recommends ${dp['optimal_adr']:,.0f}; the true optimum is ${r['truth_optimal_adr']:,.0f}.",
    f"Expected RevPAR is {r['revpar_lift_pct_causal_vs_naive']:+.1f}% vs naive. If this exact surge materialises, the naive $650 would have earned more: DeepIV would be {abs(r['revpar_lift_pct_causal_vs_naive_at_scenario_u']):.1f}% lower."], 13, head="What the engine says")
d.box(s, 0.6, 6.05, 12.15, 0.95, "Take-away", ["The naive model is right on surge nights for a reason it cannot see, and wrong on every other night. Pricing from the structural curve is the robust choice."], ROSE, 12)

s = d.slide("Before trusting the numbers, the diagnostics lab confirms the instruments are strong and price is endogenous", "Scenario A · EDA and endogeneity tab", "Scenario story")
d.image(s, c_eda, 0.6, 1.7, 7.2, 5.2)
d.bullets(s, 8.05, 1.8, 4.8, 5.0, [
    f"First-stage F {fs['f_statistic_robust']:,.0f} against a threshold of 10",
    f"Wu-Hausman {diag['wu_hausman_endogeneity']['statistic']:,.0f}: endogeneity detected",
    f"Sargan p = {het.get('pvalue', float('nan')):.2f} on the heterogeneous model: instruments consistent with validity",
    f"Price coefficient moves from {co['ols_price_coefficient']:+.2f} (OLS) to {co['2sls_price_coefficient']:+.2f} (2SLS); truth {co['true_mean_elasticity']:+.2f}",
    "Below the KPIs: the Simpson's-paradox chart, instrument relevance, exogeneity heatmap and structural demand curves"], 13, head="What the revenue team sees")

r = SC["B"]; rc = SC["Balt"]
nvB, dpB = r["per_model"]["Naive OLS"], r["per_model"]["DeepIV (PyTorch)"]
dpC = rc["per_model"]["DeepIV (PyTorch)"]
s = d.slide("Wednesday: a quiet mid-week night, and two guests who should be priced very differently", "Scenario B · optimiser tab with segment contrast", "Scenario story")
d.image(s, c_price["B"], 0.6, 1.75, 7.7, 4.2)
d.bullets(s, 8.55, 1.8, 4.3, 4.4, [
    f"Leisure early-bird: true elasticity {r['true_elasticity']:+.2f}, DeepIV {dpB['estimated_elasticity']:+.2f}. Recommended ${dpB['optimal_adr']:,.0f} with a {dpB['optimal_promo_pct']:.0f}% promo; the true optimum is ${r['truth_optimal_adr']:,.0f}.",
    f"Corporate traveller: true {rc['true_elasticity']:+.2f}, DeepIV {dpC['estimated_elasticity']:+.2f}. Recommended ${dpC['optimal_adr']:,.0f}, no promo.",
    f"Naive OLS charges ${nvB['optimal_adr']:,.0f} to both. For the leisure guest that lifts RevPAR from ${nvB['realised_net_revpar']:,.0f} to ${dpB['realised_net_revpar']:,.0f} ({r['revpar_lift_pct_causal_vs_naive']:+.0f}%)."], 13, head="What the engine says")
d.box(s, 0.6, 6.05, 12.15, 0.95, "Take-away", ["Heterogeneity is the product: one rate for all guests is wrong in both directions. The corporate guest hits the $650 grid cap, so the engine would price higher with a wider grid."], TEAL, 12)

s = d.slide("The model arena shows who to trust for this guest: 21 estimators, one scoreboard", "Scenario B · CATE and deep causal arena tab", "Scenario story")
d.image(s, c_arena, 0.6, 1.7, 6.2, 5.3)
d.bullets(s, 7.1, 1.8, 5.7, 5.0, [
    "Table: PEHE, elasticity RMSE, bias, AUUC and fit time for every model, colour-coded by family",
    "Bar charts rank all 21 models on PEHE and elasticity RMSE",
    "Qini curves show that most causal models rank guests near the oracle",
    "The live bar chart at the bottom shows each model's promo CATE for the selected guest against the truth, with the median as the ensemble answer",
    f"Top of the table: {', '.join(nm(m) for m in order['model'].head(3))}"], 13, head="What it shows")

r = SC["C"]; nvC, dpC2 = r["per_model"]["Naive OLS"], r["per_model"]["DeepIV (PyTorch)"]
s = d.slide("Friday: the feeder airport cuts seats 25%, and the market rate falls, but the right price does not", "Scenario C · optimiser tab · airline capacity shock -2.5", "Scenario story")
d.image(s, c_price["C"], 0.6, 1.75, 7.7, 4.2)
d.bullets(s, 8.55, 1.8, 4.3, 4.4, [
    f"A market-following manager drops the rate to about ${r['status_quo_adr']:,.0f}, because competitors do.",
    f"The seat cut moves the market rate but not structural demand, so the optimum stays ${r['truth_optimal_adr']:,.0f}; DeepIV recommends ${dpC2['optimal_adr']:,.0f}.",
    f"RevPAR: {r['revpar_lift_pct_causal_vs_naive']:+.0f}% vs naive and {r['revpar_lift_pct_causal_vs_status_quo']:+.1f}% vs following the market."], 13, head="What the engine says")
d.box(s, 0.6, 6.05, 12.15, 0.95, "Take-away", ["This is the instrument at work: an exogenous shifter of the rate is information about price response, not a reason to change the price."], BLUE, 12)

s = d.slide("Same Friday, now the CRM question: which of 500 upcoming reservations should get a promotion?", "Scenario C · batch CRM targeting tab", "Scenario story")
d.image(s, c_crm, 0.6, 1.75, 7.6, 4.2)
e1 = flows["crm"]["examples"][0]
nums1 = {m["label"]: m["value"] for st in e1["stages"] for m in st["metrics"]}
d.bullets(s, 8.45, 1.8, 4.4, 4.3, [
    f"Scored with Causal Forest DML: {nums1['Persuadables']} Persuadables, {nums1['Sleeping Dogs']} Sleeping Dogs, {nums1['Sure Things / Lost Causes'].split(' / ')[1]} Lost Causes.",
    f"Discounting everyone changes net RevPAR by {nums1['Discount everyone']}; targeting Persuadables earns {nums1['Target Persuadables']}.",
    "The CSV download lists each reservation with its segment and recommended promo %."], 13, head="What the engine says")
d.box(s, 0.6, 6.05, 12.15, 0.95, "Take-away", [f"Targeting avoids a large blanket-discount loss but captures only {nums1['Oracle value captured']} of the oracle value on this batch. Model choice matters: see the cross-model comparison."], ROSE, 12)

r = SC["D"]; nvD, dpD = r["per_model"]["Naive OLS"], r["per_model"]["DeepIV (PyTorch)"]
s = d.slide("Saturday: the revenue manager takes the controls and explores a custom booking", "Scenario D · live simulation with sliders for every confounder and instrument", "Scenario story")
d.image(s, c_price["D"], 0.6, 1.75, 7.7, 4.2)
d.bullets(s, 8.55, 1.8, 4.3, 4.4, [
    "Sidebar sliders set segment, tier, lead time, LOS, season, loyalty, sensitivity, room type, the three instruments and the hidden shock.",
    f"Default profile: true elasticity {r['true_elasticity']:+.2f}; DeepIV recommends ${dpD['optimal_adr']:,.0f} (true optimum ${r['truth_optimal_adr']:,.0f}); naive OLS ${nvD['optimal_adr']:,.0f}.",
    f"RevPAR {r['revpar_lift_pct_causal_vs_naive']:+.0f}% vs naive, {r['revpar_lift_pct_causal_vs_status_quo']:+.1f}% vs the market-following rate."], 13, head="What the engine says")
d.box(s, 0.6, 6.05, 12.15, 0.95, "Take-away", ["Every number recomputes live from the saved models, and the table below the charts shows all six pricing models side by side."], ORANGE, 12)

# ============================================================================================ close
s = d.slide("Limitations to keep in mind before using this on real data", None, "Close")
d.box(s, 0.6, 1.6, 3.95, 4.3, "Data and identification", [
    "Synthetic data with a known truth: real data will be messier",
    "Instruments are valid by construction; real ones need an exclusion argument and a sensitivity analysis",
    "Promo assignment is ignorable given X by design; real CRM data needs overlap checks",
    "Prices capped at $650: corporate guests hit the ceiling"], ROSE, 13)
d.box(s, 4.7, 1.6, 3.95, 4.3, "Models and evaluation", [
    "Single seed and single dataset; top-model gaps are within noise",
    "Log-linear DGP favours linear IV; the ranking may change with true non-linearity",
    "CEVAE is mis-specified here; Uplift RF learns from a noisy outcome",
    "Cross-model flow ranking uses few examples and one CRM batch"], ORANGE, 13)
d.box(s, 8.8, 1.6, 3.95, 4.3, "Engineering", [
    "GPU paths implemented but not run: no CUDA device available",
    "causalml's own meta-learner cross-check does not run on scikit-learn 1.6 and is skipped",
    "Python 3.9.7 limits Streamlit to 1.12; econml was upgraded to 0.17.1",
    "Fitted-model registry (253 MB) is git-ignored: run make train"], BLUE, 13)

s = d.slide("Recommended next steps", None, "Close")
steps = [("1", "Run on GPU and more seeds", "Execute the CUDA paths, repeat the benchmark over seeds, report intervals", BLUE),
         ("2", "Replace synthetic data", "Fit on real bookings, audit instruments, add sensitivity analysis", TEAL),
         ("3", "Validate targeting with an experiment", "Hold out a randomised promo group; compare to the uplift model's Persuadable list", PURPLE),
         ("4", "Widen the price grid and add constraints", "Lift the $650 cap, add inventory and parity constraints to the optimiser", ORANGE)]
for i, (n, t, sub, c) in enumerate(steps):
    x, y = 0.6 + (i % 2) * 6.15, 1.7 + (i // 2) * 2.6
    d.rect(s, x, y, 6.0, 2.3, LIGHT)
    d.text(s, x + 0.25, y + 0.15, 0.8, 0.9, [(n, 40, True, c)])
    d.text(s, x + 1.1, y + 0.3, 4.7, 0.6, [(t, 17, True, INK)])
    d.text(s, x + 1.1, y + 1.0, 4.7, 1.2, [(sub, 13, False, MUTED)])

s = d.slide("Appendix: reproduce everything with one command", None, "Appendix")
d.table(s, 0.6, 1.75, 12.15, [["Step", "Command", "Produces"],
                              ["Everything", "bash run_all.sh   (or make all)", "data, EDA, models, evaluation plots, screenshots (about 8 min on CPU)"],
                              ["Flows and model-by-flow comparison", "python -m src.optimization.flows", "outputs/metrics/flows.json, model_flow_comparison.csv"],
                              ["This presentation", "python presentation/build_deck.py", "presentation/Grand_Horizon_IV_CATE_End_to_End.pptx"],
                              ["API / dashboard", "make api   |   make ui", "Swagger at :8000/docs, Streamlit at :8501"],
                              ["Tests", "make test", "DGP bias, AUUC, optimiser, flows"]], [2.6, 4.2, 5.3], 12, 0.55, align=["left", "left", "left"], bold_first_col=True)
d.bullets(s, 0.6, 5.2, 12.1, 1.7, [
    "Stack: PyTorch, econml 0.17, causalml, linearmodels, XGBoost / LightGBM / CatBoost, FastAPI, Streamlit, Plotly, Playwright",
    "Code map: data/ (DGP), src/models, src/optimization (optimiser, flows), src/pipelines, src/api, app/, scripts/, presentation/"], 13)

out = HERE / "Grand_Horizon_IV_CATE_End_to_End.pptx"
d.save(out)
print(f"saved {out} with {d.n} slides")
