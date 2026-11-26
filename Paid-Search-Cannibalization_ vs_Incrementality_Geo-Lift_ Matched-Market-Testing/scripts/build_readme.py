"""Generate README.md from outputs/metrics/* (never hand-edit the README; edit this file)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

M, P, S = ROOT / "outputs" / "metrics", ROOT / "outputs" / "plots", ROOT / "outputs" / "screenshots"
rj = lambda n: json.loads((M / n).read_text())


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                     else ("" if isinstance(v, float) else str(v)) for v in r.values) + " |")
    return "\n".join(out)


def usd(x):
    return f"${x:,.0f}"


def img(rel, alt):
    return f"![{alt}](outputs/{rel})" if (ROOT / "outputs" / rel).exists() else f"*(missing: {rel})*"


def flows_md() -> str:
    """Programme scorecard, roll-up and per-stage metrics of the 8 end-to-end flows (from outputs/metrics/flows.json)."""
    d = rj("flows.json")
    out = ["### Programme scorecard (overall metrics)", ""]
    sc = [{"group": g["group"], "metrics": "; ".join(f"{m['label']}: **{m['value']}**" for m in g["metrics"])} for g in d["programme"]["groups"]]
    out.append(md(pd.DataFrame(sc)))
    out += ["", "### Flow roll-up (4 flow types x 2 examples)", "", md(pd.DataFrame(d["programme"]["rollup"])), ""]
    for fid in d["order"]:
        f = d["flows"][fid]
        out += [f"#### {f['id']} - {f['title']}", "", f"*{f['type_title']}: {f['question']}* {f['story']}", "",
                f"**Verdict: {f['verdict']}.** {f['action']}", "",
                md(pd.DataFrame([{"stage": st["title"], "status": (st.get("flag") or "").upper(),
                                  "metrics": "; ".join(f"{m['label']}: {m['value']}" for m in st["metrics"])} for st in f["stages"]])), ""]
        shot = sorted(S.glob(f"*_streamlit_flow_{fid}.png"))
        if shot:
            out += [f"![{fid}](outputs/screenshots/{shot[0].name})", ""]
    return "\n".join(out)


def main():
    mac, prog, val = rj("macro_results.json"), rj("programme_economics.json"), rj("policy_value.json")
    hw, meta = rj("hardware.json"), rj("experiment_meta.json")
    mc = pd.read_csv(M / "macro_montecarlo.csv")
    mc_s = mc.groupby("method").agg(bias=("err", "mean"), rmse=("err", lambda e: float(np.sqrt((e ** 2).mean()))),
                                    ci_coverage=("covered", "mean"), n_experiments=("seed", "count")).reset_index()
    rows = []
    for oc, lab in (("direct_bookings", "Direct bookings"), ("ota_bookings", "OTA bookings")):
        for m in ("MMT", "SDID", "BSTS"):
            r = mac[oc][m]
            rows.append({"outcome": lab, "method": m, "ATT per DMA-day": r["att"], "95% CI low": r["ci_low"], "95% CI high": r["ci_high"],
                         "truth": mac[oc]["truth"], "covers truth": "yes" if r["covers_truth"] else "NO",
                         "cumulative (10 DMAs x 28d)": r["cumulative"]})
    macro_tbl = md(pd.DataFrame(rows), "{:.2f}")
    pe = pd.DataFrame({m: {
        "Last-click ROAS": prog[m]["last_click_roas"], "Incremental ROAS (iROAS)": prog[m]["iroas"],
        "Cannibalization rate": prog[m]["cannibalization_rate"],
        "Incremental direct bookings from keeping SEM": prog[m]["incremental_direct_bookings_from_keeping"],
        "OTA leakage bookings when paused": prog[m]["ota_leakage_bookings_when_paused"],
        "OTA commission leakage ($)": prog[m]["ota_commission_leakage_usd"],
        "Branded spend avoided ($)": prog[m]["branded_spend_avoided_usd"],
        "Net margin change if paused ($)": prog[m]["net_margin_change_if_paused_usd"]} for m in ("MMT", "SDID", "BSTS", "Truth")})
    pe_tbl = md(pe.reset_index().rename(columns={"index": "metric"}), "{:,.2f}")
    lbd = pd.read_csv(M / "cate_leaderboard_direct.csv", index_col=0).reset_index()
    lbo = pd.read_csv(M / "cate_leaderboard_ota.csv", index_col=0).reset_index()
    keep = ["model", "pehe", "pehe_norm", "auuc", "qini", "qini_ratio", "rank_corr", "ate_hat", "ate_true"]
    pl_t = pd.read_csv(M / "placebo_in_time.csv")
    rules = pd.read_csv(M / "policy_rules.csv")
    bytier = pd.read_csv(M / "policy_by_tier.csv")
    bal = pd.read_csv(M / "balance_table.csv")
    shots = sorted(S.glob("*.png"))
    shot_md = "\n\n".join(f"**{p.stem}**\n\n![{p.stem}](outputs/screenshots/{p.name})" for p in shots if '_streamlit_flow_' not in p.name) or "*(run `make verify`)*"
    sdid_d = mac["direct_bookings"]["SDID"]
    best_d = lbd[lbd.model != "Constant ATE (DiD)"].iloc[0]
    const_d = lbd[lbd.model == "Constant ATE (DiD)"].iloc[0]
    n_better = int((lbd[lbd.model != "Constant ATE (DiD)"].pehe_norm < const_d.pehe_norm).sum())

    flows = flows_md()
    readme = f"""# Paid Search Cannibalization vs. Incrementality Engine
**Grand Horizon Hotels & Resorts** - does bidding on our own brand terms buy bookings, or does it just rent traffic we would
have received for free while keeping OTAs (Expedia, Booking.com) off the top of the results page?

*This README is generated by `scripts/build_readme.py` from `outputs/metrics/*`. All data are synthetic with a known
data-generating process, so every estimate can be checked against ground truth.*

## Headline results (seed {meta['seed']})

* Pausing branded paid search in 10 of 50 DMAs for 4 weeks cut direct bookings by **{abs(sdid_d['att']):.1f} per DMA-day**
  (SDID; 95% interval {sdid_d['ci_low']:.1f} to {sdid_d['ci_high']:.1f}; truth {mac['direct_bookings']['truth']:.1f}) and sent
  **{mac['ota_bookings']['SDID']['att']:.1f} per DMA-day** to OTAs (truth {mac['ota_bookings']['truth']:.1f}).
* Last-click ROAS says **{prog['SDID']['last_click_roas']:.1f}x**; the incremental ROAS is **{prog['SDID']['iroas']:.1f}x**
  (truth {prog['Truth']['iroas']:.1f}x). Cannibalization rate **{prog['SDID']['cannibalization_rate']:.0%}** (truth {prog['Truth']['cannibalization_rate']:.0%}).
* Pausing would have cost **{usd(-prog['SDID']['net_margin_change_if_paused_usd'])}** of net contribution margin by the SDID estimate over the 4 weeks
  across the 10 holdout DMAs (truth {usd(-prog['Truth']['net_margin_change_if_paused_usd'])}), including
  {usd(prog['SDID']['ota_commission_leakage_usd'])} of OTA commission leakage. This net figure is **near break-even and uncertain** (its interval spans zero, and the simulated truth is on the other side of zero), so the evidence neither supports pausing on average nor shows branded search is highly profitable; re-test before changing budgets.
* **DMA-level pause lists are not reliable with this design.** Average effects are recovered well; heterogeneity is not
  (see CATE section). The best individual learner has PEHE/sd(true CATE) = {best_d.pehe_norm:.2f}
  ({best_d['model']}), and only {n_better} of 9 learners beat simply predicting the average effect (ratio {const_d.pehe_norm:.2f}). Realised
  value of the model policy: {usd(val['model_policy_usd'])} vs always-keep $0 and an unattainable oracle of {usd(val['oracle_usd'])}.

{img('plots/iroas_waterfall.png', 'iROAS waterfall')}

## Experiment

| item | value |
|---|---|
| DMAs | 50 (10 Tier 1 Major Hub, 20 Tier 2 Regional, 20 Tier 3 Drive-to) |
| Horizon | 20-week pre-period + 4-week intervention, daily ({meta['intervention_start']} intervention start) |
| Assignment | Stratified randomisation: 20% of each tier paused (2/4/4 = 10 holdout, 40 control) |
| OTA commission | 15-20% variable (Tier 1 20%, Tier 2 17.5%, Tier 3 15%, DMA-level noise) |
| Outcomes | Direct bookings & GBV; OTA bookings & commission; Net Contribution Margin |
| Hardware | `{json.dumps(hw)}` |

`Net Contribution Margin = Direct GBV - Branded SEM spend + OTA GBV x (1 - commission rate)`. Value of branded search = -ATT(holdout).

```mermaid
flowchart LR
  G[synthetic_geo_generator<br/>50 DMAs, known truth] --> F[feature_engineering<br/>DTW + Mahalanobis, SMD]
  F --> MG[macro_geolift<br/>MMT - BSTS - SDID]
  F --> CT[CATE suite<br/>S T X R DR - LinearDML CausalForestDML - TARNet DragonNet]
  MG --> E[cannibalization_roi<br/>iROAS, commission, policy]
  CT --> E
  E --> FL[flows.py<br/>8 flows, stage + programme metrics]
  MG --> FL
  CT --> FL
  FL --> UI
  FL --> API
  E --> API[FastAPI<br/>/predict /simulate /metrics]
  E --> UI[Streamlit<br/>5 tabs, 3 presets, 8 flows]
  API --> PW[Playwright e2e]
  UI --> PW
```

### Data-generating process (what "truth" means)
With branded paid search paused, an OTA can take the top results slot. Let `s` = last-click share of direct bookings,
`c` = cannibalization (share that would arrive via the free organic listing anyway), `q` = OTA hijack rate of those organic-bound
clickers, `v` = share of genuinely incremental bookings that re-book on an OTA.

* Direct effect of pausing: `-E0_direct * s * [(1-c) + c*q]`
* OTA effect of pausing: `+E0_direct * s * [(1-c)*v + c*q]` (hijacked bookings are *diverted*, not new)
* `c` falls with worse organic rank and OTA pressure and rises with loyalty penetration; `q` rises with OTA pressure and worse rank.
* Common latent time factors, DMA drifts, day-of-week, holidays and seasonality make the problem non-trivial.

## Macro geo-lift estimators (ATT, bookings per DMA-day)

{macro_tbl}

* **MMT** - DTW + Mahalanobis one-to-one pairing (Hungarian), ratio-adjusted counterfactual, exact sign-flip permutation p-value. Balance after matching: max |SMD| = {bal.smd_matched.abs().max():.2f} (all-controls: {bal.smd_all_controls.abs().max():.2f}).
* **BSTS** - NumPyro NUTS: local linear trend + day-of-week + holiday + regression on control-DMA principal components, tier-reweighted control mean and flight index; forecast counterfactual with 95% credible band.
* **SDID** - ridge-regularised simplex unit weights, time weights, placebo-in-space standard error.

{img('plots/geolift_direct_bookings.png', 'direct geo-lift')}
{img('plots/geolift_ota_bookings.png', 'OTA geo-lift')}
{img('plots/att_estimators.png', 'estimator comparison')}

### How much to trust each estimator
Monte-Carlo over fresh draws of the *whole experiment* (new DMAs, new randomisation), direct bookings:

{md(mc_s, "{:.2f}")}

BSTS is the weakest and is **biased toward zero** here (it extrapolates a 28-day forecast from ~4 PCs of controls; with only 10 treated
DMAs aggregated to one series the regression cannot absorb their idiosyncratic drift). MMT and SDID are close to unbiased. I did not tune the
BSTS priors to the headline seed.

### Refutation tests
Placebo-in-time (pretend the pause started 4 weeks early; pre-data only; truth = 0):

{md(pl_t[['outcome','method','placebo_att','ci_low','ci_high','p_value','covers_zero']], "{:.3f}")}

A placebo effect that excludes zero (MMT, direct bookings) flags a pre-period drift between holdout and matched controls, and the headline
MMT estimate is more negative than truth by about the same amount. Placebo-in-space (SDID on fake holdouts among controls): SDID p = {mac['direct_bookings']['placebo_space_sdid']['p_value']:.3f}.

{img('plots/placebo_tests.png', 'placebo tests')}
{img('plots/montecarlo_error.png', 'monte carlo')}
{img('plots/balance_smd.png', 'balance')}

## Economics (4-week holdout programme, 10 DMAs)

{pe_tbl}

## CATE meta-learners: 9 algorithms, GPU-accelerated

Learners target the **relative lift in bookings** (outcome / DMA pre-period mean) because the net-margin effect nets a large direct
loss against an OTA gain and is far too noisy to learn from 10 treated DMAs; the economics layer composes the two booking effects into dollars.
All cross-fitting and evaluation folds are **grouped by DMA** (treatment is DMA-level, so rows of one DMA are not independent); the
propensity is the known 0.2 of the stratified design. Base learners: XGBoost `device="cuda"` (S, T, R, DR, DML nuisances), CatBoost
`task_type="GPU"` (X); EconML LinearDML / CausalForestDML; custom PyTorch CUDA TARNet and DragonNet (targeted regularisation).

Direct-booking lift (truth: average {const_d.ate_true*100:.1f}% of baseline). PEHE is in lift units (0.05 = 5 points); `pehe_norm` = PEHE / sd(true CATE), where 1.0 means "as bad as knowing nothing about heterogeneity":

{md(lbd[keep], "{:.3f}")}

OTA-booking lift:

{md(lbo[keep], "{:.3f}")}

Honest reading: ATEs are recovered well, the *ranking* of DMAs is only modestly better than chance (rank correlation ~0.2-0.55) and no learner beats the
constant-effect reference on PEHE. That is the real information content of a 10-treated-DMA experiment, not a modelling bug: effective sample size is 50 DMAs.

{img('plots/qini_direct.png', 'qini')}
{img('plots/leaderboard_direct.png', 'leaderboard')}
{img('plots/cate_heterogeneity.png', 'heterogeneity')}

## Optimal bidding policy

`pi*(x) = Pause` iff predicted change in net contribution margin from pausing exceeds an evidence margin (k x disagreement between the nine learners, k=1 fixed in advance), else `Keep`.
Realised value (4 weeks, scored against ground truth that a real experiment would not have):

{md(rules, "{:.2f}")}

True value of pausing by tier (negative = pausing loses money):

{md(bytier, "{:,.0f}")}

{img('plots/policy_by_dma.png', 'policy by DMA')}

## End-to-end flows (Streamlit tab 5, `GET /flows`)

Four flow types, **two worked examples each**. Every flow is a sequence of stages and every stage reports its own metrics, followed by flow-level metrics and
a verdict; the programme scorecard rolls the whole engine up. Statuses: PASS / CAUTION / FAIL are rule-based (interval covers truth, placebo
covers zero, Monte-Carlo bias, value against always-keep), not judgement. Built by `src/flows.py` (`make flows`) into `outputs/metrics/flows.json`, which the
dashboard, the API and this README all read.

| flow type | question | example 1 | example 2 |
|---|---|---|---|
| A. Holdout read-out | incremental and worth the money? | A1 all 10 holdout DMAs | A2 the two Tier-1 hubs (thin evidence) |
| B. Validity & refutation | which estimator can we trust? | B1 direct bookings | B2 OTA bookings |
| C. DMA what-if (CATE) | what does pausing do in one market? | C1 Miami Beach (Tier 1) | C2 Asheville (Tier 3) |
| D. Bidding policy | which DMAs pause, and does the rule pay? | D1 evidence margin k=1 | D2 naive k=0 |

{flows}

## Presentation

`presentation/Paid_Search_Incrementality_End_to_End.pptx` (built by `presentation/build_deck.py`, `make deck`): documentation of the end-to-end uplift flow, the overall programme scorecard, a profile and metrics for all 12 models (3 macro estimators + 9 CATE learners), model-by-flow comparison matrices, flow-level and stage-level metrics for the 8 flows, and the Streamlit screenshots as scenario stories. Every number is read from `outputs/`.

## Interfaces

* **FastAPI** (`make serve`, Swagger at `/docs`): `GET /health`, `POST /predict/dma-cate`, `POST /simulate/holdout-policy`, `GET /metrics/summary`, plus `GET /flows` and `GET /flows/{{id}}` for the end-to-end flows.
* **Streamlit** (`make app`): Tab 1 geo-lift & cannibalization (DMA selector, counterfactual bands, last-click ROAS vs iROAS); Tab 2 leaderboard, interactive Qini and importance; Tab 3 policy simulator with 3 one-click presets (Miami Beach Tier 1 / Nashville Tier 2 / Asheville Tier 3); Tab 4 policy table and refutations; **Tab 5 end-to-end flows** (programme scorecard + 8 flows with stage-by-stage metrics).
* Dashboard and API call the same functions (`src/api/service.py`); the Playwright test asserts the UI equals the API for every preset.

{shot_md}

## Reproduce

```bash
make setup      # pip install -r requirements.txt && playwright install chromium
make train      # scripts/run_full_pipeline.py: panel -> macro -> cate -> econ -> plots -> flows   (~15 min on an RTX 3090)
make flows      # rebuild only the 8 end-to-end flows + scorecard (~2 min)
make test       # unit tests
make verify     # Playwright e2e (API + Streamlit screenshots) + artefact checks
make deck       # rebuild the 74-slide presentation in presentation/
make readme     # regenerate this file
```

Use the `causal-gpu` conda interpreter (`PY=` in the Makefile). Hardware is auto-detected and logged; without CUDA everything falls back to CPU.

## Limitations (read before using the numbers)

* **Synthetic data.** Real branded-search effects will not follow this DGP; the value of the repo is the method, the diagnostics and the honesty checks.
* **JAX/NumPyro runs on CPU** on native Windows (no JAX CUDA wheels), so BSTS is CPU. **RAPIDS cuDF/cuML are not available on native Windows**; DTW runs on PyTorch CUDA instead and the code falls back to pandas/scikit-learn when RAPIDS is absent. LightGBM uses CPU (GPU build not installed).
* **CausalPy is not installed**, so BSTS is implemented directly in NumPyro.
* **Only 50 DMAs / 10 treated**: confidence intervals from DMA-day rows would be far too narrow; macro estimators use DMA-level inference and CATE metrics are DMA-grouped out-of-fold.
* **Policy value is evaluated with ground truth**; in a real test use cross-fitted doubly robust policy value estimates and a follow-up holdout.
* Constant 2.4-night stays and a 2.5% branded-click conversion rate are simplifications.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("README.md written,", len(readme), "chars")


if __name__ == "__main__":
    main()
