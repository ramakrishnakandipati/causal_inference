"""Generate README.md from outputs/metrics/* (never hand-edit the README numbers)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

M = ROOT / "outputs" / "metrics"
ate = json.load(open(M / "ate_comparison.json"))
lb = pd.read_csv(M / "cate_leaderboard.csv")
dev = json.load(open(M / "device_report.json"))
ds = json.load(open(M / "data_summary.json"))
camp = json.load(open(M / "campaign_summary.json"))
ref = json.load(open(M / "refutation.json"))
fl = json.load(open(M / "flows.json"))


def flows_md():
    nl = "\n"
    o = ["Open the dashboard's **tab 5 · End-to-end flows**. Pipeline: Data → Propensity → Balance → ATE/ATT → Refutation → CATE models → Policy. "
         "Four flow types × two examples; each flow walks through 5–6 stages and every stage shows its own metrics "
         "(full-page shots in `outputs/screenshots/flows/`)." + nl]
    o.append("**Programme-level metrics (shared by all flows)**" + nl + nl + "| Metric | Value |" + nl + "|---|---|")
    o += [f"| {k} | {v} |" for k, v in fl["overall"].items()]
    for f in fl["flows"]:
        o.append(nl + f"### {f['id']} · {f['title']}" + nl + nl + f"_{f['story']}_" + nl)
        o.append("| Stage | Key metrics |" + nl + "|---|---|")
        for st_ in f["stages"]:
            km = "; ".join(f"{k}: {v}" for k, v in list(st_["metrics"].items())[:5])
            o.append(f"| {st_['name']} | {km} |")
        res = ", ".join(f"{k} = {v}" for k, v in f["summary"].items())
        o.append(nl + f"**Result:** {res}" + nl + nl + f"![{f['id']}](outputs/screenshots/flows/{f['id']}.png)")
    return nl.join(o)


def ate_table(o):
    v = ate["outcomes"][o]
    rows = ["| Estimator | ATE | 95% CI | ATE bias | ATT |", "|---|---|---|---|---|"]
    for k, s in v["estimators"].items():
        rows.append(f"| {k} | {s['ate']:.3f} | [{s['ate_lo']:.3f}, {s['ate_hi']:.3f}] | {s['ate_bias']:+.3f} | {s['att']:.3f} |")
    return f"True ATE = **{v['true_ate']:.3f}**, true ATT = {v['true_att']:.3f}\n\n" + "\n".join(rows)


def lb_table(o):
    d = lb[lb.outcome == o]
    rows = ["| Model | Qini | AUUC | PEHE | corr(τ̂,τ) | Policy value (IPW) | Policy value (true) | Fit s | ms/1k |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        f = lambda x, n=3: "–" if pd.isna(x) else f"{x:.{n}f}"
        rows.append(f"| {r.model} | {r.qini:.4f} | {r.auuc:.3f} | {r.pehe:.3f} | {r.corr_true:.3f} | {r.policy_value_ipw:.3f} | {r.policy_value_true:.3f} | {f(r.fit_seconds, 1)} | {f(r.latency_ms_per_1k, 1)} |")
    return "\n".join(rows)


b = ate["balance"]
dw = ref["ancillary"]["dowhy"]
txt = f"""# Pre-Arrival Ancillary Upsells & Direct-Channel Shift — Causal / Uplift Engine

End-to-end, GPU-accelerated causal inference for a hospitality case study: estimate the effect of a pre-arrival app-download & upsell campaign on
**ancillary spend ($)** and **direct re-booking within 180 days** from confounded campaign logs, then target the guests whose behaviour the
campaign actually changes. Data is synthetic with a known structural truth, so every estimator can be scored against the real τ(x).

> Generated from `outputs/metrics/*` by `scripts/build_readme.py`. Compute used: **{'CUDA – ' + str(dev['gpu_name']) if dev['cuda'] else 'CPU'}**
> (XGBoost `{dev['xgb_device']}`, PyTorch `{dev['torch_device']}`, CatBoost `{dev['catboost_task']}`, LightGBM `{dev['lightgbm_device']}`, RAPIDS cuML: {'yes' if dev['cuml'] else 'not available → scikit-learn kNN fallback'}).

## 1. Problem
OTA guests cost 15–20% commission and rarely buy upgrades. Guests who engage with a pre-arrival offer are more tech-savvy, affluent and book earlier,
so a naive treated-vs-untreated comparison is badly inflated. N = {ds['n']:,} stays ({ds['treated_share']:.0%} treated).
Confounders: channel, room tier, lead time, party size, device, historical spend, loyalty tier, LOS, resort.

## 2. Observational ATE / ATT (propensity-based)
Naive lift over-states the ancillary effect by **{ate['outcomes']['ancillary']['estimators']['Naive']['ate'] / ate['outcomes']['ancillary']['true_ate'] - 1:+.0%}**.

### Ancillary spend ($)
{ate_table('ancillary')}

### Direct re-booking (probability)
{ate_table('direct')}

Balance (|SMD|<0.10): before **{b['share_below_0p1_before']:.0%}** of covariates → after PSM **{b['share_below_0p1_psm']:.0%}**, after IPW **{b['share_below_0p1_ipw']:.0%}**
(max |SMD| {b['max_abs_smd_before']:.2f} → {b['max_abs_smd_ipw']:.3f}). Common support keeps {ate['common_support_kept']:.1%} of stays.

Honest note: PSM/IPW/SIPW retain a small positive bias (~+1 $ on ancillary spend); AIPW (doubly robust, cross-fitted) is closest to truth. CIs for IPW/PSM
use a bootstrap with the propensity fixed, so they are somewhat narrow.

![Love plot](outputs/plots/love_plot.png)
![Overlap](outputs/plots/propensity_overlap.png)
![ATE comparison](outputs/plots/ate_comparison.png)

**Refutation** (AIPW, ancillary): baseline {ref['ancillary']['baseline_aipw_ate']:.2f}; random common cause → {ref['ancillary']['builtin']['random_common_cause']['new_effect']:.2f};
placebo treatment → {ref['ancillary']['builtin']['placebo_treatment']['new_effect']:.2f}; data subset → {ref['ancillary']['builtin']['data_subset']['new_effect']:.2f}.
DoWhy (backdoor linear regression): {'available — ' + ', '.join(f"{k}: {v.get('new_effect', float('nan')):.3f}" for k, v in dw['refuters'].items()) if dw.get('available') else dw.get('reason')}.
Details in `outputs/metrics/refutation.json`. These refuters only probe the *observed*-confounder model; they cannot rule out unmeasured confounding
(here none exists by construction).

## 3. CATE leaderboard
Qini/AUUC are computed from observed outcomes with IPW weights (no truth). PEHE, correlation and "policy value (true)" use the synthetic ground truth.
Policy value = net $ per guest when contacting the top {camp['budget_frac']:.0%} (margin × incremental outcome − offer cost). The oracle row ranks by the true τ.

### Ancillary spend
{lb_table('ancillary')}

### Direct re-booking
{lb_table('direct')}

Qini differences between the top models are small relative to the sampling noise of IPW-based curves on {int(ds['n'] * 0.3):,} test stays; PEHE and
policy-value-true are the more reliable ranking. Uplift RF (KL) is only defined for the binary outcome. LightGBM ran on CPU (GPU build absent).

![Qini ancillary](outputs/plots/qini_curves_ancillary.png)
![Qini direct](outputs/plots/qini_curves_direct.png)

## 4. Campaign policy
Best models by Qini: ancillary **{camp['best_model']['ancillary']}**, direct **{camp['best_model']['direct']}**.
Contacting the top {camp['budget_frac']:.0%} yields true net value ${camp['true_value_targeted']:,.0f} vs oracle ${camp['true_value_oracle_top']:,.0f}
(estimated ${camp['est_value_targeted']:,.0f}); contacting everyone yields ${camp['true_value_all']:,.0f}. With these cost assumptions most guests have positive
expected value, so a 30% budget leaves money on the table — the optimiser is most valuable when offers are expensive or budgets are tight (see the scenarios in tab 4).

## 5. End-to-end flows (dashboard tab 5)
{flows_md()}

## 6. Screenshots
![Swagger](docs/screenshots/api_swagger_docs.png)
![CATE response](docs/screenshots/api_cate_response.png)
![Tab 1](docs/screenshots/streamlit_tab1_ate_balance.png)
![Tab 2](docs/screenshots/streamlit_tab2_cate_uplift.png)
![Tab 3](docs/screenshots/streamlit_tab3_guest_simulator.png)
![Tab 4](docs/screenshots/streamlit_tab4_campaign_optimizer.png)
![Tab 5 flows](outputs/screenshots/flows/A1.png)

All dashboard screenshots are also written to `outputs/screenshots/`.

## 7. Run it
Use an environment with CUDA PyTorch (tested: Python 3.12, RTX 3090). `pip install -r requirements.txt`, `playwright install chromium`.
```bash
python scripts/run_pipeline.py            # data -> propensity -> CATE -> evaluate -> score (~4 min on GPU); --fast for a smoke run
uvicorn src.api.main:app --port 8000      # /health /predict/cate /optimize/campaign /models/metrics  (docs at /docs)
streamlit run src/app/streamlit_app.py    # 5-tab dashboard
python scripts/capture_screenshots.py     # regenerates docs/screenshots + outputs/screenshots (+ flows)
python scripts/run_pipeline.py --stage flows   # rebuild outputs/metrics/flows.json only
python scripts/build_readme.py            # regenerates this README
python presentation/build_deck.py         # rebuilds presentation/Pre_Arrival_Upsell_Uplift_End_to_End.pptx (78 slides)
pytest tests -q
```
Set `CAUSAL_OUT=<dir>` to redirect outputs. Config (seeds, N, hyper-parameters, margins, budget) lives in `config/config.yaml`.

## 8. Layout
`src/data` DGP · `src/models` propensity/ATE, CATE learners, TARNet/DragonNet · `src/evaluation` metrics & plots · `src/serving.py` shared scoring/optimiser
(used by both API and UI) · `src/api` FastAPI · `src/flows.py` end-to-end flows · `src/app` Streamlit · `presentation` 78-slide deck + builder · `scripts` orchestration · `tests`.

## 9. Limitations
- Synthetic data; unconfoundedness holds by construction. Real logs need sensitivity analysis for unmeasured confounding.
- Selection of "best model" by Qini on one test split is noisy; no repeated-seed Monte-Carlo was run.
- DragonNet's targeted regularisation is applied on the probability scale for the binary outcome (an approximation).
- Predicted segments are action rules on τ̂ (Persuadable / Sleeping Dog / …); Always-Taker vs Lost Cause cannot be separated without a baseline-response model.
- cuML is not installable on native Windows; kNN matching used scikit-learn here.
"""
(ROOT / "README.md").write_text(txt, encoding="utf-8")
print("README.md written", len(txt))
