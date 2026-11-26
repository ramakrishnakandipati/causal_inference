"""End-to-end pipeline. Run with the causal-gpu interpreter:

    python scripts/run_full_pipeline.py                 # all stages
    python scripts/run_full_pipeline.py --stage macro   # one stage

Stages (in order): panel, macro, cate, econ, plots, flows, verify.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import feature_engineering as fe  # noqa: E402
from src.data.synthetic_geo_generator import generate_panel, load_panel  # noqa: E402
from src.economics import cannibalization_roi as econ  # noqa: E402
from src.models import macro_geolift as mg  # noqa: E402
from src.models.model_registry import ALGOS, train_and_evaluate  # noqa: E402
from src.utils.config import load_config, path  # noqa: E402
from src.utils.device import get_hardware  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("pipeline")
STAGES = ["panel", "macro", "cate", "econ", "plots", "flows", "verify"]


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if not isinstance(v, np.ndarray) or v.size <= 50}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def stage_panel():
    gp = generate_panel(save=True)
    pairs, _ = fe.match_markets(gp.panel, gp.dma)
    bal = fe.balance_table(gp.dma, pairs)
    m = path("metrics")
    pairs.to_csv(m / "matched_pairs.csv", index=False)
    bal.to_csv(m / "balance_table.csv", index=False)
    gp.dma.to_csv(m / "dma_table.csv", index=False)
    (m / "hardware.json").write_text(json.dumps(get_hardware().summary(), indent=2))
    (m / "experiment_meta.json").write_text(json.dumps(gp.meta, indent=2))
    log.info("panel %s rows, %d holdout DMAs; max |SMD| matched = %.2f", len(gp.panel), gp.meta["n_holdout"],
             bal.smd_matched.abs().max())


def stage_macro(mc_seeds: int = 10, mc_bsts_seeds: int = 5):
    gp = load_panel()
    pairs = pd.read_csv(path("metrics") / "matched_pairs.csv")
    tr = gp.dma[gp.dma["T"] == 1]
    truth = {"direct_bookings": tr.true_post_tau_direct.mean(), "ota_bookings": tr.true_post_tau_ota.mean()}
    res, series, refut = {}, [], []
    for oc in ("direct_bookings", "ota_bookings"):
        gd = mg.GeoData(gp.panel, gp.dma, oc)
        cut = gd.T - 28
        runs = {"MMT": mg.mmt(gd, pairs, cut), "SDID": mg.sdid(gd, cut), "BSTS": mg.bsts(gd, cut)}
        res[oc] = {"truth": float(truth[oc])}
        for k, r in runs.items():
            res[oc][k] = {kk: vv for kk, vv in r.items() if np.ndim(vv) == 0 or kk == "cumulative_ci"}
            res[oc][k]["covers_truth"] = bool(r["ci_low"] <= truth[oc] <= r["ci_high"])
            ser = pd.DataFrame({"day": np.arange(gd.T)[:len(r["observed"])], "observed": r["observed"],
                                "counterfactual": r["counterfactual"], "cf_low": r["cf_low"], "cf_high": r["cf_high"]})
            ser["method"], ser["outcome"] = k, oc
            series.append(ser)
        # truth counterfactual on treated DMAs for plotting
        refut += mg.placebo_in_time(gd, pairs, weeks=4, run_bsts=True)
        sp = mg.placebo_in_space(gd, {k: runs[k]["att"] for k in ("MMT", "SDID")}, n_draws=300)
        sp["outcome"] = oc
        sp.to_csv(path("metrics") / f"placebo_space_{oc}.csv", index=False)
        sd = sp.placebo_att.std(ddof=1)
        res[oc]["placebo_space_sdid"] = {"sd": float(sd), "p_value": float(((sp.placebo_att.abs() >= abs(runs["SDID"]["att"])).sum() + 1) / (len(sp) + 1))}
        log.info("%s truth %.2f | MMT %.2f SDID %.2f BSTS %.2f", oc, truth[oc], runs["MMT"]["att"], runs["SDID"]["att"], runs["BSTS"]["att"])
    m = path("metrics")
    pd.concat(series).to_csv(m / "macro_series.csv", index=False)
    pd.DataFrame(refut).to_csv(m / "placebo_in_time.csv", index=False)
    (m / "macro_results.json").write_text(json.dumps(_jsonable(res), indent=2))
    # Monte-Carlo bias/coverage across fresh draws of the whole experiment
    rows = []
    for sd_ in range(1000, 1000 + mc_seeds):
        g2 = generate_panel(seed=sd_)
        p2, _ = fe.match_markets(g2.panel, g2.dma)
        t2 = g2.dma[g2.dma["T"] == 1].true_post_tau_direct.mean()
        gd2 = mg.GeoData(g2.panel, g2.dma, "direct_bookings")
        c2 = gd2.T - 28
        for k, f in (("MMT", lambda: mg.mmt(gd2, p2, c2)), ("SDID", lambda: mg.sdid(gd2, c2, n_placebo=100)),
                     ("BSTS", (lambda: mg.bsts(gd2, c2, num_warmup=300, num_samples=400)) if sd_ < 1000 + mc_bsts_seeds else None)):
            if f is None:
                continue
            r = f()
            rows.append({"seed": sd_, "method": k, "truth": t2, "est": r["att"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    mc = pd.DataFrame(rows)
    mc["err"] = mc.est - mc.truth
    mc["covered"] = (mc.ci_low <= mc.truth) & (mc.truth <= mc.ci_high)
    mc.to_csv(m / "macro_montecarlo.csv", index=False)
    log.info("Monte-Carlo (direct bookings):\n%s", mc.groupby("method").agg(bias=("err", "mean"), rmse=("err", lambda e: np.sqrt((e ** 2).mean())),
                                                                       coverage=("covered", "mean"), n=("seed", "count")).round(2))


def stage_cate():
    gp = load_panel()
    for target in ("direct", "ota"):
        r = train_and_evaluate(gp.panel, target=target)
        log.info("CATE leaderboard (%s)\n%s", target, r["leaderboard"][["pehe", "pehe_norm", "rank_corr", "qini_ratio"]].round(3))


def _learner_matrix(target: str):
    """DMA x learner matrix of out-of-fold mean lifts (all 9 learners)."""
    m = path("metrics")
    oof = pd.read_csv(m / f"cate_oof_predictions_{target}.csv")
    cols = [f"tau_hat_{a}" for a in ALGOS]
    return oof.groupby("dma_id")[cols].mean(), oof.groupby("dma_id").tau_true.mean()


POLICY_K = 1.0  # default evidence margin: pause only if predicted gain > 1 x learner disagreement (fixed a priori)


def stage_econ():
    gp = load_panel()
    m = path("metrics")
    mac = json.loads((m / "macro_results.json").read_text())
    prog = {}
    for meth in ("MMT", "SDID", "BSTS"):
        d, o = mac["direct_bookings"][meth], mac["ota_bookings"][meth]
        prog[meth] = econ.programme_economics(gp.panel, gp.dma, d["att"], o["att"], (d["ci_low"], d["ci_high"]),
                                              (o["ci_low"], o["ci_high"]), label=meth)
    tr = gp.dma[gp.dma["T"] == 1]
    prog["Truth"] = econ.programme_economics(gp.panel, gp.dma, tr.true_post_tau_direct.mean(), tr.true_post_tau_ota.mean(), label="Truth")
    (m / "programme_economics.json").write_text(json.dumps(_jsonable(prog), indent=2))

    # ---- DMA policy: mean of the 9 out-of-fold learners, evidence margin = k x learner disagreement
    D, _ = _learner_matrix("direct")
    O, _ = _learner_matrix("ota")
    dma = gp.dma
    D, O = D.reindex(dma.dma_id).values, O.reindex(dma.dma_id).values
    G = np.stack([econ.gain_per_day(dma, D[:, j], O[:, j]) for j in range(D.shape[1])], 1)
    sd = G.std(1)
    ld, lo = D.mean(1), O.mean(1)
    pol = econ.policy_table(dma, ld, lo, margin_day=POLICY_K * sd)
    pol["learner_disagreement_sd_day"] = sd
    pol.to_csv(m / "dma_policy_recommendations.csv", index=False)
    val = econ.policy_value(pol)
    val.update({"rule": f"mean of 9 OOF learners; pause iff gain > {POLICY_K} x learner SD", "k": POLICY_K})
    # ---- alternative rules and a sweep over k, reported side by side (no cherry-picking)
    rows = []
    true28 = dma.true_post_tau_ncm.values * 28
    for name, pause in (("Always keep", np.zeros(len(dma), bool)), ("Always pause", np.ones(len(dma), bool)),
                        ("Oracle (true effects)", true28 > 0)):
        rows.append({"rule": name, "n_pause": int(pause.sum()), "value_usd": float(true28[pause].sum()),
                     "decision_accuracy": float((pause == (true28 > 0)).mean())})
    for k in (0, 0.5, 1, 1.5, 2, 3):
        pause = (G.mean(1) - k * sd) > 0
        rows.append({"rule": f"CATE mean - {k} x learner SD", "n_pause": int(pause.sum()), "value_usd": float(true28[pause].sum()),
                     "decision_accuracy": float((pause == (true28 > 0)).mean())})
    pd.DataFrame(rows).to_csv(m / "policy_rules.csv", index=False)
    by_tier = (pd.DataFrame({"tier": dma.tier, "true_28d": true28, "pred_28d": G.mean(1) * 28}).groupby("tier")
               .agg(true_value_of_pausing_usd=("true_28d", "sum"), predicted_usd=("pred_28d", "sum")))
    by_tier.to_csv(m / "policy_by_tier.csv")
    (m / "policy_value.json").write_text(json.dumps(_jsonable(val), indent=2))
    arche = {str(t): vars(econ.archetype_inputs(gp.dma, t)) for t in (1, 2, 3)}
    (m / "archetypes.json").write_text(json.dumps(arche, indent=2))
    log.info("policy value %s", val)
    log.info("policy rules:\n%s", pd.DataFrame(rows).round(0).to_string())
    log.info("by tier:\n%s", by_tier.round(0).to_string())


def stage_plots():
    from src.visualization import plots
    plots.make_all()


def stage_flows():
    from src import flows
    flows.build_all()


def stage_verify():
    from src.visualization import plots
    ok = plots.verify_outputs()
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES)
    a = ap.parse_args()
    cfg = load_config()
    get_hardware()
    for s in (STAGES if a.stage == "all" else [a.stage]):
        log.info("=== stage: %s ===", s)
        globals()[f"stage_{s}"]()


if __name__ == "__main__":
    main()
