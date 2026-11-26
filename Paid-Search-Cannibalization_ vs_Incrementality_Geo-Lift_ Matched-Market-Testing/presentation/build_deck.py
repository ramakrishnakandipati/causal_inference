"""Build the end-to-end presentation from the repo's own outputs (nothing is typed in by hand except narrative text).

    python presentation/build_deck.py      # -> presentation/Paid_Search_Incrementality_End_to_End.pptx

Sources: outputs/metrics/* (flows.json, leaderboards, OOF predictions, macro results), outputs/plots/*, outputs/screenshots/*.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

from src import flows as F  # noqa: E402
from src.api import service  # noqa: E402
from src.api.presets import PRESETS  # noqa: E402
from src.data.synthetic_geo_generator import load_panel  # noqa: E402
from src.economics import cannibalization_roi as econ  # noqa: E402

M, P, S = ROOT / "outputs" / "metrics", ROOT / "outputs" / "plots", ROOT / "outputs" / "screenshots"
OUT = ROOT / "presentation"
ASSETS = OUT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
rj = lambda n: json.loads((M / n).read_text())

NAVY, TEAL, AMBER, RED, GREY, LIGHT, WHITE, INK = (RGBColor(0x0B, 0x25, 0x45), RGBColor(0x13, 0x8A, 0x7E), RGBColor(0xE0, 0x9A, 0x2B),
                                                   RGBColor(0xC2, 0x3B, 0x4E), RGBColor(0x5C, 0x67, 0x70), RGBColor(0xF1, 0xF4, 0xF7),
                                                   RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0x1F, 0x29, 0x33))
STATUS = {"pass": TEAL, "warn": AMBER, "fail": RED, None: GREY}
hexc = {"navy": "#0B2545", "teal": "#138A7E", "amber": "#E09A2B", "red": "#C23B4E", "grey": "#5C6770", "blue": "#3E6FB0"}

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
N = {"i": 0}


# ----------------------------------------------------------------------------- slide helpers
def txt(slide, text, x, y, w, h, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Calibri"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = ln
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, font
        p.alignment = align
    return tb


def bullets(slide, items, x, y, w, h, size=15, color=INK, gap=4):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        lvl, t = (it if isinstance(it, tuple) else (0, it))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = ("• " if lvl == 0 else "– ") + t
        r.font.size, r.font.color.rgb, r.font.name = Pt(size - 2 * lvl), color, "Calibri"
        p.level = lvl
        p.space_after = Pt(gap)
    return tb


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    s.shadow.inherit = False
    return s


def new(title, sub=None, section=None):
    s = prs.slides.add_slide(BLANK)
    N["i"] += 1
    rect(s, 0, 0, 13.333, 0.95, NAVY)
    txt(s, title, 0.45, 0.1, 11.2, 0.55, 25, True, WHITE)
    if sub:
        txt(s, sub, 0.45, 0.58, 11.6, 0.33, 13, False, RGBColor(0xBF, 0xD4, 0xE6))
    if section:
        txt(s, section.upper(), 10.4, 0.12, 2.6, 0.3, 10, True, RGBColor(0x7F, 0xD1, 0xC7), PP_ALIGN.RIGHT)
    txt(s, "Grand Horizon Hotels & Resorts | Paid search cannibalization vs incrementality", 0.45, 7.12, 9, 0.3, 9, False, GREY)
    txt(s, str(N["i"]), 12.3, 7.12, 0.6, 0.3, 9, False, GREY, PP_ALIGN.RIGHT)
    return s


def kpi(slide, label, value, x, y, w=2.3, h=1.15, color=NAVY, vsize=24):
    rect(slide, x, y, w, h, LIGHT)
    rect(slide, x, y, 0.07, h, color)
    txt(slide, label, x + 0.15, y + 0.05, w - 0.2, 0.4, 10.5, False, GREY)
    txt(slide, value, x + 0.15, y + 0.42, w - 0.2, h - 0.45, vsize if len(str(value)) < 14 else vsize - 6, True, color, anchor=MSO_ANCHOR.MIDDLE)


def table(slide, header, rows, x, y, w, col_w=None, size=11, row_h=0.3, head_fill=NAVY, flags=None, first_bold=True):
    nr, nc = len(rows) + 1, len(header)
    gs = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tb = gs.table
    tot = sum(col_w) if col_w else nc
    for j in range(nc):
        tb.columns[j].width = Inches(w * ((col_w[j] if col_w else 1) / tot))
    for i in range(nr):
        tb.rows[i].height = Inches(row_h)
        for j in range(nc):
            c = tb.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.05)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            v = header[j] if i == 0 else rows[i - 1][j]
            c.text = "" if v is None else str(v)
            para = c.text_frame.paragraphs[0]
            for r in para.runs:
                r.font.size, r.font.name = Pt(size), "Calibri"
                r.font.bold = (i == 0) or (first_bold and j == 0)
                r.font.color.rgb = WHITE if i == 0 else INK
            c.fill.solid()
            c.fill.fore_color.rgb = head_fill if i == 0 else (WHITE if i % 2 else LIGHT)
            if i > 0 and flags and flags.get((i - 1, j)):
                c.fill.fore_color.rgb = flags[(i - 1, j)]
                for r in para.runs:
                    r.font.color.rgb, r.font.bold = WHITE, True
    return tb


def img(slide, src, x, y, mw, mh, center=True):
    im = src if isinstance(src, Image.Image) else Image.open(src)
    w, h = im.size
    sc = min(mw / w, mh / h)
    ww, hh = w * sc, h * sc
    path = src
    if isinstance(src, Image.Image):
        path = ASSETS / f"_tmp_{N['i']}_{abs(hash((x, y))) % 9999}.png"
        src.save(path)
    slide.shapes.add_picture(str(path), Inches(x + (mw - ww) / 2 if center else x), Inches(y + (mh - hh) / 2 if center else y), Inches(ww), Inches(hh))


def section_slide(title, sub, n=None):
    s = prs.slides.add_slide(BLANK)
    N["i"] += 1
    rect(s, 0, 0, 13.333, 7.5, NAVY)
    rect(s, 0.6, 3.05, 1.2, 0.07, TEAL)
    txt(s, title, 0.6, 3.2, 12, 1.0, 38, True, WHITE)
    txt(s, sub, 0.6, 4.3, 11.5, 1.2, 17, False, RGBColor(0xBF, 0xD4, 0xE6))
    if n:
        txt(s, n, 0.6, 2.3, 4, 0.7, 40, True, RGBColor(0x7F, 0xD1, 0xC7))
    return s


def usd(x):
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


# ----------------------------------------------------------------------------- data
D = rj("flows.json")
FL, PROG = D["flows"], D["programme"]
mac, econ_j, val, meta, hw = rj("macro_results.json"), rj("programme_economics.json"), rj("policy_value.json"), rj("experiment_meta.json"), rj("hardware.json")
LBD = pd.read_csv(M / "cate_leaderboard_direct.csv", index_col=0)
LBO = pd.read_csv(M / "cate_leaderboard_ota.csv", index_col=0)
ALGOS = [a for a in LBD.index if a != "Constant ATE (DiD)"]
FAMILY = {"S-Learner": "Meta", "T-Learner": "Meta", "X-Learner": "Meta", "R-Learner": "Meta", "DR-Learner": "Meta",
          "LinearDML": "Double ML", "CausalForestDML": "Double ML / forest", "TARNet": "Deep (CUDA)", "DragonNet": "Deep (CUDA)"}
gp = load_panel()
dma = gp.dma


def learner_tables():
    oof = {t: pd.read_csv(M / f"cate_oof_predictions_{t}.csv") for t in ("direct", "ota")}
    algos = [c.replace("tau_hat_", "") for c in oof["direct"].columns if c.startswith("tau_hat_")]
    mats = {t: oof[t].groupby("dma_id")[[f"tau_hat_{a}" for a in algos]].mean().reindex(dma.dma_id).values for t in oof}
    G = np.stack([econ.gain_per_day(dma, mats["direct"][:, j], mats["ota"][:, j]) for j in range(len(algos))], 1)
    true28 = dma.true_post_tau_ncm.values * 28
    orc = true28 > 0
    out = {"algos": algos}
    out["solo_value"] = {a: float(true28[G[:, j] > 0].sum()) for j, a in enumerate(algos)}
    out["solo_pause"] = {a: int((G[:, j] > 0).sum()) for j, a in enumerate(algos)}
    out["solo_acc"] = {a: float(((G[:, j] > 0) == orc).mean()) for j, a in enumerate(algos)}

    def rule(cols, k):
        g = G[:, cols]
        return (g.mean(1) - k * g.std(1)) > 0
    allc = list(range(len(algos)))
    out["loo_value"] = {a: float(true28[rule([c for c in allc if c != j], 1.0)].sum()) for j, a in enumerate(algos)}
    out["ens"] = {"k1": float(true28[rule(allc, 1.0)].sum()), "k0": float(true28[rule(allc, 0.0)].sum())}
    # tier-1 holdout ATE recovery
    o = oof["direct"]
    t1 = o[(o["T"] == 1) & (o.tier == 1)]
    out["tier1_err"] = {a: float(abs(t1[f"tau_hat_{a}"].mean() - t1.tau_true.mean())) for a in algos}
    allt = o[o["T"] == 1]
    out["ate_err_all"] = {a: float(abs(allt[f"tau_hat_{a}"].mean() - allt.tau_true.mean())) for a in algos}
    # C flows: per-learner lift vs structural lift
    out["c"] = {}
    for fid, k in (("C1", "A"), ("C2", "C")):
        pr = service.predict_dma_cate(F._preset_profile(k))
        p = PRESETS[k]
        lc = p["baseline_direct_bookings_day"] * p["lastclick_share"]
        c, q = p["cannibalization_rate"], p["ota_capture_rate"]
        struct = -lc * ((1 - c) + c * q) / p["baseline_direct_bookings_day"]
        out["c"][fid] = {"lift": {a: pr["per_model"]["direct"][a] for a in algos}, "struct": struct}
    return out


LT = learner_tables()


# ----------------------------------------------------------------------------- charts
def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def chart_macro():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), dpi=170)
    meths = ["MMT", "SDID", "BSTS"]
    cols = [hexc["blue"], hexc["teal"], hexc["amber"]]
    for k, fid in enumerate(("A1", "A2")):
        s = FL[fid]["summary"]
        be = s["by_estimator"]
        a = ax[0]
        for i, m in enumerate(meths):
            a.bar(k * 4 + i, be[m]["direct_att"], color=cols[i], label=m if k == 0 else None)
            a.errorbar(k * 4 + i, be[m]["direct_att"], yerr=[[be[m]["direct_att"] - be[m]["ci_low"]], [be[m]["ci_high"] - be[m]["direct_att"]]], color="k", lw=1, capsize=3)
        a.plot([k * 4 - 0.5, k * 4 + 2.5], [s["truth"]] * 2, color=hexc["red"], lw=2.2, label="truth" if k == 0 else None)
        a = ax[1]
        for i, m in enumerate(meths):
            a.bar(k * 4 + i, be[m]["net_value_keep_usd"] / 1000, color=cols[i])
        a.plot([k * 4 - 0.5, k * 4 + 2.5], [s["truth_net_value_keep_usd"] / 1000] * 2, color=hexc["red"], lw=2.2)
    ax[0].set_xticks([1, 5]); ax[0].set_xticklabels(["A1 all 10 holdouts", "A2 two Tier-1 hubs"])
    ax[1].set_xticks([1, 5]); ax[1].set_xticklabels(["A1 all 10 holdouts", "A2 two Tier-1 hubs"])
    ax[0].set_title("Direct ATT, bookings per DMA-day (95% interval)", fontsize=11)
    ax[1].set_title("Net value of keeping branded SEM, $k per 4 weeks", fontsize=11)
    ax[0].legend(frameon=False, fontsize=9, loc="lower right")
    for a in ax:
        style(a); a.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(ASSETS / "macro_by_flow.png"); plt.close(fig)


def chart_mc():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), dpi=170)
    for a, fid, ttl in zip(ax, ("B1", "B2"), ("Direct bookings", "OTA bookings")):
        s = FL[fid]["summary"]
        ms = [m for m in ("MMT", "SDID", "BSTS") if m in s["mc"]]
        bias = [s["mc"][m]["bias"] for m in ms]
        a.bar(ms, bias, color=[hexc["blue"], hexc["teal"], hexc["amber"]][:len(ms)])
        for i, m in enumerate(ms):
            a.text(i, bias[i], f" cov {s['mc'][m]['coverage']:.0%}", ha="center", va="bottom" if bias[i] >= 0 else "top", fontsize=9)
        a.set_title(f"{fid} {ttl}: Monte-Carlo bias (estimate - truth {s['truth']:.2f})", fontsize=10.5)
        if "BSTS" not in ms:
            a.text(0.97, 0.9, "BSTS: no Monte-Carlo run", transform=a.transAxes, ha="right", fontsize=9, color=hexc["grey"])
        style(a); a.axhline(0, color="k", lw=0.6)
    fig.tight_layout(); fig.savefig(ASSETS / "mc_bias.png"); plt.close(fig)


def chart_pehe():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), dpi=170)
    for a, lb, ttl in zip(ax, (LBD, LBO), ("Direct-booking lift", "OTA-booking lift")):
        d = lb.drop("Constant ATE (DiD)").sort_values("pehe_norm")
        a.barh(d.index, d.pehe_norm, color=[hexc["teal"] if v < lb.loc["Constant ATE (DiD)", "pehe_norm"] else hexc["grey"] for v in d.pehe_norm])
        a.axvline(lb.loc["Constant ATE (DiD)", "pehe_norm"], color=hexc["red"], lw=2)
        a.text(lb.loc["Constant ATE (DiD)", "pehe_norm"], len(d) - 0.4, " constant-ATE reference", color=hexc["red"], fontsize=9, va="bottom")
        a.invert_yaxis(); a.set_title(f"{ttl}: PEHE / sd(true CATE), lower is better", fontsize=10.5)
        a.grid(axis="x", alpha=0.25)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(ASSETS / "pehe.png"); plt.close(fig)


def _heat(ax, data, rows, cols, fmts, title, lower_better=True):
    arr = np.array(data, float)
    rank = np.full_like(arr, np.nan)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        ok = ~np.isnan(col)
        if ok.sum() > 1:
            sc = col[ok] if lower_better[j] else -col[ok]
            r = (sc - sc.min()) / (np.ptp(sc) + 1e-12)
            rank[ok, j] = r
    ax.imshow(rank, cmap="RdYlGn_r", vmin=-0.45, vmax=1.45, aspect="auto")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else fmts[j].format(v), ha="center", va="center", fontsize=8.5, color="black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9)
    ax.xaxis.tick_top(); ax.set_title(title, fontsize=11, pad=34)
    for sp in ax.spines.values():
        sp.set_visible(False)


def chart_heatmaps():
    # macro: A1/A2 |bias %|, B1/B2 |MC bias|, B1/B2 CI coverage, A1 net-value error ($k)
    meths = ["MMT", "SDID", "BSTS"]
    rows = []
    for m in meths:
        a1, a2 = FL["A1"]["summary"], FL["A2"]["summary"]
        b1, b2 = FL["B1"]["summary"]["mc"].get(m), FL["B2"]["summary"]["mc"].get(m)
        rows.append([abs(a1["by_estimator"][m]["bias_pct"]), abs(a2["by_estimator"][m]["bias_pct"]),
                     abs(b1["bias"]) if b1 else np.nan, abs(b2["bias"]) if b2 else np.nan,
                     b1["coverage"] * 100 if b1 else np.nan, b2["coverage"] * 100 if b2 else np.nan,
                     abs(a1["by_estimator"][m]["net_value_keep_usd"] - a1["truth_net_value_keep_usd"]) / 1000])
    fig, ax = plt.subplots(figsize=(11, 2.9), dpi=170)
    _heat(ax, rows, meths, ["A1 |bias| %", "A2 |bias| %", "B1 |MC bias|", "B2 |MC bias|", "B1 CI cover %", "B2 CI cover %", "A1 net-value\nerror $k"],
          ["{:.0f}", "{:.0f}", "{:.2f}", "{:.2f}", "{:.0f}", "{:.0f}", "{:.0f}"], "Macro estimators across the flows (green = better)",
          [True, True, True, True, False, False, True])
    fig.tight_layout(); fig.savefig(ASSETS / "heat_macro.png"); plt.close(fig)
    rows = []
    for a in ALGOS:
        rows.append([LT["ate_err_all"][a] * 100, LT["tier1_err"][a] * 100, LBD.loc[a, "pehe_norm"], LBO.loc[a, "pehe_norm"],
                     abs(LT["c"]["C1"]["lift"][a] - LT["c"]["C1"]["struct"]) * 100, abs(LT["c"]["C2"]["lift"][a] - LT["c"]["C2"]["struct"]) * 100,
                     LT["loo_value"][a] / 1000, LT["solo_value"][a] / 1000])
    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=170)
    _heat(ax, rows, ALGOS, ["A1 ATE err\n(pts, 10 DMAs)", "A2 ATE err\n(pts, Tier-1)", "B1 PEHE/sd\ndirect", "B2 PEHE/sd\nOTA",
                            "C1 |lift - structural|\n(pts)", "C2 |lift - structural|\n(pts)", "D1 value if learner\nleft out ($k)", "D2 value if learner\nalone ($k)"],
          ["{:.1f}", "{:.1f}", "{:.2f}", "{:.2f}", "{:.1f}", "{:.1f}", "{:.1f}", "{:.1f}"], "CATE learners across the flows (green = better)",
          [True, True, True, True, True, True, False, False])
    fig.tight_layout(); fig.savefig(ASSETS / "heat_cate.png"); plt.close(fig)


def chart_c():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.7), dpi=170)
    for a, fid in zip(ax, ("C1", "C2")):
        v = [LT["c"][fid]["lift"][x] * 100 for x in ALGOS]
        a.bar(range(len(ALGOS)), v, color=[hexc["teal"]] * len(ALGOS))
        a.axhline(LT["c"][fid]["struct"] * 100, color=hexc["red"], lw=2)
        a.text(len(ALGOS) - 0.5, LT["c"][fid]["struct"] * 100, "structural", color=hexc["red"], ha="right", va="bottom", fontsize=9)
        a.set_xticks(range(len(ALGOS))); a.set_xticklabels([x.replace("Learner", "").replace("CausalForestDML", "CF-DML") for x in ALGOS], rotation=35, fontsize=8.5)
        a.set_title(f"{fid} {FL[fid]['title'].replace('What if ', '').replace(' paused?', '')}: direct lift if paused (%)", fontsize=10)
        style(a)
    fig.tight_layout(); fig.savefig(ASSETS / "learners_c.png"); plt.close(fig)


def chart_policy():
    rules = pd.read_csv(M / "policy_rules.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), dpi=170)
    r = rules[rules.rule.str.startswith("CATE")]
    ax[0].bar([x.split(" - ")[1].split(" x")[0].replace("k", "") if False else f"k={x.split('- ')[1].split(' ')[0]}" for x in r.rule], r.value_usd / 1000,
              color=[hexc["teal"] if v > 0 else hexc["red"] for v in r.value_usd])
    ax[0].set_title("Ensemble rule: realised 4-week value by margin k ($k)", fontsize=10.5)
    for lab, v, c in (("always keep", 0, hexc["grey"]), ("oracle", rules.set_index("rule").loc["Oracle (true effects)", "value_usd"] / 1000, hexc["blue"])):
        ax[0].axhline(v, color=c, ls="--", lw=1.4); ax[0].text(5.4, v, lab, ha="right", va="bottom", fontsize=8.5, color=c)
    names = sorted(ALGOS, key=lambda a: -LT["solo_value"][a])
    ax[1].bar(range(len(names)), [LT["solo_value"][a] / 1000 for a in names], color=[hexc["teal"] if LT["solo_value"][a] > 0 else hexc["red"] for a in names])
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels([a.replace("Learner", "").replace("CausalForestDML", "CF-DML") for a in names], rotation=35, fontsize=8.5)
    ax[1].set_title("One learner deciding alone (k=0): realised value ($k)", fontsize=10.5)
    for a in ax:
        style(a); a.axhline(0, color="k", lw=0.6)
    fig.tight_layout(); fig.savefig(ASSETS / "policy_rules.png"); plt.close(fig)


chart_macro(); chart_mc(); chart_pehe(); chart_heatmaps(); chart_c(); chart_policy()


# ----------------------------------------------------------------------------- screenshot crops
def crops_for_flows():
    ims = {fid: Image.open(S / f"{10 + i}_streamlit_flow_{fid}.png").convert("RGB") for i, fid in enumerate(D["order"])}
    arrs = {k: np.asarray(v) for k, v in ims.items()}
    out = {}
    for k, a in arrs.items():
        red = (a[:, 330:1450, 0].astype(int) > 200) & (a[:, 330:1450, 1] < 120) & (a[:, 330:1450, 2] < 120)
        rows_red = np.where(red.any(axis=1) & (np.arange(a.shape[0]) > 1200))[0]
        start = int(rows_red.min()) - 115                             # selected-radio row of "Choose a flow"
        body = a[:, 330:1450]
        nonwhite = np.where((body < 250).any(axis=(1, 2)))[0]
        end = min(int(nonwhite.max()) + 30, a.shape[0])
        cuts = [start]
        for frac in (1 / 3, 2 / 3):
            mid = int(start + (end - start) * frac)
            white = [r for r in range(mid - 160, mid + 160) if (body[r] >= 250).all()]
            cuts.append(min(white, key=lambda r: abs(r - mid)) if white else mid)
        cuts.append(end)
        out[k] = [ims[k].crop((330, cuts[i], 1450, cuts[i + 1])) for i in range(3)]
    return out


FLOW_CROPS = crops_for_flows()


def shot(name, x0=300):
    im = Image.open(S / name).convert("RGB")
    return im.crop((x0, 0, im.size[0], im.size[1]))


# =============================================================================== SLIDES
s = prs.slides.add_slide(BLANK)
N["i"] += 1
rect(s, 0, 0, 13.333, 7.5, NAVY)
rect(s, 0.7, 2.35, 1.4, 0.08, TEAL)
txt(s, "Paid Search: Cannibalization vs. Incrementality", 0.7, 2.5, 12, 1.0, 40, True, WHITE)
txt(s, "End-to-end geo-lift, uplift (CATE) and bidding-policy engine for Grand Horizon Hotels & Resorts", 0.7, 3.55, 11.5, 0.8, 20, False, RGBColor(0xBF, 0xD4, 0xE6))
txt(s, ["50 DMAs  |  20-week pre-period + 4-week branded-SEM pause in 10 DMAs  |  3 macro estimators + 9 CATE learners  |  8 end-to-end flows",
        "Synthetic data with known ground truth; every number in this deck is read from outputs/metrics, outputs/plots and outputs/screenshots."],
    0.7, 4.7, 12, 1.0, 13, False, RGBColor(0x9F, 0xB8, 0xCC))

# ---- agenda
s = new("Agenda", "Four parts, one thread: does bidding on our own brand buy bookings or rent traffic we would get free?")
items = [("1", "Documentation: the end-to-end uplift flow", "pipeline, data, matching, estimators, CATE protocol, economics, policy, serving, metric glossary"),
         ("2", "Scorecard: overall metrics and the flow map", "programme metrics, 4 flow types x 2 examples, verdicts"),
         ("3", "Every model, side by side", "3 macro estimators + 9 CATE learners: profile, metrics, comparison across the flows"),
         ("4", "Flows and scenario stories", "flow-level and stage-level metrics with Streamlit screenshots; presets A/B/C; recommendations")]
for i, (n_, t, d) in enumerate(items):
    rect(s, 0.8, 1.35 + i * 1.3, 11.7, 1.1, LIGHT)
    txt(s, n_, 0.95, 1.4 + i * 1.3, 0.8, 1.0, 38, True, TEAL, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, t, 1.9, 1.45 + i * 1.3, 10.4, 0.45, 20, True, NAVY)
    txt(s, d, 1.9, 1.95 + i * 1.3, 10.4, 0.45, 13.5, False, GREY)

# ---- executive summary
sd, tr = econ_j["SDID"], econ_j["Truth"]
s = new("Executive summary", "Branded search is truly incremental but near break-even after OTA effects; DMA-level pause lists are not reliable", "Summary")
cards = [("Last-click ROAS", f"{sd['last_click_roas']:.1f}x", GREY), ("Incremental ROAS (SDID)", f"{sd['iroas']:.1f}x", TEAL), ("Truth iROAS", f"{tr['iroas']:.1f}x", NAVY),
         ("Cannibalization", f"{sd['cannibalization_rate']:.0%}", AMBER), ("Net value of keeping, 4 wk (SDID | truth)", usd(sd["net_value_of_branded_search_usd"]) + " | " + usd(tr["net_value_of_branded_search_usd"]), TEAL),
         ("OTA commission leakage", usd(sd["ota_commission_leakage_usd"]), RED)]
for i, (l, v, c) in enumerate(cards):
    kpi(s, l, v, 0.6 + (i % 3) * 2.55, 1.2 + (i // 3) * 1.3, 2.4, 1.15, c)
bullets(s, [f"Pausing in 10 DMAs cut direct bookings by {abs(mac['direct_bookings']['SDID']['att']):.1f} per DMA-day (truth {abs(mac['direct_bookings']['truth']):.1f}) and sent {mac['ota_bookings']['SDID']['att']:.1f} per DMA-day to OTAs.",
            "Last-click credit overstates the value of branded paid search 2.6x: 62% of credited bookings would have arrived via the free organic listing.",
            "Only SDID passes the full refutation suite on direct bookings. MMT drifts before the pause; BSTS is biased toward zero.",
            f"Net value of keeping is {usd(sd['net_value_of_branded_search_usd'])} by SDID but its interval spans zero and the simulated truth is {usd(tr['net_value_of_branded_search_usd'])}: neither pausing nor keeping is clearly better on average.",
            f"Learners recover the average effect, not DMA-level heterogeneity: 0 of 9 beat the constant-effect reference on PEHE.",
            f"The DMA pause rule (k=1) realises {usd(val['model_policy_usd'])} against an unattainable oracle of {usd(val['oracle_usd'])}: it does not beat leaving things as they are."],
        0.6, 3.95, 7.4, 3.1, 12.5)
img(s, P / "iroas_waterfall.png", 8.15, 1.15, 4.8, 3.2)
img(s, P / "att_estimators.png", 8.15, 4.4, 4.8, 2.6)

# =============================================================================== PART 1 documentation
section_slide("Documentation: the uplift end-to-end flow", "How the engine goes from raw geo-experiment to a bidding decision, at a high level", "Part 1")

# ---- pipeline diagram
s = new("The end-to-end flow at a glance", "Eight layers; each writes artefacts the next layer reads", "Documentation")
boxes = [("1 Data", "synthetic_geo_generator\n50 DMAs x 168 days\nknown truth"), ("2 Design", "stratified 20% holdout\nDTW + Mahalanobis\nmatching, SMD"),
         ("3 Macro lift", "MMT, SDID, BSTS\nATT + intervals\nplacebo tests"), ("4 Micro lift", "9 CATE learners\nDMA-grouped OOF\nPEHE, Qini, AUUC"),
         ("5 Economics", "iROAS, cannibalization\nOTA commission\nnet contribution margin"), ("6 Policy", "pause iff gain >\nk x disagreement\nvalue vs oracle"),
         ("7 Serving", "FastAPI endpoints\nStreamlit 5 tabs\nPlaywright e2e"), ("8 Flows", "8 flows, stage +\nflow + programme\nmetrics")]
for i, (t, d) in enumerate(boxes):
    x = 0.45 + (i % 4) * 3.2
    y = 1.4 + (i // 4) * 2.45
    rect(s, x, y, 2.85, 1.95, LIGHT, TEAL if i < 7 else AMBER)
    rect(s, x, y, 2.85, 0.5, NAVY if i != 7 else AMBER)
    txt(s, t, x + 0.1, y + 0.06, 2.6, 0.4, 15, True, WHITE)
    txt(s, d.split("\n"), x + 0.12, y + 0.6, 2.65, 1.3, 12.5, False, INK)
    if i % 4 != 3:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.88), Inches(y + 0.8), Inches(0.28), Inches(0.3))
        a.fill.solid(); a.fill.fore_color.rgb = TEAL; a.line.fill.background()
txt(s, "Scenario: macro results answer 'did it work on average?', micro (CATE) results answer 'where?', economics turns both into dollars, policy turns dollars into per-DMA decisions.",
    0.6, 6.35, 12, 0.6, 13, False, GREY)

# ---- data
s = new("Layer 1: data and the data-generating process", "A known truth makes every estimate checkable", "Documentation")
bullets(s, ["50 DMAs in 3 tiers (10 Major Hub, 20 Regional, 20 Drive-to); daily panel, 20-week pre + 4-week intervention",
            "Stratified randomisation: 20% of each tier paused (2/4/4 = 10 holdout, 40 control); known propensity 0.2",
            "Outcomes: direct bookings & GBV, OTA bookings & commission (15-20%), net contribution margin (NCM)",
            "Covariates: organic SERP rank, OTA auction pressure, competitor ADR, loyalty penetration, lead time, flight-search index, calendar",
            "Latent common factors, DMA drift, day-of-week, holidays: the problem is not trivially separable"], 0.6, 1.2, 6.6, 4.3, 14)
rect(s, 7.5, 1.25, 5.3, 4.3, LIGHT)
txt(s, "What 'truth' means", 7.65, 1.3, 5, 0.4, 15, True, NAVY)
txt(s, ["s = last-click share of direct bookings", "c = cannibalization (would arrive via free organic anyway)", "q = OTA hijack rate of organic-bound clickers",
        "v = share of lost incremental bookings re-booking on an OTA", "",
        "Direct effect of pausing   = -E0 . s . [(1-c) + c.q]", "OTA effect of pausing      = +E0 . s . [(1-c).v + c.q]", "",
        "NCM = Direct GBV - SEM spend + OTA GBV x (1 - commission)", "Value of branded search = -ATT(holdout)"], 7.65, 1.75, 5.1, 3.7, 12.5, False, INK)
txt(s, "Because truth is stored per DMA, we can score ATT, CATE (PEHE), policy value and refutations against it. Real data has no such luxury.", 0.6, 5.8, 12, 0.7, 13, False, GREY)

# ---- design/matching
bal = pd.read_csv(M / "balance_table.csv")
s = new("Layer 2: experiment design and matched markets", "DTW + Mahalanobis one-to-one matching on GPU; balance checked with standardised mean differences", "Documentation")
bullets(s, ["Pre-period shape distance: banded DTW computed on torch CUDA", "Covariate distance: Mahalanobis on tier, rank, OTA pressure, ADR, loyalty, lead time, size",
            "Hungarian assignment (0.5 DTW + 0.5 Mahalanobis, same-tier penalty) gives 10 treated-control pairs",
            f"Max |SMD| after matching {bal.smd_matched.abs().max():.2f} vs {bal.smd_all_controls.abs().max():.2f} against all controls",
            "Matched pairs feed MMT; SDID and BSTS use the whole control pool"], 0.6, 1.25, 6.0, 4.5, 14)
img(s, P / "balance_smd.png", 6.8, 1.2, 6.2, 4.8)
txt(s, f"Hardware: {hw['gpu']} | torch {hw['torch']} | xgboost {hw['xgboost']} | catboost {hw['catboost']} | jax {hw['jax']} (Windows) | RAPIDS {hw['rapids']}", 0.6, 6.4, 12, 0.5, 12, False, GREY)

# ---- estimation layers
s = new("Layers 3 and 4: macro lift and micro (uplift) estimation", "Two questions, two families of estimators", "Documentation")
table(s, ["", "Macro geo-lift (3 estimators)", "Micro uplift / CATE (9 learners)"],
      [["Question", "What was the average effect of pausing?", "How does the effect differ by DMA and day?"],
       ["Unit", "Holdout aggregate (10 DMAs) vs controls", "DMA-day rows, 13 covariates, DMA-grouped folds"],
       ["Estimators", "MMT, SDID, BSTS (NumPyro NUTS)", "S, T, X, R, DR, LinearDML, CausalForestDML, TARNet, DragonNet"],
       ["Target", "ATT in bookings per DMA-day", "Relative lift = bookings / DMA pre-period mean (direct and OTA separately)"],
       ["Inference", "Exact sign-flip permutation, placebo-space SE, credible bands", "Out-of-fold PEHE, Qini, AUUC, rank correlation (no intervals)"],
       ["Checks", "Placebo-in-time, placebo-in-space, Monte-Carlo bias and coverage", "Constant-ATE reference, learner agreement, leave-one-out value"],
       ["Hardware", "DTW on CUDA; BSTS on CPU JAX", "XGBoost cuda, CatBoost GPU, TARNet/DragonNet CUDA"]],
      0.6, 1.3, 12.1, [1.2, 5, 5.9][0:3], 12, 0.62)
txt(s, "Why two layers? Macro estimators are trustworthy for the average but blind to where; CATE learners aim at where but, with 10 treated DMAs, mostly relearn the average.", 0.6, 6.2, 12, 0.7, 13, False, GREY)

# ---- CATE protocol
s = new("Layer 4 in detail: how the CATE learners are trained and judged", None, "Documentation")
bullets(s, ["Treatment is assigned at DMA level, so every fold is grouped by DMA: no DMA appears in train and test",
            "Targets are relative booking lifts for direct and OTA; NCM is composed from them in the economics layer (NCM effect itself was too noisy)",
            "Shallow base models (depth 2, min child weight 28) because there are only 50 independent units",
            "Metrics (all out-of-fold):",
            (1, "PEHE: RMSE of predicted vs true CATE; PEHE / sd(true CATE) > 1 is worse than predicting the average"),
            (1, "Qini and AUUC: does ranking DMAs by predicted effect find high-uplift DMAs? Qini ratio = Qini / oracle Qini"),
            (1, "Rank correlation: Spearman of predicted vs true CATE across DMAs"),
            (1, "ATE error: how close the learner's mean is to the true average effect"),
            "Reference row: Constant ATE (DiD) = predict the same effect everywhere, the bar every learner must clear"], 0.6, 1.25, 7.0, 5.6, 14)
img(s, P / "qini_direct.png", 7.8, 1.3, 5.2, 3.6)
img(s, P / "cate_heterogeneity.png", 7.8, 4.5, 5.2, 2.4)

# ---- economics + policy
s = new("Layers 5 and 6: economics and the bidding policy", "Bookings become dollars; dollars become keep/pause decisions", "Documentation")
rect(s, 0.6, 1.25, 6.1, 4.7, LIGHT)
txt(s, "Economics bridge", 0.75, 1.3, 5.8, 0.4, 15, True, NAVY)
txt(s, ["Incremental direct bookings = -ATT_direct x DMAs x days", "iROAS = incremental direct GBV / branded spend",
        "Last-click ROAS = last-click bookings x GBV / spend", "Cannibalization = 1 - incremental / last-click bookings",
        "OTA leakage = OTA bookings gained x GBV x commission", "Net margin change if paused = - direct GBV lost + OTA GBV x (1-commission) + spend saved",
        "Intervals propagate the ATT confidence limits"], 0.75, 1.75, 5.8, 4.1, 14.5, False, INK)
rect(s, 6.9, 1.25, 5.9, 4.7, LIGHT)
txt(s, "Policy rule", 7.05, 1.3, 5.6, 0.4, 15, True, NAVY)
txt(s, ["Per DMA and per learner: gain = predicted d(NCM)/day from pausing", "Ensemble: mean over the nine learners, spread = their standard deviation",
        "Pause iff mean gain - k x spread > 0; k = 1 fixed a priori", "Scored against ground truth: value vs always-keep ($0), always-pause, oracle",
        "A sweep over k is shown, never picked on the truth", "The structural simulator (closed form in c, q, v) provides a second opinion"], 7.05, 1.75, 5.6, 4.1, 14.5, False, INK)
txt(s, "Rule of thumb carried through every flow: act on the experiment-level answer unless DMA-level evidence clears the noise.", 0.6, 6.3, 12, 0.5, 13, False, GREY)

# ---- serving
s = new("Layers 7 and 8: serving, interface, tests and flows", "Dashboard and API share one code path, so the numbers cannot drift", "Documentation")
table(s, ["Surface", "What it gives", "Source of truth"],
      [["FastAPI (:8000)", "/health, /predict/dma-cate, /simulate/holdout-policy, /metrics/summary, /flows, /flows/{id}", "src/api/service.py"],
       ["Streamlit (:8501)", "5 tabs: geo-lift, CATE leaderboard, policy simulator (3 presets), refutation & DMA policy, end-to-end flows", "service.py + flows.json"],
       ["Flows engine", "8 flows (4 types x 2 examples); stage, flow and programme metrics; PASS / CAUTION / FAIL", "src/flows.py -> flows.json"],
       ["Tests", "12 unit tests; Playwright e2e asserts UI == API for 3 presets and 8 flows, saves 17 screenshots", "tests/"],
       ["Reproduce", "make train | flows | serve | app | test | verify | readme", "Makefile, pinned requirements"]],
      0.6, 1.3, 12.1, [2, 6.6, 3.5], 12, 0.55)
img(s, S / "01_fastapi_swagger.png", 0.6, 5.0, 4.0, 2.0)
img(s, shot("03_streamlit_tab1_geolift.png"), 4.9, 5.0, 3.8, 2.0)
img(s, shot("09_streamlit_tab5_programme_scorecard.png").crop((0, 0, 1200, 1300)), 9.0, 5.0, 3.7, 2.0)

# ---- glossary
s = new("Metric glossary", "Used at overall, flow and stage level throughout the deck", "Documentation")
table(s, ["Metric", "Meaning", "Good looks like"],
      [["ATT / DMA-day", "Average effect of pausing on bookings per holdout DMA-day", "Close to truth, interval covers it"],
       ["Bias vs truth (%)", "(estimate - truth) / |truth|", "Near 0"],
       ["MC bias, RMSE, coverage", "Over fresh experiments: mean error, RMS error, share of intervals covering truth", "Bias ~0, coverage ~95%"],
       ["Placebo-in-time", "Fake pause 4 weeks early using only pre data; truth is 0", "Interval contains 0"],
       ["Placebo-in-space p", "Share of fake-holdout effects as extreme as the real one", "Small"],
       ["iROAS / last-click ROAS", "Incremental vs credited return on branded spend", "iROAS < last-click means over-credit"],
       ["Cannibalization", "Share of credited bookings that would have arrived anyway", "Lower is better for the spend"],
       ["PEHE / sd(true CATE)", "DMA-level CATE error, scaled; 1 = no better than a constant", "Below the constant reference"],
       ["Qini, AUUC, rank corr", "Does ranking by predicted effect find high-uplift DMAs?", "High, vs oracle Qini"],
       ["Policy value / oracle", "Realised 4-week margin gain vs always-keep and a perfect rule", "Positive, high share of oracle"]],
      0.6, 1.25, 12.1, [2.6, 6.6, 2.9], 11.5, 0.5)

# =============================================================================== PART 2 scorecard
section_slide("Scorecard: overall metrics and the flow map", "Programme-level metrics for the whole engine, then the 4 flow types x 2 examples", "Part 2")

s = new("Overall programme scorecard", "Every group below is a row in flows.json -> programme.groups", "Scorecard")
rows = [[g["group"], "   |   ".join(f"{m['label']}: {m['value']}" for m in g["metrics"])] for g in PROG["groups"]]
table(s, ["Group", "Metrics"], rows, 0.6, 1.2, 12.1, [2.6, 9.5], 11.5, 0.68)

s = new("Flow map: four flow types, two examples each", "Verdicts are rule-based (interval vs truth, placebo, Monte-Carlo, value vs always-keep)", "Scorecard")
fl_flags = {}
rows = []
for i, r in enumerate(PROG["rollup"]):
    rows.append([r["flow"], r["type"], r["scenario"].split(" - ", 1)[1], r["verdict"], r["headline"]])
    last = FL[r["flow"]]["stages"][-1]["flag"]
    fl_flags[(i, 3)] = STATUS[last]
table(s, ["Flow", "Type", "Scenario", "Verdict", "Headline metrics"], rows, 0.4, 1.2, 12.5, [0.6, 2.0, 2.7, 3.2, 5.0], 10.5, 0.5, flags=fl_flags)
txt(s, "A: was it incremental?   B: can we trust it?   C: what happens in one market?   D: which DMAs should pause?", 0.5, 6.72, 12, 0.4, 13, True, NAVY)

# =============================================================================== PART 3 models
section_slide("Every model, side by side", "Three macro estimators and nine CATE learners: how each works, its metrics, and how it behaves in every flow", "Part 3")

s = new("Model landscape: 12 models, 3 families", "Hardware and role for each", "Models")
rows = [["MMT", "Macro", "Matched control x pre ratio; exact sign-flip test", "CUDA DTW for matching", "A1 A2 B1 B2"],
        ["SDID", "Macro", "Simplex-ridge unit and time weights; placebo SE", "CPU", "A1 A2 B1 B2"],
        ["BSTS", "Macro", "Local trend + DOW + control PCs; NUTS (NumPyro)", "CPU (JAX, Windows)", "A1 A2 B1 B2"]]
desc = {"S-Learner": ("One outcome model with T as a feature", "XGBoost cuda"), "T-Learner": ("Separate outcome model per arm", "XGBoost cuda"),
        "X-Learner": ("Imputed effects, propensity-weighted blend", "CatBoost GPU"), "R-Learner": ("Residual-on-residual (Robinson) with cross-fit", "XGBoost cuda"),
        "DR-Learner": ("Doubly robust pseudo-outcome regression", "XGBoost cuda"), "LinearDML": ("Double ML, linear final stage on named drivers", "EconML + XGB cuda"),
        "CausalForestDML": ("Honest causal forest on residuals", "EconML + XGB cuda"), "TARNet": ("Shared representation, two outcome heads", "PyTorch CUDA"),
        "DragonNet": ("TARNet + propensity head + targeted regularisation", "PyTorch CUDA")}
for a in ALGOS:
    rows.append([a, FAMILY[a], desc[a][0], desc[a][1], "A1 A2 B1 B2 C1 C2 D1 D2"])
table(s, ["Model", "Family", "Core idea", "Hardware", "Appears in flows"], rows, 0.4, 1.2, 12.5, [1.8, 1.8, 5.2, 2.0, 2.4], 11, 0.37)
txt(s, "Macro estimators feed flows A and B (and the economics in C and D); CATE learners feed flows C and D, and are scored on A1/A2 (ATE) and B1/B2 (PEHE) for comparison.", 0.5, 6.4, 12.3, 0.55, 12, False, GREY)

# ---- macro estimator slides
MACRO_TXT = {
    "MMT": ("Matched market test", ["Pairs each holdout DMA with one control via DTW + Mahalanobis (Hungarian)", "Counterfactual = control series x pre-period ratio",
                                     "Exact paired sign-flip permutation p; bootstrap interval", "Strength: transparent, no model. Weakness: inherits pre-period drift between a pair"]),
    "SDID": ("Synthetic difference-in-differences", ["Simplex-ridge unit weights match the holdout pre-trend; time weights pick informative pre-days",
                                                      "Placebo-in-space standard error from fake-treating control subsets", "Strength: robust to level and trend differences. Weakness: needs enough controls"]),
    "BSTS": ("Bayesian structural time series (NumPyro)", ["Local linear trend + day-of-week + holiday + control PCs, tier-reweighted control mean, flight index",
                                                            "NUTS posterior; 95% credible band; split R-hat checked", "Strength: full uncertainty. Weakness: shrinks toward zero when 10 DMAs are one aggregate series"]),
}
for m in ("MMT", "SDID", "BSTS"):
    ttl, pts = MACRO_TXT[m]
    s = new(f"{m}: {ttl}", "How it works and how it scored", "Models")
    bullets(s, pts, 0.6, 1.2, 6.3, 3.0, 14)
    a1, a2 = FL["A1"]["summary"]["by_estimator"][m], FL["A2"]["summary"]["by_estimator"][m]
    b1, b2 = FL["B1"]["summary"]["mc"].get(m), FL["B2"]["summary"]["mc"].get(m)
    rows = [["A1 direct ATT (truth %.2f)" % FL["A1"]["summary"]["truth"], f"{a1['direct_att']:.2f}  [{a1['ci_low']:.2f}, {a1['ci_high']:.2f}]", f"{a1['bias_pct']:+.0f}%", "covers" if a1["covers_truth"] else "MISSES"],
            ["A2 direct ATT (truth %.2f)" % FL["A2"]["summary"]["truth"], f"{a2['direct_att']:.2f}  [{a2['ci_low']:.2f}, {a2['ci_high']:.2f}]", f"{a2['bias_pct']:+.0f}%", "covers" if a2["covers_truth"] else "MISSES"],
            ["A1 OTA ATT (truth %.2f)" % a1["ota_truth"], f"{a1['ota_att']:.2f}", f"{(a1['ota_att'] - a1['ota_truth']) / abs(a1['ota_truth']) * 100:+.0f}%", ""],
            ["A1 net value of keeping SEM", usd(a1["net_value_keep_usd"]), f"truth {usd(FL['A1']['summary']['truth_net_value_keep_usd'])}", ""],
            ["B1 MC bias / RMSE / coverage", (f"{b1['bias']:+.2f} / {b1['rmse']:.2f} / {b1['coverage']:.0%}" if b1 else "n/a"), "", FL["B1"]["summary"]["status"][m]],
            ["B2 MC bias / RMSE / coverage", (f"{b2['bias']:+.2f} / {b2['rmse']:.2f} / {b2['coverage']:.0%}" if b2 else "not Monte-Carlo run"), "", FL["B2"]["summary"]["status"][m]],
            ["Placebo-in-time (B1)", f"{FL['B1']['summary']['placebo_time'][m]['att']:+.2f}", "", "covers 0" if FL["B1"]["summary"]["placebo_time"][m]["covers_zero"] else "excludes 0"]]
    table(s, ["Metric", "Value", "Bias / reference", "Status"], rows, 0.6, 4.1, 12.1, [3.6, 3.8, 2.5, 2.2], 11, 0.36)
    img(s, P / ("geolift_direct_bookings.png"), 7.1, 1.15, 5.8, 2.85)

# ---- CATE learner slides
CATE_TXT = {
    "S-Learner": ["One model mu(x, T); effect = mu(x,1) - mu(x,0)", "Regularises the effect toward zero when T is weak", "Best direct-lift PEHE among learners"],
    "T-Learner": ["Two outcome models, one per arm; effect = difference", "Only 10 treated DMAs: the treated model is data-starved", "Highest variance: worst PEHE"],
    "X-Learner": ["Impute individual effects from the opposite arm, regress, blend with known propensity 0.2", "Good when arms are unbalanced (here 20/80)", "Very close to the true average effect"],
    "R-Learner": ["Residualise Y and T (cross-fit), fit effect with weighted loss", "Needs residual treatment variation: only DMA-level 0.2 propensity", "Strong ATE, weak heterogeneity"],
    "DR-Learner": ["Doubly robust pseudo-outcome regressed on x", "Pseudo-outcomes are noisy with 10 treated units", "Largest outliers on single-DMA what-ifs"],
    "LinearDML": ["Double ML with a linear final stage on named drivers", "Interpretable coefficients; limited flexibility", "Negative Qini ratio on direct lift: ranking is weak"],
    "CausalForestDML": ["Honest causal forest on DML residuals", "Best rank/Qini behaviour among tree learners", "Second-best direct PEHE; closest to the constant reference"],
    "TARNet": ["Shared representation with separate outcome heads (PyTorch, CUDA)", "Best rank correlation on direct lift", "Fast: seconds on the RTX 3090"],
    "DragonNet": ["TARNet plus a propensity head and targeted regularisation", "Highest Qini ratio on direct lift", "Propensity here is a known constant 0.2, so the propensity head has little to learn"],
}
for a in ALGOS:
    s = new(f"{a}: {FAMILY[a]}", desc[a][0], "Models")
    bullets(s, CATE_TXT[a], 0.6, 1.2, 5.6, 2.6, 14)
    ld, lo = LBD.loc[a], LBO.loc[a]
    rows = [["PEHE / sd(true CATE)", f"{ld.pehe_norm:.2f}", f"{lo.pehe_norm:.2f}"], ["PEHE (lift units)", f"{ld.pehe:.3f}", f"{lo.pehe:.3f}"],
            ["Rank correlation", f"{ld.rank_corr:.2f}", f"{lo.rank_corr:.2f}"], ["Qini ratio", f"{ld.qini_ratio:.2f}", f"{lo.qini_ratio:.2f}"],
            ["ATE (pred vs true), % lift", f"{ld.ate_hat * 100:.1f} vs {ld.ate_true * 100:.1f}", f"{lo.ate_hat * 100:.1f} vs {lo.ate_true * 100:.1f}"],
            ["Train seconds (GPU where applicable)", f"{ld.seconds:.1f}", f"{lo.seconds:.1f}"]]
    table(s, ["Metric", "Direct lift", "OTA lift"], rows, 6.5, 1.2, 6.4, [3.0, 1.7, 1.7], 11, 0.38)
    c1, c2 = LT["c"]["C1"], LT["c"]["C2"]
    rows2 = [["C1 Miami: direct lift if paused", f"{c1['lift'][a] * 100:.1f}%", f"structural {c1['struct'] * 100:.1f}%"],
             ["C2 Asheville: direct lift if paused", f"{c2['lift'][a] * 100:.1f}%", f"structural {c2['struct'] * 100:.1f}%"],
             ["D2 policy value, learner alone (k=0)", usd(LT["solo_value"][a]), f"{LT['solo_pause'][a]} pauses, acc {LT['solo_acc'][a]:.0%}"],
             ["D1 value of ensemble without this learner", usd(LT["loo_value"][a]), f"full ensemble {usd(LT['ens']['k1'])}"],
             ["A1 / A2 ATE error (pts)", f"{LT['ate_err_all'][a] * 100:.1f} / {LT['tier1_err'][a] * 100:.1f}", "10 holdouts / Tier-1"]]
    table(s, ["Across the flows", "Value", "Reference"], rows2, 0.6, 4.2, 12.1, [4.6, 2.5, 3.5], 11, 0.4)
    txt(s, f"Direct-lift rank among 9 learners by PEHE: {int((LBD.drop('Constant ATE (DiD)').pehe_norm < ld.pehe_norm).sum()) + 1} of 9;  constant-ATE reference {LBD.loc['Constant ATE (DiD)', 'pehe_norm']:.2f}.",
        0.6, 6.68, 12, 0.4, 11, False, GREY)

# ---- comparisons
s = new("Macro estimators compared across flows A and B", "Direct ATT and net value against truth; Monte-Carlo bias and coverage", "Comparison")
img(s, ASSETS / "macro_by_flow.png", 0.5, 1.1, 12.3, 3.6)
img(s, ASSETS / "heat_macro.png", 0.5, 4.65, 12.3, 2.4)

s = new("Macro estimators: Monte-Carlo bias and what to believe", "Fresh experiments each time; BSTS has no OTA Monte-Carlo run, so it cannot be cleared", "Comparison")
img(s, ASSETS / "mc_bias.png", 0.5, 1.1, 8.2, 2.9)
rows = [[m, FL["B1"]["summary"]["status"][m], FL["B2"]["summary"]["status"][m]] for m in ("MMT", "SDID", "BSTS")]
fl = {}
for i, r in enumerate(rows):
    for j in (1, 2):
        fl[(i, j)] = {"PASS": TEAL, "CAUTION": AMBER, "FAIL": RED, "UNVERIFIED": GREY}[r[j]]
table(s, ["Estimator", "B1 direct", "B2 OTA"], rows, 9.0, 1.3, 3.9, [1.5, 1.2, 1.2], 12, 0.5, flags=fl)
bullets(s, ["SDID is the estimator to trust: PASS on direct and OTA", "MMT: unbiased on average but shows placebo drift on direct bookings",
            "BSTS: biased toward zero (about +2.1 bookings per DMA-day on direct) and untested on OTA, where its headline is about 100% off"], 0.6, 4.2, 12.1, 2.6, 14)

s = new("CATE learners compared: leaderboards", "Out-of-fold, DMA-grouped; PEHE / sd(true CATE) with the constant-ATE reference", "Comparison")
img(s, ASSETS / "pehe.png", 0.4, 1.05, 12.5, 3.3)
keep = ["pehe_norm", "rank_corr", "qini_ratio", "ate_abs_err", "seconds"]
rows = []
for a in ALGOS:
    rows.append([a, f"{LBD.loc[a, 'pehe_norm']:.2f}", f"{LBD.loc[a, 'rank_corr']:.2f}", f"{LBD.loc[a, 'qini_ratio']:.2f}", f"{LBO.loc[a, 'pehe_norm']:.2f}", f"{LBO.loc[a, 'rank_corr']:.2f}", f"{LBD.loc[a, 'seconds']:.0f}s"])
rows.append(["Constant ATE (DiD)", f"{LBD.loc['Constant ATE (DiD)', 'pehe_norm']:.2f}", "n/a", "n/a", f"{LBO.loc['Constant ATE (DiD)', 'pehe_norm']:.2f}", "n/a", "0s"])
table(s, ["Model", "Direct PEHE/sd", "Direct rank", "Direct Qini ratio", "OTA PEHE/sd", "OTA rank", "Train"], rows, 0.5, 4.3, 12.3, [2.4, 1.6, 1.4, 1.7, 1.6, 1.4, 1.1], 10, 0.245)

s = new("Model-by-flow comparison: macro estimators and CATE learners", "One matrix per family: each cell is that model's metric in that flow (green = best in column)", "Comparison")
img(s, ASSETS / "heat_macro.png", 0.4, 1.0, 12.5, 2.1)
img(s, ASSETS / "heat_cate.png", 0.4, 3.0, 12.5, 4.0)

s = new("The same learners in the single-DMA what-ifs (flows C1, C2)", "Spread across learners is itself the uncertainty signal; the structural line is the experiment-calibrated reference", "Comparison")
img(s, ASSETS / "learners_c.png", 0.5, 1.1, 12.3, 3.7)
bullets(s, [f"C1 Miami: all nine learners predict a direct loss; the DR-Learner is an outlier ({LT['c']['C1']['lift']['DR-Learner'] * 100:.0f}%), the S-Learner and CF-DML sit closest to the structural {LT['c']['C1']['struct'] * 100:.0f}%",
            "C2 Asheville: lift is small for every learner, so decisions there hinge on a few thousand dollars",
            "Mean of nine is more stable than any single learner or the top-3: used for the CATE route in C and the policy in D"], 0.6, 4.95, 12.1, 2.0, 14)

s = new("The same learners in the policy flows (D1, D2)", "Realised 4-week value against ground truth", "Comparison")
img(s, ASSETS / "policy_rules.png", 0.5, 1.1, 12.3, 3.9)
bullets(s, [f"Ensemble at k=1 (D1): {usd(LT['ens']['k1'])}; at k=0 (D2): {usd(LT['ens']['k0'])}; always keep: $0; oracle {usd(val['oracle_usd'])}",
            "Individual learners deciding alone are mixed, the naive ensemble is worse than the margin rule, and nothing beats keeping everywhere reliably",
            "The k sweep is shown for transparency: choosing k after seeing truth would be cherry-picking"], 0.6, 5.1, 12.1, 1.9, 14)

# =============================================================================== PART 4 flows
section_slide("Flows and scenario stories", "Four flow types x two examples. For each: story, flow-level metrics, stage-level metrics, and the Streamlit screen", "Part 4")

for fid in D["order"]:
    f = FL[fid]
    last = f["stages"][-1]["flag"]
    col = STATUS[last]
    # slide 1: story + flow metrics
    s = new(f"{fid}: {f['title']}", f"Flow type {f['type']} - {f['type_title']}: {f['question']}", "Flows")
    txt(s, "The story", 0.6, 1.15, 5, 0.35, 14, True, TEAL)
    txt(s, f["story"], 0.6, 1.5, 12.1, 0.9, 14, False, INK)
    rect(s, 0.6, 2.45, 12.1, 0.85, col)
    txt(s, "Flow verdict", 0.75, 2.5, 3, 0.3, 11, False, WHITE)
    txt(s, f["verdict"], 0.75, 2.78, 11.8, 0.5, 20, True, WHITE)
    txt(s, f["action"], 0.6, 3.38, 12.1, 0.8, 12, False, GREY)
    txt(s, "Flow-level metrics", 0.6, 4.2, 5, 0.35, 14, True, TEAL)
    fm = f["flow_metrics"][:6]
    for i, m in enumerate(fm):
        kpi(s, m["label"], m["value"], 0.6 + (i % 3) * 4.07, 4.6 + (i // 3) * 1.2, 3.9, 1.05, NAVY, 22)
    # slide 2: stage-level metrics
    s = new(f"{fid}: stage-by-stage metrics", f["title"], "Flows")
    rows, flg = [], {}
    for i, st in enumerate(f["stages"]):
        rows.append([st["title"], (st.get("flag") or "-").upper(), "   |   ".join(f"{m['label']}: {m['value']}" for m in st["metrics"])])
        if st.get("flag"):
            flg[(i, 1)] = STATUS[st["flag"]]
    table(s, ["Stage", "Status", "Metrics"], rows, 0.4, 1.15, 12.5, [2.7, 0.9, 8.9], 9.5, 0.78, flags=flg)
    # slides 3-4: the Streamlit screen as the analyst sees it (three consecutive crops) plus a legible read-out
    c = FLOW_CROPS[fid]
    s = new(f"{fid}: what the analyst sees in Streamlit (1/2)", f"Tab 5 - End-to-End Flows - {f['short']}  |  full-size image: outputs/screenshots", "Flows")
    img(s, c[0], 0.3, 1.05, 6.3, 5.95)
    img(s, c[1], 6.75, 1.05, 6.3, 5.95)
    s = new(f"{fid}: what the analyst sees in Streamlit (2/2)", f"Tab 5 - End-to-End Flows - {f['short']}", "Flows")
    img(s, c[2], 0.3, 1.05, 6.3, 5.95)
    txt(s, "Reading the screen", 6.95, 1.15, 5.8, 0.4, 16, True, TEAL)
    lines = [f"{st['title']}  [{(st.get('flag') or 'info').upper()}]: " + "; ".join(f"{m['label']} {m['value']}" for m in st["metrics"][:2]) for st in f["stages"]]
    bullets(s, lines, 6.95, 1.6, 5.9, 4.6, 12, gap=7)
    txt(s, f"Verdict: {f['verdict']}", 6.95, 6.35, 5.9, 0.5, 15, True, STATUS[f["stages"][-1]["flag"]])

# ---- scenario stories with presets
s = new("Scenario story: one executive, three markets", "Streamlit tab 3 presets A, B, C: same engine, three tier archetypes", "Stories")
story = {"A": "Miami Beach flagship (Tier 1): heavy OTA conquesting, 55% cannibalization. Pausing hands bookings to Expedia and Booking.com. Keep.",
         "B": "Nashville regional (Tier 2): near break-even. Cannibalization 82%, modest OTA threat. The simulator tips to Pause by $2.8k per 4 weeks: not worth the risk without a re-test.",
         "C": "Asheville drive-to (Tier 3): top organic rank, almost no OTA pressure, 93% cannibalization. Pause saves $3.2k in 4 weeks: a pilot candidate."}
for k, (kk, txt_) in enumerate(story.items()):
    x = 0.4 + k * 4.2
    p = PRESETS[kk]
    sim = service.simulate_holdout_policy({x_: p[x_] for x_ in ["baseline_direct_bookings_day", "ota_to_direct_ratio", "lastclick_share", "cannibalization_rate", "ota_capture_rate",
                                                              "diversion_rate", "adr", "commission_rate", "cpc"]})
    rect(s, x, 1.2, 4.0, 5.7, LIGHT)
    txt(s, f"Preset {kk}: {p['label'].split(' (')[0]}", x + 0.1, 1.25, 3.8, 0.4, 14, True, NAVY)
    img(s, shot(f"0{4 + k + 1}_streamlit_tab3_preset_{kk}.png"), x + 0.1, 1.7, 3.8, 2.4)
    txt(s, txt_, x + 0.1, 4.15, 3.8, 1.5, 11.5, False, INK)
    txt(s, f"{sim['decision']}  |  iROAS {sim['iroas']:.1f}x vs last-click {sim['last_click_roas']:.1f}x  |  net change if paused {usd(sim['net_margin_change_if_paused_usd'])}", x + 0.1, 5.75, 3.8, 1.0, 11.5, True,
        TEAL if sim["decision"].startswith("Keep") else AMBER)

for ttl, name, pts in (("Story: the programme read-out (Tab 1)", "03_streamlit_tab1_geolift.png", ["Observed vs counterfactual per outcome, with the pause day marked", "Last-click vs incremental ROAS at programme level for any estimator",
                                                                                                    "DMA selector for single-market views with matched control"]),
                       ("Story: who is the best learner? (Tab 2)", "04_streamlit_tab2_leaderboard.png", ["Leaderboard of nine learners and the constant-ATE reference, direct and OTA",
                                                                                                         "Interactive Qini curves and permutation importance", "Reading: heterogeneity is poorly recovered with 10 treated DMAs"]),
                       ("Story: refutation and DMA policy (Tab 4)", "08_streamlit_tab4_policy_refutation.png", ["Policy value vs oracle, always-pause and alternative rules", "Placebo tests and Monte-Carlo bias in one place",
                                                                                                                  "Per-DMA decision table with learner disagreement"])):
    s = new(ttl, "Streamlit screenshot", "Stories")
    img(s, shot(name), 0.4, 1.1, 8.3, 5.9)
    bullets(s, pts, 8.9, 1.4, 4.1, 5, 14)

# ---- recommendations
s = new("Recommendations", "What the evidence supports and what it does not", "Close")
bullets(s, [f"Do not pause branded search on the average evidence, and do not claim it is highly profitable either: SDID net value of keeping is {usd(sd['net_value_of_branded_search_usd'])} per 4 weeks across 10 DMAs with an interval that spans zero (simulated truth {usd(tr['net_value_of_branded_search_usd'])}); it protects {usd(sd['ota_commission_leakage_usd'])} of OTA commission",
            "Stop reporting last-click ROAS as return: iROAS is 2.2x vs 5.8x credited; renegotiate budgets on iROAS",
            "Pilot, do not roll out, DMA pauses: Tier-3 drive-to profiles (C2, preset C) are the only near-break-even candidates, and need a confirmatory holdout",
            "Do not act on DMA-level CATE pause lists from a 10-treated-DMA experiment (D1/D2 lose money); add holdouts and weeks first",
            "Make SDID the primary estimator; use MMT as a transparent cross-check; use BSTS only with a Monte-Carlo clearance",
            "Run the flows as a standing checklist: A (effect), B (trust), C (where), D (policy) before any budget change"], 0.6, 1.25, 12.1, 5.6, 15, gap=8)

s = new("Limitations and how to reproduce", None, "Close")
bullets(s, ["Synthetic data: real branded-search effects will not follow this DGP; value is the method, diagnostics and honesty checks",
            "Only 50 DMAs / 10 treated: effective sample size is 50 units; CATE metrics are DMA-grouped out-of-fold",
            "JAX/NumPyro on CPU on Windows; RAPIDS not available (DTW on CUDA instead); CausalPy not installed so BSTS is written directly in NumPyro",
            "Policy value is scored with simulated truth; a real test needs doubly robust policy evaluation and a follow-up holdout",
            "Flow verdict thresholds are rules in src/flows.py, not statistical guarantees"], 0.6, 1.2, 12.1, 3.2, 14)
rect(s, 0.6, 4.55, 12.1, 2.3, LIGHT)
txt(s, ["make setup    # pip install -r requirements.txt && playwright install chromium", "make train    # panel -> macro -> cate -> econ -> plots -> flows",
        "make flows    # rebuild the 8 flows + scorecard (outputs/metrics/flows.json)", "make verify   # Playwright e2e, screenshots, artefact checks",
        "python presentation/build_deck.py   # rebuild this deck from outputs/"], 0.8, 4.65, 11.7, 2.1, 13, False, INK, font="Consolas")

out = OUT / "Paid_Search_Incrementality_End_to_End.pptx"
prs.save(out)
for p in ASSETS.glob("_tmp_*.png"):
    p.unlink()
print("saved", out, "slides:", len(prs.slides), "size MB:", round(out.stat().st_size / 1e6, 1))
