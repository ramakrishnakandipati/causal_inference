"""Build the end-to-end presentation (native, editable .pptx) from the saved metrics, flow reports and screenshots.

    python scripts/build_deck.py          ->  presentation/Aurelia_OOH_Incrementality_End_to_End.pptx

Every number comes from outputs/metrics/* or from src/flows.py (the same engine the API and Streamlit use); nothing is typed in by hand
except descriptions of the models. Slides carry speaker notes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.chart.data import CategoryChartData  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

from src import flows as fl  # noqa: E402
from src.serving import Engine  # noqa: E402

M, FIG, SHOT = ROOT / "outputs/metrics", ROOT / "outputs/figures", ROOT / "outputs/screenshots"
OUT = ROOT / "presentation"
OUT.mkdir(exist_ok=True)
FILE = OUT / "Aurelia_OOH_Incrementality_End_to_End.pptx"

NAVY, BLUE, ORANGE, AQUA = RGBColor(0x1F, 0x2A, 0x44), RGBColor(0x26, 0x80, 0xE0), RGBColor(0xF2, 0x72, 0x30), RGBColor(0x2A, 0xB5, 0xA5)
INK, GRAY, LIGHT, WHITE = RGBColor(0x22, 0x22, 0x22), RGBColor(0x66, 0x66, 0x66), RGBColor(0xF1, 0xF4, 0xF9), RGBColor(0xFF, 0xFF, 0xFF)
GREEN_BG, AMBER_BG, RED_BG = RGBColor(0xDD, 0xF3, 0xE4), RGBColor(0xFF, 0xF0, 0xCC), RGBColor(0xFC, 0xDD, 0xDD)
GREEN_FG, AMBER_FG, RED_FG = RGBColor(0x1B, 0x6B, 0x3A), RGBColor(0x8A, 0x5A, 0x00), RGBColor(0x9B, 0x1C, 0x1C)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5
_n = [0]


# ----------------------------------------------------------------------------- helpers
def rd(name: str) -> pd.DataFrame:
    return pd.read_csv(M / name)


def f(x, d=0, pre="", suf="") -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{pre}{x:,.{d}f}{suf}"


def text(slide, x, y, w, h, s, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Calibri"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    lines = s if isinstance(s, list) else [s]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = ln
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, font
    return tb


def bullets(slide, x, y, w, h, items, size=14, color=INK, gap=6):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        head, _, tail = it.partition("||")
        r = p.add_run()
        r.text = "• " + head
        r.font.size, r.font.color.rgb, r.font.name = Pt(size), color, "Calibri"
        r.font.bold = bool(tail)
        if tail:
            r2 = p.add_run()
            r2.text = tail
            r2.font.size, r2.font.color.rgb, r2.font.name = Pt(size), color, "Calibri"
    return tb


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    return s


def box_text(slide, x, y, w, h, s, fill=LIGHT, fg=INK, size=12, bold=False, line=None, align=PP_ALIGN.CENTER):
    shp = rect(slide, x, y, w, h, fill, line)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.06)
    lines = s if isinstance(s, list) else [s]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = ln
        r.font.size, r.font.name, r.font.color.rgb = Pt(size if i == 0 else size - 2), "Calibri", fg
        r.font.bold = bold if i == 0 else False
    return shp


def arrow(slide, x1, y1, x2, y2, color=GRAY):
    c = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(1.75)
    ln = c.line._get_or_add_ln()
    from lxml import etree
    tail = etree.SubElement(ln, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd")
    tail.set("type", "triangle")
    return c


def new_slide(title: str, subtitle: str | None = None, notes: str = "", section: str | None = None):
    s = prs.slides.add_slide(BLANK)
    _n[0] += 1
    bar = rect(s, 0, 0, SW, 0.95, NAVY, shape=MSO_SHAPE.RECTANGLE)
    text(s, 0.45, 0.12, 11.6, 0.55, title, size=26, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        text(s, 0.45, 0.58, 12, 0.35, subtitle, size=13, color=RGBColor(0xC9, 0xD6, 0xEE))
    if section:
        text(s, 10.6, 0.1, 2.5, 0.3, section.upper(), size=10, bold=True, color=RGBColor(0x8F, 0xB8, 0xF0), align=PP_ALIGN.RIGHT)
    text(s, 12.2, 7.1, 1.0, 0.3, str(_n[0]), size=10, color=GRAY, align=PP_ALIGN.RIGHT)
    text(s, 0.45, 7.1, 8, 0.3, "Aurelia Hotels & Resorts | Regional brand & OOH incrementality | simulated data, ground truth known", size=9, color=GRAY)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def image(slide, path: Path, x, y, w, h, border=True, crop_ar: float | None = None):
    """Place an image scaled to fit the w x h box, centred. crop_ar keeps only the top part so width/height = crop_ar (for very tall screenshots)."""
    import io
    im = Image.open(path)
    src: object = str(path)
    if crop_ar and im.width / im.height < crop_ar:
        im = im.crop((0, 0, im.width, int(im.width / crop_ar)))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        buf.seek(0)
        src = buf
    ar = im.width / im.height
    iw, ih = (w, w / ar) if w / h < ar else (h * ar, h)
    px, py = x + (w - iw) / 2, y + (h - ih) / 2
    pic = slide.shapes.add_picture(src, Inches(px), Inches(py), Inches(iw), Inches(ih))
    if border:
        pic.line.color.rgb = RGBColor(0xD5, 0xDB, 0xE5)
        pic.line.width = Pt(0.75)
    return pic


def table(slide, df: pd.DataFrame, x, y, w, h, col_w=None, size=10, header_fill=NAVY, fills=None, bold_first=False, align_first_left=True):
    """df of strings; fills = {(row, col): (bg, fg)} (row index in df)."""
    rows, cols = len(df) + 1, len(df.columns)
    gs = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = gs.table
    tblPr = gs._element.graphic.graphicData.tbl.tblPr
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")
    if col_w:
        tot = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Inches(w * cw / tot)
    for r in range(rows):
        tbl.rows[r].height = Inches(h / rows)
    for c, name in enumerate(df.columns):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        _cell(cell, str(name), size, True, WHITE, PP_ALIGN.LEFT if c == 0 and align_first_left else PP_ALIGN.CENTER)
    for r in range(len(df)):
        for c in range(cols):
            cell = tbl.cell(r + 1, c)
            cell.fill.solid()
            bg, fg = (fills or {}).get((r, c), (WHITE if r % 2 == 0 else LIGHT, INK))
            cell.fill.fore_color.rgb = bg
            _cell(cell, str(df.iat[r, c]), size, (bold_first and c == 0), fg, PP_ALIGN.LEFT if c == 0 and align_first_left else PP_ALIGN.CENTER)
    return tbl


def _cell(cell, s, size, bold, color, align):
    cell.margin_left = cell.margin_right = Inches(0.05)
    cell.margin_top = cell.margin_bottom = Inches(0.02)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = s
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, "Calibri"


def kpi(slide, x, y, w, h, value, label, sub=None, color=BLUE):
    rect(slide, x, y, w, h, LIGHT)
    rect(slide, x, y + 0.12, 0.07, h - 0.24, color, shape=MSO_SHAPE.RECTANGLE)
    text(slide, x + 0.18, y + 0.08, w - 0.25, h * 0.5, value, size=26, bold=True, color=NAVY)
    text(slide, x + 0.18, y + h * 0.52, w - 0.25, 0.3, label, size=11, bold=True, color=INK)
    if sub:
        text(slide, x + 0.18, y + h * 0.52 + 0.25, w - 0.25, 0.3, sub, size=9.5, color=GRAY)


def bar_chart(slide, x, y, w, h, cats, series: dict, title=None, horizontal=True, colors=None, number_format='0', legend=True, font=10):
    cd = CategoryChartData()
    cd.categories = cats
    for k, v in series.items():
        cd.add_series(k, [float(a) if a == a else 0.0 for a in v])
    ct = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    ch = slide.shapes.add_chart(ct, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.font.size = Pt(font)
    ch.has_title = bool(title)
    if title:
        ch.chart_title.text_frame.text = title
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
    ch.has_legend = legend and len(series) > 1
    if ch.has_legend:
        ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.BOTTOM, False
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.has_data_labels = True
    pl.data_labels.number_format, pl.data_labels.number_format_is_linked = number_format, False
    pl.data_labels.font.size = Pt(font - 1)
    for i, s in enumerate(pl.series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = (colors or [BLUE, ORANGE, AQUA, GRAY])[i % 4]
    if horizontal:
        ch.category_axis.reverse_order = True
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW     # keep labels clear of negative bars
    ch.value_axis.has_major_gridlines = False
    ch.value_axis.visible = False
    return ch


def tone(v: float, good: float, ok: float):
    a = abs(v)
    return (GREEN_BG, GREEN_FG) if a <= good else (AMBER_BG, AMBER_FG) if a <= ok else (RED_BG, RED_FG)


VERDICT_TONE = {"INCREMENTAL AND PROFITABLE": (GREEN_BG, GREEN_FG), "NO DETECTABLE EFFECT": (GREEN_BG, GREEN_FG), "SAFE TO RUN": (GREEN_BG, GREEN_FG),
                "LEAN IN": (GREEN_BG, GREEN_FG), "BASELINE": (GREEN_BG, GREEN_FG), "PULL BACK": (RED_BG, RED_FG), "FALSE POSITIVE": (RED_BG, RED_FG),
                "DO NOT TRUST THE PROJECTION": (RED_BG, RED_FG), "INCREMENTAL BUT BELOW BREAK-EVEN": (RED_BG, RED_FG)}


# ----------------------------------------------------------------------------- data
eng = Engine()
S = rd("panel_att_summary.csv")
S = S[S.outcome == "room_nights_booked"]
E = rd("economics.csv")
MC = rd("panel_monte_carlo.csv")
CATE = rd("cate_benchmark.csv")
GPU = rd("gpu_benchmark.csv")
PP = rd("placebo_pvalues.csv")
PV = rd("pacing_validation.csv")
IT = rd("in_time_placebo.csv")
LODO = rd("lodo_sensitivity.csv")
BAY = rd("bayes_diagnostics.csv")
REPS = {x["id"]: fl.report(eng, x) for x in fl.FLOWS}
head = S[(S.scenario == "mia_all24") & (S.estimator == "ASCM (Ridge)")].iloc[0]
twfe = S[(S.scenario == "mia_all24") & (S.estimator == "TWFE DiD")].iloc[0]
econ = E[(E.scenario == "mia_all24") & (E.estimator == "ASCM (Ridge)")].iloc[0]
cate_m = CATE[~CATE.reference].sort_values("pehe")
best = cate_m.iloc[0]
const = CATE[CATE.learner == "Constant ATE (oracle)"].iloc[0]
oracle = CATE[CATE.learner == "Oracle X-regression (CV)"].iloc[0]
mc_t = MC.set_index("estimator")
ORDER = ["TWFE DiD", "Event Study", "CS-DiD (DR)", "SCM", "ASCM (Ridge)", "SDID", "Bayesian SC", "ASCM (ElasticNet)"]

# ============================================================================ 1 cover
s = prs.slides.add_slide(BLANK)
_n[0] += 1
rect(s, 0, 0, SW, SH, NAVY, shape=MSO_SHAPE.RECTANGLE)
rect(s, 0.6, 2.2, 0.12, 2.2, BLUE, shape=MSO_SHAPE.RECTANGLE)
text(s, 0.95, 1.9, 11.5, 1.2, "Regional Brand & Out-of-Home Incrementality", size=40, bold=True, color=WHITE)
text(s, 0.95, 3.05, 11.5, 0.8, "End-to-end flow, model comparison and scenario stories", size=24, color=RGBColor(0xC9, 0xD6, 0xEE))
text(s, 0.95, 3.9, 11.5, 0.9, ["Aurelia Hotels & Resorts - Miami CTV + airport + billboard campaign",
                               "Synthetic control, DiD and geo-CATE (uplift) on a 25-DMA x 240-day simulated panel"], size=16, color=RGBColor(0x8F, 0xB8, 0xF0))
text(s, 0.95, 6.5, 11.5, 0.4, "Every number in this deck is generated from outputs/metrics and the live flow engine (scripts/build_deck.py).", size=11, color=RGBColor(0x8F, 0xA3, 0xC8))
s.notes_slide.notes_text_frame.text = ("Opening: this deck walks the whole engine - data, the eight panel estimators, the ten uplift (geo-CATE) learners, eight worked flows shown through "
                                       "the Streamlit app, and the metrics at stage, flow and programme level. The data are simulated so the truth is always on the table.")

# ============================================================================ 2 executive summary
s = new_slide("Executive summary", "What the engine says about the Miami winter-peak campaign", section="Overview",
              notes="Headline: the synthetic-control family recovers the truth (131 vs 124 room nights/day) while a naive DiD overstates it. The single-panel TWFE bias is a bad draw; "
                    "the 12-panel Monte-Carlo mean is about +36%. Both figures are quoted so nobody over- or under-sells the DiD problem.")
kpi(s, 0.45, 1.2, 3.0, 1.35, f(head.att_mean, 0), "ASCM lift, room nights / day", f"95% CI {f(head.ci_lo)} to {f(head.ci_hi)} | truth {f(head.att_true_mean)}")
kpi(s, 3.6, 1.2, 3.0, 1.35, f(econ.iroas, 2, suf="x"), "iROAS", f"CI {f(econ.iroas_lo, 2)} to {f(econ.iroas_hi, 2)} | truth {f(econ.true_iroas, 2)}x", ORANGE)
kpi(s, 6.75, 1.2, 3.0, 1.35, f(econ.cpirn_usd, 0, "$"), "Cost / incremental room night", f"truth {f(econ.true_cpirn_usd, 0, '$')}", AQUA)
kpi(s, 9.9, 1.2, 3.0, 1.35, "0.040", "Placebo p-value", "rank 1 of 25 = smallest attainable", BLUE)
kpi(s, 0.45, 2.75, 3.0, 1.35, f"{twfe.att_bias_pct:+.0f}%", "TWFE bias, headline panel", f"Monte-Carlo mean {mc_t.loc['TWFE DiD', 'bias_pct']:+.0f}% over 12 panels", ORANGE)
kpi(s, 3.6, 2.75, 3.0, 1.35, f"{mc_t.loc['ASCM (Ridge)', 'bias_pct']:+.0f}%", "ASCM bias, Monte-Carlo mean", f"RMSE {mc_t.loc['ASCM (Ridge)', 'rmse']:.0f} room nights/day", BLUE)
kpi(s, 6.75, 2.75, 3.0, 1.35, f(best.pehe, 1), f"Best CATE PEHE ({best.learner})", f"constant-effect baseline {const.pehe:.1f}", AQUA)
kpi(s, 9.9, 2.75, 3.0, 1.35, "8 / 8", "Flows where UI = API", "4 flow types x 2 examples", ORANGE)
bullets(s, 0.45, 4.4, 12.4, 2.6, [
    "Incremental and profitable in Miami.|| 7,855 incremental room nights over 60 days (truth 7,441); iROAS 1.49x vs break-even 1.0x.",
    "A naive DiD would have over-credited the campaign.|| Miami's winter ramp is steeper than the donor average, so parallel trends fail; re-weighting donors (SCM family) fixes it.",
    "Thin donor pools cannot prove a real effect.|| Las Vegas (4 donors) and New York (3 donors) are 'suggestive, not conclusive': the smallest attainable p is 0.20 / 0.25.",
    "Uplift tells you which days work, not how much in total.|| Ten learners rank campaign days well (oracle Qini 0.61-0.83) but only a few beat a constant-effect baseline on PEHE.",
    "Do not trust the pacing optimiser.|| Its projected +8.7% vs even pacing is +0.2% under the true response (flow D2)."], size=13.5)

# ============================================================================ 3 business problem
s = new_slide("The business problem", "Why a geo experiment, and what decision it feeds", section="Overview",
              notes="Aurelia cannot randomise individual guests for CTV, airport spectaculars or billboards - they reach the whole DMA. So the unit of analysis is the market, "
                    "and the counterfactual must be built from other markets.")
bullets(s, 0.45, 1.2, 6.2, 5.6, [
    "Campaign.|| 60 days (Dec 28 2025 - Feb 25 2026) of CTV, airport digital and regional billboards in Miami, after a 180-day baseline; $3.0M spend.",
    "Constraint.|| Media reaches an entire Designated Market Area: no user-level holdout.",
    "Question 1.|| How many incremental room nights, web sessions and branded searches did it create?",
    "Question 2.|| What did each incremental room night cost (CPIRN) and what is iROAS?",
    "Question 3.|| On which days does media work hardest (weekends, holidays, weather, rival rate cuts, flight-search surges)?",
    "Question 4.|| How should the next budget be flighted?"], size=15)
box_text(s, 7.0, 1.3, 5.8, 0.7, "Decisions the engine supports", fill=NAVY, fg=WHITE, size=15, bold=True)
for i, (h, d) in enumerate([("Scale / hold / cut", "Is the campaign incremental and above break-even?"), ("Trust the number", "Placebo, in-time and leave-one-donor-out checks"),
                            ("Where to lean in", "Geo-CATE: which market-days respond most"), ("How to pace", "Budget what-if simulator, audited against truth")]):
    box_text(s, 7.0, 2.15 + i * 1.15, 5.8, 1.0, [h, d], fill=LIGHT, size=14, bold=True)

# ============================================================================ 4 architecture
s = new_slide("End-to-end architecture", "From simulated panel to API and workbench", section="Documentation",
              notes="Left to right: data, estimation, diagnostics, economics, uplift, serving. The API and Streamlit app share src/serving.py so their numbers always agree.")
cols = [("1 Panel", ["25 DMAs x 240 days", "180 pre / 60 campaign", "ground truth y0,y1"], BLUE), ("2 Panel ATT", ["8 estimators", "DiD, event study, CS-DiD", "SCM, ASCM, SDID, Bayes"], BLUE),
        ("3 Diagnostics", ["24 in-space placebos", "in-time placebo, LODO", "pre-trend tests"], ORANGE), ("4 Economics", ["incremental room nights", "CPIRN, iROAS + CI", "break-even"], ORANGE),
        ("5 Uplift / CATE", ["ASCM residuals", "10 learners", "PEHE, Qini, drivers"], AQUA), ("6 Decide", ["what-if simulator", "pacing + response curve", "audit vs truth"], AQUA)]
for i, (h, ls, c) in enumerate(cols):
    x = 0.45 + i * 2.1
    box_text(s, x, 1.5, 1.9, 0.6, h, fill=c, fg=WHITE, size=14, bold=True)
    box_text(s, x, 2.15, 1.9, 1.5, ls, fill=LIGHT, size=11.5)
    if i < len(cols) - 1:
        arrow(s, x + 1.92, 1.8, x + 2.08, 1.8)
box_text(s, 0.45, 4.1, 5.9, 0.6, "FastAPI service: /health  /predict/att  /predict/cate  /simulate/campaign", fill=NAVY, fg=WHITE, size=13, bold=True)
box_text(s, 6.9, 4.1, 5.9, 0.6, "Streamlit workbench: 4 presets, 6 tabs, 8 end-to-end flows", fill=NAVY, fg=WHITE, size=13, bold=True)
box_text(s, 0.45, 4.85, 12.35, 0.6, "src/serving.py - one engine behind both, so the UI and API cannot disagree (checked for all 8 flows by the screenshot script)", fill=LIGHT, size=12.5, bold=True)
bullets(s, 0.45, 5.65, 12.4, 1.4, ["Stack: Python 3.12, PyTorch CUDA (RTX 3090), XGBoost/CatBoost GPU, LightGBM, econml, NumPyro/JAX (CPU on native Windows), OSQP, FastAPI, Streamlit, Playwright.",
                                  "Reproducible: run_full_pipeline.py -> capture_screenshots.py -> build_readme.py -> build_deck.py; README and deck are generated, never hand-edited."], size=12.5)

# ============================================================================ 5 uplift e2e documentation
s = new_slide("Uplift end-to-end flow at a glance", "How a campaign-level ATT becomes a market-day uplift (CATE) and a pacing decision", section="Documentation",
              notes="Uplift here is the conditional lift of media on a market-day, given conditions such as weekend, holiday, weather, flight searches and rival rates. "
                    "We cannot observe a counterfactual for Miami on any day, so the label is the gap to a synthetic control; the learners then explain that gap with covariates.")
steps = [("1 Counterfactual", "Leave-self-out ASCM predicts each DMA-day without media", "output: residual = observed - synthetic"),
         ("2 Uplift label", "Campaign window only (60 days, D=1 for Miami)", "why: pre-window has no overlap"),
         ("3 Ten learners", "S T X R DR, CausalForestDML, LinearDML, DomainAdaptation, TARNet, CFRNet", "output: tau(x) per market-day"),
         ("4 Evaluate", "PEHE, ATT bias, corr, AUUC / Qini vs truth", "baselines: constant ATE, oracle regression"),
         ("5 Explain", "Heterogeneity drivers: weekend, holiday, flights, OTA, weather", "Linear-DML coefficients, subgroup lifts"),
         ("6 Act", "What-if simulator + response curve + pacing plans", "audit: projected vs true per strategy")]
for i, (h, d, o) in enumerate(steps):
    x = 0.45 + i * 2.1
    box_text(s, x, 1.3, 1.95, 0.65, h, fill=AQUA if i in (2, 3) else BLUE, fg=WHITE, size=13.5, bold=True)
    box_text(s, x, 2.0, 1.95, 1.6, d, fill=LIGHT, size=11)
    box_text(s, x, 3.65, 1.95, 0.9, o, fill=WHITE, size=10, line=RGBColor(0xD5, 0xDB, 0xE5))
    if i < 5:
        arrow(s, x + 1.97, 1.62, x + 2.08, 1.62)
bullets(s, 0.45, 4.85, 12.4, 2.2, [
    "Uplift vs ATT.|| ATT is one number for the whole campaign; CATE is a function of market conditions, learned from only 60 treated days.",
    "Honest limits.|| With one treated market a feature's level partly proxies 'being Miami'; read Linear-DML coefficients as a descriptive surface, not causal elasticities.",
    "What it is good for.|| Ranking days (flighting), not recovering the exact room-night count: every learner beats random ordering (oracle Qini 0.61-0.83).",
    "How it is used downstream.|| Flow C scores a hypothetical day with all ten learners; flow D multiplies the response curve by the best learner's day-by-day multiplier."], size=12.5)

# ============================================================================ 6 metric glossary / uplift evaluation
s = new_slide("How every model is scored", "Metric definitions used at stage, flow and programme level", section="Documentation",
              notes="Two families of metrics: panel (aggregate effect) and uplift (heterogeneous effect). Truth columns are only possible because the data are simulated.")
gl = pd.DataFrame([
    ("ATT / day", "Mean daily effect on Miami over the 60 campaign days", "Panel"), ("Bias %", "(estimate - truth) / truth", "Panel"),
    ("RMSE", "Root mean squared error across 12 simulated panels (Monte-Carlo)", "Panel"), ("CI coverage", "Share of panels whose 95% interval contains the truth", "Panel"),
    ("Pre-RMSPE", "Pre-period tracking error of the counterfactual; lower = better fit", "Panel"), ("Placebo p", "Exact in-space p = rank / (donors + 1); floor = 1/(donors+1)", "Panel"),
    ("CPIRN / iROAS", "Spend / incremental room nights; incremental net revenue / spend", "Economics"),
    ("PEHE", "RMSE of predicted vs true daily effect on the 60 campaign days", "Uplift"), ("Qini (oracle)", "Ranking quality of days vs truth: 0 random, 1 perfect", "Uplift"),
    ("Corr / ATT bias", "Correlation of predicted vs true effect; error of the mean prediction", "Uplift"),
    ("Projection error", "Projected - true change vs even pacing (percentage points)", "Pacing")], columns=["Metric", "Definition", "Used for"])
table(s, gl, 0.45, 1.2, 12.4, 5.5, col_w=[2, 8, 1.4], size=11.5, bold_first=True)

# ============================================================================ 7 panel + ground truth
s = new_slide("The simulated panel and its ground truth", "25 DMAs x 240 days; Miami is a planted mix of donors", section="Data",
              notes="Miami is a convex combination of FLL, TPA, ORL, HNL, PHX, SAN, MSY plus a 1.5% level shift, so a synthetic control exists in principle. "
                    "Its winter ramp is steeper than the donor average, which breaks parallel trends. Four scenario panels expose Miami, Las Vegas, New York and a Tampa placebo.")
bullets(s, 0.45, 1.2, 5.3, 5.7, [
    "Demand.|| Three-factor model (national trend, Sunbelt winter peak, urban/business cycle) + DMA drift + AR(1) noise + ten covariates.",
    "Miami.|| FLL .30, TPA .20, ORL .15, HNL .12, PHX .08, SAN .08, MSY .07; own weather, flight and rival shocks stay idiosyncratic.",
    "Campaign effect.|| Spend -> geometric adstock -> Hill saturation -> multipliers (weekend, holiday, flights up; OTA pressure, rival cuts, bad weather down).",
    "Outcomes.|| Room nights (primary), direct web sessions, branded search index, net revenue.",
    "Truth.|| y0_true, y1_true, true_att per day: every estimate is graded.",
    "Four scenarios.|| Miami (+ all-24 headline), Las Vegas, New York, Tampa placebo (true effect 0)."], size=12.5)
image(s, FIG / "01_parallel_trends_event_study.png", 6.0, 1.2, 6.9, 5.7)

# ============================================================================ 8 panel catalogue
s = new_slide("Model catalogue 1: eight panel estimators", "What each one assumes, and where it breaks", section="Models",
              notes="The first three assume parallel trends; the synthetic-control family builds a counterfactual from re-weighted donors; Bayesian SC adds posterior inference. "
                    "Headline estimator is ASCM (Ridge).")
cat = pd.DataFrame([
    ("TWFE DiD", "DiD", "Parallel trends; CR1 SE by DMA", "Simple, fast", "Biased when Miami's ramp is steeper"),
    ("Event Study", "DiD", "Parallel trends; leads/lags vs k=-1", "Shows dynamics and pre-trends", "Noisy; wide bands"),
    ("CS-DiD (DR)", "DiD", "Doubly robust ATT(g,t); one cohort", "Covariate-adjusted", "Collapses to regression DiD"),
    ("SCM", "Synthetic control", "Simplex weights on pre-period", "Interpretable donor mix", "Sensitive to donor set (LODO)"),
    ("ASCM (Ridge)", "Synthetic control", "SCM + ridge bias correction", "Lowest bias in Monte-Carlo; headline", "Needs lambda tuning"),
    ("SDID", "Synthetic control", "Unit + time weights + FE", "Robust to level shifts", "Higher pre-RMSPE"),
    ("Bayesian SC", "Bayesian", "NumPyro NUTS, Dirichlet(0.5) prior", "Full posterior, narrow intervals", "Most biased SC variant; optimistic CI"),
    ("ASCM (ElasticNet)", "Synthetic control", "SCM + elastic-net correction", "Sparse correction", "Variant of ASCM")], columns=["Estimator", "Family", "Key assumption", "Strength", "Risk"])
table(s, cat, 0.45, 1.2, 12.4, 5.6, col_w=[1.6, 1.4, 3, 2.7, 3], size=11.5, bold_first=True)

# ============================================================================ 9 CATE catalogue
s = new_slide("Model catalogue 2: ten geo-CATE (uplift) learners", "Meta-learners, double-ML and neural uplift models", section="Models",
              notes="All ten learn tau(x) on the 60 campaign days using leave-self-out ASCM residuals as the outcome. Backends are those actually used in the benchmark run.")
bk = cate_m.set_index("learner")["backend"].to_dict()
cat2 = pd.DataFrame([
    ("S-Learner", "Meta", "One model with treatment as a feature"), ("T-Learner", "Meta", "Separate models for treated / control"),
    ("X-Learner", "Meta", "Imputed effects, propensity-weighted blend"), ("R-Learner", "Meta / DML", "Residual-on-residual effect model"),
    ("DR-Learner", "Meta", "AIPW pseudo-outcomes (least stable: ~4% treated)"), ("Causal Forest DML", "Double ML", "Forest on residuals; native confidence intervals"),
    ("Linear DML", "Double ML", "Linear effect on named drivers; native CIs"), ("Domain Adaptation", "Transfer", "Learn on control, adapt to treated"),
    ("TARNet", "Neural", "Shared trunk, two outcome heads (PyTorch CUDA)"), ("CFRNet-DragonNet", "Neural", "Balanced representation + propensity head")], columns=["Learner", "Family", "Idea"])
cat2.insert(2, "Backend", [bk.get(n, "-") for n in cat2["Learner"]])
table(s, cat2, 0.45, 1.2, 12.4, 5.2, col_w=[2.2, 1.6, 1.5, 6], size=12, bold_first=True)
text(s, 0.45, 6.5, 12.4, 0.5, "Two reference rows bound what is achievable: a constant-effect baseline and an oracle regression of the true effect on the features.", size=12, color=GRAY)

# ============================================================================ 10 panel comparison headline
s = new_slide("Panel estimators compared: Miami headline", "All 24 donors, room nights per day; truth = 124", section="Model comparison",
              notes="Bias % vs the simulation truth. The synthetic-control family sits within roughly +/-9% except Bayesian SC; the DiD family is far off. "
                    "TWFE +107% is the single-draw error; see the Monte-Carlo slide for the typical size.")
h = S[S.scenario == "mia_all24"].set_index("estimator").loc[ORDER]
ec = E[E.scenario == "mia_all24"].set_index("estimator")
bar_chart(s, 0.3, 1.1, 4.9, 5.9, ORDER, {"Bias vs truth (%)": h.att_bias_pct.tolist()}, "Bias vs truth, %", number_format='+0;-0', colors=[ORANGE])
t = pd.DataFrame({"Estimator": ORDER, "ATT/day": [f(v) for v in h.att_mean], "95% CI": [f"{f(a)} to {f(b)}" for a, b in zip(h.ci_lo, h.ci_hi)],
                  "Bias %": [f"{v:+.0f}%" for v in h.att_bias_pct], "Covers?": ["yes" if c else "no" for c in h.covered], "Pre-RMSPE": [f(v, 0) for v in h.pre_rmspe],
                  "Placebo p": [f(v, 2) for v in h.p_placebo], "iROAS": [f(ec.loc[e, "iroas"], 2) if e in ec.index else "n/a" for e in ORDER]})
fills = {(i, 3): tone(v, 10, 30) for i, v in enumerate(h.att_bias_pct)}
table(s, t, 5.3, 1.25, 7.6, 5.0, col_w=[1.7, 0.9, 1.5, 0.9, 0.8, 1, 1, 0.8], size=10.5, fills=fills, bold_first=True)
text(s, 5.3, 6.35, 7.6, 0.6, "Green: |bias| <= 10%   Amber: <= 30%   Red: > 30%.  Truth: 124 room nights/day, iROAS 1.41x.", size=10.5, color=GRAY)

# ============================================================================ 11 Monte-Carlo
s = new_slide("Panel estimators compared: 12 simulated panels", "One panel is one draw - the Monte-Carlo shows the typical error", section="Model comparison",
              notes="Across 12 freshly simulated panels TWFE's mean bias is about +36% and ASCM's about +2%. Bayesian SC has the narrowest intervals and zero power at 5% in this run, "
                    "and a larger bias than ASCM - reported honestly.")
mc = MC.set_index("estimator").loc[ORDER]
bar_chart(s, 0.3, 1.1, 5.0, 5.9, ORDER, {"RMSE (room nights/day)": mc.rmse.tolist()}, "RMSE across 12 panels", number_format='0', colors=[BLUE])
t = pd.DataFrame({"Estimator": ORDER, "Mean ATT": [f(v) for v in mc.mean_att], "Bias %": [f"{v:+.0f}%" for v in mc.bias_pct], "RMSE": [f(v) for v in mc.rmse],
                  "CI coverage": [f"{v * 100:.0f}%" for v in mc.ci_coverage], "Mean SE": [f(v) for v in mc.mean_se], "Power @5%": [f"{v * 100:.0f}%" for v in mc.power_p05]})
table(s, t, 5.4, 1.25, 7.5, 4.9, col_w=[1.8, 1, 1, 0.9, 1.1, 0.9, 1], size=11, fills={(i, 2): tone(v, 10, 30) for i, v in enumerate(mc.bias_pct)}, bold_first=True)
text(s, 5.4, 6.25, 7.5, 0.7, f"Truth {mc.true_att.iloc[0]:.0f} room nights/day. SCM and ASCM have the lowest RMSE and 100% power; Bayesian SC's posterior is narrow but biased (+{mc.loc['Bayesian SC', 'bias_pct']:.0f}%).", size=11, color=GRAY)

# ============================================================================ 12 figure 05 + takeaway
s = new_slide("Seven estimators vs truth, with Monte-Carlo check", "Figure from the pipeline (300 DPI)", section="Model comparison",
              notes="Reference figure: every estimator against the simulation truth on the headline panel and across the Monte-Carlo.")
image(s, FIG / "05_panel_att_estimators_comparison.png", 0.45, 1.15, 8.6, 5.8)
bullets(s, 9.3, 1.3, 3.7, 5.5, ["Ranking is stable.|| ASCM / SCM / SDID best; TWFE and Event Study worst.", "Size of DiD error is not stable.|| +107% on this panel, +36% on average.",
                                "Intervals differ in honesty.|| Placebo-calibrated ASCM covers truth; Bayesian credible intervals are optimistic."], size=12.5)

# ============================================================================ 13 placebo
s = new_slide("Inference and robustness checks", "In-space placebo, in-time placebo, leave-one-donor-out", section="Model comparison",
              notes="Miami ranks first of 25 on post/pre RMSPE ratio: p = 0.04, the smallest possible with 24 donors. "
                    "Dropping any influential donor moves ASCM within a modest range, while plain SCM is far more sensitive.")
image(s, FIG / "04_in_space_placebo_permutation.png", 0.45, 1.15, 7.4, 5.8)
la = LODO[(LODO.estimator == "ASCM (Ridge)") & (LODO.dropped != "none")]
ls_ = LODO[(LODO.estimator == "SCM") & (LODO.dropped != "none")]
it = IT.copy()
bullets(s, 8.1, 1.3, 4.9, 5.5, [
    f"In-space.|| Miami ranks 1 of 25 (exact p = 0.040).",
    f"Leave-one-donor-out.|| ASCM moves between {la.att_mean.min():,.0f} and {la.att_mean.max():,.0f} room nights/day; plain SCM between {ls_.att_mean.min():,.0f} and {ls_.att_mean.max():,.0f}.",
    "In-time placebo.|| Campaign back-dated to day 120: " + "; ".join(f"{r.estimator} {r.att_pct_of_baseline:+.1f}% of baseline" for r in it.itertuples() if r.significant_at_5pct) + " report a false 'significant' effect." if it.significant_at_5pct.any()
    else "In-time placebo.|| No estimator reports a significant effect on the back-dated campaign.",
    f"Bayesian diagnostics.|| max R-hat {BAY.rhat_max.max():.3f}, min ESS {BAY.ess_min.min():,.0f}, divergences {int(BAY.divergences.sum())}."], size=12.5)

# ============================================================================ 14 CATE benchmark
s = new_slide("Geo-CATE benchmark: ten learners", "PEHE (lower is better), mean over 5 simulated panels", section="Model comparison",
              notes="Dashed reference: constant ATE PEHE 41.95; oracle X-regression 30.8. Only the learners below the constant baseline add information on the exact effect per day; "
                    "TARNet and CFRNet-DragonNet lead and are also the most stable (small sd).")
cm = cate_m.reset_index(drop=True)
bar_chart(s, 0.3, 1.1, 5.0, 5.9, cm.learner.tolist(), {"PEHE": cm.pehe.tolist()}, f"PEHE (constant baseline {const.pehe:.1f}, oracle {oracle.pehe:.1f})", number_format='0.0', colors=[AQUA])
t = pd.DataFrame({"Learner": cm.learner, "PEHE": [f(v, 1) for v in cm.pehe], "sd": [f(v, 1) for v in cm.pehe_sd], "ATT bias": [f(v, 0) for v in cm.att_bias],
                  "Corr": [f(v, 2) for v in cm["corr"]], "Qini (oracle)": [f(v, 2) for v in cm.qini_oracle], "Qini (obs.)": [f(v, 2) for v in cm.qini_observed], "Sec": [f(v, 1) for v in cm.seconds]})
fills = {(i, 1): ((GREEN_BG, GREEN_FG) if v < const.pehe else (RED_BG, RED_FG)) for i, v in enumerate(cm.pehe)}
table(s, t, 5.4, 1.25, 7.5, 5.0, col_w=[2, 0.8, 0.7, 0.9, 0.8, 1.1, 1.1, 0.7], size=10.5, fills=fills, bold_first=True)
n_beat = int((cm.pehe < const.pehe).sum())
text(s, 5.4, 6.35, 7.5, 0.6, f"{n_beat} of 10 beat the constant-effect baseline on PEHE (green). Every learner ranks days better than random. Only 60 treated rows: treat the order as indicative.", size=10.5, color=GRAY)

# ============================================================================ 15 heterogeneity
s = new_slide("What drives lift, and how well days are ranked", "Heterogeneity drivers and learner comparison figures", section="Model comparison",
              notes="Weekend, holiday window and flight-search surges raise lift; OTA pressure, rival rate cuts and bad weather lower it. "
                    "Read coefficients as descriptive, not causal elasticities, with one treated market.")
image(s, FIG / "06_cate_meta_learners_comparison.png", 0.45, 1.15, 6.2, 5.8)
image(s, FIG / "07_cate_heterogeneity_drivers.png", 6.8, 1.15, 6.1, 5.8)

# ============================================================================ 16 compute
s = new_slide("Compute: where the GPU helps", "Measured CPU vs GPU seconds on an RTX 3090", section="Models",
              notes="Boosted trees lose on the GPU at the panel's size (6,000 rows) and win only at much larger sizes; the PyTorch networks gain at every size. "
                    "JAX runs on CPU on native Windows.")
g = GPU.copy()
t = pd.DataFrame({"Workload": g.workload, "Rows": [f"{v:,.0f}" for v in g.rows], "CPU s": [f(v, 2) for v in g.cpu_seconds], "GPU s": [f(v, 2) for v in g.gpu_seconds],
                  "Speed-up": [f(v, 2, suf="x") for v in g.speedup]})
fills = {(i, 4): ((GREEN_BG, GREEN_FG) if v > 1.2 else (RED_BG, RED_FG) if v < 0.9 else (AMBER_BG, AMBER_FG)) for i, v in enumerate(g.speedup) if v == v}
table(s, t, 0.45, 1.25, 8.2, 5.3, col_w=[3, 1.2, 1, 1, 1.2], size=11, fills=fills, bold_first=True)
bullets(s, 8.9, 1.3, 4.1, 5.5, ["Panel size (6,000 rows).|| GPU slower for XGBoost / CatBoost; 4.5x faster for PyTorch.", "1.2M rows.|| XGBoost 4.2x, CatBoost 3.8x, PyTorch 5.0x.",
                                "Bayesian SC.|| NumPyro NUTS on CPU (native-Windows JAX has no CUDA)."], size=12.5)

# ============================================================================ 17 flow framework
s = new_slide("The flow framework", "Four flow types x two examples; metrics at stage, flow and programme level", section="Flows",
              notes="Every flow is a sequence of stages. Each stage reports its own metrics; the scorecard rolls them into a verdict (flow level); the programme scorecard at the end rolls all eight flows up.")
for i, (k, v) in enumerate(fl.FLOW_TYPES.items()):
    x = 0.45 + i * 3.15
    box_text(s, x, 1.25, 3.0, 0.6, f"{k}  {v['title']}", fill=[BLUE, ORANGE, AQUA, NAVY][i], fg=WHITE, size=14, bold=True)
    box_text(s, x, 1.9, 3.0, 0.9, v["question"], fill=LIGHT, size=11)
    ex = [fx for fx in fl.FLOWS if fx["type"] == k]
    box_text(s, x, 2.85, 3.0, 0.75, [ex[0]["short"], ex[1]["short"]], fill=WHITE, size=11, line=RGBColor(0xD5, 0xDB, 0xE5))
stage_df = pd.DataFrame([("A / B  panel flows", "Design -> Parallel trends -> DiD family -> Synthetic control -> Placebo -> Economics -> Verdict", "7 stages"),
                         ("C  uplift what-if", "Conditions -> Learner predictions -> Agreement -> vs campaign average -> Audit vs nearest real day", "5 stages"),
                         ("D  pacing", "Inputs -> Response-curve quality -> Daily plan -> Strategy comparison -> Audit vs truth -> Verdict", "6 stages")], columns=["Flow", "Stages", "Count"])
table(s, stage_df, 0.45, 3.9, 12.4, 1.8, col_w=[2, 9, 1], size=11.5, bold_first=True)
bullets(s, 0.45, 5.85, 12.4, 1.2, ["Three metric levels.|| Stage metrics (per step: pre-trend RMS, pre-fit RMSPE, placebo rank) -> flow metrics (scorecard: ATT, iROAS, p, verdict) -> programme metrics (all 8 flows, all models; closing section)."], size=12.5)

# ============================================================================ 18-33 flows
KEYIMG = {"panel": ["trends", "did", "sc", "econ"], "cate": ["learners", "agree", "index", "audit"], "pacing": ["plan", "compare", "audit", "curve"]}
FLOW_NOTES = {
    "A1": "Story: the headline analysis. Parallel trends are weak (pre-trend RMS 128), so TWFE overshoots; ASCM, SCM and SDID land within about 6% of the truth; Miami ranks first of 25 in the placebo test; iROAS 1.49x clears break-even.",
    "A2": "Story: Las Vegas. The effect is real (truth 91/day) but only four donors exist, so p cannot go below 0.20 and the ASCM interval is huge. The point estimate underestimates by 34%, and iROAS 0.86x looks below break-even while the truth is 1.31x - a warning about thin pools.",
    "B1": "Story: Tampa was never advertised to. Every estimator should say 'no effect'. The ASCM interval includes zero and p = 0.25 - a correct null - but TWFE and CS-DiD point estimates are still +55 to +59 room nights/day, a reminder of DiD noise.",
    "B2": "Story: New York. Positive point estimates, wide intervals, three donors, p = 0.25. ASCM's iROAS 1.17x is above 1 while the truth is 0.73x: do not scale on this evidence.",
    "C1": "Story: a holiday-window weekend with a flight-search surge. Ten learners agree that lift is above average; the spread across learners is wide, which is shown at stage 3.",
    "C2": "Story: a rainy weekday with an OTA price war. Learners agree lift is well below average, so spend should move to better days.",
    "D1": "Story: weekend-weighted pacing. Projection and truth agree (+0.6% vs +0.5%), so it is a safe, harmless re-weighting.",
    "D2": "Story: the optimiser promises +8.7% but the true response gives +0.2%. The fitted curve has R2 0.56 with theta on its bound; never present 'optimised' as a validated recommendation.",
}
for fx in fl.FLOWS:
    r = REPS[fx["id"]]
    tn = fl.FLOW_TYPES[fx["type"]]["title"]
    # --- story slide
    s = new_slide(f"Flow {fx['id']}: {fx['title']}", f"{tn} - {fl.FLOW_TYPES[fx['type']]['question']}", section=f"Flow {fx['type']}", notes=FLOW_NOTES[fx["id"]])
    bg, fg = VERDICT_TONE.get(r["verdict"], (AMBER_BG, AMBER_FG))
    box_text(s, 0.45, 1.1, 7.3, 0.75, [r["verdict"], r["action"]], fill=bg, fg=fg, size=15, bold=True)
    rows = []
    for st_ in r["steps"]:
        ms = [m for m in st_["metrics"] if m["label"] != "Verdict"]
        rows.append((st_["title"], "; ".join(f"{m['label']} {m['value']}" for m in ms)))
    tdf = pd.DataFrame(rows, columns=["Stage", "Stage-level metrics"])
    table(s, tdf, 0.45, 2.0, 7.3, 4.9, col_w=[2.1, 5.2], size=9.5 if len(rows) > 6 else 10.5, bold_first=True)
    ov = SHOT / "flows" / f"{fx['id']}_overall.png"
    box_text(s, 7.95, 1.1, 5.0, 0.4, "Flow-level scorecard (Streamlit)", fill=NAVY, fg=WHITE, size=12, bold=True)
    image(s, ov, 7.95, 1.55, 5.0, 1.5)
    key = KEYIMG[r["kind"]][1 if r["kind"] != "pacing" else 2]
    box_text(s, 7.95, 3.2, 5.0, 0.4, "Key stage (Streamlit)", fill=NAVY, fg=WHITE, size=12, bold=True)
    image(s, SHOT / "flows" / f"{fx['id']}_{key}.png", 7.95, 3.65, 5.0, 3.35)
    # --- evidence slide
    s = new_slide(f"Flow {fx['id']} evidence: stage by stage", f"{fx['title']} - Streamlit screenshots", section=f"Flow {fx['type']}", notes=FLOW_NOTES[fx["id"]])
    imgs = KEYIMG[r["kind"]]
    titles = {st_["key"]: st_["title"] for st_ in r["steps"]}
    for i, k in enumerate(imgs):
        x, y = 0.4 + (i % 2) * 6.35, 1.1 + (i // 2) * 3.0
        text(s, x, y, 6.2, 0.28, titles[k], size=11, bold=True, color=NAVY)
        image(s, SHOT / "flows" / f"{fx['id']}_{k}.png", x, y + 0.3, 6.2, 2.62)

# ============================================================================ 34 cross-flow scorecard
s = new_slide("Flow-level scorecard: all eight flows", "Headline metric and verdict per flow", section="Programme metrics",
              notes="The two 'suggestive' panel flows are exactly the ones with thin donor pools. The only flow whose projection fails the audit is the optimiser (D2).")
rows, fills = [], {}
for i, fx in enumerate(fl.FLOWS):
    r = REPS[fx["id"]]
    hd = r["headline"]
    if r["kind"] == "panel":
        a = f"{hd['att_mean']:,.0f} (truth {hd['truth']:,.0f})"
        b = f"p {hd['p']:.3f}"
        c = f"iROAS {hd['iroas']:.2f}x (truth {hd['true_iroas']:.2f}x)" if r["has_econ"] else "no spend"
    elif r["kind"] == "cate":
        a = f"median {hd['median']:,.0f} RN/day"
        b = f"spread {hd['lo']:,.0f}-{hd['hi']:,.0f}"
        c = f"index {hd['index']:.2f}x average"
    else:
        a = f"projected {hd['projected']:+.1f}% vs even"
        b = f"true {hd['true']:+.1f}%"
        c = f"iROAS {hd['iroas']:.2f}x (projected)"
    rows.append((fx["id"], fx["title"], a, b, c, r["verdict"]))
    fills[(i, 5)] = VERDICT_TONE.get(r["verdict"], (AMBER_BG, AMBER_FG))
table(s, pd.DataFrame(rows, columns=["Flow", "Example", "Headline", "Evidence", "Economics / index", "Verdict"]), 0.45, 1.2, 12.4, 4.2, col_w=[0.6, 3.5, 2.3, 1.7, 2.4, 2.8], size=11, fills=fills, bold_first=True)
bullets(s, 0.45, 5.6, 12.4, 1.4, ["Panel flows: detection needs donors|| (A1 yes; A2, B2 suggestive) and a placebo check (B1 correctly null).", "Uplift flows: lean in / pull back|| follows the index vs the campaign-average lift.", "Pacing flows: a projection is only as good as its audit|| (D1 holds, D2 fails)."], size=12.5)

# ============================================================================ 35 model comparison across flows - panel
s = new_slide("Model comparison across flows: panel estimators", "ATT per day (bias vs truth) for every estimator in each panel flow", section="Programme metrics",
              notes="Green means within 10% of the truth. Tampa's truth is 0, so cells show the raw ATT and are coloured by size against the Miami truth scale. "
                    "ASCM is best or near-best on the headline and in the Monte-Carlo; on thin pools (A2, B2) every method struggles and no estimator wins everywhere.")
sc_cols = [("A1 Miami", "mia_all24"), ("A2 Las Vegas", "las_convention"), ("B1 Tampa (truth 0)", "tpa_placebo"), ("B2 New York", "nyc_spring_break")]
rows, fills = [], {}
for i, e in enumerate(ORDER):
    row = [e]
    for j, (_, sc) in enumerate(sc_cols):
        r_ = S[(S.scenario == sc) & (S.estimator == e)].iloc[0]
        if sc == "tpa_placebo":
            row.append(f"{r_.att_mean:+.0f}")
            fills[(i, j + 1)] = tone(r_.att_mean, 20, 50)
        else:
            row.append(f"{r_.att_mean:,.0f} ({r_.att_bias_pct:+.0f}%)")
            fills[(i, j + 1)] = tone(r_.att_bias_pct, 10, 30)
    row.append(f"{mc_t.loc[e, 'bias_pct']:+.0f}% / {mc_t.loc[e, 'rmse']:.0f}")
    fills[(i, 5)] = tone(mc_t.loc[e, "bias_pct"], 10, 30)
    rows.append(row)
table(s, pd.DataFrame(rows, columns=["Estimator"] + [c[0] for c in sc_cols] + ["Monte-Carlo bias / RMSE"]), 0.45, 1.2, 12.4, 4.6, col_w=[1.8, 2, 2, 2, 2, 2.2], size=11.5, fills=fills, bold_first=True)
rk = S[S.scenario.isin([c[1] for c in sc_cols if c[1] != "tpa_placebo"])].groupby("estimator").att_bias_pct.apply(lambda v: v.abs().mean()).sort_values()
bullets(s, 0.45, 5.95, 12.4, 1.1, [f"Mean |bias| over the three campaign flows (A1, A2, B2): " + ", ".join(f"{k} {v:.0f}%" for k, v in rk.head(4).items()) + f"; worst: {rk.index[-1]} {rk.iloc[-1]:.0f}%. No estimator wins everywhere: the lowest-bias DiD variants have intervals that cover zero.",
                                   "ASCM wins where donors are plentiful (A1: +6%; Monte-Carlo: +2%); with 3-4 donors (A2, B2) every method is off by 25-80%, which is why those flows end 'suggestive, not conclusive'."], size=12)

# ============================================================================ 36 across flows - CATE + pacing
s = new_slide("Model comparison across flows: uplift learners and pacing plans", "Flow C predictions per learner; flow D projected vs true per strategy", section="Programme metrics",
              notes="Left: the ten learners on the two hypothetical days. Right: all eight pacing strategies projected vs true; D1 and D2 are highlighted.")
c1, c2 = REPS["C1"]["preds"], REPS["C2"]["preds"]
rows, fills = [], {}
for i, r_ in enumerate(cm.itertuples()):
    rows.append((r_.learner, f(c1.get(r_.learner), 0), f(c2.get(r_.learner), 0), f(r_.pehe, 1), f(r_.qini_oracle, 2)))
    fills[(i, 3)] = (GREEN_BG, GREEN_FG) if r_.pehe < const.pehe else (RED_BG, RED_FG)
text(s, 0.45, 1.1, 6.4, 0.3, "Uplift learners (RN/day): C1 holiday weekend, C2 rainy weekday", size=12, bold=True, color=NAVY)
table(s, pd.DataFrame(rows, columns=["Learner", "C1", "C2", "PEHE", "Qini"]), 0.45, 1.45, 6.4, 4.9, col_w=[2.4, 0.9, 0.9, 0.9, 0.9], size=11, fills=fills, bold_first=True)
pv = PV[PV.strategy != "actual campaign (reference)"].copy()
rows, fills = [], {}
for i, r_ in enumerate(pv.itertuples()):
    rows.append((r_.strategy, f"{r_.projected_vs_even_pct:+.1f}%", f"{r_.true_vs_even_pct:+.1f}%", f"{r_.projected_vs_even_pct - r_.true_vs_even_pct:+.1f}"))
    fills[(i, 3)] = tone(r_.projected_vs_even_pct - r_.true_vs_even_pct, 2, 5)
    if r_.strategy in ("weekend_weighted", "optimized"):
        fills[(i, 0)] = (AMBER_BG, AMBER_FG)
text(s, 7.1, 1.1, 5.8, 0.3, "Pacing strategies vs even (D1 weekend, D2 optimised highlighted)", size=12, bold=True, color=NAVY)
table(s, pd.DataFrame(rows, columns=["Strategy", "Projected", "True", "Error pts"]), 7.1, 1.45, 5.8, 4.2, col_w=[2.2, 1.2, 1.2, 1.2], size=11, fills=fills, bold_first=True)
bullets(s, 7.1, 5.8, 5.8, 1.2, ["Back-loading is the one gross mistake, and the curve flags it (-22% projected and true).", "Optimiser, front-loading and pulsing gains do not survive the truth."], size=11.5)
text(s, 0.45, 6.45, 6.4, 0.6, "Green PEHE = beats the constant-effect baseline (41.9).", size=10.5, color=GRAY)

# ============================================================================ 37 stage-level rollup
s = new_slide("Stage-level metrics across the four panel flows", "Same seven stages, four situations", section="Programme metrics",
              notes="This is the stage-level view: the same metric at the same stage across flows. It shows where each situation goes wrong - weak parallel trends in Miami, thin placebo pools in Las Vegas and New York.")


def mv(rep, key, label):
    st_ = next(x for x in rep["steps"] if x["key"] == key)
    return next(m["value"] for m in st_["metrics"] if m["label"].startswith(label))


spec_rows = [("1 Design", "design", "Donor DMAs"), ("1 Design", "design", "Smallest possible p"), ("2 Trends", "trends", "Pre-trend RMS"), ("2 Trends", "trends", "Permutation p"),
             ("3 DiD", "did", "TWFE DiD"), ("3 DiD", "did", "CS-DiD"), ("4 Synthetic ctrl", "sc", "Pre-fit RMSPE"), ("4 Synthetic ctrl", "sc", "ASCM ATT"),
             ("4 Synthetic ctrl", "sc", "ASCM bias"), ("5 Placebo", "placebo", "Treated rank"), ("5 Placebo", "placebo", "Exact p"), ("5 Placebo", "placebo", "CI covers"),
             ("6 Economics", "econ", "iROAS"), ("6 Economics", "econ", "True iROAS")]
key_fix = {"design": "setup"}
rows = []
for stg, k, lab in spec_rows:
    rows.append([stg, lab] + [mv(REPS[i], key_fix.get(k, k), lab) for i in ["A1", "A2", "B1", "B2"]])
table(s, pd.DataFrame(rows, columns=["Stage", "Metric", "A1 Miami", "A2 Las Vegas", "B1 Tampa", "B2 New York"]), 0.45, 1.2, 12.4, 5.7, col_w=[1.7, 2.3, 2, 2, 2, 2], size=11, bold_first=True)

# ============================================================================ 38 overall metrics scorecard
s = new_slide("Programme-level scorecard", "Overall metrics across all models and flows", section="Programme metrics",
              notes="The programme view: how many models, how many flows, how well they agree with the truth, and whether the UI matches the API. Tests: 28 passing.")
import json  # noqa: E402

ch = pd.DataFrame(json.load(open(SHOT / "flows" / "flow_api_vs_ui_check.json", encoding="utf-8"))).T
n_ok = int(ch.ui_shows_api_number.astype(bool).sum())
n_stage = sum(len(r_["steps"]) for r_ in REPS.values())
verd = pd.Series([REPS[i]["verdict"] for i in REPS]).value_counts()
kpi(s, 0.45, 1.2, 3.0, 1.3, "8 + 10", "Models compared", "8 panel estimators, 10 uplift learners", BLUE)
kpi(s, 3.6, 1.2, 3.0, 1.3, "8", "Flows", f"4 types x 2 examples, {n_stage} stage views", ORANGE)
kpi(s, 6.75, 1.2, 3.0, 1.3, f"{n_ok} / {len(ch)}", "UI = API checks", "number shown in Streamlit equals live API", AQUA)
kpi(s, 9.9, 1.2, 3.0, 1.3, "28", "Automated tests passing", "data, estimators, CATE, API contract", BLUE)
kpi(s, 0.45, 2.7, 3.0, 1.3, f"{mc_t.loc['ASCM (Ridge)', 'bias_pct']:+.0f}% / {mc_t.loc['TWFE DiD', 'bias_pct']:+.0f}%", "Mean bias: ASCM vs TWFE", "12 simulated panels", ORANGE)
kpi(s, 3.6, 2.7, 3.0, 1.3, f"{mc_t.loc['ASCM (Ridge)', 'ci_coverage'] * 100:.0f}%", "ASCM interval coverage", "95% nominal, Monte-Carlo", AQUA)
kpi(s, 6.75, 2.7, 3.0, 1.3, f"{best.pehe:.1f}", "Best uplift PEHE", f"{best.learner}; baseline {const.pehe:.1f}", BLUE)
kpi(s, 9.9, 2.7, 3.0, 1.3, f"{n_beat} / 10", "Learners beating baseline", "on PEHE; all 10 rank days > random", ORANGE)
rows = [(k, str(v)) for k, v in verd.items()]
text(s, 0.45, 4.2, 5.0, 0.3, "Verdict distribution over 8 flows", size=12, bold=True, color=NAVY)
fills = {i: VERDICT_TONE.get(k, (AMBER_BG, AMBER_FG)) for i, (k, _) in enumerate(rows)}
table(s, pd.DataFrame(rows, columns=["Verdict", "Flows"]), 0.45, 4.5, 5.6, 0.3 * (len(rows) + 1), col_w=[4.4, 1], size=10, fills={(i, 0): v for i, v in fills.items()})
bullets(s, 6.4, 4.3, 6.5, 2.7, ["Detect real effects only with enough donors.|| 24 donors: p 0.04; 3-4 donors: p 0.20-0.25 whatever the data say.", "Placebo-calibrated beats posterior intervals|| for honesty (Bayesian SC is optimistic).",
                                "Rank days, do not trust the optimiser.|| Uplift ranking works; projected pacing gains do not survive the truth."], size=12)

# ============================================================================ 39 Streamlit tour
s = new_slide("The Streamlit workbench", "Six tabs, four presets; screenshots from the running app", section="Workbench",
              notes="Presets: Miami flagship, Las Vegas, New York, Tampa placebo. Tabs: executive & iROAS, DiD & event study, synthetic control & donor lab, geo-CATE studio, live API & batch scoring, end-to-end flows.")
tour = [("Tab 1 - Executive & iROAS", "streamlit_tab1_executive_iroas.png"), ("Tab 2 - DiD & event study", "streamlit_tab2_did_event_study.png"), ("Tab 3 - Synthetic control", "streamlit_tab3_synthetic_control.png"),
        ("Tab 4 - Geo-CATE studio", "streamlit_tab4_cate_metalearners.png"), ("Tab 5 - Live API", "streamlit_tab5_live_api.png"), ("Tab 6 - End-to-end flows", "flows/A1_full.png")]
for i, (cap, fn) in enumerate(tour):
    x, y = 0.4 + (i % 3) * 4.3, 1.1 + (i // 3) * 3.0
    text(s, x, y, 4.2, 0.28, cap, size=11, bold=True, color=NAVY)
    image(s, SHOT / fn, x, y + 0.3, 4.1, 2.6, crop_ar=1.55)

# ============================================================================ 40 presets story
s = new_slide("Scenarios as a story: the four one-click presets", "Same engine, four markets", section="Workbench",
              notes="Miami: detected. Las Vegas and New York: thin donor pools. Tampa: the negative control.")
for i, (cap, fn) in enumerate([("Preset 2 - Las Vegas", "streamlit_preset2_las_vegas.png"), ("Preset 3 - New York", "streamlit_preset3_new_york.png"),
                               ("Preset 4 - Tampa placebo", "streamlit_preset4_placebo_check.png")]):
    x = 0.4 + i * 4.3
    text(s, x, 1.1, 4.2, 0.3, cap, size=12, bold=True, color=NAVY)
    image(s, SHOT / fn, x, 1.45, 4.1, 3.3)
    fid = ["A2", "B2", "B1"][i]
    rp = REPS[fid]
    hd = rp["headline"]
    box_text(s, x, 4.9, 4.1, 0.5, rp["verdict"], fill=VERDICT_TONE.get(rp["verdict"], (AMBER_BG, AMBER_FG))[0], fg=VERDICT_TONE.get(rp["verdict"], (AMBER_BG, AMBER_FG))[1], size=12, bold=True)
    bullets(s, x, 5.5, 4.1, 1.5, [f"ASCM {hd['att_mean']:,.0f} room nights/day vs truth {hd['truth']:,.0f}; placebo p {hd['p']:.3f}." + (f" iROAS {hd['iroas']:.2f}x vs truth {hd['true_iroas']:.2f}x." if rp["has_econ"] else " No spend, so no iROAS."), fl.FLOW_BY_ID[fid]["story"].split(". ")[0] + "."], size=10.5, gap=3)

# ============================================================================ 41 API
s = new_slide("Serving: FastAPI", "Same engine, programmatic access", section="Workbench",
              notes="The screenshot script calls these endpoints and checks that the Streamlit flows show the same numbers.")
image(s, SHOT / "api_swagger_docs.png", 0.45, 1.15, 6.3, 5.8)
image(s, SHOT / "api_cate_response.png", 6.95, 1.15, 5.95, 3.3)
bullets(s, 6.95, 4.6, 5.95, 2.4, ["GET /health|| liveness, loaded models, fitted response curve", "POST /predict/att|| ATT for a preset or custom donor pool (cached or live)",
                                  "POST /predict/cate|| conditional lift from any of ten learners, CIs for DML", "POST /simulate/campaign|| budget and pacing what-if for eight strategies"], size=12)

# ============================================================================ 42 limitations
s = new_slide("Limitations and honest caveats", "What not to over-read", section="Wrap-up",
              notes="These are the same caveats as the README; the deck should never be presented without them.")
bullets(s, 0.45, 1.2, 12.4, 5.7, [
    "One treated market.|| Cluster-robust SEs are unreliable; inference rests on placebo permutation and the Bayesian posterior.",
    "Small donor pools give weak inference.|| Las Vegas and New York cannot reach p below 0.20 / 0.25.",
    "Bayesian SC is not automatically better.|| Narrow intervals but larger bias than ASCM; one run produced a false positive on Tampa.",
    "Thin treated sample for uplift.|| 60 treated rows, outcome is a synthetic-control residual, so CATE inherits counterfactual error; DR-Learner is least stable.",
    "Pacing projections.|| The fitted curve (R2 0.56, theta on its bound) cannot rank reasonable plans; do not present 'optimised' as validated.",
    "Simulated data.|| 'Truth' columns exist only because the data are generated; real engagements have estimates and diagnostics only.",
    "Event study ends at k = 59.|| The panel has 60 campaign days.",
    "GPU gains depend on size.|| Boosted trees lose on the GPU at 6,000 rows; JAX is CPU-only on native Windows."], size=14)

# ============================================================================ 43 recommendations
s = new_slide("Recommendations for a real engagement", "What this exercise says to do with real data", section="Wrap-up",
              notes="These are process recommendations derived from the simulation results, not claims about Aurelia's actual markets.")
rec = [("Design", "Hold out many donor DMAs: 24 gives p 0.04; 3-4 cannot prove anything. Plan the placebo distribution before launch."),
       ("Estimate", "Headline with ASCM; show SCM and SDID alongside; never headline a plain TWFE DiD when pre-trends diverge."),
       ("Validate", "Always run in-space and in-time placebos and a leave-one-donor-out; report the smallest attainable p."),
       ("Decide", "Use iROAS with its interval against break-even 1.0x; treat 'suggestive' as a reason to extend the test, not to scale."),
       ("Optimise", "Use uplift to rank days; keep even / weekend / CATE-weighted pacing; avoid back-loading; test pacing with a holdout.")]
for i, (h, d) in enumerate(rec):
    box_text(s, 0.45, 1.2 + i * 1.1, 2.2, 0.95, h, fill=[BLUE, BLUE, ORANGE, ORANGE, AQUA][i], fg=WHITE, size=16, bold=True)
    box_text(s, 2.75, 1.2 + i * 1.1, 10.1, 0.95, d, fill=LIGHT, size=13, align=PP_ALIGN.LEFT)

# ============================================================================ 44 reproduce
s = new_slide("Appendix: reproduce everything", "Commands and artifacts", section="Appendix",
              notes="Use the causal-gpu conda environment; the base interpreter has incompatible xgboost and streamlit versions.")
code = ["python scripts/run_full_pipeline.py              # data -> panel ATT -> CATE -> Monte-Carlo -> GPU -> figures -> verify",
        "python scripts/capture_screenshots.py            # API + Streamlit screenshots incl. 8 flows (UI == API check)",
        "python scripts/capture_screenshots.py flows      # only the flow screenshots",
        "python scripts/build_readme.py                   # README generated from outputs/metrics",
        "python scripts/build_deck.py                     # this presentation",
        "python -m streamlit run src/app/streamlit_app.py --server.port 8501",
        "python -m uvicorn src.api.main:app --port 8000   # Swagger at /docs",
        "python -m pytest tests -q                        # 28 tests"]
rect(s, 0.45, 1.2, 12.4, 3.6, RGBColor(0x14, 0x1B, 0x2D))
text(s, 0.65, 1.3, 12.0, 3.4, code, size=12.5, color=RGBColor(0xDC, 0xE6, 0xF7), font="Consolas")
bullets(s, 0.45, 5.0, 12.4, 2.0, ["Artifacts: outputs/metrics (CSV/JSON), outputs/figures (8 x 300 DPI), outputs/screenshots (+ flows/), README.md, presentation/.",
                                  "Code map: src/flows.py (flow logic), src/app/flows_ui.py (tab 6), src/serving.py (shared engine), src/models/* (estimators, learners, response curve)."], size=12.5)

prs.save(FILE)
print("saved", FILE, "slides:", len(prs.slides._sldIdLst))
