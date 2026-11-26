"""Build the end-to-end presentation (native, editable .pptx) from the saved metrics, flow report and screenshots.

    python scripts/build_deck.py     ->  presentation/Loyalty_Gold_RDD_Uplift_End_to_End.pptx

Every number comes from outputs/metrics/* (leaderboard, Optuna trials, flows_report.json) or from the flows module that the API
and the Streamlit tab also use; only the descriptions of the models and the narrative are written here. Slides carry speaker notes.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lxml import etree  # noqa: E402
from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.chart.data import CategoryChartData, XyChartData  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

from src import flows as fl  # noqa: E402

M, PLOTS, SHOTS = ROOT / "outputs/metrics", ROOT / "outputs/plots", ROOT / "outputs/screenshots"
OUT = ROOT / "presentation"
OUT.mkdir(exist_ok=True)
FILE = OUT / "Loyalty_Gold_RDD_Uplift_End_to_End.pptx"
ASSETS = Path(tempfile.mkdtemp(prefix="deck_assets_"))

# ----------------------------------------------------------------------------- theme
NAVY, BLUE, ORANGE, AQUA = RGBColor(0x14, 0x21, 0x3D), RGBColor(0x25, 0x7D, 0xE0), RGBColor(0xF4, 0x72, 0x30), RGBColor(0x00, 0xB8, 0x94)
INK, GRAY, MID, LIGHT, WHITE = RGBColor(0x1B, 0x1B, 0x1B), RGBColor(0x5F, 0x66, 0x73), RGBColor(0xC9, 0xD1, 0xDE), RGBColor(0xF2, 0xF5, 0xFA), RGBColor(0xFF, 0xFF, 0xFF)
TINT, GRID, AMBER = RGBColor(0xE3, 0xEE, 0xFC), RGBColor(0xE4, 0xE8, 0xEF), RGBColor(0xF2, 0xA6, 0x00)
SOFT = {"PASS": (RGBColor(0xDD, 0xF3, 0xE4), RGBColor(0x0B, 0x6B, 0x2B)), "WARN": (RGBColor(0xFF, 0xF0, 0xCC), RGBColor(0x8A, 0x5A, 0x00)),
        "FAIL": (RGBColor(0xFC, 0xDD, 0xDD), RGBColor(0x9B, 0x1C, 0x1C)), "INFO": (RGBColor(0xEC, 0xEF, 0xF4), RGBColor(0x55, 0x5E, 0x6C))}
HEAD, BODY = "Cambria", "Calibri"
SW, SH = 13.333, 7.5
LM = 0.6                      # left/right margin

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
BLANK = prs.slide_layouts[6]
_count = [0]

R = json.loads((M / "flows_report.json").read_text(encoding="utf-8"))
FLOWS = {f["id"]: f for f in R["flows"]}
MF = R["model_flow"]
PROG = R["programme"]
ASSUME = R["assumptions"]
META = json.loads((ROOT / "outputs/models/meta.json").read_text(encoding="utf-8"))
LB = pd.read_csv(M / "cate_leaderboard_room_nights.csv")
LB = LB[LB["status"] == "ok"].copy()
PRETTY = {"s_learner": "S-Learner", "t_learner": "T-Learner", "x_learner": "X-Learner", "r_learner": "R-Learner", "dr_learner": "DR-Learner",
          "linear_dml": "LinearDML", "nonparam_dml": "NonParamDML", "causal_forest_dml": "CausalForestDML",
          "orthogonal_forest": "Orthogonal Forest", "dmliv": "DMLIV", "tarnet": "TARNet", "dragonnet": "DragonNet"}
LBI = LB.set_index("learner")
CODE = {"s_learner": "S", "t_learner": "T", "x_learner": "X", "r_learner": "R", "dr_learner": "DR", "linear_dml": "Lin", "nonparam_dml": "NP",
        "causal_forest_dml": "CF", "orthogonal_forest": "ORF", "dmliv": "IV", "tarnet": "TAR", "dragonnet": "DRG"}


# ----------------------------------------------------------------------------- low-level helpers
def rgb_hex(c: RGBColor) -> str:
    return str(c)


def tb(slide, x, y, w, h, paras, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=BODY,
       italic=False, space_after=0, line=None, margin=0.0):
    """Text box. ``paras`` is a string or a list of strings / dicts {t, size, bold, color, bullet, runs, space_after, italic}."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    if isinstance(paras, str):
        paras = [paras]
    for i, item in enumerate(paras):
        d = {"t": item} if isinstance(item, str) else dict(item)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = d.get("align", align)
        p.space_after = Pt(d.get("space_after", space_after))
        if line:
            p.line_spacing = line
        runs = d.get("runs") or [(d.get("t", ""), d.get("bold", bold), d.get("color", color))]
        for text, b, col in runs:
            r = p.add_run()
            r.text = text
            r.font.size = Pt(d.get("size", size))
            r.font.bold = b
            r.font.italic = d.get("italic", italic)
            r.font.name = d.get("font", font)
            r.font.color.rgb = col
        if d.get("bullet"):
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.22))))
            pPr.set("indent", str(-int(Inches(0.22))))
            for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
                for e in pPr.findall(qn(tag)):
                    pPr.remove(e)
            bu = etree.SubElement(pPr, qn("a:buChar"))
            bu.set("char", "•")
    return box


def shape(slide, x, y, w, h, fill=LIGHT, line=None, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, name=None):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    s.shadow.inherit = False
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if name:
        s.name = name
    s.text_frame.text = ""
    return s


def label(slide, x, y, w, h, text, fill, color, size=12, bold=True, align=PP_ALIGN.CENTER, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.2, font=BODY):
    s = shape(slide, x, y, w, h, fill, kind=kind, radius=radius)
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
    r.font.color.rgb = color
    return s


def arrow(slide, x1, y1, x2, y2, color=GRAY, width=2.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


def notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def img(path: Path, max_w: int = 1900) -> str:
    """Downscale a screenshot/plot into the temp asset folder (keeps the deck small) and return the new path."""
    if ASSETS in Path(path).parents:
        return str(path)
    out = ASSETS / (path.parent.name + "_" + path.name)
    if not out.exists():
        im = Image.open(path)
        if im.width > max_w:
            im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
        im.save(out, optimize=True)
    return str(out)


def top_crop(path: Path, aspect: float, key: str) -> Path:
    """Keep the top part of a tall screenshot (page header, tabs, first content) at the given width/height ratio."""
    im = Image.open(path)
    w, h = im.size
    out = ASSETS / f"crop_{key}.png"
    im.crop((0, 0, w, min(h, int(w / aspect)))).save(out, optimize=True)
    return out


def picture(slide, path: Path, x, y, w, h, frame=True, align="center", max_w=1900):
    """Place an image inside the box (x,y,w,h) keeping its aspect ratio; returns (placed_w, placed_h)."""
    p = img(path, max_w)
    iw, ih = Image.open(p).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px = x + (w - pw) / 2 if align == "center" else x
    py = y + (h - ph) / 2 if align == "center" else y
    if frame:
        shape(slide, px - 0.05, py - 0.05, pw + 0.1, ph + 0.1, WHITE, MID, MSO_SHAPE.RECTANGLE)
    slide.shapes.add_picture(p, Inches(px), Inches(py), Inches(pw), Inches(ph))
    return pw, ph


# ----------------------------------------------------------------------------- slide scaffolding
def new_slide(kicker: str = "", title: str = "", dark: bool = False, number: bool = True):
    s = prs.slides.add_slide(BLANK)
    _count[0] += 1
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY if dark else WHITE
    if kicker:
        tb(s, LM, 0.32, 9, 0.3, kicker.upper(), size=11, bold=True, color=ORANGE if dark else BLUE)
    if title:
        tb(s, LM, 0.58, SW - 2 * LM, 0.95, title, size=28, bold=True, color=WHITE if dark else NAVY, font=HEAD, anchor=MSO_ANCHOR.TOP, line=0.95)
    if number and not dark:
        tb(s, LM, 7.05, 8, 0.25, "Loyalty Gold RDD + uplift engine", size=10, color=GRAY)
        tb(s, SW - LM - 1.0, 7.05, 1.0, 0.25, str(_count[0]), size=10, color=GRAY, align=PP_ALIGN.RIGHT)
    return s


def card(slide, x, y, w, h, title, body, fill=LIGHT, title_color=NAVY, body_size=13, title_size=15, body_color=INK):
    shape(slide, x, y, w, h, fill)
    tb(slide, x + 0.2, y + 0.15, w - 0.4, 0.4, title, size=title_size, bold=True, color=title_color)
    tb(slide, x + 0.2, y + 0.58, w - 0.4, h - 0.7, body, size=body_size, color=body_color, space_after=4)


def stat(slide, x, y, w, h, value, caption, color=BLUE, fill=LIGHT, vsize=36, csize=12):
    shape(slide, x, y, w, h, fill)
    tb(slide, x + 0.15, y + 0.12, w - 0.3, h * 0.55, value, size=vsize, bold=True, color=color, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
    tb(slide, x + 0.15, y + h * 0.62, w - 0.3, h * 0.36, caption, size=csize, color=GRAY)


def verdict_colors(v: str):
    if v in fl.GOOD:
        return SOFT["PASS"]
    if v in fl.BAD:
        return SOFT["FAIL"]
    return SOFT["WARN"]


def _cell_border(cell, color="D6DCE6", w=9525, sides="b"):
    tcPr = cell._tc.get_or_add_tcPr()
    for side in reversed(sides):                       # schema order is lnL, lnR, lnT, lnB, then the fill
        tag = {"l": "a:lnL", "r": "a:lnR", "t": "a:lnT", "b": "a:lnB"}[side]
        ln = etree.Element(qn(tag), w=str(w), cap="flat", cmpd="sng", algn="ctr")
        sf = etree.SubElement(ln, qn("a:solidFill"))
        etree.SubElement(sf, qn("a:srgbClr"), val=color)
        etree.SubElement(ln, qn("a:prstDash"), val="solid")
        tcPr.insert(0, ln)


def table(slide, x, y, col_w, rows, row_h=0.4, size=12, hsize=12, header=True, fills=None, colors=None, bolds=None, aligns=None,
          header_fill=NAVY, first_col_bold=False, zebra=False, row_heights=None):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * nr))
    t = shp.table
    tblPr = t._tbl.tblPr
    for a in ("firstRow", "bandRow"):
        tblPr.set(a, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"          # No Style, No Grid
    for j, w in enumerate(col_w):
        t.columns[j].width = Inches(w)
    for i in range(nr):
        t.rows[i].height = Inches(row_heights[i] if row_heights else row_h)
        for j in range(nc):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.07)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            is_h = header and i == 0
            fill = header_fill if is_h else (fills or {}).get((i, j), (LIGHT if (zebra and i % 2 == 0) else None))
            if fill is not None:
                c.fill.solid()
                c.fill.fore_color.rgb = fill
            else:
                c.fill.background()
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (aligns or {}).get(j, PP_ALIGN.LEFT) if not is_h else (aligns or {}).get(j, PP_ALIGN.LEFT)
            r = p.add_run()
            r.text = str(rows[i][j])
            r.font.name = BODY
            r.font.size = Pt(hsize if is_h else size)
            r.font.bold = True if is_h else bool((bolds or {}).get((i, j), first_col_bold and j == 0))
            r.font.color.rgb = WHITE if is_h else (colors or {}).get((i, j), INK)
            if not is_h:
                _cell_border(c)
    return shp


def style_chart(ch, size=11, legend=None):
    ch.font.size = Pt(size)
    ch.font.name = BODY
    ch.font.color.rgb = GRAY
    ch.has_title = False
    if legend:
        ch.has_legend = True
        ch.legend.position = legend
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(size)
    else:
        ch.has_legend = False


def bar_chart(slide, x, y, w, h, cats, series, colors, horizontal=False, fmt="0.00", legend=None, labels=True, size=11,
              vmin=None, vmax=None, gap=55, point_colors=None, overlap=None, stacked=False):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series.items():
        cd.add_series(name, vals)
    kind = (XL_CHART_TYPE.BAR_STACKED if stacked else XL_CHART_TYPE.BAR_CLUSTERED) if horizontal else \
           (XL_CHART_TYPE.COLUMN_STACKED if stacked else XL_CHART_TYPE.COLUMN_CLUSTERED)
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    style_chart(ch, size, legend)
    plot = ch.plots[0]
    plot.gap_width = gap
    if overlap is not None:
        plot.overlap = overlap
    for s, col in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = col
        s.invert_if_negative = False
    if point_colors:
        s0 = plot.series[0]
        for i, col in enumerate(point_colors):
            pt = s0.points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = col
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format = fmt
        dl.number_format_is_linked = False
        dl.font.size = Pt(size - 1)
        dl.font.color.rgb = INK
        dl.position = XL_LABEL_POSITION.CENTER if stacked else XL_LABEL_POSITION.OUTSIDE_END
    va, ca = ch.value_axis, ch.category_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = GRID
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size - 1)
    va.tick_labels.number_format = fmt
    va.tick_labels.number_format_is_linked = False
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    ca.has_major_gridlines = False
    ca.tick_labels.font.size = Pt(size)
    ca.format.line.color.rgb = MID
    return ch


def money(v, d=0):
    return fl._money(v, d)


# ============================================================================= slides
def slide_title():
    s = new_slide(dark=True, number=False)
    tb(s, LM, 0.9, 7.0, 0.35, "CAUSAL INFERENCE · UPLIFT MODELLING · HOTEL LOYALTY", size=12, bold=True, color=ORANGE)
    tb(s, LM, 1.45, 6.9, 2.6, "Does Gold status change how guests behave?", size=44, bold=True, color=WHITE, font=HEAD, line=0.95)
    tb(s, LM, 4.15, 6.7, 1.3, "An end-to-end causal and uplift engine: regression discontinuity at 20 nights, 12 heterogeneous-effect models, "
       "and eight worked decision flows", size=18, color=MID, line=1.05)
    tb(s, LM, 6.3, 7, 0.6, f"All data synthetic with a known truth · {META['n_panel']:,} members · GPU-trained on {META['hardware'].get('gpu_name', 'CPU')}",
       size=12, color=MID)
    # real RDD curves from the A1 flow as the visual
    ch = next(st for st in FLOWS["A1"]["stages"] if st["key"] == "estimates")["chart"]["outcomes"]["next_year_room_nights"]
    cd = XyChartData()
    for name, side in (("Below the Gold threshold", "left"), ("At or above 20 nights", "right")):
        ser = cd.add_series(name)
        for xv, yv in zip(ch[side]["x"], ch[side]["f"]):
            ser.add_data_point(xv, yv)
    gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(7.9), Inches(1.2), Inches(4.9), Inches(4.7), cd)
    c = gf.chart
    style_chart(c, 11, XL_LEGEND_POSITION.BOTTOM)
    c.font.color.rgb = MID
    c.legend.font.color.rgb = MID
    for ser, col in zip(c.plots[0].series, (BLUE, ORANGE)):
        ser.format.line.color.rgb = col
        ser.format.line.width = Pt(4)
        ser.smooth = False
    va, ca = c.value_axis, c.category_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0x2B, 0x3A, 0x5C)
    va.format.line.fill.background()
    ca.format.line.color.rgb = MID
    ca.minimum_scale, ca.maximum_scale = 4, 36
    va.tick_labels.font.size = ca.tick_labels.font.size = Pt(10)
    va.has_title = ca.has_title = True
    va.axis_title.text_frame.text = "Next-year room nights"
    ca.axis_title.text_frame.text = "Qualifying nights in year t"
    for ax in (va, ca):
        r = ax.axis_title.text_frame.paragraphs[0].runs[0]
        r.font.size, r.font.color.rgb, r.font.bold = Pt(11), MID, False
    notes(s, "Opening. The question: do Gold perks cause guests to stay and spend more, or do high-tier guests simply spend more anyway? "
             "The chart is the real fuzzy-RDD fit from the headline flow: next-year room nights jump at 20 qualifying nights.")


def slide_exec():
    a1 = FLOWS["A1"]
    est = next(st for st in a1["stages"] if st["key"] == "estimates")["table"]
    s = new_slide("Executive summary", "Gold causes about +3.7 room nights, and it pays for its perks")
    vals = [(est[0]["Estimate"], "extra room nights (95% CI " + est[0]["95% CI"] + ")"), (est[1]["Estimate"], "ancillary spend per Gold member"),
            (est[2]["Estimate"], "direct-booking share"), (money(a1["kpi"]["net_per_member"]), f"net per Gold member per year at {ASSUME['contribution_margin']:.0%} margin")]
    for i, (v, cap) in enumerate(vals):
        stat(s, LM + i * 3.07, 1.65, 2.9, 1.55, v, cap, color=BLUE if i < 3 else AQUA, vsize=34, csize=12)
    best = LB.sort_values("pehe_rmse").iloc[0]
    c1, c2 = FLOWS["C1"]["kpi"], FLOWS["C2"]["kpi"]
    d1, d2 = FLOWS["D1"]["kpi"], FLOWS["D2"]["kpi"]
    cols = [
        ("Can we trust it?", [f"Density, balance, robustness and placebo checks pass on the clean design (3 of 3 gates).",
                              f"A gamed threshold is caught in {PROG['groups'][2]['metrics'][5]['value']} of re-draws.",
                              f"Estimates land within 1 standard error of the known truth for ancillary and direct share; room nights is {a1['kpi']['per_outcome']['next_year_room_nights']['err_se']:+.1f} SE."]),
        ("Who responds?", [f"{c1['effect']:+.1f} nights for a bleisure couple vs {c2['effect']:+.1f} for a corporate road warrior.",
                           f"12 models trained and tuned; {int((LB['pearson'] >= 0.85).sum())} of 12 correlate 0.85+ with the true individual effect.",
                           f"Best single model by error: {PRETTY[best['learner']]} (RMSE {best['pehe_rmse']:.2f} nights)."]),
        ("What should we do?", [f"Keep Gold at 20 nights; lowering to 17 adds about {money(d1['net_margin'])} a year at ${ASSUME['perk_cost_per_member']:.0f} perks.",
                                f"Tighten only if perks cost more than about {money(d2['break_even_cost'])} per member.",
                                "Steer proactive offers to leisure and bleisure guests, not corporate."])]
    for i, (t, bl) in enumerate(cols):
        x = LM + i * 4.1
        shape(s, x, 3.55, 3.95, 3.3, WHITE, MID)
        tb(s, x + 0.2, 3.7, 3.55, 0.4, t, size=17, bold=True, color=NAVY, font=HEAD)
        tb(s, x + 0.2, 4.2, 3.55, 2.6, [{"t": b, "bullet": True, "space_after": 6} for b in bl], size=14)
    notes(s, "The answer in one slide. All numbers are generated from the saved artifacts. The room-night estimate is a little above the known truth of "
             f"{a1['kpi']['true_effect']:+.2f}; the interval covers it. Money is on a revenue basis scaled by an assumed margin, so treat the dollar figures as illustrative.")


def slide_agenda():
    s = new_slide("Storyline", "Four parts, from method to decisions")
    parts = [("1", "Problem, approach and the uplift flow", "Why a discontinuity, the end-to-end pipeline documented stage by stage, how uplift is measured, architecture."),
             ("2", "The 12 models and how they compare", "RDD evidence, every model explained, tuning, leaderboard, and the comparison across the four model-driven flows."),
             ("3", "Flows, stages and metrics", "Eight flows in four types, stage-level gates, flow-level verdicts and the programme scorecard."),
             ("4", "Scenarios as a story", "Eight scenarios told with the Streamlit dashboard: read-out, validity, guest what-ifs and threshold policy.")]
    for i, (n, t, d) in enumerate(parts):
        y = 1.7 + i * 1.28
        shape(s, LM, y, SW - 2 * LM, 1.12, LIGHT)
        label(s, LM + 0.25, y + 0.22, 0.68, 0.68, n, BLUE, WHITE, size=24, kind=MSO_SHAPE.OVAL, font=HEAD)
        tb(s, LM + 1.25, y + 0.15, 10.6, 0.4, t, size=19, bold=True, color=NAVY, font=HEAD)
        tb(s, LM + 1.25, y + 0.58, 10.6, 0.5, d, size=14, color=GRAY)
    notes(s, "Roadmap. Parts 1 and 2 are the documentation of method and models; parts 3 and 4 show how the engine is used to make decisions.")


def divider(num: str, title: str, points: list):
    s = new_slide(dark=True, number=False)
    tb(s, LM, 1.3, 3, 1.4, num, size=96, bold=True, color=BLUE, font=HEAD)
    tb(s, LM, 3.0, 9.5, 1.5, title, size=40, bold=True, color=WHITE, font=HEAD, line=0.95)
    tb(s, LM, 4.9, 10.5, 1.8, [{"t": p, "bullet": True, "space_after": 6} for p in points], size=17, color=MID)
    return s


def slide_problem():
    s = new_slide("Part 1 · Problem", "High-tier guests spend more, but that does not prove Gold caused it")
    card(s, LM, 1.65, 5.9, 2.35, "The trap: a naive comparison", [
        {"t": "Frequent travellers reach Gold and also book more, spend more and book direct more.", "bullet": True, "space_after": 4},
        {"t": "Comparing Gold with non-Gold mixes the effect of perks with who the guests already are.", "bullet": True}], fill=SOFT["FAIL"][0], title_color=SOFT["FAIL"][1])
    card(s, LM + 6.2, 1.65, 5.9, 2.35, "The fix: compare guests either side of 20 nights", [
        {"t": "A guest with 19 nights and one with 20 are alike in everything except Gold status.", "bullet": True, "space_after": 4},
        {"t": "Any jump in next-year behaviour right at the cutoff is the causal effect of Gold.", "bullet": True}], fill=SOFT["PASS"][0], title_color=SOFT["PASS"][1])
    # figure: two guests either side of the cutoff
    shape(s, LM, 4.3, SW - 2 * LM, 2.55, LIGHT)
    xs = [LM + 0.4, LM + 3.5, LM + 6.7, LM + 9.8]
    items = [("19 nights", "Silver", MID, INK), ("20 nights", "Gold", BLUE, WHITE), ("Same age mix, ADR, lead time, tenure, card holding", "balance check", TINT, NAVY),
             ("Jump in next-year nights, spend, direct share", "the causal effect", AQUA, WHITE)]
    for (t, sub, fill, fg), x in zip(items, xs):
        w = 2.6 if x < LM + 5 else 2.4
        label(s, x, 4.75, w, 1.0, t, fill, fg, size=16, radius=0.12)
        tb(s, x, 5.85, w, 0.4, sub, size=13, color=GRAY, align=PP_ALIGN.CENTER)
    for x in (xs[0] + 2.6, xs[1] + 2.6, xs[2] + 2.6):
        arrow(s, x + 0.05, 5.25, x + 0.45, 5.25)
    tb(s, LM + 0.4, 6.3, 11.3, 0.4, "Two things can break this: guests gaming the threshold, and guests who differ across it. The validity suite tests both.", size=13, color=GRAY, italic=True)
    notes(s, "The core identification idea. We rely on the threshold being as-good-as-random locally, which is why every flow starts with density and balance checks.")


def slide_strategy():
    s = new_slide("Part 1 · Approach", "Two layers: the average effect at the cutoff, then who responds most")
    shape(s, LM, 1.7, 5.7, 4.9, LIGHT)
    label(s, LM + 0.25, 1.9, 1.3, 0.45, "LAYER 1", BLUE, WHITE, size=12)
    tb(s, LM + 1.7, 1.88, 3.9, 0.5, "Regression discontinuity (RDD)", size=19, bold=True, color=NAVY, font=HEAD)
    tb(s, LM + 0.25, 2.6, 5.2, 3.9, [
        {"t": "Question: what does Gold cause for guests at exactly 20 nights?", "bold": True, "space_after": 8},
        {"t": "Sharp design: Gold iff 20+ nights.", "bullet": True, "space_after": 4},
        {"t": "Fuzzy design: 12% below / 94% above hold Gold (card fast-track, status match, sync lag); eligibility instruments Gold (2SLS).", "bullet": True, "space_after": 4},
        {"t": "Local polynomial fits, IK and CCT bandwidths, triangular / Epanechnikov / uniform kernels.", "bullet": True, "space_after": 4},
        {"t": "Output: one number per outcome, with a validity suite.", "bullet": True}], size=14)
    shape(s, LM + 6.4, 1.7, 5.7, 4.9, LIGHT)
    label(s, LM + 6.65, 1.9, 1.3, 0.45, "LAYER 2", ORANGE, WHITE, size=12)
    tb(s, LM + 8.1, 1.88, 3.9, 0.5, "Uplift / CATE learners", size=19, bold=True, color=NAVY, font=HEAD)
    tb(s, LM + 6.65, 2.6, 5.2, 3.9, [
        {"t": "Question: which guests respond most to Gold perks?", "bold": True, "space_after": 8},
        {"t": "Window |nights - 20| <= 10 with RDD kernel weights.", "bullet": True, "space_after": 4},
        {"t": "12 learners: meta-learners, orthogonal learners, double ML, forests, IV, deep nets.", "bullet": True, "space_after": 4},
        {"t": "Scored without ground truth (R-loss, AUUC, Qini) and, because the data are synthetic, against the true effect.", "bullet": True, "space_after": 4},
        {"t": "Output: an effect for every guest, a targeting ranking, a policy value.", "bullet": True}], size=14)
    arrow(s, LM + 5.8, 4.15, LM + 6.35, 4.15, ORANGE, 3)
    notes(s, "Layer 1 gives a credible average. Layer 2 reuses the same local weighting to estimate individual effects, so the heterogeneity we report is still anchored at the cutoff.")


def slide_data():
    s = new_slide("Part 1 · Data", "A synthetic panel with a known truth, so every estimate can be checked")
    rows = [["Item", "Specification"],
            ["Members", f"{META['n_panel']:,}, two consecutive qualification years"],
            ["Running variable", "Qualifying nights in year t (1 to 45), smooth density, optional bunching toggle"],
            ["Cutoff and treatment", "20 nights = Gold; sharp rule, or fuzzy (12% below, 94% above)"],
            ["Covariates (10)", "ADR, business-travel share, lead time, ancillary spend, direct share, tenure, co-brand card, region, age band, weekend share"],
            ["Outcomes (3)", "Next-year room nights, ancillary spend (USD), direct-booking ratio"],
            ["True effects at the cutoff", f"+{gen_t('next_year_room_nights')} nights, +${gen_t('next_year_ancillary_spend')}, +{gen_t('next_year_direct_booking_ratio')} direct share"],
            ["Heterogeneity", "High for mid-tier leisure and bleisure, low for rigid corporate travellers"]]
    table(s, LM, 1.7, [2.6, 9.5], rows, row_h=0.56, size=13, hsize=13, first_col_bold=True)
    notes(s, "Why synthetic: the generator calibrates individual effects so the cutoff-local average is exactly the stated truth. That lets us measure bias, coverage and model accuracy, which is impossible with real loyalty data.")


def gen_t(oc):
    from src.data.synthetic_generator import TARGETS
    v = TARGETS[oc]
    return f"{v:g}"


def slide_pipeline():
    s = new_slide("Part 1 · Uplift end-to-end flow", "The pipeline in eight stages, from raw panel to a served decision")
    steps = [("1", "Generate data", "Panel with known effects; sharp and fuzzy; optional gaming", "synthetic_generator.py"),
             ("2", "RDD + validity", "Local polynomial, 2SLS; density, balance, placebo, donut", "rdd_estimators.py"),
             ("3", "Window and weights", "|nights - 20| <= 10, kernel weights K((R-c)/h)", "CateData (nuisance.py)"),
             ("4", "Remove the trend", "Side-specific slope in R removed from the outcome", "CateData.Y"),
             ("5", "Train 12 learners", "GPU XGBoost / CatBoost, econml, PyTorch CUDA", "cate_learners.py"),
             ("6", "Tune with Optuna", "Minimise validation R-loss; trial 0 = defaults", "tune_learner()"),
             ("7", "Evaluate out-of-fold", "AUUC, Qini, Uplift@K, policy value, accuracy vs truth", "causal_metrics.py"),
             ("8", "Decide and serve", "Flows, FastAPI, Streamlit, threshold simulator", "flows.py · api/ · ui/")]
    for i, (n, t, d, art) in enumerate(steps):
        col, row = i % 4, i // 4
        x, y = LM + col * 3.07, 1.75 + row * 2.6
        shape(s, x, y, 2.75, 2.2, LIGHT)
        label(s, x + 0.15, y + 0.15, 0.5, 0.5, n, BLUE if row == 0 else ORANGE, WHITE, size=16, kind=MSO_SHAPE.OVAL, font=HEAD)
        tb(s, x + 0.78, y + 0.17, 1.9, 0.5, t, size=15, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        tb(s, x + 0.18, y + 0.8, 2.4, 0.85, d, size=12.5, color=INK)
        tb(s, x + 0.18, y + 1.72, 2.45, 0.35, art, size=11, color=GRAY, italic=True)
        if col < 3:
            arrow(s, x + 2.78, y + 1.1, x + 3.04, y + 1.1)
    arrow(s, LM + 3 * 3.07 + 1.35, 3.97, LM + 3 * 3.07 + 1.35, 4.33, GRAY)
    notes(s, "High-level documentation of the uplift flow. Stages 1 to 4 prepare a locally valid sample; stages 5 to 7 estimate and score effects; stage 8 turns them into decisions. "
             "Stage 4 matters: in a sharp design treatment is a function of the running variable, so the trend in nights must be removed before learners see the data.")


def slide_stage_doc():
    s = new_slide("Part 1 · Uplift end-to-end flow", "Each stage has an input, a design choice, an artifact and a metric to watch")
    rows = [["Stage", "What happens", "Key design choice", "Artifact", "Metric to watch"],
            ["1 Data", "50,000-member panel, two years", "Effects calibrated to exact truth", "panel.csv.gz", "Take-up jump 0.81"],
            ["2 RDD", "Estimate the jump; run four falsification tests", "Quadratic density fit, Poisson variance", "rdd_*.csv, rdd_diagnostics.json", "McCrary p, Holm p, placebo nulls"],
            ["3 Window", "Keep +-10 nights, triangular weights", "R is not an effect modifier", "CateData", "Guests in window"],
            ["4 Trend", "Remove side-specific slope in nights", "Keeps the level at the cutoff", "CateData.Y", "Slope per side"],
            ["5 Learners", "Fit 12 estimators on weighted sample", "GPU for trees and nets; CPU fallback", "models/room_nights/*.joblib", "Training seconds"],
            ["6 Tuning", "Optuna, 8 trials per learner", "R-loss needs no ground truth", "optuna_*.csv, best params", "R-loss gain vs defaults"],
            ["7 Evaluate", "3-fold out-of-fold predictions", "Truth used for reporting only", "cate_leaderboard, oof_cate", "AUUC, Qini, RMSE, corr"],
            ["8 Decide", "8 flows with gates and verdicts", "Money = revenue x margin - perks", "flows_report.json", "Gates passed, net value"]]
    table(s, LM, 1.7, [1.35, 3.3, 3.0, 2.55, 1.9], rows, row_h=0.56, size=12, hsize=12, first_col_bold=True, zebra=True)
    notes(s, "Reference table for the flow. Artifact names are the files written to outputs/. Everything on the later slides is read from those files.")


def slide_weights():
    s = new_slide("Part 1 · Uplift end-to-end flow", "Kernel weights keep the learners local to the cutoff")
    u = np.linspace(-1, 1, 41)
    cd = XyChartData()
    for name, fn in (("Triangular (default)", lambda v: 1 - abs(v)), ("Epanechnikov", lambda v: 0.75 * (1 - v ** 2)), ("Uniform", lambda v: 0.5 + 0 * v)):
        ser = cd.add_series(name)
        for v in u:
            ser.add_data_point(float(v), float(fn(v)))
    gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(LM), Inches(1.65), Inches(6.2), Inches(4.5), cd)
    c = gf.chart
    style_chart(c, 11, XL_LEGEND_POSITION.BOTTOM)
    for ser, col in zip(c.plots[0].series, (BLUE, ORANGE, AQUA)):
        ser.format.line.color.rgb, ser.format.line.width, ser.smooth = col, Pt(3.5), False
    c.value_axis.major_gridlines.format.line.color.rgb = GRID
    c.value_axis.format.line.fill.background()
    c.category_axis.has_title = c.value_axis.has_title = True
    c.category_axis.axis_title.text_frame.text = "Distance from the cutoff, (R - 20) / h"
    c.value_axis.axis_title.text_frame.text = "Weight K(u)"
    for ax in (c.category_axis, c.value_axis):
        r = ax.axis_title.text_frame.paragraphs[0].runs[0]
        r.font.size, r.font.bold, r.font.color.rgb = Pt(11), False, GRAY
    sl = FLOWS["A1"]
    card(s, LM + 6.55, 1.65, 5.55, 2.1, "Window and weights", [
        {"t": f"Guests within +-{META['cate_bandwidth']:.0f} nights of 20 enter; weight falls linearly to zero at the edge.", "bullet": True, "space_after": 4},
        {"t": "The same kernel is used by the RDD and by every learner's loss.", "bullet": True}], body_size=13)
    card(s, LM + 6.55, 3.95, 5.55, 2.2, "Why remove the trend first", [
        {"t": "Nights drive both Gold (sharp rule) and next-year nights. Without adjustment a learner reads the slope as an effect.", "bullet": True, "space_after": 4},
        {"t": "We subtract each side's linear trend in nights, so only the jump at 20 remains.", "bullet": True}], body_size=13)
    tb(s, LM, 6.35, 11.9, 0.5, "Effect modifiers are the 10 guest covariates; nights itself is never a modifier, so conclusions stay local to guests near 20 nights.", size=13, color=GRAY, italic=True)
    notes(s, "Kernel weights are the bridge between layer 1 and layer 2. Extrapolating a learner far from 20 nights is not supported by this design, which is why the threshold simulator warns beyond 5 nights.")


def slide_metrics_explained():
    s = new_slide("Part 1 · Uplift end-to-end flow", "How uplift is measured: ranking quality, accuracy and policy value")
    top = LB.sort_values("auuc_over_random", ascending=False).iloc[0]
    best_rmse = LB.sort_values("pehe_rmse").iloc[0]
    cards = [("AUUC and Qini", "Rank guests by predicted effect; how much extra outcome do we gain by targeting the top share instead of random?", f"best AUUC over random {LB['auuc_over_random'].max():.3f} ({PRETTY[top['learner']]})"),
             ("Uplift@K", "Average effect among the top K% ranked guests, from observed treated minus control.", f"best uplift@10% {LB['uplift_at_10'].max():+.2f} nights"),
             ("R-loss", "Robinson residual loss; lower is better and needs no ground truth. Used to tune and to rank without truth.", f"OOF R-loss {LB['r_loss_oof'].min():.2f} to {LB['r_loss_oof'].max():.2f}"),
             ("PEHE / RMSE", "Error of the individual effect against the known truth (synthetic data only).", f"best RMSE {best_rmse['pehe_rmse']:.2f} nights ({PRETTY[best_rmse['learner']]})"),
             ("Correlation", "Pearson correlation of predicted and true individual effects.", f"best {LB['pearson'].max():.2f}, median {LB['pearson'].median():.2f}"),
             ("Policy value", "Average gain of treating guests the model ranks above zero, using doubly robust scores.", f"{LB['policy_value_dr'].min():.2f} to {LB['policy_value_dr'].max():.2f} nights per guest")]
    for i, (t, d, r) in enumerate(cards):
        col, row = i % 3, i // 3
        x, y = LM + col * 4.1, 1.65 + row * 2.45
        shape(s, x, y, 3.95, 2.25, LIGHT)
        tb(s, x + 0.2, y + 0.12, 3.5, 0.4, t, size=16, bold=True, color=NAVY, font=HEAD)
        tb(s, x + 0.2, y + 0.58, 3.55, 1.1, d, size=12.5, color=INK)
        tb(s, x + 0.2, y + 1.72, 3.55, 0.45, r, size=12.5, bold=True, color=BLUE)
    tb(s, LM, 6.55, 12, 0.4, "Ground-truth metrics are for reporting; model selection and tuning use only R-loss and the observed-outcome uplift metrics.", size=13, color=GRAY, italic=True)
    notes(s, "Two families of metrics. Observed-outcome metrics (AUUC, Qini, uplift@K, R-loss, policy value) work on real data. Truth-based metrics (RMSE, correlation) exist only because the data are synthetic.")


def slide_architecture():
    s = new_slide("Part 1 · Architecture", "One trained artifact set feeds the API, the dashboard and the flows")
    cols = [("Data", ["synthetic_generator", "sharp / fuzzy / gamed / placebo"], BLUE), ("Train", ["RDD + validity suite", "12 learners, Optuna", "out-of-fold evaluation"], BLUE),
            ("outputs/", ["models/*.joblib", "metrics/*.csv, flows_report.json", "plots, screenshots, logs"], NAVY), ("ModelService", ["loads artifacts", "CPU inference", "late / cate / simulate"], ORANGE),
            ("Consumers", ["FastAPI (Swagger)", "Streamlit, 6 tabs", "8 flows + scorecard"], AQUA)]
    w = 2.3
    for i, (t, items, col) in enumerate(cols):
        x = LM + i * 2.45
        label(s, x, 1.8, w, 0.6, t, col, WHITE, size=16, font=HEAD, radius=0.15)
        shape(s, x, 2.5, w, 2.0, LIGHT)
        tb(s, x + 0.12, 2.6, w - 0.24, 1.9, [{"t": it, "bullet": True, "space_after": 4} for it in items], size=12.5)
        if i < 4:
            arrow(s, x + w + 0.02, 2.1, x + 2.43, 2.1, GRAY, 2.5)
    shape(s, LM, 4.8, SW - 2 * LM, 2.0, TINT)
    hw = META["hardware"]
    pts = [f"Training: {hw.get('gpu_name', 'CPU')} - GPU XGBoost, CatBoost and PyTorch CUDA; LightGBM on CPU; automatic CPU fallback.",
           f"Full run: {META['train_seconds'] / 60:.1f} minutes for 12 tuned learners with 3-fold out-of-fold evaluation.",
           "Serving runs on CPU: XGBoost-CUDA aborts the process when a short-lived thread exits, so models are switched to CPU on load (predictions identical).",
           "Environment: Python 3.9.7, Streamlit 1.12 (newer releases exclude 3.9.7), plotly below 6."]
    tb(s, LM + 0.25, 4.95, SW - 2 * LM - 0.5, 1.8, [{"t": p, "bullet": True, "space_after": 4} for p in pts], size=13)
    notes(s, "Everything downstream of training reads the same artifacts, so the dashboard, the API and the flows cannot disagree; the capture script verifies that the numbers on screen equal the API's numbers.")


def slide_run():
    s = new_slide("Part 1 · Reproducibility", "One command per stage, all artifacts regenerated from the config")
    rows = [["Command", "What it does"],
            ["make setup", "Create the venv, install CUDA torch and dependencies, Playwright Chromium"],
            ["make train", "Data, RDD and diagnostics, 12 tuned learners, artifacts, flows report"],
            ["make flows", "Rebuild flows_report.json (8 flows, scorecard, model-by-flow matrix)"],
            ["make serve / make app", "FastAPI on :8000 (Swagger at /docs); Streamlit on :8501"],
            ["make test", "pytest suite: RDD recovery, validity, all learners, metrics, API, flows"],
            ["make verify-ui", "Screenshot the API, every tab and every flow stage; check UI equals API"],
            ["python scripts/build_deck.py", "Rebuild this presentation from the saved outputs"]]
    table(s, LM, 1.65, [3.6, 8.5], rows, row_h=0.52, size=13, hsize=13, first_col_bold=True, zebra=True)
    stat(s, LM, 6.0, 2.8, 0.95, "63", "automated tests pass", color=AQUA, vsize=28, csize=12)
    stat(s, LM + 3.0, 6.0, 2.8, 0.95, f"{META['train_seconds'] / 60:.1f} min", "full training run", color=BLUE, vsize=28, csize=12)
    stat(s, LM + 6.0, 6.0, 2.8, 0.95, "8 + 65", "flows and stage screenshots", color=ORANGE, vsize=28, csize=12)
    stat(s, LM + 9.0, 6.0, 3.1, 0.95, "UI = API", "verified on every flow", color=NAVY, vsize=28, csize=12)
    notes(s, "Reproducibility. The deck itself is generated by scripts/build_deck.py from the same artifacts.")
    return s


# ---------------------------------------------------------------- Part 2
def slide_rdd():
    a1 = FLOWS["A1"]
    est = next(st for st in a1["stages"] if st["key"] == "estimates")
    s = new_slide("Part 2 · Causal estimate", "The jump at 20 nights is visible in every outcome")
    picture(s, PLOTS / "01_rdd_discontinuity_main.png", LM, 1.65, SW - 2 * LM, 3.3, max_w=2300)
    rows = [["Outcome", "Fuzzy RDD estimate", "95% CI", "True effect", "Error (SE)", "CI covers truth"]] + \
           [[r["Outcome"], r["Estimate"], r["95% CI"], r["True effect"], r["Error (SE)"], r["CI covers truth"]] for r in est["table"]]
    table(s, LM, 5.05, [2.6, 2.1, 2.7, 1.7, 1.4, 1.6], rows, row_h=0.42, size=13, hsize=13, first_col_bold=True)
    tb(s, LM, 6.75, 12, 0.3, f"Headline design is fuzzy; first-stage jump {a1['kpi']['first_stage']:.2f}. Source: flow A1.", size=11, color=GRAY)
    notes(s, "Dots are mean outcomes per qualifying-night count; lines are local polynomial fits with 95% bands. The vertical gap at the dashed line is the effect.")


def slide_validity():
    a1 = FLOWS["A1"]
    st_d = next(x for x in a1["stages"] if x["key"] == "density")
    st_b = next(x for x in a1["stages"] if x["key"] == "balance")
    st_r = next(x for x in a1["stages"] if x["key"] == "robustness")
    b2 = FLOWS["B2"]["kpi"]["replication"]
    s = new_slide("Part 2 · Causal estimate", "Density test: smooth on clean data, bunching caught when the threshold is gamed")
    picture(s, PLOTS / "02_mccrary_density_test.png", LM, 1.7, SW - 2 * LM, 3.9, max_w=2400)
    tiles = [(st_d["metrics"][1]["value"], "McCrary p-value on the clean panel (A1): no bunching", AQUA),
             (f"{b2['density_flag_rate']:.0%}", "of gamed re-draws flagged (flow B2)", ORANGE),
             (f"{b2['clean_density_flag_rate']:.0%}", "false alarms on clean re-draws (nominal 5%)", AQUA)]
    for i, (v, c, col) in enumerate(tiles):
        stat(s, LM + i * 4.1, 5.75, 3.95, 1.15, v, c, color=col, vsize=26, csize=12)
    notes(s, "The density test was rebuilt during development: the first version rejected a true null 37 to 100 percent of the time; the quadratic fit with Poisson variance brought it to about 7 percent.")
    s = new_slide("Part 2 · Causal estimate", "Balance and placebo tests: nothing else jumps at 20 nights")
    picture(s, PLOTS / "04_covariate_balance_placebo.png", LM, 1.7, 8.6, 5.0, max_w=2000)
    tiles2 = [(st_b["metrics"][0]["value"], "covariates balanced (smallest Holm p " + st_b["metrics"][1]["value"] + ")"),
              (st_r["metrics"][2]["value"], "placebo cutoffs null (12, 15, 25, 30)"),
              (st_r["metrics"][3]["value"], "donut estimate vs baseline, 19 and 20 dropped")]
    for i, (v, c) in enumerate(tiles2):
        stat(s, LM + 8.9, 1.7 + i * 1.7, 3.2, 1.55, v, c, color=AQUA, vsize=24, csize=12)
    notes(s, "Balance: no pre-treatment covariate jumps at the cutoff after Holm correction. Placebo: false cutoffs show no effect. Donut: dropping the guests at exactly 19 and 20 nights leaves the estimate unchanged.")


def slide_robust():
    a1 = FLOWS["A1"]
    st_r = next(x for x in a1["stages"] if x["key"] == "robustness")
    s = new_slide("Part 2 · Causal estimate", "The estimate is stable across bandwidths, kernels and polynomial orders")
    picture(s, PLOTS / "03_bandwidth_sensitivity.png", LM, 1.7, SW - 2 * LM, 3.6, max_w=2400)
    for i, m in enumerate(st_r["metrics"]):
        stat(s, LM + i * 3.07, 5.45, 2.9, 1.1, m["value"], m["label"], color=BLUE, vsize=20, csize=11.5)
    tb(s, LM, 6.65, 12, 0.35, "A quadratic fit at h = 3 is not estimable with whole-night data (only two usable nights per side), so that cell is blank by design.", size=12, color=GRAY, italic=True)
    notes(s, "Robustness gate: pass if the largest deviation from the headline is within 3 SE, at least 3 of 4 placebo cutoffs are null, and the donut estimate is within 3 SE.")


MODELS = {
    "s_learner": ("Meta-learner", "One outcome model with Gold and Gold-by-covariate interaction terms; the effect is prediction with Gold on minus off.", "GPU XGBoost or LightGBM (CPU)", "Fast; shares information across arms", "Can shrink a weak treatment signal"),
    "t_learner": ("Meta-learner", "Separate outcome models for treated and control guests; effect is the difference of their predictions.", "GPU XGBoost or CatBoost GPU", "Flexible; no interaction engineering", "Each arm sees less data; noisy at the edges"),
    "x_learner": ("Meta-learner", "Imputes each guest's missing outcome with the other arm's model, fits effect models on those, blends by propensity.", "GPU XGBoost + propensity model", "Strong when one arm is smaller", "More moving parts"),
    "r_learner": ("Orthogonal", "Robinson residual-on-residual: regress outcome residual on treatment residual with cross-fitted nuisances.", "GPU XGBoost, 3-fold cross-fit", "Neyman-orthogonal; robust to nuisance error", "Weights by residual squared can be noisy"),
    "dr_learner": ("Orthogonal", "Doubly robust AIPW pseudo-outcomes regressed on covariates, with cross-fitting.", "GPU XGBoost, 3-fold cross-fit", "Consistent if either outcome or propensity model is right", "Propensity extremes inflate variance"),
    "linear_dml": ("Double ML", "Double machine learning with a linear (polynomial) effect model on top of cross-fitted nuisances.", "econml, GPU XGBoost nuisances", "Interpretable coefficients", "Cannot capture a bell-shaped effect"),
    "nonparam_dml": ("Double ML", "Double machine learning with a flexible XGBoost final stage.", "econml, GPU XGBoost nuisance and final", "Orthogonal and non-parametric", "Final stage can overfit small cells"),
    "causal_forest_dml": ("Forest", "Honest causal forest on double-ML residuals; gives confidence intervals.", "econml, GPU XGBoost nuisances", "Best uncertainty quantification; strong ranking", "Slower; many trees"),
    "orthogonal_forest": ("Forest", "Orthogonal random forest (econml DMLOrthoForest), distilled into a small XGBoost surrogate for prediction.", "econml on a 3,000-row subsample; XGBoost surrogate", "True ORF estimator", "0.25 s per prediction, so subsampled; weakest here"),
    "dmliv": ("IV", "Non-parametric DML-IV: eligibility 1(R>=20) instruments actual Gold, so the effect is for compliers.", "econml NonParamDMLIV, GPU final stage", "Handles fuzzy compliance explicitly", "Needs a strong first stage"),
    "tarnet": ("Deep", "Shared representation network with separate mu0 and mu1 heads, trained with kernel sample weights.", "PyTorch CUDA, early stopping", "Learns shared structure across arms", "Needs tuning; slower"),
    "dragonnet": ("Deep", "TARNet plus a propensity head and targeted regularization.", "PyTorch CUDA, early stopping", "Propensity-aware representation", "Most hyperparameters; slowest"),
}


def slide_models_overview():
    s = new_slide("Part 2 · The 12 models", "Twelve estimators across five families, all kernel-weighted")
    order = ["s_learner", "t_learner", "x_learner", "r_learner", "dr_learner", "linear_dml", "nonparam_dml", "causal_forest_dml", "orthogonal_forest", "dmliv", "tarnet", "dragonnet"]
    rows = [["#", "Model", "Family", "Implementation", "Tuned by Optuna"]]
    for i, k in enumerate(order, 1):
        trials = LBI.loc[k, "n_trials"] if k in LBI.index else 0
        rows.append([str(i), PRETTY[k], MODELS[k][0], MODELS[k][2], f"{int(trials)} trials" if trials and not pd.isna(trials) and trials > 0 else "fixed settings"])
    table(s, LM, 1.65, [0.5, 2.5, 1.8, 5.4, 1.9], rows, row_h=0.385, size=12, hsize=12, zebra=True, first_col_bold=False)
    tb(s, LM, 6.75, 12, 0.3, "GPU: XGBoost (device=cuda), CatBoost (task_type=GPU), PyTorch CUDA. LightGBM and the orthogonal forest's econml core run on CPU.", size=11, color=GRAY)
    notes(s, "All twelve implement the same fit/predict interface and are weighted by the RDD kernel. The next five slides explain them family by family with their results.")


def model_card(s, x, y, w, h, key):
    fam, idea, impl, plus, minus = MODELS[key]
    r = LBI.loc[key]
    shape(s, x, y, w, h, LIGHT)
    tb(s, x + 0.2, y + 0.12, w - 0.4, 0.45, PRETTY[key], size=19, bold=True, color=NAVY, font=HEAD)
    tb(s, x + 0.2, y + 0.62, w - 0.4, 1.5, idea, size=14, color=INK)
    tb(s, x + 0.2, y + 2.2, w - 0.4, 1.2, [{"runs": [("Strength: ", True, SOFT["PASS"][1]), (plus, False, INK)], "space_after": 4},
                                           {"runs": [("Watch-out: ", True, SOFT["WARN"][1]), (minus, False, INK)]}], size=13)
    narrow = w < 4.5                       # three-up slides: three chips, training time moves into the implementation line
    tb(s, x + 0.2, y + h - 1.05, w - 0.4, 0.4, f"{impl} · {r['seconds']:.0f} s training" if narrow else impl, size=11.5, color=GRAY, italic=True)
    chips = [(f"corr {r['pearson']:.2f}", BLUE), (f"RMSE {r['pehe_rmse']:.2f}", BLUE), (f"AUUC {r['auuc_over_random']:.3f}", AQUA)]
    if not narrow:
        chips.append((f"{r['seconds']:.0f} s", GRAY))
    cw = (w - 0.4 - 0.08 * (len(chips) - 1)) / len(chips)
    for i, (t, col) in enumerate(chips):
        label(s, x + 0.2 + i * (cw + 0.08), y + h - 0.58, cw, 0.42, t, WHITE, col, size=11.5, radius=0.3)


def family_slide(title, keys, kicker_note, note):
    s = new_slide("Part 2 · The 12 models", title)
    n = len(keys)
    w = (SW - 2 * LM - 0.25 * (n - 1)) / n
    h = 4.4 if n >= 3 else 3.9
    for i, k in enumerate(keys):
        model_card(s, LM + i * (w + 0.25), 1.7, w, h, k)
    tb(s, LM, 1.7 + h + 0.2, 12, 0.6, kicker_note, size=13, color=GRAY, italic=True)
    notes(s, note)
    return s


def slide_tuning():
    s = new_slide("Part 2 · The 12 models", "Optuna tunes on R-loss; trial 0 is always the defaults")
    rows = []
    for k in PRETTY:
        p = M / f"optuna_room_nights_{k}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        v0 = float(d.loc[d["number"] == 0, "value"].iloc[0])
        gain = (v0 - float(d["value"].min())) / v0 * 100
        rows.append((PRETTY[k], gain, len(d)))
    rows.sort(key=lambda r: -r[1])
    ch = bar_chart(s, LM, 1.7, 7.2, 4.9, [r[0] for r in rows][::-1], {"R-loss gain over defaults (%)": [r[1] for r in rows][::-1]}, [BLUE],
                   horizontal=True, fmt="0.00", size=11, gap=40)
    ch.has_title = True
    ch.chart_title.text_frame.text = "Validation R-loss improvement of the best trial vs the default (%)"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = False
    card(s, LM + 7.5, 1.65, 4.6, 2.35, "Why R-loss", [
        {"t": "Needs no ground truth, so the same recipe works on real data.", "bullet": True, "space_after": 4},
        {"t": "Nuisances are fitted on the training part only, so validation residuals are honest.", "bullet": True}], body_size=13)
    card(s, LM + 7.5, 4.2, 4.6, 2.3, "Reading the gains", [
        {"t": "Gains are small (well under 1%): the R-loss is dominated by outcome noise, so tuning rarely moves the ranking.", "bullet": True, "space_after": 4},
        {"t": "LinearDML is the exception: tuning chose the quadratic variant.", "bullet": True}], body_size=13)
    notes(s, "Tuning is defensive: trial 0 always uses the defaults, so a tuned model is never worse on validation than the untuned one. The orthogonal forest is not tuned because each trial would cost minutes.")


def heat(err, lo=0.15, hi=0.4):
    a = abs(err)
    return SOFT["PASS"][0] if a <= lo else (SOFT["WARN"][0] if a <= hi else SOFT["FAIL"][0])


def slide_leaderboard():
    cs = np.sort(LB["pearson"].to_numpy())[::-1]
    s = new_slide("Part 2 · Model comparison", f"Leaderboard: nine models cluster within {cs[0] - cs[8]:.2f} correlation; three trail")
    d = LB.sort_values("pehe_rmse")
    rows = [["Model", "Corr.", "RMSE", "Avg effect", "AUUC gain", "Qini", "Uplift@10%", "Policy value", "R-loss", "Sec."]]
    fills, bolds = {}, {}
    for i, (_, r) in enumerate(d.iterrows(), 1):
        rows.append([PRETTY[r["learner"]], f"{r['pearson']:.3f}", f"{r['pehe_rmse']:.3f}", f"{r['ate_hat']:.2f}", f"{r['auuc_over_random']:.3f}",
                     f"{r['qini_coefficient']:.3f}", f"{r['uplift_at_10']:+.2f}", f"{r['policy_value_dr']:.3f}", f"{r['r_loss_oof']:.2f}", f"{r['seconds']:.0f}"])
    for j, col in enumerate(["pearson", "pehe_rmse", "ate_hat", "auuc_over_random", "qini_coefficient", "uplift_at_10", "policy_value_dr", "r_loss_oof", "seconds"], start=1):
        ser = d[col].reset_index(drop=True)
        best = ser.idxmin() if col in ("pehe_rmse", "r_loss_oof", "seconds") else (ser.sub(3.4).abs().idxmin() if col == "ate_hat" else ser.idxmax())
        fills[(best + 1, j)] = SOFT["PASS"][0]
        bolds[(best + 1, j)] = True
    table(s, LM, 1.65, [2.4, 0.95, 0.95, 1.2, 1.2, 1.0, 1.35, 1.35, 0.95, 0.85], rows, row_h=0.37, size=12, hsize=11.5, fills=fills, bolds=bolds, first_col_bold=True,
          aligns={j: PP_ALIGN.CENTER for j in range(1, 10)})
    tb(s, LM, 6.62, 12, 0.4, f"Out-of-fold, {len(LB)} learners, room nights, truth average effect {LB['ate_true'].iloc[0]:.2f}. Green marks the best value per column "
       "(for average effect, the closest to the truth). Sorted by RMSE.", size=11.5, color=GRAY)
    notes(s, "Global leaderboard on room nights. The spread among the top ten is small relative to the noise, so rankings in this table are not decisive; the next slides test whether they hold across flows.")


def slide_charts_acc():
    s = new_slide("Part 2 · Model comparison", "Accuracy: tree and orthogonal learners lead; the linear and forest-ORF trail")
    d = LB.sort_values("pearson", ascending=False)
    names = [PRETTY[k] for k in d["learner"]][::-1]
    cols = [BLUE if p >= 0.85 else MID for p in d["pearson"]][::-1]
    ch = bar_chart(s, LM, 1.65, 5.9, 4.9, names, {"Correlation": d["pearson"].tolist()[::-1]}, [BLUE], horizontal=True, fmt="0.00", size=11, gap=35, point_colors=cols, vmin=0, vmax=1)
    ch.has_title = True
    ch.chart_title.text_frame.text = "Correlation with the true individual effect (blue = 0.85+)"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = False
    d2 = LB.sort_values("pehe_rmse")
    ch2 = bar_chart(s, LM + 6.2, 1.65, 5.9, 4.9, [PRETTY[k] for k in d2["learner"]][::-1], {"RMSE": d2["pehe_rmse"].tolist()[::-1]}, [ORANGE], horizontal=True, fmt="0.00", size=11, gap=35)
    ch2.has_title = True
    ch2.chart_title.text_frame.text = "Individual-effect error, RMSE in nights (lower is better)"
    ch2.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    ch2.chart_title.text_frame.paragraphs[0].runs[0].font.bold = False
    tb(s, LM, 6.65, 12, 0.3, "Out-of-fold predictions on the kernel-weighted window; truth known because the data are synthetic.", size=11, color=GRAY)
    notes(s, "Correlation measures ranking agreement; RMSE penalises scale errors too. The orthogonal forest and LinearDML are the two clear laggards.")


def slide_charts_uplift():
    ratio = LB["uplift_at_10"].max() / LB["ate_true"].iloc[0]
    s = new_slide("Part 2 · Model comparison", f"Targeting: the top 10% of guests respond about {ratio:.1f}x the average")
    d = LB.sort_values("auuc_over_random", ascending=False)
    cats = [PRETTY[k] for k in d["learner"]]
    ch = bar_chart(s, LM, 1.65, 7.6, 3.95, cats, {"Uplift@10%": d["uplift_at_10"].tolist(), "Uplift@30%": d["uplift_at_30"].tolist()}, [BLUE, ORANGE], fmt="0.0",
                   legend=XL_LEGEND_POSITION.TOP, size=10, gap=40, overlap=-5)
    ch.has_title = True
    ch.chart_title.text_frame.text = "Average effect among the top-ranked 10% and 30% (nights)"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = False
    picture(s, PLOTS / "07_uplift_qini_curves.png", LM + 7.8, 1.65, 4.3, 3.95, max_w=1700)
    stat(s, LM, 5.75, 3.7, 1.05, f"{LB['uplift_at_10'].max():+.2f} nights", "best uplift among the top 10%", color=BLUE, vsize=22, csize=12)
    stat(s, LM + 3.9, 5.75, 3.7, 1.05, f"{LB['ate_true'].iloc[0]:+.2f} nights", "average effect if everyone is treated", color=GRAY, vsize=22, csize=12)
    stat(s, LM + 7.8, 5.75, 4.3, 1.05, f"{LB['auuc_over_random'].max():.3f}", "best AUUC gain over random targeting", color=AQUA, vsize=22, csize=12)
    notes(s, "Uplift@K is observed treated-minus-control within the top-ranked slice, so it works on real data. The curves on the right show every model beating random targeting.")


def slide_cost_accuracy():
    s = new_slide("Part 2 · Model comparison", "Cost versus accuracy: the cheap models are already near the frontier")
    cd = XyChartData()
    for k in LB["learner"]:
        ser = cd.add_series(PRETTY[k])
        ser.add_data_point(float(LBI.loc[k, "seconds"]), float(LBI.loc[k, "pehe_rmse"]))
    gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(LM), Inches(1.65), Inches(8.2), Inches(5.0), cd)
    c = gf.chart
    style_chart(c, 11, None)
    pal = [BLUE, ORANGE, AQUA, AMBER, RGBColor(0x6A, 0x4C, 0xE2), RGBColor(0xEE, 0x7F, 0xAC), RGBColor(0x00, 0x8A, 0x17), RGBColor(0xEC, 0x46, 0x47),
           GRAY, NAVY, RGBColor(0x84, 0xB9, 0xF3), RGBColor(0xB0, 0xB0, 0xB0)]
    order_s = sorted(LB["learner"], key=lambda k: LBI.loc[k, "seconds"])
    where = {k: (XL_LABEL_POSITION.ABOVE if i % 2 == 0 else XL_LABEL_POSITION.BELOW) for i, k in enumerate(order_s)}
    for ser, col, k in zip(c.plots[0].series, pal, LB["learner"]):
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 11
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = col
        ser.marker.format.line.fill.background()
        ser.format.line.fill.background()
        dl = ser.points[0].data_label
        dl.has_text_frame = True
        dl.text_frame.text = CODE[k]
        dl.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
        dl.text_frame.paragraphs[0].runs[0].font.bold = True
        dl.position = where[k]
    c.value_axis.has_title = c.category_axis.has_title = True
    c.value_axis.axis_title.text_frame.text = "RMSE of individual effect (nights, lower is better)"
    c.category_axis.axis_title.text_frame.text = "Training seconds incl. tuning and out-of-fold fits (log scale)"
    for ax in (c.value_axis, c.category_axis):
        r = ax.axis_title.text_frame.paragraphs[0].runs[0]
        r.font.size, r.font.bold, r.font.color.rgb = Pt(11), False, GRAY
    c.value_axis.major_gridlines.format.line.color.rgb = GRID
    sc = c.category_axis._element.find(qn("c:scaling"))
    lg = etree.Element(qn("c:logBase"))
    lg.set("val", "10")
    sc.insert(0, lg)
    c.category_axis.minimum_scale, c.category_axis.maximum_scale = 1, 1000
    fast = LB.sort_values("seconds").head(3)
    card(s, LM + 8.5, 1.65, 3.6, 5.0, "What it says", [
        {"t": f"{', '.join(PRETTY[k] for k in fast['learner'])} train in under {fast['seconds'].max():.0f} s and sit close to the best RMSE.", "bullet": True, "space_after": 5},
        {"t": f"Deep nets and forests cost {LBI.loc['dragonnet', 'seconds']:.0f} to {LBI['seconds'].max():.0f} s for no accuracy gain here.", "bullet": True, "space_after": 5},
        {"t": "The orthogonal forest is the slowest and the least accurate.", "bullet": True, "space_after": 8},
        {"t": "Key: " + ", ".join(f"{CODE[k]} = {PRETTY[k]}" for k in LB["learner"]), "size": 10.5, "color": GRAY}], body_size=13)
    notes(s, "Seconds cover Optuna tuning (8 trials), three out-of-fold fits and the final fit on one RTX 3090. Tree learners are cheap because the data fit comfortably on the GPU.")


def slide_model_flow_matrix():
    s = new_slide("Part 2 · Comparison across flows", "Each model scored on the guests each flow is about")
    flows = ["C1", "C2", "D1", "D2"]
    heads = {"C1": "C1 bleisure\nlook-alikes", "C2": "C2 corporate\nlook-alikes", "D1": "D1 guests at\n17-19 nights", "D2": "D2 guests at\n20-23 nights"}
    rows = [["Model"] + [heads[f] for f in flows] + ["Mean |error|"]]
    truth = ["Truth (average effect, nights)"] + [f"{MF['populations'][f]['true']:+.2f}  (n={MF['populations'][f]['n']:,})" for f in flows] + [""]
    rows.append(truth)
    order = sorted(MF["learners"], key=lambda l: MF["summary"][l]["mean_abs_err"])
    fills, bolds, colors = {}, {(1, j): True for j in range(6)}, {}
    for j in range(6):
        fills[(1, j)] = TINT
    for i, l in enumerate(order, 2):
        row = [PRETTY[l]]
        for j, f in enumerate(flows, 1):
            e = MF["err"][l][f]
            row.append(f"{MF['est'][l][f]:+.2f}  ({e:+.2f})")
            fills[(i, j)] = heat(e)
        row.append(f"{MF['summary'][l]['mean_abs_err']:.2f}")
        fills[(i, 5)] = heat(MF["summary"][l]["mean_abs_err"], 0.12, 0.3)
        rows.append(row)
    table(s, LM, 1.65, [2.5, 2.25, 2.25, 2.25, 2.25, 1.0], rows, row_h=0.32, size=12, hsize=11.5, fills=fills, bolds=bolds, first_col_bold=True,
          aligns={j: PP_ALIGN.CENTER for j in range(1, 6)}, row_heights=[0.6] + [0.32] * (len(rows) - 1))
    tb(s, LM, 6.55, 12, 0.45, "Cell = estimated average effect (error vs truth), out-of-fold, in room nights. Green error <= 0.15, amber <= 0.4, red above. "
       "Models are sorted by mean absolute error across the four flows.", size=11.5, color=GRAY)
    notes(s, "A new view of model quality: instead of a global score, each model is judged on the specific guests each flow concerns. Most models under-estimate the bleisure couple (shrinkage toward the mean), and the orthogonal forest is off everywhere.")


def slide_model_flow_net():
    s = new_slide("Part 2 · Comparison across flows", "All 12 models agree on the sign of both threshold decisions")
    order = sorted(MF["learners"], key=lambda l: -MF["net"][l]["D1"])
    cats = [PRETTY[l] for l in order]
    ch = bar_chart(s, LM, 1.65, 8.0, 4.9, cats, {"D1 lower to 17 nights": [MF["net"][l]["D1"] / 1000 for l in order], "D2 raise to 24 nights": [MF["net"][l]["D2"] / 1000 for l in order]},
                   [BLUE, ORANGE], fmt="0", legend=XL_LEGEND_POSITION.TOP, size=10, gap=40, overlap=-5)
    ch.has_title = True
    ch.chart_title.text_frame.text = "Net value per year by model, $ thousands (margin-adjusted)"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = False
    nets1 = [MF["net"][l]["D1"] for l in order]
    nets2 = [MF["net"][l]["D2"] for l in order]
    stat(s, LM + 8.3, 1.65, 3.8, 1.45, f"{sum(1 for v in nets1 if v > 0)} of {len(nets1)}", "models say lowering to 17 nights pays (D1)", color=BLUE, vsize=30, csize=12)
    stat(s, LM + 8.3, 3.25, 3.8, 1.45, f"{sum(1 for v in nets2 if v > 0)} of {len(nets2)}", "models say raising to 24 nights pays at $300 perks (D2)", color=ORANGE, vsize=30, csize=12)
    spread1 = (max(nets1) - min(nets1)) / MF["median_net"]["D1"] * 100
    spread2 = (max(nets2) - min(nets2)) / MF["median_net"]["D2"] * 100
    card(s, LM + 8.3, 4.85, 3.8, 1.7, "Spread across models", [{"t": f"D1 range {spread1:.0f}% of the median; D2 range {spread2:.0f}%. The decision does not depend on which model is used.", "bullet": False}], body_size=13)
    notes(s, "Decision robustness: if the verdict changed with the model we would not trust it. It does not. The orthogonal forest sits lowest on D1 and highest on D2, as expected from its under-estimate.")


def slide_model_ranking():
    s = new_slide("Part 2 · Comparison across flows", "The best model depends on the question: global winner and flow winner differ")
    glob = LB.sort_values("pehe_rmse").reset_index(drop=True)
    grank = {r["learner"]: i + 1 for i, r in glob.iterrows()}
    order = sorted(MF["learners"], key=lambda l: MF["summary"][l]["rank"])
    rows = [["Model", "Cross-flow rank", "Mean |error| across 4 flows", "Worst flow error", "Global RMSE rank", "Global corr."]]
    fills = {}
    for i, l in enumerate(order, 1):
        sm = MF["summary"][l]
        rows.append([PRETTY[l], str(sm["rank"]), f"{sm['mean_abs_err']:.2f}", f"{sm['max_abs_err']:.2f}", str(grank[l]), f"{LBI.loc[l, 'pearson']:.3f}"])
        if sm["rank"] <= 3:
            fills[(i, 1)] = SOFT["PASS"][0]
        if grank[l] <= 3:
            fills[(i, 4)] = SOFT["PASS"][0]
    table(s, LM, 1.65, [2.5, 1.5, 2.3, 1.7, 1.7, 1.6], rows, row_h=0.32, size=12, hsize=11.5, fills=fills, first_col_bold=True, aligns={j: PP_ALIGN.CENTER for j in range(1, 6)},
          row_heights=[0.5] + [0.32] * 12)
    best_f, best_g = order[0], glob.iloc[0]["learner"]
    shape(s, LM, 6.4, 12.1, 0.62, TINT)
    tb(s, LM + 0.25, 6.43, 11.6, 0.58, [{"runs": [(f"{PRETTY[best_g]}", True, BLUE), (f" has the lowest global RMSE; ", False, INK), (f"{PRETTY[best_f]}", True, ORANGE),
                                              (f" has the lowest error on the four flow populations. The gaps are small (differences of a few hundredths of a night), so use a top-3 group, not a single winner.", False, INK)]}],
       size=13)
    notes(s, "Why compare across flows: a global metric averages over every guest, while each decision concerns one population. Both T-learner and X-learner rank first on the flow populations; the R-learner leads the global RMSE. We therefore recommend an ensemble of the top group for high-stakes decisions.")


def slide_model_reco():
    s = new_slide("Part 2 · Comparison across flows", "Which model for which job")
    top_unc = "CausalForestDML"
    best_auuc = PRETTY[LB.sort_values("auuc_over_random", ascending=False).iloc[0]["learner"]]
    best_up = PRETTY[LB.sort_values("uplift_at_10", ascending=False).iloc[0]["learner"]]
    flow_best = [PRETTY[l] for l in sorted(MF["learners"], key=lambda l: MF["summary"][l]["rank"])[:3]]
    glob_best = PRETTY[LB.sort_values("pehe_rmse").iloc[0]["learner"]]
    rows = [["Job", "Recommended", "Evidence"],
            ["Headline effect for one guest", f"{glob_best} (current API default), cross-checked by the accuracy-weighted mean", "Lowest global RMSE; weighted mean shown in every guest flow"],
            ["Effect for a segment or policy band", ", ".join(flow_best), "Lowest mean error across the four flow populations"],
            ["Rank guests for targeting", f"{best_auuc} / {best_up}", "Highest AUUC gain and uplift@10% on observed outcomes"],
            ["Effect with a confidence interval", top_unc, "Honest forest gives a 95% interval; used in the guest flows"],
            ["Fuzzy compliance explicitly", "DMLIV", "Instruments Gold with eligibility; effect is for compliers"],
            ["High-stakes decision", "Ensemble of the top group", "All 12 agree on the sign of both threshold decisions"],
            ["Avoid for effect sizes", "LinearDML, Orthogonal Forest", "Lowest correlation and highest error on every view"]]
    table(s, LM, 1.7, [3.3, 4.4, 4.4], rows, row_h=0.62, size=13, hsize=13, first_col_bold=True, zebra=True)
    notes(s, "Decision guide. The dashboard shows every model's estimate for a guest so analysts can see the agreement themselves.")


# ---------------------------------------------------------------- Part 3
STAGE_SHORT = {"setup": "Setup", "density": "Density", "balance": "Balance", "estimates": "Estimates", "robustness": "Robustness", "replication": "Replication",
               "economics": "Economics", "profile": "Profile", "context": "Context", "models": "Models", "evidence": "Evidence", "today": "Today", "candidate": "Candidate",
               "who": "Who", "impact": "Impact", "net": "Net value", "sensitivity": "Sensitivity"}


def slide_framework():
    s = new_slide("Part 3 · Flows", "Four flow types, two examples each, every stage gated")
    types = [("A", "Gold effect read-out", "Does Gold change behaviour, and is it worth the perks?", "A1 fuzzy design  ·  A2 sharp design"),
             ("B", "Validity & negative control", "Can we trust the design, and tell 'no effect' from an effect?", "B1 placebo programme  ·  B2 gamed threshold"),
             ("C", "Guest what-if", "How much does Gold do for this particular guest?", "C1 bleisure couple  ·  C2 corporate road warrior"),
             ("D", "Threshold policy", "What happens if the Gold threshold moves?", "D1 lower to 17  ·  D2 raise to 24")]
    for i, (k, t, q, ex) in enumerate(types):
        y = 1.7 + i * 1.27
        shape(s, LM, y, SW - 2 * LM, 1.15, LIGHT)
        label(s, LM + 0.2, y + 0.2, 0.75, 0.75, k, [BLUE, ORANGE, AQUA, NAVY][i], WHITE, size=26, kind=MSO_SHAPE.OVAL, font=HEAD)
        tb(s, LM + 1.15, y + 0.1, 3.6, 0.4, t, size=16, bold=True, color=NAVY, font=HEAD)
        tb(s, LM + 1.15, y + 0.5, 3.6, 0.6, q, size=11.5, color=GRAY)
        stages = [st["key"] for st in next(f for f in R["flows"] if f["type"] == k)["stages"]]
        cw = 0.93
        for j, st_key in enumerate(stages):
            label(s, LM + 4.95 + j * (cw + 0.04), y + 0.3, cw, 0.5, STAGE_SHORT[st_key], WHITE, NAVY, size=10, radius=0.25)
        tb(s, LM + 4.95, y + 0.84, 7.1, 0.28, ex, size=11, color=GRAY, italic=True)
    notes(s, "Every stage carries a pass, warn, fail or info gate; every flow ends in a verdict. The same JSON drives the dashboard tab, the README and this deck.")


def slide_scorecard():
    s = new_slide("Part 3 · Overall metrics", "Programme scorecard: data, estimates, validity, models, decisions and compute")
    for i, g in enumerate(PROG["groups"]):
        col, row = i % 3, i // 3
        x, y = LM + col * 4.1, 1.7 + row * 2.62
        shape(s, x, y, 3.95, 2.5, LIGHT)
        tb(s, x + 0.18, y + 0.1, 3.6, 0.35, g["title"].replace("Heterogeneity models (room nights, out-of-fold)", "Models (room nights, out-of-fold)"), size=14, bold=True, color=NAVY, font=HEAD)
        mets = g["metrics"][:6]
        tb(s, x + 0.18, y + 0.5, 3.62, 2.0, [{"runs": [(m["label"] + ": ", False, GRAY), (m["value"], True, INK)], "space_after": 2} for m in mets], size=12)
    notes(s, "Overall metrics across all eight flows and all twelve models. Same numbers as the dashboard's programme scorecard.")


def slide_all_flows():
    s = new_slide("Part 3 · Flow-level metrics", "All eight flows at a glance: verdict and headline metric")
    rows = [["Flow", "Verdict", "Headline metrics", "Gates"]]
    fills, colors, bolds = {}, {}, {}
    for i, c in enumerate(PROG["compare"], 1):
        f = FLOWS[c["Flow"].split(" ")[0]]
        gates = " ".join({"PASS": "P", "WARN": "W", "FAIL": "F", "INFO": "i"}[st["status"]] for st in f["stages"])
        rows.append([c["Flow"], c["Verdict"], c["Headline"], gates])
        bg, fg = verdict_colors(c["Verdict"])
        fills[(i, 1)], colors[(i, 1)], bolds[(i, 1)] = bg, fg, True
    table(s, LM, 1.7, [2.65, 2.6, 5.45, 1.4], rows, row_h=0.52, size=12, hsize=12, fills=fills, colors=colors, bolds=bolds, first_col_bold=True)
    tb(s, LM, 6.55, 12, 0.3, "Gates per stage in order: P pass, W warn, F fail, i information only. Money at the assumed margin and perk cost.", size=11, color=GRAY)
    notes(s, "B1 is a false positive by design of the seed: a placebo programme that happens to look significant (p = 0.033); its replication stage shows the method rejects only ~3% of re-draws.")


def slide_gate_map():
    s = new_slide("Part 3 · Stage-level metrics", "Stage gates across all flows: where each flow passes, warns or fails")
    maxst = 7
    rows = [["Flow"] + [str(i) for i in range(1, maxst + 1)] + ["Verdict"]]
    fills, colors, bolds = {}, {}, {}
    for r, fid in enumerate(FLOWS, 1):
        f = FLOWS[fid]
        row = [f["short"].split(" - ")[0] + " " + f["short"].split(" - ")[1][:26]]
        for j in range(maxst):
            if j < len(f["stages"]):
                st = f["stages"][j]
                row.append(STAGE_SHORT[st["key"]])
                bg, fg = SOFT[st["status"]]
                fills[(r, j + 1)], colors[(r, j + 1)] = bg, fg
            else:
                row.append("")
        row.append(f["summary"]["verdict"].title()[:26])
        bg, fg = verdict_colors(f["summary"]["verdict"])
        fills[(r, maxst + 1)], colors[(r, maxst + 1)], bolds[(r, maxst + 1)] = bg, fg, True
        rows.append(row)
    table(s, LM, 1.7, [3.0] + [0.97] * maxst + [2.3], rows, row_h=0.46, size=11, hsize=12, fills=fills, colors=colors, bolds=bolds, first_col_bold=True,
          aligns={j: PP_ALIGN.CENTER for j in range(1, maxst + 1)})
    for i, (k, lab) in enumerate((("PASS", "pass"), ("WARN", "warn"), ("FAIL", "fail"), ("INFO", "information"))):
        label(s, LM + i * 1.6, 6.2, 1.45, 0.4, lab, SOFT[k][0], SOFT[k][1], size=12)
    tb(s, LM + 6.6, 6.2, 5.5, 0.4, "Stage names are those of each flow type; B flows have a seventh (replication) stage.", size=11, color=GRAY, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "Stage-level view. B2 fails density and balance; D flows warn on sensitivity because the decision flips with the perk cost; C2 warns on economics because it is break-even.")


def slide_money():
    s = new_slide("Part 3 · Flow-level metrics", "From revenue to net value: the economics stage in every flow")
    cats, g, c, p, n = [], [], [], [], []
    for fid, lab in (("A1", "A1 Gold member"), ("A2", "A2 sharp design"), ("C1", "C1 bleisure couple"), ("C2", "C2 corporate")):
        eco = next(st for st in FLOWS[fid]["stages"] if st["key"] == "economics")["chart"]["values"]
        cats.append(lab)
        g.append(eco[0]), c.append(eco[1]), p.append(eco[2]), n.append(eco[3])
    ch = bar_chart(s, LM, 1.7, 7.6, 4.8, cats, {"Revenue": g, "Contribution": c, "Perk cost": p, "Net": n}, [MID, BLUE, ORANGE, AQUA], fmt="$#,##0", legend=XL_LEGEND_POSITION.TOP, size=10, gap=50, overlap=-5)
    ch.has_title = True
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    ch.chart_title.text_frame.text = "Per Gold member per year (USD)"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = False
    card(s, LM + 7.9, 1.65, 4.2, 2.35, "How money is computed", [
        {"t": "Revenue = extra room nights x ADR (+ ancillary for panel flows).", "bullet": True, "space_after": 3},
        {"t": f"Contribution = revenue x {ASSUME['contribution_margin']:.0%}.", "bullet": True, "space_after": 3},
        {"t": f"Net = contribution - ${ASSUME['perk_cost_per_member']:.0f} perk cost.", "bullet": True}], body_size=13)
    d1, d2 = FLOWS["D1"]["kpi"], FLOWS["D2"]["kpi"]
    card(s, LM + 7.9, 4.2, 4.2, 2.25, "Break-even perk cost", [
        {"t": f"D1 lower to 17: adopt while perks cost under {money(d1['break_even_cost'])} per member.", "bullet": True, "space_after": 3},
        {"t": f"D2 raise to 24: adopt when perks cost more than {money(d2['break_even_cost'])}.", "bullet": True}], body_size=13)
    notes(s, "Both margin and perk cost are assumptions, editable under flows: in the config. Direct-booking gains are not monetised, which makes the net conservative.")


# ---------------------------------------------------------------- Part 4: scenarios
SCENARIO = {
    "A1": ("Monday: the VP asks whether Gold is worth it", ["estimates", "robustness"], "Is Gold worth the perks?"),
    "A2": ("Tuesday: what if the rule were strict?", ["setup", "estimates"], "Do sharp and fuzzy designs agree?"),
    "B1": ("Wednesday: the CFO tests the method on a dud programme", ["estimates", "replication"], "Would the method cry wolf on a programme that does nothing?"),
    "B2": ("Thursday: marketing sees guests topping up stays", ["density", "replication"], "Can we still trust the estimate if the threshold is gamed?"),
    "C1": ("Friday: who gets the proactive Gold offer?", ["models", "economics"], "How much does Gold do for a bleisure couple?"),
    "C2": ("Friday: and the corporate road warrior?", ["context", "economics"], "Should we spend perks on a rigid corporate traveller?"),
    "D1": ("Next quarter: open Gold to 17 nights?", ["who", "net"], "What if we lower the threshold by three nights?"),
    "D2": ("Next quarter: tighten Gold to cut perk cost?", ["net", "sensitivity"], "What if we raise the threshold to 24 nights at $300 perks?"),
}


def key_metrics(st, n=3):
    return " · ".join(f"{m['label']} {m['value']}" for m in st["metrics"][:n])


def scenario_slides(fid):
    f = FLOWS[fid]
    headline, shot_keys, question = SCENARIO[fid]
    ftype = R["flow_types"][f["type"]]["title"]
    # ---- slide 1: story, verdict, flow metrics, stage table
    s = new_slide(f"Part 4 · Scenario {fid} · {ftype}", headline)
    tb(s, LM, 1.7, 4.7, 0.7, [{"runs": [("The question: ", True, NAVY), (question, False, INK)]}], size=14)
    tb(s, LM, 2.45, 4.7, 1.4, f["story"], size=12.5, color=GRAY)
    bg, fg = verdict_colors(f["summary"]["verdict"])
    shape(s, LM, 3.9, 4.7, 1.0, bg)
    tb(s, LM + 0.18, 3.96, 4.35, 0.35, f["summary"]["verdict"], size=16, bold=True, color=fg, font=HEAD)
    tb(s, LM + 0.18, 4.3, 4.35, 0.6, f["summary"]["reason"], size=11, color=fg)
    mets = f["summary"]["metrics"][:6]
    for i, m in enumerate(mets):
        col, row = i % 3, i // 3
        x, y = LM + col * 1.6, 5.05 + row * 0.95
        shape(s, x, y, 1.5, 0.88, LIGHT)
        tb(s, x + 0.07, y + 0.03, 1.38, 0.4, m["value"], size=14, bold=True, color=BLUE, anchor=MSO_ANCHOR.MIDDLE)
        tb(s, x + 0.07, y + 0.43, 1.38, 0.42, m["label"], size=10, color=GRAY)
    rows = [["#", "Stage", "Gate", "Stage-level metrics"]]
    fills, colors, bolds = {}, {}, {}
    for i, st in enumerate(f["stages"], 1):
        rows.append([str(i), STAGE_SHORT[st["key"]], st["status"], key_metrics(st, 3)])
        bg2, fg2 = SOFT[st["status"]]
        fills[(i, 2)], colors[(i, 2)], bolds[(i, 2)] = bg2, fg2, True
    n = len(rows)
    rh = [0.38] + [min(0.78, 4.9 / (n - 1))] * (n - 1)
    table(s, LM + 5.0, 1.7, [0.4, 1.3, 0.75, 4.65], rows, size=11, hsize=11.5, fills=fills, colors=colors, bolds=bolds, aligns={0: PP_ALIGN.CENTER, 2: PP_ALIGN.CENTER},
          row_heights=rh)
    notes(s, f"Scenario {fid}: {f['title']}. {f['story']} Verdict: {f['summary']['verdict']}. {f['summary']['reason']}")
    # ---- slide 2: dashboard evidence
    s2 = new_slide(f"Part 4 · Scenario {fid} · In the dashboard", f"{fid}: what the Streamlit flow showed")
    w = (SW - 2 * LM - 0.3) / 2
    for i, k in enumerate(shot_keys):
        st = next(x for x in f["stages"] if x["key"] == k)
        x = LM + i * (w + 0.3)
        tb(s2, x, 1.65, w, 0.35, f"Stage {f['stages'].index(st) + 1}: {st['title']}", size=14, bold=True, color=NAVY)
        picture(s2, SHOTS / "flows" / f"{fid}_{k}.png", x, 2.05, w, 3.7, max_w=1500)
    tk = [next(x for x in f["stages"] if x["key"] == k) for k in shot_keys]
    tb(s2, LM, 5.95, SW - 2 * LM, 1.0, [{"runs": [(f"{STAGE_SHORT[st['key']]}: ", True, BLUE), (st["takeaway"], False, INK)], "space_after": 4} for st in tk], size=12.5)
    notes(s2, "Screenshots are captured from the live Streamlit app by scripts/capture_screenshots.py; the numbers on screen equal the API's.")


# ---------------------------------------------------------------- wrap-up
def slide_dashboard():
    s = new_slide("Part 4 · Dashboard and API", "The dashboard has six tabs; the API serves the same models")
    tabs = [("10_ui_rdd_effect.png", "Gold effect (RDD)"), ("11_ui_validity_checks.png", "Validity checks"), ("12_ui_guest_explorer.png", "Guest explorer"),
            ("14_ui_threshold_simulator.png", "Threshold simulator"), ("15_ui_model_leaderboard.png", "Model leaderboard"), ("16_ui_end_to_end_flows.png", "End-to-end flows")]
    w, h = 3.55, 2.2
    for i, (f, t) in enumerate(tabs):
        col, row = i % 3, i // 3
        x, y = LM + col * 4.2, 1.7 + row * 2.6
        picture(s, top_crop(SHOTS / f, w / h, f[:2]), x, y, w, h, max_w=1500)
        tb(s, x, y + h + 0.07, w, 0.3, t, size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    notes(s, "A tour of the six tabs (top of each page). The last tab is the end-to-end flows used in part 4.")


def slide_api():
    s = new_slide("Part 4 · Dashboard and API", "Five endpoints, validated inputs, identical numbers to the dashboard")
    picture(s, top_crop(SHOTS / "03_api_predict_cate.png", 1.05, "api"), LM, 1.7, 4.6, 4.5, max_w=1400)
    rows = [["Endpoint", "What it returns"],
            ["GET /health", "status, cutoff, regime, best learner, GPU info"],
            ["POST /predict/late", "RDD effect for any outcome, kernel, bandwidth, order, design"],
            ["POST /predict/cate", "individual effect, percentile, segment, advice, every learner's estimate"],
            ["POST /simulate/threshold", "guests affected, incremental nights, revenue, net value, warnings"],
            ["GET /leaderboard", "out-of-fold learner metrics"]]
    table(s, LM + 5.0, 1.7, [2.6, 4.5], rows, row_h=0.6, size=12, hsize=12, first_col_bold=True, zebra=True)
    card(s, LM + 5.0, 5.55, 7.1, 1.35, "Guard rails", [{"t": "Pydantic v2 domain validation (ADR 30 to 2000 USD, shares in [0,1], known regions); 422 on bad input, 404 unknown learner, 503 without models. "
                                                  "Every flow's UI number is checked against the API in the capture report.", "bullet": False}], body_size=12, title_size=14)
    notes(s, "Swagger screenshot is the executed predict/cate call for the bleisure-couple profile.")


def slide_takeaways():
    s = new_slide("Wrap-up", "Five takeaways")
    d1 = FLOWS["D1"]["kpi"]
    c1, c2 = FLOWS["C1"]["kpi"], FLOWS["C2"]["kpi"]
    items = [("Causal", f"Gold raises room nights by about {FLOWS['A1']['kpi']['headline_effect']:+.1f}, spend by $424 and direct share by 0.11; the design passes every validity check."),
             ("Heterogeneous", f"Effects range from {c2['effect']:+.1f} (corporate) to {c1['effect']:+.1f} (bleisure) nights; the top 10% of guests respond about {LB['uplift_at_10'].max() / LB['ate_true'].iloc[0]:.1f}x the average."),
             ("Robust to model choice", "All 12 models agree on the sign of both threshold decisions; use the top group, not one winner."),
             ("Decision-ready", f"Lowering to 17 nights adds about {money(d1['net_margin'])} a year at the assumed margin; tightening pays only if perks cost over {money(FLOWS['D2']['kpi']['break_even_cost'])}."),
             ("Honest about limits", "Synthetic data, assumed margin and perk cost, extrapolation beyond 5 nights, and one placebo draw that looks significant.")]
    for i, (t, d) in enumerate(items):
        y = 1.7 + i * 1.02
        shape(s, LM, y, SW - 2 * LM, 0.92, LIGHT)
        label(s, LM + 0.2, y + 0.17, 0.6, 0.6, str(i + 1), BLUE, WHITE, size=18, kind=MSO_SHAPE.OVAL, font=HEAD)
        tb(s, LM + 1.05, y + 0.08, 2.6, 0.76, t, size=16, bold=True, color=NAVY, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
        tb(s, LM + 3.7, y + 0.08, 8.2, 0.76, d, size=13, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "Summary slide for the decision-makers.")


def slide_limits():
    s = new_slide("Wrap-up", "Limits, caveats and next steps")
    card(s, LM, 1.7, 5.9, 5.1, "Caveats", [
        {"t": "All data are synthetic; real loyalty data will add sorting, noisy tier credit and gaming that the diagnostics are built to catch.", "bullet": True, "space_after": 6},
        {"t": "B1 (placebo) looks significant with the default seed; over 30 re-draws only about 3% do. One draw proves nothing either way.", "bullet": True, "space_after": 6},
        {"t": "Money uses an assumed 40% margin and $120 perk cost; direct-booking gains are not monetised.", "bullet": True, "space_after": 6},
        {"t": "Learner effects are local to guests near 20 nights; beyond about 5 nights the simulator is directional only.", "bullet": True, "space_after": 6},
        {"t": "CATE models exist for room nights only; the RDD covers all three outcomes.", "bullet": True}], body_size=15)
    card(s, LM + 6.2, 1.7, 5.9, 5.1, "Next steps", [
        {"t": "Run the pipeline on real loyalty data: start with the density and balance gates.", "bullet": True, "space_after": 6},
        {"t": "Train learners for ancillary spend and direct-booking ratio.", "bullet": True, "space_after": 6},
        {"t": "Replace the assumed margin and perk cost with finance's numbers; add a margin slider to the flows tab.", "bullet": True, "space_after": 6},
        {"t": "Replace the single headline model with the top-3 ensemble in the API.", "bullet": True, "space_after": 6},
        {"t": "Add a geographic or time dimension to the flows once the panel supports it.", "bullet": True}], body_size=15, fill=TINT)
    notes(s, "Closing slide: be explicit about what the evidence does and does not support.")


def slide_appendix_files():
    s = new_slide("Appendix", "Where everything lives")
    rows = [["Path", "Contents"],
            ["src/data/synthetic_generator.py", "Panel generator: sharp, fuzzy, gamed, placebo (effect_scale)"],
            ["src/models/rdd_estimators.py", "Local polynomial, 2SLS, IK/CCT, McCrary, balance, placebo, donut"],
            ["src/models/cate_learners.py", "The 12 learners and the Optuna tuner"],
            ["src/models/trainer.py", "End-to-end training, out-of-fold evaluation, artifacts"],
            ["src/evaluation/", "causal_metrics.py (AUUC, Qini, uplift, policy, R-loss), visualizer.py (8 plots)"],
            ["src/flows.py, src/ui/flows_ui.py", "8 flows, scorecard, model-by-flow matrix; the dashboard tab"],
            ["src/api/", "FastAPI app, Pydantic schemas, shared ModelService"],
            ["outputs/metrics, models, plots, screenshots", "Leaderboards, Optuna trials, flows_report.json; models; 300-DPI plots; 75 screenshots"],
            ["presentation/", "This deck and its build script output"]]
    table(s, LM, 1.7, [4.6, 7.5], rows, row_h=0.5, size=12, hsize=12, first_col_bold=True, zebra=True)
    notes(s, "File map for reviewers.")


def slide_appendix_glossary():
    s = new_slide("Appendix", "Glossary")
    rows = [["Term", "Meaning"],
            ["RDD", "Regression discontinuity design: compare guests just above and below a cutoff"],
            ["Sharp / fuzzy", "Gold assigned strictly by the cutoff / cutoff changes the probability of Gold"],
            ["LATE / CATE", "Local average effect at the cutoff / effect conditional on guest covariates"],
            ["McCrary test", "Checks that the density of nights is continuous at the cutoff (no gaming)"],
            ["Holm p", "p-value adjusted for testing 10 covariates at once"],
            ["Donut hole", "Re-estimate after dropping guests with exactly 19 and 20 nights"],
            ["AUUC / Qini", "Area under the uplift curve / Radcliffe's incremental-gain curve"],
            ["R-loss / PEHE", "Residual loss usable without truth / error of individual effects against truth"],
            ["Out-of-fold", "Predictions for guests the model did not train on (3-fold cross-fitting)"]]
    table(s, LM, 1.7, [2.6, 9.5], rows, row_h=0.5, size=12.5, hsize=12.5, first_col_bold=True, zebra=True)
    notes(s, "Reference.")


# ============================================================================= assemble
def build():
    slide_title()
    slide_exec()
    slide_agenda()

    divider("1", "Problem, approach and the uplift end-to-end flow",
            ["Why a discontinuity answers a causal question", "The pipeline documented stage by stage, with artifacts and metrics", "How uplift is measured, and the architecture behind it"])
    slide_problem()
    slide_strategy()
    slide_data()
    slide_pipeline()
    slide_stage_doc()
    slide_weights()
    slide_metrics_explained()
    slide_architecture()
    slide_run()

    divider("2", "The 12 models and how they compare",
            ["RDD evidence, then every model explained family by family", "Tuning and the leaderboard on every metric", "Comparison across the four model-driven flows"])
    slide_rdd()
    slide_validity()
    slide_robust()
    slide_models_overview()
    family_slide("Meta-learners: S, T and X turn any regressor into an effect estimator", ["s_learner", "t_learner", "x_learner"],
                 "Meta-learners reuse standard GPU gradient boosting, so they are cheap to train and the strongest value for money here.",
                 "S, T and X differ in how they split the data between treated and control guests.")
    family_slide("Orthogonal learners: R and DR protect against nuisance error", ["r_learner", "dr_learner"],
                 "Both cross-fit their nuisance models, so errors in the outcome or propensity models do not bias the effect to first order.",
                 "R-learner has the lowest global RMSE; DR is close behind and doubly robust.")
    family_slide("Double ML and forests: econml estimators with intervals", ["linear_dml", "nonparam_dml", "causal_forest_dml"],
                 "CausalForestDML adds confidence intervals; LinearDML cannot represent the bell-shaped heterogeneity, so it trails by design.",
                 "The linear model is kept as an interpretable baseline.")
    family_slide("Orthogonal forest and DMLIV: the slowest estimator and the compliance-aware one", ["orthogonal_forest", "dmliv"],
                 "The orthogonal forest is distilled because the library predicts at about 0.25 s per row; DMLIV is the only learner that uses eligibility as an instrument.",
                 "ORF is retained for completeness and is the weakest model.")
    family_slide("Deep uplift networks: TARNet and DragonNet on CUDA", ["tarnet", "dragonnet"],
                 "Both train on the GPU with kernel-weighted losses and early stopping; the loss curves are in outputs/plots/08.",
                 "DragonNet adds a propensity head and targeted regularization on top of TARNet.")
    slide_tuning()
    slide_leaderboard()
    slide_charts_acc()
    slide_charts_uplift()
    slide_cost_accuracy()
    slide_model_flow_matrix()
    slide_model_flow_net()
    slide_model_ranking()
    slide_model_reco()

    divider("3", "Flows, stages and metrics",
            ["Eight flows in four types", "Stage-level gates, flow-level verdicts, the programme scorecard", "The money logic behind every verdict"])
    slide_framework()
    slide_scorecard()
    slide_all_flows()
    slide_gate_map()
    slide_money()

    divider("4", "Scenarios as a story",
            ["Eight scenarios told with the Streamlit dashboard", "Each: the question, the stage metrics, the verdict, and screenshots of the stages that matter", "From a Monday read-out to next quarter's threshold decision"])
    for fid in ("A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"):
        scenario_slides(fid)
    slide_dashboard()
    slide_api()
    slide_takeaways()
    slide_limits()
    slide_appendix_files()
    slide_appendix_glossary()
    prs.save(FILE)
    print(f"saved {FILE} with {len(prs.slides)} slides")


if __name__ == "__main__":
    build()
