"""Small python-pptx helper layer: consistent styling, tables, charts, chevrons, image fitting."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

INK, MUTED, LIGHT, WHITE = "1F2430", "6B7280", "F3F5F9", "FFFFFF"
BLUE, ORANGE, TEAL, PURPLE, ROSE, GREY = "2A6FDB", "E8833A", "1F9D8A", "7B5CD6", "C8466B", "8A8F98"
GOOD = "DDF3EA"
FAMILY_COLOR = {"naive": ROSE, "iv": BLUE, "deep_iv": PURPLE, "cate_meta": TEAL, "cate_deep": ORANGE}
FAMILY_LABEL = {"naive": "Naive (biased)", "iv": "IV / econometric", "deep_iv": "Deep IV / GMM",
                "cate_meta": "Meta-learner / forest", "cate_deep": "Deep CATE net"}
FONT = "Calibri"
W, H = 13.333, 7.5


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


class Deck:
    def __init__(self, footer: str):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.footer = footer
        self.n = 0

    # ---------------------------------------------------------------- slides
    def slide(self, title: str, kicker: Optional[str] = None, section: Optional[str] = None):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        if section:
            self.text(s, 0.6, 0.28, 8, 0.3, [(section.upper(), 10, True, BLUE)])
        self.text(s, 0.6, 0.52, 12.1, 0.9, [(title, 22, True, INK)], anchor="top")
        if kicker:
            self.text(s, 0.6, 1.27, 12.1, 0.4, [(kicker, 12, False, MUTED)], anchor="top")
        self.text(s, 0.6, 7.08, 9, 0.3, [(self.footer, 9, False, GREY)])
        self.text(s, 12.0, 7.08, 0.8, 0.3, [(str(self.n), 9, False, GREY)], align="right")
        return s

    def title_slide(self, title: str, subtitle: str, meta: str):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(H))
        bg.fill.solid(); bg.fill.fore_color.rgb = rgb(INK); bg.line.fill.background()
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(2.2), Inches(0.12), Inches(2.2))
        bar.fill.solid(); bar.fill.fore_color.rgb = rgb(ORANGE); bar.line.fill.background()
        self.text(s, 1.2, 2.1, 11, 1.6, [(title, 40, True, WHITE)], anchor="top")
        self.text(s, 1.2, 3.75, 11, 0.9, [(subtitle, 18, False, "C9D1E0")], anchor="top")
        self.text(s, 1.2, 6.4, 11, 0.5, [(meta, 12, False, "9AA4B8")])
        return s

    def section_slide(self, num: str, title: str, sub: str):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(H))
        bg.fill.solid(); bg.fill.fore_color.rgb = rgb(BLUE); bg.line.fill.background()
        self.text(s, 1.0, 2.3, 3, 1.2, [(num, 60, True, "FFFFFF")])
        self.text(s, 1.0, 3.5, 11, 1.0, [(title, 34, True, WHITE)], anchor="top")
        self.text(s, 1.0, 4.4, 11, 1.0, [(sub, 16, False, "DCE6FA")], anchor="top")
        return s

    # ---------------------------------------------------------------- primitives
    def text(self, s, x, y, w, h, paras: Sequence[Tuple], anchor: str = "top", align: str = "left", bullet: bool = False):
        """paras: (text, size, bold, color[, italic])."""
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.04)
        tf.margin_top = tf.margin_bottom = Inches(0.02)
        tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[anchor]
        for i, p in enumerate(paras):
            text, size, bold, color = p[0], p[1], p[2], p[3]
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = {"left": PP_ALIGN.LEFT, "right": PP_ALIGN.RIGHT, "center": PP_ALIGN.CENTER}[align]
            para.space_after = Pt(size * 0.45)
            run = para.add_run()
            run.text = ("•  " + text) if bullet else text
            f = run.font
            f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, (p[4] if len(p) > 4 else False)
            f.color.rgb = rgb(color)
        return tb

    def bullets(self, s, x, y, w, h, items: Sequence[str], size: int = 15, color: str = INK, head: Optional[str] = None):
        paras = []
        if head:
            paras.append((head, size + 1, True, BLUE))
        paras += [("•  " + t, size, False, color) for t in items]
        return self.text(s, x, y, w, h, paras)

    def rect(self, s, x, y, w, h, fill: str = LIGHT, line: Optional[str] = None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
        r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        r.fill.solid(); r.fill.fore_color.rgb = rgb(fill)
        if line:
            r.line.color.rgb = rgb(line); r.line.width = Pt(1)
        else:
            r.line.fill.background()
        if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
            r.adjustments[0] = 0.08
        r.shadow.inherit = False
        return r

    def card(self, s, x, y, w, h, value: str, label: str, color: str = BLUE, sub: Optional[str] = None, vsize: int = 28):
        self.rect(s, x, y, w, h, LIGHT)
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.07), Inches(h))
        bar.fill.solid(); bar.fill.fore_color.rgb = rgb(color); bar.line.fill.background()
        paras = [(value, vsize, True, INK), (label, 11, False, MUTED)]
        if sub:
            paras.append((sub, 10, False, GREY))
        self.text(s, x + 0.18, y + 0.08, w - 0.25, h - 0.1, paras, anchor="middle")

    def box(self, s, x, y, w, h, title: str, body: Sequence[str], color: str = BLUE, size: int = 12, fill: str = LIGHT):
        self.rect(s, x, y, w, h, fill)
        hdr = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.38))
        hdr.fill.solid(); hdr.fill.fore_color.rgb = rgb(color); hdr.line.fill.background()
        self.text(s, x + 0.08, y + 0.03, w - 0.16, 0.32, [(title, 12, True, WHITE)], anchor="middle")
        self.text(s, x + 0.1, y + 0.45, w - 0.2, h - 0.5, [("•  " + b, size, False, INK) for b in body])

    def chevrons(self, s, x, y, w, h, labels: Sequence[str], colors: Optional[Sequence[str]] = None, size: int = 12):
        n = len(labels)
        seg = w / n
        for i, lab in enumerate(labels):
            shp = s.shapes.add_shape(MSO_SHAPE.CHEVRON if i else MSO_SHAPE.PENTAGON, Inches(x + i * seg), Inches(y), Inches(seg + 0.12), Inches(h))
            shp.fill.solid(); shp.fill.fore_color.rgb = rgb((colors or [BLUE] * n)[i % len(colors or [BLUE])])
            shp.line.color.rgb = rgb(WHITE)
            shp.shadow.inherit = False
            # text goes in an overlaid text box: a chevron's own text area is too narrow and wraps mid-word
            lines = [(ln, size, True, WHITE) for ln in lab.split("\n")]
            self.text(s, x + i * seg + (0.3 if i else 0.08), y, seg - 0.2 - (0.1 if i == 0 else 0.0), h, lines,
                      anchor="middle", align="center")

    def arrow(self, s, x1, y1, x2, y2, color: str = MUTED):
        c = s.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        c.line.color.rgb = rgb(color); c.line.width = Pt(2)
        ln = c.line._get_or_add_ln()
        from lxml import etree
        tail = etree.SubElement(ln, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd")
        tail.set("type", "triangle")
        return c

    def node(self, s, x, y, w, h, label: str, fill: str = BLUE, size: int = 11, dashed: bool = False, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
        shp = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        shp.fill.solid(); shp.fill.fore_color.rgb = rgb(fill)
        shp.line.color.rgb = rgb(INK if dashed else fill)
        if dashed:
            shp.line.dash_style = 4
            shp.line.width = Pt(1.5)
        shp.shadow.inherit = False
        tf = shp.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.05)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = label
        r.font.name, r.font.size, r.font.bold = FONT, Pt(size), True
        r.font.color.rgb = rgb(WHITE if fill not in (LIGHT, WHITE, GOOD) else INK)
        return shp

    # ---------------------------------------------------------------- table
    def table(self, s, x, y, w, rows: List[List[str]], col_w: Optional[Sequence[float]] = None, size: int = 10,
              row_h: float = 0.28, header_fill: str = INK, highlight: Optional[Dict[Tuple[int, int], str]] = None,
              bold_first_col: bool = False, align: Optional[Sequence[str]] = None, fills: Optional[Dict[int, str]] = None):
        nr, nc = len(rows), len(rows[0])
        gs = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
        tbl = gs.table
        tblPr = gs._element.graphic.graphicData.tbl.tblPr
        tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")
        if col_w:
            tot = sum(col_w)
            for i, cw in enumerate(col_w):
                tbl.columns[i].width = Inches(w * cw / tot)
        for r in range(nr):
            tbl.rows[r].height = Inches(row_h)
            for c in range(nc):
                cell = tbl.cell(r, c)
                cell.margin_left = cell.margin_right = Inches(0.05)
                cell.margin_top = cell.margin_bottom = Inches(0.015)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.fill.solid()
                if r == 0:
                    cell.fill.fore_color.rgb = rgb(header_fill)
                elif highlight and (r, c) in highlight:
                    cell.fill.fore_color.rgb = rgb(highlight[(r, c)])
                elif fills and r in fills:
                    cell.fill.fore_color.rgb = rgb(fills[r])
                else:
                    cell.fill.fore_color.rgb = rgb(LIGHT if r % 2 == 0 else WHITE)
                tf = cell.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                a = (align[c] if align else ("left" if c == 0 else "center"))
                p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[a]
                # an empty run falls back to the 18pt default paragraph size and inflates the row height
                run = p.add_run(); run.text = str(rows[r][c]) or " "
                run.font.name, run.font.size = FONT, Pt(size)
                run.font.bold = (r == 0) or (bold_first_col and c == 0)
                run.font.color.rgb = rgb(WHITE if r == 0 else INK)
        return tbl

    # ---------------------------------------------------------------- image
    def image(self, s, path: Path, x: float, y: float, w: float, h: float, border: bool = True):
        """Fit the image inside the (w, h) box, preserving aspect ratio, centred."""
        iw, ih = Image.open(path).size
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        px, py = x + (w - pw) / 2, y + (h - ph) / 2
        pic = s.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
        if border:
            pic.line.color.rgb = rgb("D5DAE3"); pic.line.width = Pt(0.75)
        return pic

    # ---------------------------------------------------------------- charts
    def bar_chart(self, s, x, y, w, h, cats: Sequence[str], series: Dict[str, Sequence[float]], colors: Sequence[str],
                  horizontal: bool = True, title: Optional[str] = None, fmt: str = "0.00", point_colors: Optional[Sequence[str]] = None,
                  legend: bool = False, font: int = 10, max_val: Optional[float] = None, labels: bool = True):
        cd = CategoryChartData()
        cd.categories = list(cats)
        for name, vals in series.items():
            cd.add_series(name, [float(v) for v in vals])
        kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
        gf = s.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
        ch = gf.chart
        ch.font.size, ch.font.name = Pt(font), FONT
        ch.has_title = bool(title)
        if title:
            ch.chart_title.text_frame.text = title
            ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(font + 2)
            ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
        ch.has_legend = legend
        if legend:
            ch.legend.position = XL_LEGEND_POSITION.BOTTOM
            ch.legend.include_in_layout = False
        va = ch.value_axis
        va.has_major_gridlines = True
        va.major_gridlines.format.line.color.rgb = rgb("E5E7EB")
        va.format.line.fill.background()
        va.tick_labels.font.color.rgb = rgb(MUTED)
        if max_val:
            va.maximum_scale = max_val
        va.minimum_scale = 0
        ca = ch.category_axis
        ca.tick_labels.font.size = Pt(font)
        ca.format.line.color.rgb = rgb("D5DAE3")
        if horizontal:
            ca.reverse_order = True
            va.tick_label_position  # noqa: B018
        plot = ch.plots[0]
        plot.gap_width = 60
        plot.vary_by_categories = False
        for i, ser in enumerate(plot.series):
            ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(colors[i % len(colors)])
            if labels:
                ser.data_labels.show_value = True
                ser.data_labels.number_format = fmt
                ser.data_labels.number_format_is_linked = False
                ser.data_labels.font.size = Pt(font - 1)
                ser.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        if point_colors and len(plot.series) == 1:
            for i, c in enumerate(point_colors):
                pt = plot.series[0].points[i]
                pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(c)
        return ch

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(path))
