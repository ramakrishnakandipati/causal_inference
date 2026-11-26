"""Shared matplotlib style: thin marks, recessive grid, fixed categorical order (never cycled by rank)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order; colour follows the entity (model family / segment), not its rank.
CAT = ["#2A6FDB", "#E8833A", "#1F9D8A", "#7B5CD6", "#C8466B", "#8A8F98"]
FAMILY_COLOR = {"naive": "#C8466B", "iv": "#2A6FDB", "deep_iv": "#7B5CD6", "cate_meta": "#1F9D8A", "cate_deep": "#E8833A"}
FAMILY_LABEL = {"naive": "Naive (biased)", "iv": "IV / econometric", "deep_iv": "Deep IV / GMM",
                "cate_meta": "Meta-learner / forest", "cate_deep": "Deep CATE net"}
SEG_COLOR = {"leisure": CAT[1], "corporate": CAT[0], "group": CAT[2]}
INK, MUTED, GRID = "#1F2430", "#6B7280", "#E5E7EB"


def apply() -> None:
    # causalml/econml switch matplotlib to a global style on import; reset it before applying ours
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titleweight": "semibold", "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
    })
