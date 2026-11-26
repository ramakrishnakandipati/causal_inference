"""Build outputs/metrics/flows.json (8 end-to-end flows + programme scorecard).  python scripts/build_flows.py"""
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from src.flows import build_all  # noqa: E402

if __name__ == "__main__":
    out = build_all()
    for fid in out["order"]:
        f = out["flows"][fid]
        print(f"{fid}: {f['verdict']:40s} | " + " | ".join(f"{m['label']}={m['value']}" for m in f["flow_metrics"][:3]))
