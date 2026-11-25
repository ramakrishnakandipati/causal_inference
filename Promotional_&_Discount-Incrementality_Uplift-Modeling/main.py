"""Master CLI orchestrator.

python main.py --stage all               # data -> train -> evaluate -> plots -> screenshots
python main.py --stage all --quick       # fast smoke run (15k guests, few trials)
python main.py --stage serve-api | serve-ui
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import get_logger, load_config  # noqa: E402

log = get_logger()
STAGES = ["data", "train", "evaluate", "plots", "screenshots"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", *STAGES, "serve-api", "serve-ui", "test"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ablation", action="store_true", help="also benchmark LightGBM / CatBoost base learners")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-screenshots", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if a.no_gpu:
        cfg["gpu"]["enabled"] = False
    todo = STAGES if a.stage == "all" else [a.stage]
    if a.no_screenshots and "screenshots" in todo:
        todo.remove("screenshots")
    for st in todo:
        log.info("===== stage: %s =====", st)
        if st == "data":
            from data.synthetic_generator import main as gen
            gen(n=15000 if a.quick else None)
        elif st == "train":
            from src.train import run
            run(cfg, quick=a.quick, ablation=a.ablation or a.stage == "all")
        elif st == "evaluate":
            from src.evaluate import run
            run(cfg)
        elif st == "plots":
            from src.visualize import run
            run(cfg)
        elif st == "screenshots":
            subprocess.run([sys.executable, str(ROOT / "scripts" / "capture_screenshots.py")], check=True)
        elif st == "serve-api":
            subprocess.run([sys.executable, "-m", "uvicorn", "api.app:app", "--port", "8000"], cwd=ROOT)
        elif st == "serve-ui":
            subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py"], cwd=ROOT)
        elif st == "test":
            subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT)


if __name__ == "__main__":
    main()
