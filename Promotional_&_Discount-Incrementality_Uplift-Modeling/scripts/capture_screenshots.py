"""Start FastAPI + Streamlit, capture Swagger / API call / Streamlit tabs & presets with Playwright, then tear down."""
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_config, resolve  # noqa: E402

API, UI = 8000, 8501


def wait(url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            time.sleep(1.5)
    raise TimeoutError(url)


def idle(pg, timeout=180):
    """Wait until Streamlit finished (re)running: the status widget disappears and the app is not 'stale'."""
    t0 = time.time()
    pg.wait_for_selector("[data-testid=stAppViewContainer]", timeout=timeout * 1000)
    while time.time() - t0 < timeout:
        running = pg.locator("[data-testid=stStatusWidget]").count() > 0
        stale = pg.locator("[data-stale=true]").count() > 0
        if not running and not stale:
            break
        time.sleep(0.5)
    pg.wait_for_timeout(2000)


def capture_flows(pg, out, ui_url, api_url):
    """Screenshot every step of the 5 end-to-end flows and check that the UI decision equals the API decision."""
    from src.flows import FLOWS, guest_payload, load_flow_data, pick_guest
    te, _ = load_flow_data()
    fdir = out / "flows"; fdir.mkdir(parents=True, exist_ok=True)
    best = requests.get(f"{api_url}/health").json()["default_estimator"]
    pg.set_viewport_size({"width": 1500, "height": 5200})   # tall enough that a whole flow is on screen (element shots clip at the viewport)
    pg.goto(ui_url); idle(pg)
    pg.add_style_tag(content="header[data-testid=stHeader]{display:none !important}")
    pg.get_by_role("tab", name="End-to-End Flows").click(); pg.wait_for_timeout(2500); idle(pg)
    checks = {}
    for f in FLOWS:
        pg.locator("[data-testid=stRadio]").get_by_text(f["short"], exact=True).click()
        pg.wait_for_timeout(2000); idle(pg)
        fid = f["id"]
        pg.locator(f".st-key-{fid}_full").wait_for(timeout=60000)
        pg.locator(f".st-key-{fid}_full").screenshot(path=fdir / f"{fid}_{f['slug']}_full.png")
        for k in range(1, 6):
            pg.locator(f".st-key-{fid}_step{k}").screenshot(path=fdir / f"{fid}_{f['slug']}_step{k}.png")
        if f["kind"] == "guest":
            payload = guest_payload(pick_guest(te, f["persona"]))
            api = requests.post(f"{api_url}/score", json={"guest": payload, "estimator": best}).json()
            ui_text = pg.locator(f".st-key-{fid}_step4").inner_text()
            checks[fid] = {"persona": f["persona"], "api_quadrant": api["quadrant"], "api_decision": api["decision"],
                           "api_tau_margin_hat": round(api["tau_margin_hat"], 3),
                           "ui_shows_same_quadrant": api["quadrant"] in ui_text, "ui_shows_same_decision": api["decision"] in ui_text}
    (fdir / "flow_api_vs_ui_check.json").write_text(json.dumps(checks, indent=2))
    print("flows ->", fdir, "| UI==API:", all(v["ui_shows_same_decision"] and v["ui_shows_same_quadrant"] for v in checks.values()))


def main(only: str = "all"):
    out = resolve(load_config()["paths"]["screenshots"]); out.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    procs = [subprocess.Popen([sys.executable, "-m", "uvicorn", "api.app:app", "--port", str(API)], cwd=ROOT, creationflags=flags,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
             subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.port", str(UI),
                               "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT, creationflags=flags,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    try:
        wait(f"http://localhost:{API}/health"); wait(f"http://localhost:{UI}/_stcore/health")
        from playwright.sync_api import sync_playwright
        presets = requests.get(f"http://localhost:{API}/presets").json()
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1500, "height": 1000})
            if only == "flows":
                capture_flows(pg, out, f"http://localhost:{UI}", f"http://localhost:{API}")
                b.close()
                return
            pg.goto(f"http://localhost:{API}/docs"); pg.wait_for_selector(".opblock")
            pg.screenshot(path=out / "api_01_swagger_docs.png", full_page=True)
            # live API call for each preset -> JSON rendered page
            results = {}
            for name, g in presets.items():
                body = {"guest": {k: v for k, v in g.items() if k != "persona"}, "explain": False}
                results[name] = requests.post(f"http://localhost:{API}/score", json=body).json()
            html = "<html><body style='font-family:monospace;padding:20px'><h2>POST /score responses</h2>" + "".join(
                f"<h3>{n}</h3><pre>{json.dumps({k: v for k, v in r.items() if k != 'shap'}, indent=2)}</pre>" for n, r in results.items()) + "</body></html>"
            pg.set_content(html); pg.screenshot(path=out / "api_02_score_responses.png", full_page=True)
            (out / "preset_decisions.json").write_text(json.dumps({n: {k: r[k] for k in ("quadrant", "decision", "tau_margin_hat")}
                                                                    for n, r in results.items()}, indent=2))
            pg.goto(f"http://localhost:{UI}"); idle(pg)
            pg.screenshot(path=out / "ui_01_simulator_default.png", full_page=True)
            for i, name in enumerate(presets, 1):
                pg.get_by_role("button", name=f"Example {i}").click(); pg.wait_for_timeout(1500); idle(pg)
                pg.screenshot(path=out / f"ui_02_preset{i}.png", full_page=True)
            pg.get_by_role("tab", name="Model Benchmark & Diagnostics").click(); pg.wait_for_timeout(3000); idle(pg)
            pg.screenshot(path=out / "ui_03_benchmark.png", full_page=True)
            pg.get_by_role("tab", name="Campaign ROI & Policy Simulator").click(); pg.wait_for_timeout(3000); idle(pg)
            pg.screenshot(path=out / "ui_04_roi.png", full_page=True)
            capture_flows(pg, out, f"http://localhost:{UI}", f"http://localhost:{API}")
            b.close()
        print("screenshots ->", out)
    finally:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            try:
                pr.wait(10)
            except Exception:
                pr.kill()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
