"""Start FastAPI (8000) and Streamlit (8501), capture UI/API screenshots with Playwright, shut everything down.

    python scripts/capture_screenshots.py            # everything
    python scripts/capture_screenshots.py flows      # only the end-to-end flow screenshots
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from src.utils.config import ensure_dir, load_config, save_json  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

cfg = load_config()
log = get_logger("shots")
API = f"http://127.0.0.1:{cfg['api']['port']}"
UI = f"http://127.0.0.1:{cfg['ui']['port']}"
OUT = ensure_dir(cfg["paths"]["screenshots"])
LOGS = ensure_dir(cfg["paths"]["logs"])
PROCS: list[subprocess.Popen] = []
ENGINE = None


# ----------------------------------------------------------------------------- process management
def start(cmd: list[str], logfile: str) -> subprocess.Popen:
    env = dict(os.environ, PYTHONUTF8="1", STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    fh = open(LOGS / logfile, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, creationflags=flags)
    PROCS.append(p)
    return p


def stop_all() -> None:
    for p in PROCS:
        if p.poll() is None:
            p.terminate()
    for p in PROCS:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
            else:
                p.kill()
    PROCS.clear()


atexit.register(stop_all)


def wait(url: str, timeout: float = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.5)
    raise TimeoutError(f"{url} did not become ready in {timeout}s")


def idle(pg, timeout: float = 180) -> None:
    """Wait for Streamlit to finish a rerun (no status widget, nothing stale)."""
    pg.wait_for_selector("[data-testid=stAppViewContainer]", timeout=timeout * 1000)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pg.locator("[data-testid=stStatusWidget]").count() == 0 and pg.locator("[data-stale=true]").count() == 0:
            break
        time.sleep(0.5)
    time.sleep(2.0)


def fit_viewport(pg, pad: int = 70, min_h: int = 760) -> None:
    """Resize the viewport to the rendered content so screenshots carry no blank tail (Streamlit scrolls an inner pane)."""
    pg.set_viewport_size({"width": 1600, "height": 5200})
    time.sleep(1.0)
    bottom = pg.evaluate("""() => {
        const el = document.querySelector('[data-testid="stMainBlockContainer"]');
        const r = el.getBoundingClientRect();
        return r.bottom + (el.closest('[data-testid="stMain"]')?.scrollTop || 0);
    }""")
    pg.set_viewport_size({"width": 1600, "height": int(max(min_h, bottom + pad))})
    time.sleep(1.0)


def shot(pg, name: str, fit: bool = False) -> Path:
    path = OUT / name
    pg.mouse.move(1500, 12)                     # park the pointer: no hover tooltips in the capture
    if fit:
        fit_viewport(pg)
    time.sleep(0.4)
    pg.screenshot(path=str(path))
    log.info("saved %-40s %6.0f KB", name, path.stat().st_size / 1024)
    return path


# ----------------------------------------------------------------------------- API screenshots
CATE_BODY = {"rows": [{"competitor_adr_usd": 335.0, "flight_search_index": 125.0, "temp_max_f": 80.0, "precipitation_in": 0.0,
                       "weather_severity_index": 3.0, "is_weekend": 1, "is_holiday_window": 0, "seasonality_sin": 0.40,
                       "seasonality_cos": 0.92, "ota_promo_pressure_index": 28.0, "days_since_launch": 20, "dow": 5},
                      {"competitor_adr_usd": 300.0, "flight_search_index": 95.0, "temp_max_f": 72.0, "precipitation_in": 0.8,
                       "weather_severity_index": 45.0, "is_weekend": 0, "is_holiday_window": 0, "seasonality_sin": 0.40,
                       "seasonality_cos": 0.92, "ota_promo_pressure_index": 55.0, "days_since_launch": 33, "dow": 2}],
             "learner": "Causal Forest DML", "baseline_room_nights": 1500}


def api_shots(browser) -> dict:
    ctx = browser.new_context(viewport={"width": 1400, "height": 2000}, device_scale_factor=2)
    pg = ctx.new_page()
    pg.goto(API + "/docs", wait_until="networkidle")
    pg.wait_for_selector(".opblock", timeout=60000)
    time.sleep(1.0)
    pg.mouse.move(1390, 5)
    pg.locator(".swagger-ui").first.screenshot(path=str(OUT / "api_swagger_docs.png"))
    log.info("saved %-40s %6.0f KB", "api_swagger_docs.png", (OUT / "api_swagger_docs.png").stat().st_size / 1024)
    info: dict = {}
    # expand POST /predict/cate, run it with a concrete body and capture the live response
    ok = False
    try:
        op = pg.locator("#operations-prediction-predict_cate_predict_cate_post")
        op.locator(".opblock-summary").click()
        op.get_by_role("button", name="Try it out").click()
        op.locator("textarea.body-param__text").fill(json.dumps(CATE_BODY, indent=2))
        op.get_by_role("button", name="Execute").click()
        pg.wait_for_selector("#operations-prediction-predict_cate_predict_cate_post .live-responses-table", timeout=60000)
        time.sleep(1.5)
        op.scroll_into_view_if_needed()
        op.screenshot(path=str(OUT / "api_cate_response.png"))
        log.info("saved %-40s (Swagger live response)", "api_cate_response.png")
        ok = True
    except Exception as exc:
        log.info("Swagger interaction failed (%s); rendering the live JSON instead", type(exc).__name__)
    r = requests.post(API + "/predict/cate", json=CATE_BODY, timeout=60)
    info["cate_status"] = r.status_code
    info["cate_response"] = r.json()
    if not ok:
        html = ("<html><body style='font-family:Consolas,monospace;background:#fff;padding:28px;width:1100px'>"
                "<h3 style='font-family:sans-serif'>POST /predict/cate - live response (HTTP %d)</h3><pre>%s</pre></body></html>"
                % (r.status_code, json.dumps(r.json(), indent=2)))
        pg.set_content(html)
        pg.screenshot(path=str(OUT / "api_cate_response.png"), full_page=True)
    ctx.close()
    return info


# ----------------------------------------------------------------------------- Streamlit screenshots
def click_tab(pg, prefix: str) -> None:
    pg.get_by_role("tab", name=prefix).click()
    time.sleep(1.0)
    idle(pg)


def ui_shots(browser, api_info: dict) -> dict:
    check: dict = {}
    ctx = browser.new_context(viewport={"width": 1600, "height": 2400}, device_scale_factor=1.5)
    pg = ctx.new_page()
    pg.goto(UI, wait_until="domcontentloaded")
    pg.get_by_role("button", name="1. Miami Flagship").wait_for(timeout=240000)
    pg.add_style_tag(content="header[data-testid=stHeader]{display:none !important}")
    idle(pg)
    # --- tab 1 (preset 1: Miami flagship)
    pg.get_by_role("button", name="1. Miami Flagship").click()
    idle(pg)
    shot(pg, "streamlit_tab1_executive_iroas.png", fit=True)
    body = pg.inner_text("[data-testid=stMainBlockContainer]")
    att = requests.post(API + "/predict/att", json={"scenario": "mia_flagship", "estimator": "ASCM (Ridge)", "include_series": False}, timeout=60).json()
    token = f"{att['att_total']:,.0f}"
    check["api_att_total_ascm_mia_flagship"] = att["att_total"]
    check["ui_shows_same_total"] = token in body
    log.info("API/UI consistency: ATT total %s present in UI text: %s", token, check["ui_shows_same_total"])
    # --- tab 2
    click_tab(pg, "2. DiD")
    shot(pg, "streamlit_tab2_did_event_study.png", fit=True)
    # --- tab 3
    click_tab(pg, "3. Synthetic")
    shot(pg, "streamlit_tab3_synthetic_control.png", fit=True)
    # --- tab 4 (benchmark, what-if simulator and pacing studio)
    click_tab(pg, "4. Geo-CATE")
    time.sleep(2)
    shot(pg, "streamlit_tab4_cate_metalearners.png", fit=True)
    # --- tab 5 (live API)
    click_tab(pg, "5. Live")
    pg.get_by_role("button", name="Score batch through the API").click()     # live batch call through /predict/cate
    idle(pg)
    time.sleep(1.5)
    shot(pg, "streamlit_tab5_live_api.png", fit=True)
    # --- other presets, tab 1
    click_tab(pg, "1. Executive")
    for i, (btn, fname) in enumerate([("2. Las Vegas", "streamlit_preset2_las_vegas.png"), ("3. New York", "streamlit_preset3_new_york.png"),
                                      ("4. Negative", "streamlit_preset4_placebo_check.png")], start=2):
        pg.get_by_role("button", name=btn).click()
        idle(pg)
        shot(pg, fname, fit=True)
    ctx.close()
    return check


# ----------------------------------------------------------------------------- end-to-end flow screenshots
def api_flow_numbers(rep: dict) -> dict:
    """What the live API says for this flow, to compare with the numbers the UI displays."""
    f = rep["flow"]
    if rep["kind"] == "panel":
        j = requests.post(API + "/predict/att", json={"scenario": rep["scenario"], "estimator": "ASCM (Ridge)", "include_series": False}, timeout=60).json()
        return {"label": "ATT total", "api": j["att_total"], "token": f"{j['att_total']:,.0f}"}
    if rep["kind"] == "pacing":
        j = requests.post(API + "/simulate/campaign", json={"budget_usd": f["budget"], "strategy": f["strategy"]}, timeout=60).json()
        v = j["totals"]["incremental_room_nights"]
        return {"label": "projected incremental room nights", "api": v, "token": f"{v:,.0f}"}
    from src import flows as fl
    from src.serving import Engine
    row = fl.cate_row(ENGINE, f["cond"])
    learner = rep["headline"]["best_learner"]
    j = requests.post(API + "/predict/cate", json={"rows": [row], "learner": learner}, timeout=60).json()
    v = j["incremental_room_nights"][0]
    return {"label": f"{learner} lift (room nights)", "api": v, "token": f"{v:,.0f}"}


def select_flow_ready(pg, flow_id: str, timeout: float = 120) -> None:
    """Wait until the flow just selected is fully rendered: its heading is present and nothing is still fading (stale)."""
    pg.locator(".st-key-flow_full").get_by_text(f"Flow {flow_id} -").first.wait_for(timeout=timeout * 1000)
    idle(pg)
    t0 = time.time()
    while time.time() - t0 < 30:
        faded = pg.evaluate("""() => { let el = document.querySelector('.st-key-flow_full'); let m = 1;
            while (el && el !== document.body) { m = Math.min(m, parseFloat(getComputedStyle(el).opacity)); el = el.parentElement; }
            return m; }""")
        if faded >= 0.999 and pg.locator(".st-key-flow_full img").count() >= 1:
            break
        time.sleep(0.5)
    time.sleep(2.0)


def capture_flows(browser) -> dict:
    from src import flows as fl
    global ENGINE
    from src.serving import Engine
    ENGINE = Engine()
    fdir = ensure_dir(OUT / "flows")
    ctx = browser.new_context(viewport={"width": 1500, "height": 5600}, device_scale_factor=1.5)
    pg = ctx.new_page()
    pg.goto(UI, wait_until="domcontentloaded")
    pg.get_by_role("button", name="1. Miami Flagship").wait_for(timeout=240000)
    pg.add_style_tag(content="header[data-testid=stHeader]{display:none !important}")
    idle(pg)
    click_tab(pg, "6. End-to-End")
    checks: dict = {}
    for f in fl.FLOWS:
        pg.get_by_text(f["short"], exact=True).click()
        select_flow_ready(pg, f["id"])
        rep = fl.report(ENGINE, f)
        pg.mouse.move(1490, 8)
        names = ["overall"] + [s["key"] for s in rep["steps"]]
        for n in names:
            loc = pg.locator(f".st-key-flow_{n}" if n == "overall" else f".st-key-flowstep_{n}")
            loc.scroll_into_view_if_needed()
            loc.screenshot(path=str(fdir / f"{f['id']}_{n}.png"))
        pg.locator(".st-key-flow_full").screenshot(path=str(fdir / f"{f['id']}_full.png"))
        body = pg.locator(".st-key-flow_full").inner_text()
        a = api_flow_numbers(rep)
        checks[f["id"]] = {"flow": f["title"], "verdict": rep["verdict"], "api_metric": a["label"], "api_value": a["api"],
                           "ui_shows_api_number": a["token"] in body, "ui_shows_verdict": rep["verdict"] in body}
        log.info("flow %s: %d stage shots + full | UI==API: %s | verdict %s", f["id"], len(names), checks[f["id"]]["ui_shows_api_number"], rep["verdict"])
    pg.get_by_text("A1 - Miami winter peak", exact=True).click()
    select_flow_ready(pg, "A1")
    pg.locator(".st-key-flow_compare").screenshot(path=str(fdir / "all_flows_comparison.png"))
    save_json(checks, fdir / "flow_api_vs_ui_check.json")
    ctx.close()
    return checks


def main(only: str | None = None) -> None:
    log.info("starting FastAPI and Streamlit ...")
    start([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", str(cfg["api"]["port"])], "uvicorn.log")
    start([sys.executable, "-m", "streamlit", "run", "src/app/streamlit_app.py", "--server.port", str(cfg["ui"]["port"]),
           "--server.headless", "true", "--browser.gatherUsageStats", "false"], "streamlit.log")
    try:
        wait(API + "/health")
        wait(UI + "/_stcore/health")
        log.info("services are up")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            if only == "flows":
                checks = capture_flows(browser)
                log.info("flows UI==API: %s", all(v["ui_shows_api_number"] and v["ui_shows_verdict"] for v in checks.values()))
            else:
                api_info = api_shots(browser)
                check = ui_shots(browser, api_info)
                check["flows"] = capture_flows(browser)
                save_json({"api_cate": api_info, "consistency": check}, OUT / "capture_report.json")
            browser.close()
    finally:
        stop_all()
        log.info("services stopped")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
