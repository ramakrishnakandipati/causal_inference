# Targets mirror the pipeline stages. On Windows without GNU make, run the `python ...` command shown.
PY ?= python

.PHONY: all install data panel cate mc gpu figures readme verify quick test api ui screenshots clean

all: data panel cate mc gpu figures readme verify          # python scripts/run_full_pipeline.py

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m playwright install chromium

data:
	$(PY) scripts/run_full_pipeline.py --stage data

panel:
	$(PY) scripts/run_full_pipeline.py --stage panel

cate:
	$(PY) scripts/run_full_pipeline.py --stage cate

mc:
	$(PY) scripts/run_full_pipeline.py --stage mc

gpu:
	$(PY) scripts/run_full_pipeline.py --stage gpu

figures:
	$(PY) scripts/run_full_pipeline.py --stage figures

readme:
	$(PY) scripts/build_readme.py

verify:
	$(PY) scripts/run_full_pipeline.py --stage verify

quick:
	$(PY) scripts/run_full_pipeline.py --quick

test:
	$(PY) -m pytest tests -q

api:
	$(PY) -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

ui:
	$(PY) -m streamlit run src/app/streamlit_app.py --server.port 8501

screenshots:
	$(PY) scripts/capture_screenshots.py

clean:
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('outputs/models','outputs/metrics','outputs/figures','.pytest_cache')]"
