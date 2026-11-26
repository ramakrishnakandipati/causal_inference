VENV ?= C:/venvs/loyalty
PY   ?= $(VENV)/Scripts/python

.PHONY: setup train serve app verify-ui test all

setup:
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cu126
	$(PY) -m pip install -e .
	$(PY) -m playwright install chromium

train:
	$(PY) -m src.models.trainer --config configs/config.yaml

serve:
	$(PY) -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

app:
	$(PY) -m streamlit run src/ui/app.py --server.port 8501

verify-ui:
	$(PY) scripts/capture_screenshots.py

test:
	$(PY) -m pytest -q

all: setup train test verify-ui
