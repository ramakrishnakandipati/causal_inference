PY ?= python
export PYTHONWARNINGS=ignore

.PHONY: all setup check data eda train evaluate api ui screenshots test clean

all: check data eda train evaluate screenshots

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m playwright install chromium

check:
	$(PY) -c "from src.utils.gpu_utils import hardware_summary; print(hardware_summary())"

data:
	$(PY) -m data.generate_hospitality_data

eda:
	$(PY) -m src.pipelines.eda_visualizations

train:
	$(PY) -m src.pipelines.train_benchmark

evaluate:
	$(PY) -m src.pipelines.evaluate_plots

api:
	$(PY) -m uvicorn src.api.main:app --port 8000

ui:
	$(PY) -m streamlit run app/streamlit_app.py --server.port 8501

screenshots:
	$(PY) scripts/capture_screenshots.py

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf outputs/models/* outputs/metrics/* outputs/data/* outputs/visualizations/*/*.png outputs/screenshots/*/*.png
