# Use the causal-gpu interpreter: the base conda python has old xgboost/streamlit and cannot load the saved models.
PY ?= C:/Users/RamaKrishna/anaconda3/envs/causal-gpu/python.exe

.PHONY: setup train flows deck serve app verify test readme all

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m playwright install chromium

train:            ## data -> macro geo-lift -> 9 CATE models (x2 outcomes) -> economics -> figures
	$(PY) scripts/run_full_pipeline.py

flows:            ## rebuild the 8 end-to-end flows + programme scorecard (outputs/metrics/flows.json)
	$(PY) scripts/build_flows.py

deck:             ## rebuild presentation/Paid_Search_Incrementality_End_to_End.pptx from outputs/
	$(PY) presentation/build_deck.py

serve:            ## FastAPI on :8000 (Swagger at /docs)
	$(PY) -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

app:              ## Streamlit dashboard on :8501
	$(PY) -m streamlit run app/streamlit_app.py --server.port 8501

test:
	$(PY) -m pytest tests -q --ignore=tests/verify_e2e_playwright.py

verify:           ## Playwright e2e of FastAPI + Streamlit, screenshots, then artefact checks
	$(PY) tests/verify_e2e_playwright.py
	$(PY) scripts/run_full_pipeline.py --stage verify

readme:
	$(PY) scripts/build_readme.py

all: train test verify readme
