#!/usr/bin/env bash
# One-command pipeline: env check -> data -> EDA -> training -> evaluation plots -> Playwright screenshots
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
export PYTHONWARNINGS=ignore

echo "[1/6] environment check";   $PY -c "from src.utils.gpu_utils import hardware_summary; print(hardware_summary())"
echo "[2/6] data generation";     $PY -m data.generate_hospitality_data
echo "[3/6] EDA visualizations";  $PY -m src.pipelines.eda_visualizations
echo "[4/6] model training";      $PY -m src.pipelines.train_benchmark
echo "[5/6] evaluation plots";    $PY -m src.pipelines.evaluate_plots
echo "[6/6] screenshots";         $PY scripts/capture_screenshots.py
echo "Done. Artifacts are in ./outputs"
