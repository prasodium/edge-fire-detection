#!/usr/bin/env bash
# Runs the full FP32/FP16/INT8 x 320/416/640 benchmark sweep and writes
# docs/benchmark_results.csv. Run this ON THE TARGET PI 5 for numbers that
# matter for the deployment decision (see docs/benchmark_report.md).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

python training/benchmark.py \
    --weights-dir weights \
    --images-dir datasets/processed/val/images \
    --runs 100 \
    --threads "${BENCHMARK_THREADS:-3}" \
    --out docs/benchmark_results.csv

echo "Results written to docs/benchmark_results.csv"
