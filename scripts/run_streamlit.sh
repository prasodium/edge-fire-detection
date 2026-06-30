#!/usr/bin/env bash
# Launches the Streamlit live dashboard (alternative to scripts/run_dashboard.sh's
# FastAPI dashboard - run only one of the two in production).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

exec streamlit run dashboard/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true
