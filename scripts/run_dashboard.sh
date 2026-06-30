#!/usr/bin/env bash
# Launches the dashboard (which also starts the detection pipeline on startup).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

exec uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 --workers 1
