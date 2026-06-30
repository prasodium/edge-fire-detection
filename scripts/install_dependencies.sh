#!/usr/bin/env bash
# Creates the project virtualenv and installs Python dependencies.
# Run after scripts/setup_raspberry_pi.sh, from the project root.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "== Creating virtualenv with access to system site-packages (picamera2/libcamera are"
echo "   apt-installed, not pip-installable, so the venv must see them) =="
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

pip install --upgrade pip wheel
# requirements-pi.txt is the lean, inference-only set (no torch/ultralytics) -
# that's all the Pi needs. Use requirements.txt instead only if you intend to
# train/export models directly on this machine (not recommended on a Pi).
pip install -r requirements-pi.txt

mkdir -p logs storage/snapshots storage/clips weights

echo "Done. Activate with: source .venv/bin/activate"
echo "Then run: bash scripts/run_dashboard.sh"
