# Edge AI Fire & Smoke Detection System

A CPU-only, edge-deployed fire and smoke early-detection system for the
Raspberry Pi 5 (8GB), built for indoor venues — seminar halls, lecture
halls, classrooms, conference rooms, auditoriums. No external AI
accelerator (no Coral/Hailo/NCS); inference runs on the Pi 5's CPU via ONNX
Runtime, targeting 10-15 FPS at ≤80% CPU utilization within a 15W power
budget.

## What's real vs. what's scaffolded

This repository is a **complete, working software system**: every Python
module is real, runnable code (not pseudocode), and the pure-logic core —
temporal verification, multi-object tracking, false-positive filtering,
zone mapping, config loading, bounded frame queue — has a passing unit test
suite (`pytest tests/ -v`, 20/20 passing as of this writing).

What it does **not** include, because they require resources outside this
environment:

- **A trained fire/smoke model.** `weights/` is empty by design (see
  `weights/README.md`). `training/` contains a complete, real training
  pipeline (dataset normalization, augmentation, Ultralytics training with
  transfer learning/early stopping/cosine LR, ONNX export, FP16/INT8
  quantization, benchmarking) — but actually running it requires GPU time
  and the licensed third-party datasets listed in `configs/dataset.yaml`.
- **Real Raspberry Pi 5 hardware benchmarks.** `docs/benchmark_report.md`
  is a template with a working measurement harness
  (`training/benchmark.py`, `scripts/benchmark_models.sh`); the numbers in
  `docs/model_comparison.md` are clearly labeled as literature-derived
  estimates pending on-device validation.
- **Downloaded third-party datasets.** Not redistributed here for license
  reasons; `training/dataset_prep.py` defines the normalization pipeline
  each one feeds into.

Everything else — camera capture, the 5-stage detection/decision pipeline,
GPIO alarm control, video/snapshot recording, MQTT/Firebase notifications,
SQLite event logging, and the FastAPI dashboard — is implemented and
covered by automated tests where the logic doesn't require a camera or GPU.

## Quick start (development, no Pi required)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

## Quick start (Raspberry Pi 5 deployment)

```bash
git clone <this-repo> ~/edge-fire-detection && cd ~/edge-fire-detection
bash scripts/setup_raspberry_pi.sh
bash scripts/install_dependencies.sh
# place a trained model under weights/ - see weights/README.md
bash scripts/run_dashboard.sh      # FastAPI dashboard -> http://<pi-ip>:8000
# or
bash scripts/run_streamlit.sh      # Streamlit dashboard -> http://<pi-ip>:8501
```

Run **only one** of the two dashboards in production — both start their own
copy of the camera/inference pipeline and compete for the same CPU budget.
See `dashboard/streamlit_app.py`'s docstring for the tradeoff (Streamlit
polls/reruns on a timer, ~1 fps "live" view; the FastAPI dashboard streams
true MJPEG at the configured inference FPS).

See `docs/deployment_guide.md` for the full walkthrough including systemd
setup, camera calibration, and zone configuration.

## Project structure

```
edge-fire-detection/
├── camera/         # Async Picamera2 capture (OpenCV fallback for dev)
├── inference/       # Detector, tracker, smoke/FP/temporal/motion verification, decision engine
├── gpio/            # Buzzer/relay/LED control (gpiozero, mock fallback off-Pi)
├── alarm/           # Alarm orchestration, recording, notifications
├── storage/         # SQLite event database
├── dashboard/        # Two live-view UIs (FastAPI MJPEG + Streamlit) sharing one render module
├── training/         # Dataset prep, augmentation, train, export/quantize, benchmark
├── datasets/         # Dataset registry (raw/processed data is gitignored)
├── weights/          # Exported ONNX models (gitignored - see weights/README.md)
├── utils/            # Config, logging, system monitor, frame queue, memory pool
├── configs/           # All runtime tuning - camera/model/decision/alarm/system/dataset YAML
├── scripts/          # Setup, install, run, systemd service, benchmark runner
├── tests/             # pytest unit tests (logic-only, no hardware required)
├── logs/              # Runtime logs (gitignored)
└── docs/              # Architecture, diagrams, guides, manuals (see below)
```

## Documentation index

| Doc | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System architecture, module map, sequence + class diagrams |
| [docs/flowcharts.md](docs/flowcharts.md) | Pipeline, false-positive-filter, and alarm-lifecycle flowcharts |
| [docs/model_comparison.md](docs/model_comparison.md) | YOLOv8n vs v9/v10/v11/v12-nano vs MobileNet-SSD vs EfficientDet-Lite vs PP-YOLOE-S vs NanoDet vs RTMDet-tiny |
| [docs/benchmark_report.md](docs/benchmark_report.md) | Benchmark methodology + results template |
| [docs/wiring_diagram.md](docs/wiring_diagram.md) | GPIO pin assignment, ASCII wiring diagram, power budget |
| [docs/deployment_guide.md](docs/deployment_guide.md) | Install, configure, run as a service, compliance notes |
| [docs/optimization_guide.md](docs/optimization_guide.md) | CPU/model/camera/memory optimization rationale |
| [docs/testing_guide.md](docs/testing_guide.md) | Unit/integration/soak testing, safe fire-testing procedure |
| [docs/user_manual.md](docs/user_manual.md) | For hall/facility staff using the dashboard |
| [docs/maintenance_manual.md](docs/maintenance_manual.md) | Routine maintenance, tuning, troubleshooting |
| [docs/future_improvements.md](docs/future_improvements.md) | Roadmap |

## Core design decisions worth knowing before you read the code

- **An alarm structurally cannot fire from one frame.** This isn't a
  threshold you could accidentally defeat — `inference/decision_engine.py`
  requires Stage 1-4 agreement across a tracked region over multiple
  seconds before a Stage 5 `AlarmEvent` is even constructed. See
  `docs/architecture.md` Section 6.
- **False-positive reduction is classical CV, not a second neural net.**
  Color/flicker/motion/static-light-source checks run in
  `inference/false_positive_filter.py` and `inference/motion_analyzer.py`
  using OpenCV/NumPy — deliberately, to stay inside the CPU-only budget.
- **The physical alarm never depends on the network or dashboard.**
  `alarm/alarm_manager.py` drives GPIO directly from the inference thread;
  MQTT/Firebase notifications are best-effort side channels that catch
  their own exceptions.
- **Every tunable threshold lives in `configs/*.yaml`**, not hardcoded —
  per-venue calibration (camera exposure/WB, zone polygons, false-positive
  thresholds) is a config change, not a code change.
