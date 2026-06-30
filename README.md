# Edge AI Fire & Smoke Detection System

A CPU-only, edge-deployed fire and smoke early-detection system for the
Raspberry Pi 5 (8GB), built for indoor venues — seminar halls, lecture
halls, classrooms, conference rooms, auditoriums. No external AI
accelerator (no Coral/Hailo/NCS); inference runs on the Pi 5's CPU via ONNX
Runtime, targeting 10-15 FPS at ≤80% CPU utilization within a 15W power
budget.

## What's real vs. what's scaffolded

This repository is a **complete, working software system**, and as of this
writing it ships with a **real trained model**, not just code: a YOLOv8n
detector trained on 201 real, license-clear fire/smoke images, exported to
ONNX (FP32/FP16/INT8), benchmarked, and wired into `configs/model.yaml` so
the pipeline and both dashboards run out of the box. Three real bugs were
found and fixed while getting that model trained and exported (an AdamW
learning-rate mismatch that caused training to diverge, an FP16-export
graph onnxruntime refused to load, and an opset version that broke INT8
quantization) — see `docs/benchmark_report.md` for the details and the real
measured numbers, including a significant accuracy regression found in the
INT8 export that's worth reading before you assume "INT8 is always best."

**This shipped model is a 2-class pipeline-validation demo
(`small_flame`, `smoke`), not the production 15-class detector** described
elsewhere in this README — 141 training images proves the system works
end-to-end, it does not cover electrical/paper/wood/curtain/plastic fire,
the zone-context classes, or the false-positive hard-negative set. See
`weights/README.md` and `docs/benchmark_report.md` for exactly what was
trained on, what the measured accuracy was, and how to retrain on the full
production taxonomy.

The pure-logic core (temporal verification, multi-object tracking,
false-positive filtering, zone mapping, config loading, bounded frame
queue) plus a real end-to-end integration test against the trained model
(detector finds objects in a real fire photo; decision engine stays silent
on a static frame; decision engine correctly alarms under simulated
flicker/motion) — 23/23 tests passing (`pytest tests/ -v`).

What still requires resources outside this environment:

- **The production 15-class model.** `training/` has the full pipeline
  (dataset normalization, augmentation, transfer learning, export,
  quantization, benchmarking) ready to point at a real production dataset —
  see `configs/dataset.yaml` for the licensed third-party sources to combine
  with your own on-site data collection.
- **Real Raspberry Pi 5 hardware numbers.** The benchmark numbers in
  `docs/benchmark_report.md` were measured on an Apple M2 dev machine, not
  a Pi 5 — re-run `scripts/benchmark_models.sh` on the actual device before
  trusting absolute FPS/latency against the 10-15 FPS / 80% CPU targets.

Everything else — camera capture, the 5-stage detection/decision pipeline,
GPIO alarm control, video/snapshot recording, MQTT/Firebase notifications,
SQLite event logging, and both dashboards (FastAPI + Streamlit) — is
implemented and either unit-tested or integration-tested against the real
model.

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

## Run it and test accuracy right now (with the included demo model)

On any machine (Pi or dev laptop with a webcam — `camera.yaml:backend`
auto-falls-back to OpenCV off-Pi):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_dashboard.sh        # http://localhost:8000 - live feed + bounding boxes
# or: bash scripts/run_streamlit.sh  # http://localhost:8501
```

Point the camera at fire-colored/flickering imagery (a lighter at a safe
distance, recorded fire footage on a phone, etc. — see
`docs/testing_guide.md` for safe testing procedure) and watch for bounding
boxes labeled `small_flame`/`smoke` in the live view, and an alarm after ~8
consecutive confident frames.

**To check accuracy** against the held-out test set used to validate this
model:

```bash
pip install ultralytics
yolo val model=weights/yolov8n_fire_fp32_416.onnx \
    data=datasets/processed_demo/data.yaml split=test imgsz=416
```

This reproduces the numbers in `docs/benchmark_report.md` (mAP50=0.742,
precision=0.90, recall=0.68 on 20 held-out images). To run the same
integration tests that validate the full detector→decision-engine path
against this real model:

```bash
pytest tests/test_integration_demo_model.py -v
```

To benchmark latency/CPU/memory on **your own machine** (these numbers
differ meaningfully from the Apple M2 numbers in the report — re-run on
your actual target hardware, especially the Pi 5):

```bash
python -m training.benchmark --weights-dir weights \
    --images-dir datasets/raw/libreyolo_fire_smoke_bbox/test/images --runs 100
```

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
