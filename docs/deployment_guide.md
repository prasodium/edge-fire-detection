# Deployment Guide

## 1. Prerequisites

- Raspberry Pi 5 (8GB), Raspberry Pi OS 64-bit (Bookworm or later)
- Raspberry Pi Camera Module 3 NoIR Wide, connected via CSI
- A trained + exported ONNX model in `weights/` (see `docs/benchmark_report.md`
  for the train → export pipeline; this repo ships configs and code, not a
  pretrained fire-specific model — you must train or source one before
  first real deployment)
- Network access for initial setup (apt packages); the running system does
  **not** require internet for its core alarm path

## 2. Initial setup

```bash
git clone <this-repo> ~/edge-fire-detection
cd ~/edge-fire-detection
bash scripts/setup_raspberry_pi.sh        # OS packages, camera enable, swap, CPU governor
bash scripts/install_dependencies.sh       # venv + pip requirements
```

## 3. Configuration

Edit `configs/*.yaml` for your venue before first run:

1. `configs/camera.yaml` — exposure/white-balance manual values need
   calibrating per-room (see `docs/optimization_guide.md` "Camera
   Calibration"). Auto modes are intentionally avoided (see comments in the
   file) because auto-exposure/auto-WB actively fight the false-positive
   filter under flickering stage lighting.
2. `configs/decision.yaml` — set `zones:` polygons to match your actual
   camera framing of stage/projector/podium/exit-door (normalized 0-1
   coordinates). Add any fixed bright light fixtures to `exclusion_zones`.
3. `configs/alarm.yaml` — set `notifications.mqtt.broker_host` (default
   assumes a local Mosquitto broker installed by `setup_raspberry_pi.sh`),
   and `gpio.*_pin` if your wiring differs from `docs/wiring_diagram.md`.
4. `configs/model.yaml` — `active_model` must point at a key whose
   `weights:` path exists under `weights/`.

## 4. Placing model weights

```bash
weights/
├── yolov8n_fire_fp32_640.onnx
├── yolov8n_fire_fp16_416.onnx
└── yolov8n_fire_int8_416.onnx   # <- this is configs/model.yaml's default active_model
```

If you have no trained model yet, see `training/` and `docs/benchmark_report.md`.
**Do not deploy with a COCO-pretrained (non-fire) checkpoint** — it has no
fire/smoke classes and the pipeline will never alarm; it exists only for
transfer-learning initialization in `training/train.py`.

## 5. Smoke-testing before going live

```bash
source .venv/bin/activate
python -c "from utils.config import load_config; print(load_config().active_model_spec())"
python scripts/run_headless.py   # Ctrl+C to stop after confirming no crash + camera frames flowing
```

Then run the full dashboard and visually confirm the live feed, bounding
boxes (point a phone flashlight or lighter at the camera from a safe
distance to test true-positive path — **never** start an open flame indoors
for testing; use the `/scripts` calibration approach in `docs/testing_guide.md`
instead), and CPU/temperature telemetry:

```bash
bash scripts/run_dashboard.sh
# open http://<pi-ip>:8000
```

## 6. Running as a system service

```bash
sudo cp scripts/fire-detection.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fire-detection.service
sudo systemctl status fire-detection.service
journalctl -u fire-detection.service -f
```

## 7. Multi-room / multi-camera deployment

Each Pi 5 + camera is one independent instance (`device_name` in
`configs/system.yaml` distinguishes them in MQTT topics / dashboard
titles). There is no built-in multi-camera fan-in on a single Pi 5 — the
CPU/power budget is sized for one camera stream at the target FPS. For a
multi-hall building, deploy one Pi 5 per hall and aggregate alarms
centrally via the shared MQTT topic namespace (`firealert/<device_name>`)
or by polling each unit's `/api/status` and `/api/events` from a central
dashboard (not included in this repo — see `docs/future_improvements.md`).

## 8. Regulatory & compliance

This system is a **supplementary AI-based early-warning aid**. It is not a
substitute for, and should not be the sole basis for, a code-compliant fire
detection/alarm system (e.g. ionization/photoelectric smoke detectors,
heat detectors, sprinkler systems per NFPA 72 or local equivalent). Before
wiring the relay output to any building fire alarm control panel (FACP) or
life-safety circuit, have the integration reviewed by a licensed fire
protection engineer and your local Authority Having Jurisdiction (AHJ).
Treat this system's relay output as an **auxiliary/supervisory** signal,
not a primary initiating device circuit, unless and until it has been
formally certified to the relevant standard for your jurisdiction.

## 9. Rollback / updating

```bash
cd ~/edge-fire-detection
git pull
source .venv/bin/activate
pip install -r requirements.txt --upgrade
sudo systemctl restart fire-detection.service
```

Keep the previous `weights/*.onnx` files until the new model has been
soak-tested (`docs/testing_guide.md` "Soak Test") — `configs/model.yaml`
makes rolling back a one-line `active_model` change plus a service restart.
