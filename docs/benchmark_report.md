# Benchmark Report (Template)

This report is **not pre-filled with fabricated numbers**. Run the harness
below on the actual Raspberry Pi 5 and paste the generated CSV in. See
`docs/model_comparison.md` for the architecture-selection reasoning that
doesn't depend on having hardware-measured numbers yet.

## How to generate this report

```bash
# 1. Train (on a workstation/cloud GPU - not on the Pi)
python training/train.py --base-model yolov8n --imgsz 416 --epochs 200

# 2. Export FP32 -> FP16 -> INT8
python training/export.py --weights training/runs/fire_smoke_yolov8n_416/weights/best.pt --imgsz 416
python training/export.py --weights training/runs/fire_smoke_yolov8n_416/weights/best.pt --imgsz 320
python training/export.py --weights training/runs/fire_smoke_yolov8n_416/weights/best.pt --imgsz 640

# 3. Copy the resulting weights/*.onnx files to the Raspberry Pi 5

# 4. On the Pi, with the venv active:
bash scripts/benchmark_models.sh
# -> writes docs/benchmark_results.csv
```

## Results table (fill in from docs/benchmark_results.csv)

| Model file | Precision | Input size | Mean latency (ms) | P95 latency (ms) | FPS | CPU % during | Peak RSS (MB) | Model size (MB) |
|---|---|---|---|---|---|---|---|---|
| yolov8n_fire_fp32_320.onnx | FP32 | 320 | — | — | — | — | — | — |
| yolov8n_fire_fp32_416.onnx | FP32 | 416 | — | — | — | — | — | — |
| yolov8n_fire_fp32_640.onnx | FP32 | 640 | — | — | — | — | — | — |
| yolov8n_fire_fp16_320.onnx | FP16 | 320 | — | — | — | — | — | — |
| yolov8n_fire_fp16_416.onnx | FP16 | 416 | — | — | — | — | — | — |
| yolov8n_fire_fp16_640.onnx | FP16 | 640 | — | — | — | — | — | — |
| yolov8n_fire_int8_320.onnx | INT8 | 320 | — | — | — | — | — | — |
| yolov8n_fire_int8_416.onnx | INT8 | 416 | — | — | — | — | — | — |
| yolov8n_fire_int8_640.onnx | INT8 | 640 | — | — | — | — | — | — |

## Acceptance criteria (from the project spec)

- [ ] Model size ≤ 20MB preferred, ≤ 40MB hard max
- [ ] Inference at 10-15 FPS sustained
- [ ] CPU utilization stays below 80% during continuous inference
  (cross-check against `GET /api/status` on the dashboard, or `htop`,
  while the pipeline runs for ≥15 minutes)
- [ ] No thermal throttling over a ≥1 hour soak test
  (`vcgencmd get_throttled` should report `0x0` throughout;
  `cat /sys/class/thermal/thermal_zone0/temp` should stay below the
  `thermal_throttle_temp_c` configured in `configs/system.yaml`)
- [ ] Power draw ≤ 15W measured at the USB-C input (use a USB power meter)

## Accuracy evaluation (separate from latency benchmarking)

Latency/CPU/memory are measured by `training/benchmark.py`. Detection
accuracy (mAP, per-class precision/recall, false-positive rate against the
hard-negative set in `configs/dataset.yaml:negative_hard_examples`) should
be measured with:

```bash
yolo val model=training/runs/.../weights/best.pt data=datasets/processed/data.yaml imgsz=416
```

Report per-class AP for all 15 classes, plus a dedicated false-positive rate
metric: run inference over the hard-negative clips (stage lighting, LEDs,
projector content, candles, etc.) with the full decision-engine pipeline
(not just the raw detector) and report how many of those clips incorrectly
reach Stage 5 (a real alarm) versus how many are correctly suppressed at
Stage 2/3/4. Target: **zero** confirmed alarms on the hard-negative set;
investigate and tune `configs/decision.yaml` thresholds if any pass through.
