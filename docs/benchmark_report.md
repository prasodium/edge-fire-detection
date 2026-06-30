# Benchmark Report

## Demo model: real, measured results (not on Pi 5 - read this first)

A real YOLOv8n model was trained, exported, and benchmarked end-to-end in
this repo to prove the full pipeline (training → export → inference →
decision engine → dashboard) actually works, not just that the code
compiles. Full reproduction steps are in the project README and
`docs/deployment_guide.md`. Key facts about this run:

- **Dataset**: [LibreYOLO/fire-smoke-seg](https://huggingface.co/datasets/LibreYOLO/fire-smoke-seg)
  (mirror of Roboflow Universe "fire-and-smoke-segmentation" by
  roboflow-universe-projects), CC BY 4.0. 201 images total (141 train / 40
  val / 20 test), 2 classes (fire, smoke - relabeled `small_flame`/`smoke`
  to fit this project's taxonomy). Segmentation polygons converted to
  bounding boxes by `training/convert_seg_to_bbox.py`.
- **This is NOT the production 15-class dataset** described elsewhere in
  this repo (`configs/dataset.yaml`). 141 training images is enough to
  prove the pipeline works, nowhere near enough for a deployable detector
  covering all 15 classes and the full false-positive hard-negative set.
  Treat all numbers below as a pipeline validation, not a production
  accuracy claim.
- **Training**: YOLOv8n, transfer learning from COCO weights, 80 epochs,
  416px, AdamW, cosine LR, on Apple M2 (MPS), ~11.5 minutes wall clock.
  Command: `python -m training.train --data datasets/processed_demo/data.yaml
  --base-model yolov8n --imgsz 416 --epochs 80 --batch 8 --patience 20 --device mps`
- **A real bug was caught and fixed during this run**: the default
  `optimizer="AdamW"` in `training/train.py` was paired with `lr0=0.01`,
  which is tuned for SGD. With AdamW this caused the loss to visibly diverge
  (mAP50 stuck at ~0 through epoch 8). Fixed by setting `lr0=0.001` for
  AdamW - now baked into `training/train.py` permanently.

### Accuracy (held-out test split, 20 images / 48 boxes, never seen in training or validation)

| Format | mAP50 | mAP50-95 | Precision | Recall | small_flame mAP50 | smoke mAP50 |
|---|---|---|---|---|---|---|
| PyTorch (.pt) | 0.765 | 0.415 | 0.864 | 0.707 | 0.817 | 0.712 |
| ONNX FP32 | 0.742 | 0.384 | 0.904 | 0.681 | 0.808 | 0.677 |
| ONNX FP16 | 0.742 | 0.384 | 0.905 | 0.681 | 0.808 | 0.677 |
| ONNX INT8 | **0.264** | 0.164 | 0.837 | **0.286** | 0.412 | 0.116 |

**Important finding: INT8 static quantization badly hurt accuracy on this
small model/dataset** (recall collapsed from 0.68 to 0.29). Root cause is
almost certainly the tiny calibration set (40 validation images) being
insufficient to characterize activation ranges for per-channel PTQ on a
3M-parameter model. `configs/model.yaml` was set to use **FP16** as the
demo's `active_model`, not INT8, specifically because of this measured gap
- FP16 matches FP32 accuracy exactly while halving file size. **Do not
assume INT8 is automatically the right choice** for a small/low-data model;
always validate accuracy after quantization (as done here) before deploying
it, and prefer a larger/more diverse calibration set than was available here
for any production INT8 export.

A second real bug was caught and fixed in the export pipeline itself: the
FP16 conversion (`onnxconverter_common`) produced a graph onnxruntime
refused to load (`Type Error ... Resize_output_cast0`), and the INT8 export
failed to load (`Unrecognized attribute: axis for operator DequantizeLinear`)
because the base ONNX export used opset 12, which lacks per-channel
DequantizeLinear support. Both are fixed in `training/export.py` (opset
bumped to 13; FP16 conversion re-runs ONNX shape inference after blocking
`Resize` from fp16 conversion).

### Speed/size (Apple M2, 8 cores, 3 ONNX Runtime intra-op threads — NOT a Raspberry Pi 5)

| Model file | Precision | Input | Mean latency | FPS | Model size |
|---|---|---|---|---|---|
| yolov8n_fire_fp32_416.onnx | FP32 | 416 | 39.2ms | 25.5 | 11.6MB |
| yolov8n_fire_fp16_416.onnx | FP16 | 416 | 39.4ms | 25.4 | 5.8MB |
| yolov8n_fire_int8_416.onnx | INT8 | 416 | 15.4ms | 64.9 | 3.2MB |

Raw CSV: `docs/benchmark_results.csv`. Two things worth noting:

1. **FP16 shows no CPU latency improvement over FP32** - expected on most
   CPUs (including the Pi 5's Cortex-A76), which lack native FP16 compute
   and upcast internally to FP32 for arithmetic. FP16's only benefit here is
   file size. Don't expect a speed win from FP16 on CPU-only inference.
2. **INT8 is genuinely ~2.5x faster** (real integer SIMD path), which is why
   it remains the right target for production *if* paired with a properly
   sized calibration set that doesn't tank accuracy the way it did here.
3. **These are Apple M2 numbers, not Pi 5 numbers.** The M2's cores are
   substantially faster per-clock than the Pi 5's Cortex-A76. Re-run
   `scripts/benchmark_models.sh` on the actual Pi 5 before trusting absolute
   FPS/latency for the 10-15 FPS / 80% CPU acceptance criteria below.

### End-to-end pipeline validation (real model + real decision engine)

`tests/test_integration_demo_model.py` (skips automatically if `weights/`
is empty) validates, with the real trained model:

- The production `FireDetector` class loads this exact model via
  `configs/model.yaml` and returns sensible detections on a real fire image.
- Feeding the **same static frame** repeatedly (zero motion, zero flicker)
  produces **zero** alarm events - the decision engine correctly recognizes
  there's no real-world temporal signal, even though Stage 1 detects an
  object every single frame. This is the false-positive-reduction design
  working as intended, not a gap.
- Feeding frames with **simulated realistic flicker/motion** (brightness
  oscillation + pixel jitter + noise, approximating real flame statistics)
  correctly produces exactly **one** `critical` AlarmEvent once the track
  accumulates 8+ consecutive frames at >85% average confidence - the full
  Stage 1-5 path, with a real model, confirmed working.

## How to reproduce / extend this report

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

## Production results table (fill in once trained on the full 15-class dataset)

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
