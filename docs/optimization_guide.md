# Optimization Guide

## 0. Dashboard choice and CPU budget

Two dashboards ship with this repo (`dashboard/app.py` FastAPI/MJPEG and
`dashboard/streamlit_app.py` Streamlit) — **run only one at a time in
production.** Each one creates its own `FireDetectionPipeline` instance,
meaning each opens its own camera session and runs its own ONNX inference
loop; running both simultaneously roughly doubles CPU/RAM use and will blow
the 80% CPU / 15W budget. Streamlit additionally reruns its script on
`REFRESH_SECONDS` (default 1s) which adds its own modest overhead on top of
inference — if CPU headroom is tight, raise `REFRESH_SECONDS` in
`dashboard/streamlit_app.py` before reaching for the FastAPI dashboard,
which streams frames the inference loop already produced rather than
polling for them.

## 1. CPU optimization

- **ONNX Runtime thread tuning** (`configs/model.yaml:inference.onnxruntime`):
  `intra_op_num_threads` is set to 3 (Pi 5 has 4 Cortex-A76 cores) so one
  core stays free for camera capture, the dashboard, and OS/journald
  overhead — this is what keeps total CPU utilization under the 80% target
  rather than starving everything else with a 4-thread session.
- **`execution_mode: parallel`** lets ONNX Runtime parallelize across
  independent subgraphs where the model structure allows it, rather than
  serializing the whole graph through one thread pool.
- **NEON/SIMD comes for free** from using ONNX Runtime's and OpenCV's
  official `aarch64` wheels/apt packages on Raspberry Pi OS 64-bit — no
  manual SIMD code is required in this codebase; just don't accidentally
  install x86 wheels via an emulation layer.
- **Bounded queues, not unbounded ones** (`utils/frame_queue.py`): prevents
  a CPU-bound moment from turning into an ever-growing backlog that would
  otherwise show up as a CPU usage cliff once the consumer catches up on
  stale frames.
- **Pre-allocated buffer pool** (`utils/memory_pool.py`): avoids per-frame
  `np.empty` allocation/GC churn at 15fps over a 24/7 runtime.
- **Thermal guard** (`inference/pipeline.py:_should_skip_frame_for_thermal_budget`):
  actively skips inference (not capture) when `thermal_throttle_temp_c`
  (78°C) is approached, trading a temporary FPS dip for staying clear of the
  Pi 5's actual throttle point (~80-85°C) — see `configs/system.yaml`.

## 2. Model-level optimization

| Technique | Where | Effect |
|---|---|---|
| Transfer learning from COCO weights | `training/train.py` | Faster convergence, less data needed |
| FP16 conversion | `training/export.py:convert_fp16` | ~2x size reduction vs FP32, ARM NEON has good FP16 support on Pi 5 |
| INT8 static PTQ (QDQ format) | `training/export.py:quantize_int8` | ~4x size reduction vs FP32, typically 1.5-3x CPU latency improvement; calibrated on real indoor fire/smoke imagery (`_CalibrationDataReader`), not synthetic data, to avoid mis-calibrating the warm-color-heavy activation ranges |
| Structured channel pruning | `training/train.py:prune` (`torch-pruning`) | Reduces FLOPs/size further; always fine-tune after pruning to recover accuracy |
| Knowledge distillation | `training/train.py:distill` | Use only if you have budget to train a larger teacher first; marginal gains for nano-scale students, prioritize INT8 quantization first |
| Resolution selection (320/416/640) | `configs/model.yaml:input_size` | Single highest-leverage lever — benchmark all three (`scripts/benchmark_models.sh`) before committing |

## 3. Camera & capture optimization

- **Manual exposure/white-balance** (`configs/camera.yaml`) instead of
  auto modes: auto-exposure hunts under flickering stage/LED lighting,
  which both blurs flame edges (motion blur from slow shutter) and
  actively confuses the false-positive filter's flicker-frequency check
  (the camera's own AE adjustments look like brightness variance).
  Calibrate manual values on-site once at install time (see "Camera
  Calibration" below).
- **HDR mode** on the IMX708 helps avoid blown-out highlights from
  projector beams or window glare that would otherwise saturate to white
  and get rejected/missed inconsistently by the color-consistency check.
- **Capture at 1536×864, infer at 416×416**: capturing higher than the
  inference resolution preserves detail for the saved snapshot/clip
  evidence and for the false-positive filter's region crops (color/flicker
  checks benefit from more source pixels per detected box), while keeping
  the model input small for speed.
- **`buffer_count: 4`** double/triple-buffers the Picamera2 pipeline so the
  ISP/driver never blocks waiting for the previous buffer to be consumed.

## 4. Camera calibration (do this once per venue)

1. Mount the camera in its final position, at final framing.
2. Run `python scripts/run_headless.py` and watch the dashboard live view
   under the room's normal lighting conditions (lights on, projector on,
   typical occupancy).
3. Adjust `configs/camera.yaml:exposure.exposure_time_us` and
   `analogue_gain` until the image is well-exposed without highlight
   clipping on the brightest fixtures (stage lights, projector screen).
4. Adjust `white_balance.colour_gains` `[red, blue]` until whites/grays in
   the room render neutral — incorrect WB is one of the most common causes
   of both false positives (everything looks orange) and false negatives
   (real flame's orange gets color-corrected away).
5. Set `configs/decision.yaml:zones` polygons by eyeballing the live feed
   against the stage/projector/podium/exit-door locations (coordinates are
   normalized 0-1 fractions of frame width/height).
6. Let the system run for `static_roi_learning_frames` (300 frames, ~20-30s
   at 10-15fps) before relying on the static-light-source suppression check
   — it needs that warm-up to learn which bright regions are fixed fixtures.

## 5. Memory optimization (8GB budget)

- ONNX Runtime session + INT8 model: typically well under 200MB resident.
- Frame buffers: capped by `frame_queue_size` (4) × resolution — at
  1536×864×3 bytes that's ~16MB max in the queue, plus the pre-allocated
  pool in `utils/memory_pool.py`.
- SQLite WAL mode (`storage/db.py`) keeps write latency low without an
  in-memory cache that grows unbounded — `database.retention_days` in
  `configs/alarm.yaml` bounds long-term growth.
- `recording.max_storage_gb` in `configs/alarm.yaml` is the bound on disk
  (not RAM) growth from saved snapshots/clips — implement pruning in a cron
  job or extend `EventDatabase` if this isn't already enforced by your OS's
  disk (see `docs/maintenance_manual.md`).

## 6. Quantization-aware training note

Ultralytics' training loop does not expose a first-class CPU QAT path as of
this writing. The practical, well-supported equivalent used here is
**static post-training quantization (PTQ) with a representative calibration
set** (`training/export.py:quantize_int8`), which on YOLO-nano-scale models
typically recovers within 1-2 mAP points of the FP32 baseline — validate
this gap yourself with `yolo val` on both the FP32 and INT8 ONNX exports
before committing to INT8 for production.
