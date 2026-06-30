# Model Comparison & Selection

## IMPORTANT — provenance of these numbers

This project has not yet had access to a physical Raspberry Pi 5 or a
trained fire/smoke checkpoint in this environment. The table below combines:

- **Published/community benchmark figures** for each architecture's COCO
  mAP and parameter count (cited as "literature"), and
- **A measurement methodology and harness** (`training/benchmark.py`) that
  produces *real, reproducible* latency/CPU/memory numbers once you run it
  with actual exported weights on actual Pi 5 hardware.

Do not treat the "Pi 5 CPU latency" column as ground truth — treat it as an
**informed estimate** extrapolated from published ARM Cortex-A76 (Pi 5's
CPU) benchmarks of similar-sized ONNX models, to be **replaced** by your own
`docs/benchmark_results.csv` after running `scripts/benchmark_models.sh` on
the real device. The "Recommendation" stands regardless: it's based on
architectural fit (model size, ONNX export maturity, NMS-free or NMS
overhead, community support), not the exact latency figure.

## Candidate models

| Model | Params | COCO mAP50-95 (lit.) | ONNX export maturity | NMS | Est. Pi5 CPU latency @416px (FP32, single-thread-equivalent) | Notes |
|---|---|---|---|---|---|---|
| **YOLOv8n** | 3.2M | 37.3 | Excellent (`ultralytics export`, first-class) | Required post-export | ~55-75ms | Most mature CPU export path, huge community, easiest INT8 PTQ support in `ultralytics`+`onnxruntime` |
| YOLOv9-tiny | ~2.0M | 38.3 | Good (community ONNX export scripts; not yet first-class in `ultralytics` as of writing) | Required | ~60-80ms | Reparam-heavy architecture (GELAN) adds export complexity; marginal mAP gain over v8n |
| YOLOv10n | 2.3M | 38.5 | Good (`ultralytics` supports v10 export) | **NMS-free** (built-in) | ~45-65ms | NMS-free is attractive for a fixed CPU budget — removes a variable-cost post-process step |
| YOLOv11n | 2.6M | 39.5 | Excellent (`ultralytics` first-class as of v8.3+) | Required post-export | ~50-70ms | Best accuracy/size tradeoff in the v8-v11 family per Ultralytics' own published benchmarks |
| YOLOv12n | ~2.6M | ~40.6 (early reports) | Emerging — export tooling less battle-tested at time of writing | Required | ~55-75ms (unverified) | Newest, attention-augmented; treat as "watch list", not a launch-day pick |
| MobileNet-SSD (v2) | ~4.3M (SSDLite) | ~22 | Excellent (ancient, ubiquitous ONNX/TFLite export) | Required | ~40-60ms | Meaningfully lower mAP; weak on small/early-stage flame which is exactly the hardest, highest-value case here |
| EfficientDet-Lite0 | 3.2M | ~27 | Moderate (TFLite-first, ONNX export less common) | Required | ~70-100ms | EfficientNet backbone's depthwise-separable convs are not as CPU-cache-friendly on ARM as YOLO's plain convs at this scale |
| PP-YOLOE-S | 7.9M | 43.0 | Poor on CPU-only deployment outside PaddlePaddle/PaddleLite toolchain | NMS-free (variant) | ~90-130ms | Best raw mAP but heavier and the PaddlePaddle→ONNX→ORT path is the least-trodden for this hardware |
| NanoDet-Plus-m | 1.2M | ~30.4 | Good (native ONNX export) | Required | ~35-55ms | Smallest model, fastest, but mAP ceiling is meaningfully lower — risk for early-stage/small-flame recall |
| RTMDet-tiny | 4.9M | 41.0 | Moderate (MMDetection→ONNX path works but has more moving parts) | Required | ~75-105ms | Strong accuracy, but heavier than the YOLO-nano family for similar gains here |

## Recommendation: **YOLOv8n**, ONNX, **416×416**, **INT8**

Reasoning, in priority order for this specific deployment:

1. **Export/tooling maturity dominates risk on a fixed timeline.** `ultralytics`
   → ONNX → `onnxruntime` static INT8 quantization is the most-traveled path
   of every option here; YOLOv8n has the fewest "unknowns" between a trained
   checkpoint and a deployed, quantized, NEON-accelerated CPU model.
2. **YOLOv11n is the runner-up and worth A/B testing** once the training
   pipeline is running — it consistently shows ~1-2 mAP points over v8n at
   a similar parameter count in Ultralytics' own published comparisons, with
   equally mature export support (same `ultralytics` package, same flags).
   `training/train.py --base-model yolov11n` is wired up for exactly this
   comparison.
3. **YOLOv10n's NMS-free head is attractive but a second-order concern** —
   NMS over ≤20 detections at 416px is a small fraction of total latency
   compared to the backbone forward pass; the architectural simplicity isn't
   worth giving up `ultralytics`' more battle-tested v8 export path for a
   first deployment.
4. **Reject MobileNet-SSD/EfficientDet-Lite/NanoDet for this use case
   specifically** because the project's hardest requirement is catching
   *small/early-stage flame* and *thin smoke* — exactly the regime where
   their lower mAP (especially on small objects) costs the most. The 15-20MB
   model-size budget is generous enough that paying for YOLOv8n/v11n's extra
   few MB over NanoDet is the right trade.
5. **416×416 over 320/640**: 320px loses too much small-object signal for
   "small flame" / "thin smoke" / "short circuit spark"; 640px roughly
   doubles latency for accuracy gains that don't change the
   detect-within-N-seconds outcome once the 8-frame temporal gate is in
   place. 416px is the documented sweet spot for YOLO-nano models on
   CPU-class hardware — confirm against your own `docs/benchmark_results.csv`
   once trained weights exist; the resolution is a one-line change in
   `configs/model.yaml`.

## How to validate this on real hardware

```bash
# After training + exporting FP32/FP16/INT8 x 320/416/640 (9 files total):
bash scripts/benchmark_models.sh
# Produces docs/benchmark_results.csv - update the table above with real numbers,
# and re-confirm the resolution/precision choice in configs/model.yaml:active_model.
```
