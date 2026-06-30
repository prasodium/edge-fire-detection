# weights/

Holds exported ONNX model files referenced by `configs/model.yaml`.
`.onnx`/`.pt` files are gitignored — see root `.gitignore` — because trained
weights are specific to a training run/dataset and don't belong in git
history.

## Current state: a real demo model, not production weights

This directory currently (if you've run the steps in the project README)
contains a **real, trained 2-class demo model** (`small_flame`, `smoke`),
trained on 141 public images, exported to FP32/FP16/INT8 ONNX, and wired up
as `configs/model.yaml`'s `active_model` (`yolov8n_demo_fp16_416`) so the
pipeline/dashboard run today. See `docs/benchmark_report.md` for the full
provenance, measured accuracy (mAP50=0.742 on held-out test data), and an
important finding: **the INT8 export lost significant accuracy** on this
small dataset (mAP50 dropped to 0.264) — FP16 is the demo default for that
reason, not INT8.

```
weights/
├── yolov8n_fire_fp32_416.onnx   # 11.6MB - demo model, FP32
├── yolov8n_fire_fp16_416.onnx   # 5.8MB  - demo model, FP16 - active_model default
└── yolov8n_fire_int8_416.onnx   # 3.2MB  - demo model, INT8 - accuracy regressed, do not deploy as-is
```

**This is a pipeline-validation model, not a production detector.** It only
knows 2 generic classes from ~140 training images of one public dataset —
it has not seen electrical/paper/wood/curtain/plastic fire, the
fire-near-projector/podium/stage/exit-door context classes, or any of the
hard-negative false-positive cases (stage lighting, LEDs, candles, etc.)
this project's `configs/dataset.yaml` calls for. Do not deploy it to a real
venue.

## Training the production model

```bash
python training/train.py --base-model yolov8n --imgsz 416 --epochs 200 \
    --data datasets/processed/data.yaml
python training/export.py --weights training/runs/.../best.pt --imgsz 416 \
    --calib-dir datasets/processed/val/images
```

Then restore the full 15-class list in `configs/model.yaml` (it's there,
commented out, under "PRODUCTION classes") and point `active_model` at your
new export. See `docs/deployment_guide.md` and `docs/benchmark_report.md`.

**Do not deploy a model without verifying classes match `configs/model.yaml`**
— a COCO-pretrained (non-fire) checkpoint has no fire/smoke classes and will
never alarm; a class-count/order mismatch between the model and
`configs/model.yaml:classes` will silently mislabel every detection.
