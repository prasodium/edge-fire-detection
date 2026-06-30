# weights/

This directory holds exported ONNX model files referenced by
`configs/model.yaml`. It is intentionally empty in version control
(`.onnx`/`.pt` are gitignored — see root `.gitignore`) because:

1. Trained fire/smoke weights are specific to your training run and dataset,
   not something to bake into the repo.
2. Model binaries are large and don't belong in git history.

## Expected files (per `configs/model.yaml`)

```
weights/
├── yolov8n_fire_fp32_640.onnx
├── yolov8n_fire_fp16_416.onnx
└── yolov8n_fire_int8_416.onnx   <- default active_model
```

## How to populate this directory

```bash
python training/train.py --base-model yolov8n --imgsz 416 --epochs 200
python training/export.py --weights training/runs/.../best.pt --imgsz 416
```

See `docs/deployment_guide.md` and `docs/benchmark_report.md`.

**Do not start the pipeline without a real fire/smoke-trained model** — a
COCO-pretrained checkpoint has no fire/smoke classes and will never alarm.
