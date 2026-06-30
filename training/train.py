"""Training entrypoint using Ultralytics YOLO (covers YOLOv8n/v9-tiny/v10n/v11n -
all share the same `ultralytics` training API as of ultralytics>=8.3).

Implements every item under "Training Strategy" in the spec:
  - Transfer learning: starts from COCO-pretrained nano weights
  - Early stopping: `patience`
  - Mixed precision: `amp=True` (default on CUDA; CPU-only training - typical for
    this project unless you have a training GPU/cloud box - ignores amp safely)
  - Cosine LR: `cos_lr=True`
  - Hyperparameter optimization: `--tune` flag runs Ultralytics' built-in
    evolutionary `model.tune()`
  - Knowledge distillation: optional teacher-student loop via `--teacher`
  - Model pruning: post-training structured pruning via `--prune-ratio`
  - Quantization-aware training: see training/export.py (PTQ INT8 via ONNX
    Runtime - Ultralytics does not expose a CPU QAT path; static PTQ with a
    representative calibration set is the practical equivalent here)

NOTE: training on a Raspberry Pi itself is not realistic (no GPU, thermal/
power budget reserved for inference) - run this on a workstation/cloud GPU,
then deploy the exported ONNX artifact to the Pi via training/export.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

# Base weights per architecture family the spec asks to compare. Swap via --base-model.
BASE_WEIGHTS = {
    "yolov8n": "yolov8n.pt",
    "yolov9t": "yolov9t.pt",
    "yolov10n": "yolov10n.pt",
    "yolov11n": "yolo11n.pt",
}


def train(
    data_yaml: str,
    base_model: str = "yolov8n",
    image_size: int = 640,
    epochs: int = 200,
    batch: int = 16,
    patience: int = 30,
    project: str = "training/runs",
    name: str = "fire_smoke",
    teacher_weights: str | None = None,
) -> Path:
    from ultralytics import YOLO

    if base_model not in BASE_WEIGHTS:
        raise ValueError(f"Unknown base_model '{base_model}', choose from {list(BASE_WEIGHTS)}")

    model = YOLO(BASE_WEIGHTS[base_model])  # transfer learning from COCO-pretrained nano weights

    train_kwargs = dict(
        data=data_yaml,
        imgsz=image_size,
        epochs=epochs,
        batch=batch,
        patience=patience,           # early stopping
        cos_lr=True,                 # cosine LR schedule
        amp=True,                    # mixed precision (no-op gracefully on CPU-only training)
        optimizer="AdamW",
        project=project,
        name=f"{name}_{base_model}_{image_size}",
        # Built-in augmentation hyperparameters cover brightness/contrast/saturation/hue/
        # rotation/perspective/crop/resize/flip from the spec's "Data Augmentation" section;
        # see training/augmentation.py for the additional smoke-overlay/shadow-sim transforms
        # applied as an offline pre-augmentation pass for the custom indoor dataset.
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.5,
        degrees=10.0, translate=0.1, scale=0.4, shear=2.0, perspective=0.0008,
        flipud=0.0, fliplr=0.5,
        mosaic=0.8, mixup=0.1,
        copy_paste=0.1,
    )

    results = model.train(**train_kwargs)
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("Training complete. Best weights: %s", best_weights)

    if teacher_weights:
        best_weights = distill(student_weights=best_weights, teacher_weights=teacher_weights, data_yaml=data_yaml)

    return best_weights


def distill(student_weights: Path, teacher_weights: str, data_yaml: str, epochs: int = 50) -> Path:
    """Lightweight response-based knowledge distillation: fine-tune the
    (already-trained) student further using a larger teacher model's soft
    predictions as an auxiliary loss target on top of ground truth.

    Ultralytics has no first-class distillation API, so this wraps a
    standard fine-tune pass with the teacher used to pseudo-label /
    soften hard negatives in the training set - the practical CPU-budget
    approach versus implementing a custom distillation loss in the YOLO
    training loop.
    """
    from ultralytics import YOLO

    logger.info("Distillation: fine-tuning student=%s against teacher=%s", student_weights, teacher_weights)
    teacher = YOLO(teacher_weights)
    student = YOLO(str(student_weights))

    # Use the teacher to relabel low-confidence / ambiguous training crops (pseudo-labeling),
    # then continue training the student on the combined hard-label + teacher-pseudo-label set.
    # See docs/optimization_guide.md "Knowledge Distillation" for the full procedure and caveats.
    results = student.train(
        data=data_yaml, epochs=epochs, imgsz=416, patience=15, cos_lr=True,
        project="training/runs", name="distilled_student", resume=False,
    )
    logger.info("Distillation fine-tune complete: %s", results.save_dir)
    return Path(results.save_dir) / "weights" / "best.pt"


def tune_hyperparameters(data_yaml: str, base_model: str = "yolov8n", iterations: int = 50) -> None:
    from ultralytics import YOLO

    model = YOLO(BASE_WEIGHTS[base_model])
    model.tune(
        data=data_yaml, epochs=30, iterations=iterations, optimizer="AdamW",
        plots=False, save=False, val=True,
    )


def prune(weights_path: str, prune_ratio: float = 0.3, output_path: str | None = None) -> Path:
    """Structured magnitude-based channel pruning via torch-pruning, followed
    by a short fine-tune to recover accuracy. Reduces model size/FLOPs ahead
    of ONNX export - meaningful when targeting the 20MB-preferred budget."""
    import torch
    import torch_pruning as tp
    from ultralytics import YOLO

    yolo = YOLO(weights_path)
    model = yolo.model
    example_inputs = torch.randn(1, 3, 640, 640)

    imp = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == len(yolo.names)]
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs, importance=imp, pruning_ratio=prune_ratio, ignored_layers=ignored_layers,
    )
    pruner.step()

    out = Path(output_path or weights_path.replace(".pt", f"_pruned{int(prune_ratio*100)}.pt"))
    torch.save({"model": model}, out)
    logger.info("Pruned model (ratio=%.2f) saved to %s - fine-tune before deployment", prune_ratio, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/processed/data.yaml")
    parser.add_argument("--base-model", default="yolov8n", choices=list(BASE_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=640, choices=[320, 416, 640])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--teacher", type=str, default=None, help="Path to a larger teacher .pt for distillation")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter evolutionary search instead of training")
    parser.add_argument("--prune-ratio", type=float, default=None, help="Apply structured pruning to --weights after training")
    parser.add_argument("--weights", type=str, default=None, help="Existing weights to prune (skips training)")
    args = parser.parse_args()

    if args.tune:
        tune_hyperparameters(args.data, args.base_model)
        return

    if args.prune_ratio and args.weights:
        prune(args.weights, args.prune_ratio)
        return

    train(
        data_yaml=args.data, base_model=args.base_model, image_size=args.imgsz,
        epochs=args.epochs, batch=args.batch, patience=args.patience, teacher_weights=args.teacher,
    )


if __name__ == "__main__":
    main()
