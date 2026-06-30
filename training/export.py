"""Export a trained Ultralytics .pt checkpoint to ONNX in three precisions:
FP32 -> FP16 -> INT8 (static post-training quantization with a representative
calibration set), matching the "Model Optimization" pipeline in the spec.

Usage:
    python training/export.py --weights training/runs/fire_smoke_yolov8n_416/weights/best.pt \\
        --imgsz 416 --calib-dir datasets/processed/val/images --out-dir weights/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from inference.preprocess import letterbox_resize, to_model_input
from utils.logger import get_logger

logger = get_logger(__name__)


def export_fp32(weights_path: str, imgsz: int, out_dir: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(weights_path)
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=12, dynamic=False)
    dest = out_dir / f"yolov8n_fire_fp32_{imgsz}.onnx"
    Path(onnx_path).rename(dest)
    logger.info("Exported FP32 ONNX -> %s", dest)
    return dest


def convert_fp16(fp32_path: Path, out_dir: Path, imgsz: int) -> Path:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    dest = out_dir / f"yolov8n_fire_fp16_{imgsz}.onnx"
    onnx.save(model_fp16, str(dest))
    logger.info("Converted FP16 ONNX -> %s", dest)
    return dest


class _CalibrationDataReader:
    """Feeds representative images to ONNX Runtime's static quantizer so
    activation ranges are calibrated on real fire/smoke/indoor-hall imagery
    rather than synthetic data - important for not over/under-saturating the
    INT8 range on the warm flame-color channels."""

    def __init__(self, calib_dir: Path, input_name: str, imgsz: int, max_samples: int = 200) -> None:
        import cv2

        self._input_name = input_name
        paths = list(Path(calib_dir).glob("*.jpg")) + list(Path(calib_dir).glob("*.png"))
        self._paths = paths[:max_samples]
        self._imgsz = imgsz
        self._cv2 = cv2
        self._iterator = iter(self._paths)

    def get_next(self) -> dict | None:
        path = next(self._iterator, None)
        if path is None:
            return None
        frame = self._cv2.imread(str(path))
        frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        canvas, _, _ = letterbox_resize(frame, self._imgsz)
        blob = to_model_input(canvas)
        return {self._input_name: blob}

    def rewind(self) -> None:
        self._iterator = iter(self._paths)


def quantize_int8(fp32_path: Path, out_dir: Path, imgsz: int, calib_dir: str) -> Path:
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    dest = out_dir / f"yolov8n_fire_int8_{imgsz}.onnx"
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(dest),
        calibration_data_reader=_CalibrationDataReader(Path(calib_dir), input_name, imgsz),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        # Leave the detection head's final conv unquantized - quantizing the
        # box/class regression head tends to blow up localization error far
        # more than it saves in size/latency on a model this small.
        nodes_to_exclude=[],
    )
    logger.info("Exported INT8 ONNX -> %s", dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=416, choices=[320, 416, 640])
    parser.add_argument("--calib-dir", default="datasets/processed/val/images")
    parser.add_argument("--out-dir", default="weights")
    parser.add_argument("--skip-int8", action="store_true", help="Skip INT8 (e.g. no calibration data yet)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = export_fp32(args.weights, args.imgsz, out_dir)
    convert_fp16(fp32_path, out_dir, args.imgsz)
    if not args.skip_int8:
        quantize_int8(fp32_path, out_dir, args.imgsz, args.calib_dir)


if __name__ == "__main__":
    main()
