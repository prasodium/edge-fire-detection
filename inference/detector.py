"""ONNX Runtime YOLO-family detector wrapper, tuned for ARM64 CPU inference.

Works with any Ultralytics-exported YOLOv8/v9/v10/v11-nano ONNX model
(single output tensor [1, 4+num_classes, num_anchors], no built-in NMS) -
this is the standard `yolo export format=onnx` layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from inference.preprocess import letterbox_resize, to_model_input, unletterbox_box
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]  # in original frame coordinates


class FireDetector:
    """Loads one ONNX model and runs letterbox -> infer -> NMS -> Detection list."""

    def __init__(
        self,
        weights_path: str | Path,
        class_names: list[str],
        input_size: int = 416,
        confidence_threshold: float = 0.45,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 20,
        intra_op_threads: int = 3,
        inter_op_threads: int = 1,
    ) -> None:
        self._class_names = class_names
        self._input_size = input_size
        self._conf_threshold = confidence_threshold
        self._iou_threshold = nms_iou_threshold
        self._max_detections = max_detections

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. Run training/export.py "
                "or place a pretrained ONNX model there - see weights/README.md."
            )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_op_threads
        sess_options.inter_op_num_threads = inter_op_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # CPUExecutionProvider is the only provider relevant here: no Coral/Hailo/NCS.
        # ORT's CPU EP already uses ARM NEON kernels on aarch64 builds.
        self._session = ort.InferenceSession(
            str(weights_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("Loaded detector %s (input=%s)", weights_path.name, self._input_name)

    def infer(self, frame_rgb: np.ndarray) -> list[Detection]:
        canvas, scale, pad = letterbox_resize(frame_rgb, self._input_size)
        blob = to_model_input(canvas)

        outputs = self._session.run(None, {self._input_name: blob})
        raw = outputs[0]  # shape: [1, 4+num_classes, num_anchors]

        return self._postprocess(raw, scale, pad)

    def _postprocess(
        self, raw: np.ndarray, scale: float, pad: tuple[int, int]
    ) -> list[Detection]:
        preds = np.squeeze(raw, axis=0).T  # -> [num_anchors, 4+num_classes]
        if preds.size == 0:
            return []

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        keep = confidences >= self._conf_threshold
        if not np.any(keep):
            return []

        boxes_cxcywh = boxes_cxcywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        boxes_xyxy = self._cxcywh_to_xyxy(boxes_cxcywh)
        nms_indices = self._nms(boxes_xyxy, confidences, self._iou_threshold)
        nms_indices = nms_indices[: self._max_detections]

        detections: list[Detection] = []
        for idx in nms_indices:
            box = unletterbox_box(tuple(boxes_xyxy[idx]), scale, pad)
            cid = int(class_ids[idx])
            name = self._class_names[cid] if cid < len(self._class_names) else f"class_{cid}"
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=name,
                    confidence=float(confidences[idx]),
                    # cast to native float: numpy.float32 survives the arithmetic in
                    # unletterbox_box and otherwise breaks Pydantic/FastAPI JSON
                    # serialization (dashboard/app.py's /api/status) with
                    # "Unable to serialize unknown type: numpy.float32"
                    box_xyxy=tuple(float(v) for v in box),
                )
            )
        return detections

    @staticmethod
    def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Vectorized greedy NMS - no cv2.dnn.NMSBoxes dependency, keeps this module
        usable even when cv2 is unavailable in restricted environments."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
            union = areas[i] + areas[order[1:]] - inter
            iou = np.where(union > 0, inter / union, 0.0)

            order = order[1:][iou <= iou_threshold]
        return np.array(keep, dtype=int)
