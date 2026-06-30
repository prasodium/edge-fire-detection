"""Shared frame-annotation helpers used by both dashboards (FastAPI's MJPEG
stream in dashboard/app.py and the Streamlit live view in
dashboard/streamlit_app.py) so bounding-box rendering stays in one place.
"""
from __future__ import annotations

import cv2
import numpy as np

from inference.detector import Detection

SEVERITY_BOX_COLOR = {
    "info": (200, 200, 200),
    "warning": (0, 165, 255),
    "critical": (0, 0, 255),
}


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Returns a copy of frame_bgr with bounding boxes + class/confidence labels drawn."""
    annotated = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.box_xyxy)
        color = (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.0%}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(0, y1 - text_h - 8)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return annotated
