"""Frame preprocessing: resize + normalize, with minimal copying.

Targets ONNX Runtime's expected NCHW float tensor layout. Uses cv2's
SIMD/NEON-accelerated resize (cv2 ships with ARM NEON kernels on Pi OS
64-bit) rather than a Python/numpy resize loop.
"""
from __future__ import annotations

import cv2
import numpy as np


def letterbox_resize(
    frame: np.ndarray, target_size: int, pad_value: int = 114
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize preserving aspect ratio, padding to a square target_size.

    Returns (resized_padded_image, scale, (pad_x, pad_y)) so detections can
    be mapped back to original frame coordinates.
    """
    h, w = frame.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    canvas = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def to_model_input(image: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB [0,255] -> float32 NCHW [0,1], contiguous for zero-copy ORT binding."""
    blob = image.astype(np.float32, copy=False) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
    blob = np.ascontiguousarray(blob[np.newaxis, ...])  # add batch dim
    return blob


def unletterbox_box(
    box_xyxy: tuple[float, float, float, float],
    scale: float,
    pad: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Map a box from letterboxed model space back to original frame coordinates."""
    pad_x, pad_y = pad
    x1, y1, x2, y2 = box_xyxy
    return (
        (x1 - pad_x) / scale,
        (y1 - pad_y) / scale,
        (x2 - pad_x) / scale,
        (y2 - pad_y) / scale,
    )
