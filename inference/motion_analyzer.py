"""Stage 4: scene-level motion analysis (frame differencing + sparse optical flow).

Fire/smoke events are accompanied by *some* motion in the affected region.
A region that the detector flags but that shows zero motion across frames
(e.g. a printed fire poster, a still photo on a slide) is suspicious. This
also rejects rigid waving motion (a flag, a banner) via flow-direction
consistency, distinct from the flicker check in false_positive_filter.py
which looks at brightness, not pixel displacement.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionResult:
    has_motion: bool
    motion_pixel_ratio: float
    mean_flow_divergence: float
    is_rigid_translation: bool


class MotionAnalyzer:
    def __init__(
        self,
        frame_diff_threshold: int = 25,
        min_motion_pixels_ratio: float = 0.002,
        smoke_min_divergence: float = 0.05,
        rigid_motion_rejection: bool = True,
    ) -> None:
        self._diff_threshold = frame_diff_threshold
        self._min_ratio = min_motion_pixels_ratio
        self._min_divergence = smoke_min_divergence
        self._reject_rigid = rigid_motion_rejection
        self._prev_gray: np.ndarray | None = None
        self._prev_flow_gray: np.ndarray | None = None

    def analyze(self, frame_bgr: np.ndarray, box_xyxy: tuple[float, float, float, float]) -> MotionResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = (int(max(0, v)) for v in box_xyxy)
        x2 = min(x2, gray.shape[1])
        y2 = min(y2, gray.shape[0])

        motion_ratio = self._frame_diff_ratio(gray, (x1, y1, x2, y2))
        divergence, is_rigid = self._optical_flow_divergence(gray, (x1, y1, x2, y2))

        has_motion = motion_ratio >= self._min_ratio
        return MotionResult(
            has_motion=has_motion,
            motion_pixel_ratio=motion_ratio,
            mean_flow_divergence=divergence,
            is_rigid_translation=is_rigid,
        )

    def _frame_diff_ratio(self, gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0
        diff = cv2.absdiff(gray[y1:y2, x1:x2], self._prev_gray[y1:y2, x1:x2])
        self._prev_gray = gray
        if diff.size == 0:
            return 0.0
        return float(np.mean(diff > self._diff_threshold))

    def _optical_flow_divergence(
        self, gray: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[float, bool]:
        """Cheap dense Farneback flow on the cropped region only (not full
        frame) to stay within the CPU budget at 15fps on Pi 5."""
        x1, y1, x2, y2 = box
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0, False

        if self._prev_flow_gray is None or self._prev_flow_gray.shape != region.shape:
            self._prev_flow_gray = region
            return 0.0, False

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_flow_gray, region, None, 0.5, 2, 11, 2, 5, 1.1, 0
        )
        self._prev_flow_gray = region

        fx, fy = flow[..., 0], flow[..., 1]
        divergence = float(np.mean(np.abs(np.gradient(fx)[1]) + np.abs(np.gradient(fy)[0])))

        # Rigid translation = nearly all vectors point the same direction with low spread;
        # diffusive smoke/flame flicker = vectors point many directions (high circular variance).
        angles = np.arctan2(fy, fx)
        mean_cos = np.mean(np.cos(angles))
        mean_sin = np.mean(np.sin(angles))
        resultant_length = (mean_cos**2 + mean_sin**2) ** 0.5  # 1.0 = perfectly aligned (rigid)
        is_rigid = self._reject_rigid and resultant_length > 0.85

        return divergence, is_rigid
