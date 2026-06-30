"""Classical-CV smoke verification branch.

Runs alongside the learned detector's smoke/dense_smoke/thin_smoke classes
as an independent signal (Stage 2 of the decision engine). Smoke has fairly
distinctive low-level statistics - low saturation, dark-channel attenuation,
and diffusive upward optical flow - that a cheap heuristic can check without
spending CPU on a second neural net. This is intentionally classical CV
(no second model) to stay inside the CPU/power budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SmokeVerificationResult:
    verified: bool
    dark_channel_score: float
    low_saturation_ratio: float
    diffusion_score: float


def _dark_channel(region_bgr: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """He et al. dark channel prior: smoke/haze regions have a low minimum
    across RGB channels in local patches (unlike most saturated solid colors)."""
    min_channel = np.min(region_bgr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)


class SmokeVerifier:
    def __init__(
        self,
        dark_channel_threshold: float = 0.35,
        motion_diffusion_threshold: float = 0.12,
        min_region_area_px: int = 900,
    ) -> None:
        self._dc_threshold = dark_channel_threshold
        self._diffusion_threshold = motion_diffusion_threshold
        self._min_area = min_region_area_px
        self._prev_gray: np.ndarray | None = None

    def verify(self, frame_bgr: np.ndarray, box_xyxy: tuple[float, float, float, float]) -> SmokeVerificationResult:
        x1, y1, x2, y2 = (int(max(0, v)) for v in box_xyxy)
        x2 = min(x2, frame_bgr.shape[1])
        y2 = min(y2, frame_bgr.shape[0])
        region = frame_bgr[y1:y2, x1:x2]

        if region.size == 0 or region.shape[0] * region.shape[1] < self._min_area:
            return SmokeVerificationResult(False, 0.0, 0.0, 0.0)

        dark_channel = _dark_channel(region.astype(np.float32) / 255.0)
        dc_score = float(np.mean(dark_channel))

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        low_sat_ratio = float(np.mean(saturation < 0.25))  # smoke is low-saturation, grayish

        diffusion_score = self._diffusion_score(frame_bgr, (x1, y1, x2, y2))

        verified = (
            dc_score >= self._dc_threshold
            and low_sat_ratio > 0.4
            and diffusion_score >= self._diffusion_threshold
        )
        return SmokeVerificationResult(verified, dc_score, low_sat_ratio, diffusion_score)

    def _diffusion_score(self, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> float:
        """Smoke diffuses outward/upward over time; rigid objects (orange cloth,
        a static light) don't change shape. Approximate via frame-to-frame
        region growth using cheap grayscale diffing rather than full optical flow
        (kept separate from the shared optical-flow motion analyzer for locality)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0

        x1, y1, x2, y2 = box
        diff = cv2.absdiff(gray[y1:y2, x1:x2], self._prev_gray[y1:y2, x1:x2])
        self._prev_gray = gray
        if diff.size == 0:
            return 0.0
        return float(np.mean(diff > 12) )  # fraction of pixels that changed = diffusive activity
