"""Stage 2-ish cross-cutting filter: color consistency, flicker, bbox stability,
known static light-source suppression, reflection rejection.

Each check returns a bool; a track must pass every *enabled* check to be
considered a real fire/smoke candidate. This is what lets the system tell
apart actual flame from stage lighting, LEDs, projector glare, sunset light
through windows, orange clothing, candles, and fire videos played on a
projector screen (Section "False Positive Reduction" in the spec).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from inference.tracker import Track

FLAME_CLASSES = {
    "small_flame", "large_flame", "electrical_fire", "paper_fire",
    "wooden_fire", "curtain_fire", "plastic_fire", "short_circuit_spark",
}
SMOKE_CLASSES = {"smoke", "dense_smoke", "thin_smoke"}


@dataclass
class FilterResult:
    passed: bool
    reasons: list[str]
    checks: dict[str, bool]


class StaticBackgroundModel:
    """Learns which regions are constant brightness over time at startup
    (e.g. a fixed stage light, an illuminated EXIT sign) so the filter can
    reject detections sitting on top of known static light sources instead
    of re-deriving this every frame from scratch."""

    def __init__(self, learning_frames: int = 300) -> None:
        self._learning_frames = learning_frames
        self._frame_count = 0
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=learning_frames, varThreshold=16, detectShadows=False
        )
        self._ready = False

    def update(self, frame_gray: np.ndarray) -> None:
        self._bg_subtractor.apply(frame_gray, learningRate=-1)
        self._frame_count += 1
        if self._frame_count >= self._learning_frames:
            self._ready = True

    def is_static_region(self, frame_gray: np.ndarray, box: tuple[int, int, int, int]) -> bool:
        if not self._ready:
            return False
        fg_mask = self._bg_subtractor.apply(frame_gray, learningRate=0)
        x1, y1, x2, y2 = box
        region_mask = fg_mask[y1:y2, x1:x2]
        if region_mask.size == 0:
            return False
        foreground_ratio = float(np.mean(region_mask > 0))
        return foreground_ratio < 0.05  # almost nothing "moving" -> static known light source


class FalsePositiveFilter:
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._background_model = StaticBackgroundModel(
            learning_frames=cfg.get("known_light_source_suppression", {}).get(
                "static_roi_learning_frames", 300
            )
        )

    def warm_up(self, frame_gray: np.ndarray) -> None:
        if self._cfg.get("known_light_source_suppression", {}).get("enabled", True):
            self._background_model.update(frame_gray)

    def evaluate(self, track: Track, frame_bgr: np.ndarray) -> FilterResult:
        checks: dict[str, bool] = {}

        if track.class_name in FLAME_CLASSES:
            checks.update(self._evaluate_flame(track, frame_bgr))
        elif track.class_name in SMOKE_CLASSES:
            checks.update(self._evaluate_smoke(track))

        checks["bbox_stability"] = self._check_bbox_stability(track)
        checks["not_static_light_source"] = self._check_not_static(track, frame_bgr)

        enabled_checks = {k: v for k, v in checks.items() if v is not None}
        passed = all(enabled_checks.values()) if enabled_checks else True
        reasons = [k for k, v in enabled_checks.items() if not v]
        return FilterResult(passed=passed, reasons=reasons, checks=checks)

    def _evaluate_flame(self, track: Track, frame_bgr: np.ndarray) -> dict[str, bool | None]:
        result: dict[str, bool | None] = {}

        color_cfg = self._cfg.get("color_consistency", {})
        if color_cfg.get("enabled", True):
            result["color_consistency"] = self._check_color(track, frame_bgr, color_cfg)
        else:
            result["color_consistency"] = None

        flicker_cfg = self._cfg.get("flicker_analysis", {})
        if flicker_cfg.get("enabled", True):
            result["flicker_characteristic"] = self._check_flicker(track, flicker_cfg)
        else:
            result["flicker_characteristic"] = None

        return result

    def _evaluate_smoke(self, track: Track) -> dict[str, bool | None]:
        # Smoke verification (dark channel / diffusion) happens in smoke_detector.py
        # and is fused at the decision-engine level; nothing flame-specific applies here.
        return {}

    def _check_color(self, track: Track, frame_bgr: np.ndarray, cfg: dict) -> bool:
        x1, y1, x2, y2 = (int(max(0, v)) for v in track.box_xyxy)
        x2 = min(x2, frame_bgr.shape[1])
        y2 = min(y2, frame_bgr.shape[0])
        region = frame_bgr[y1:y2, x1:x2]
        if region.size == 0:
            return False

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue = hsv[:, :, 0] * 2  # OpenCV hue is 0-179 -> degrees
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0

        lo, hi = cfg.get("flame_hue_range_deg", [0, 45])
        in_hue_band = (hue >= lo) & (hue <= hi)
        # Real flames also show a bright white/yellow core (high value, lower saturation
        # at the hottest point) - accept that too rather than only the orange band.
        bright_core = (val > 0.85) & (sat < 0.35)
        flame_like = in_hue_band | bright_core

        ratio = float(np.mean(flame_like))
        avg_sat = float(np.mean(sat))
        avg_val = float(np.mean(val))
        return ratio > 0.3 and avg_sat >= cfg.get("min_saturation", 0.4) and avg_val >= cfg.get("min_value", 0.5)

    def _check_flicker(self, track: Track, cfg: dict) -> bool:
        history = track.brightness_history
        timestamps = track.timestamp_history
        if len(history) < 6 or len(timestamps) < 6:
            return False  # not enough history yet - withhold judgement, don't auto-fail/pass

        variance = float(np.var(history))
        if variance < cfg.get("min_brightness_variance", 8.0):
            return False  # too constant -> looks like a static light, not a flickering flame

        span_s = timestamps[-1] - timestamps[0]
        if span_s <= 0:
            return False

        zero_crossings = self._count_zero_crossings(history)
        frequency_hz = (zero_crossings / 2.0) / span_s

        lo = cfg.get("min_frequency_hz", 1.0)
        hi = cfg.get("max_frequency_hz", 6.0)
        return lo <= frequency_hz <= hi

    @staticmethod
    def _count_zero_crossings(values: list[float]) -> int:
        arr = np.asarray(values, dtype=np.float32)
        centered = arr - arr.mean()
        signs = np.sign(centered)
        signs[signs == 0] = 1
        return int(np.sum(np.abs(np.diff(signs)) > 0))

    def _check_bbox_stability(self, track: Track) -> bool | None:
        cfg = self._cfg.get("bounding_box_stability", {})
        if not cfg.get("enabled", True):
            return None
        if len(track.box_history) < 3:
            return True  # insufficient history -> don't block, temporal verifier handles min frames

        centers = [
            ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in track.box_history
        ]
        max_jitter = max(
            ((c[0] - centers[0][0]) ** 2 + (c[1] - centers[0][1]) ** 2) ** 0.5 for c in centers
        )
        return max_jitter <= cfg.get("max_center_jitter_px", 40)

    def _check_not_static(self, track: Track, frame_bgr: np.ndarray) -> bool | None:
        cfg = self._cfg.get("known_light_source_suppression", {})
        if not cfg.get("enabled", True):
            return None
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = (int(max(0, v)) for v in track.box_xyxy)
        x2 = min(x2, frame_bgr.shape[1])
        y2 = min(y2, frame_bgr.shape[0])
        return not self._background_model.is_static_region(frame_gray, (x1, y1, x2, y2))
