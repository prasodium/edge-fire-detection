"""Multi-stage decision engine.

    Stage 1: AI detects fire/smoke (FireDetector, already run upstream)
    Stage 2: Smoke verification (SmokeVerifier) + flame false-positive filter
    Stage 3: Temporal verification (TemporalVerifier)
    Stage 4: Motion analysis (MotionAnalyzer)
    Stage 5: Final alarm decision (this class)

`require_all_stages: true` means every enabled stage must agree before an
AlarmEvent is raised - this is the core false-positive-reduction mechanism
described in the spec.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from inference.detector import Detection
from inference.false_positive_filter import FLAME_CLASSES, SMOKE_CLASSES, FalsePositiveFilter
from inference.motion_analyzer import MotionAnalyzer
from inference.smoke_detector import SmokeVerifier
from inference.temporal_verifier import TemporalVerifier
from inference.tracker import IoUTracker, Track
from inference.zone_mapper import ZoneMapper
from utils.logger import get_logger

logger = get_logger("alarm.decision_engine")


@dataclass
class AlarmEvent:
    track_id: int
    class_name: str
    severity: str  # "info" | "warning" | "critical"
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    zones: list[str]
    timestamp: float = field(default_factory=time.time)
    stage_results: dict = field(default_factory=dict)

    @property
    def context_label(self) -> str:
        """Builds e.g. 'large_flame near stage, exit_door' for logs/notifications."""
        if not self.zones:
            return self.class_name
        return f"{self.class_name} near {', '.join(self.zones)}"


class DecisionEngine:
    def __init__(
        self,
        decision_cfg: dict,
        model_cfg: dict,
        frame_width: int,
        frame_height: int,
    ) -> None:
        temporal_cfg = decision_cfg["temporal_verification"]
        fp_cfg = decision_cfg["false_positive_filter"]
        motion_cfg = decision_cfg["motion_analysis"]
        smoke_cfg = model_cfg.get("smoke_branch", {})

        self._tracker = IoUTracker(
            iou_match_threshold=fp_cfg.get("bounding_box_stability", {}).get("iou_match_threshold", 0.3),
            max_missed_frames=temporal_cfg.get("max_gap_frames", 2),
            history_window=temporal_cfg.get("rolling_window_size", 16),
        )
        self._temporal_verifier = TemporalVerifier(
            consecutive_frames_required=temporal_cfg.get("consecutive_frames_required", 8),
            rolling_window_size=temporal_cfg.get("rolling_window_size", 16),
            min_average_confidence=temporal_cfg.get("min_average_confidence", 0.85),
            max_gap_frames=temporal_cfg.get("max_gap_frames", 2),
        )
        self._fp_filter = FalsePositiveFilter(fp_cfg)
        self._smoke_verifier = SmokeVerifier(
            dark_channel_threshold=smoke_cfg.get("dark_channel_threshold", 0.35),
            motion_diffusion_threshold=smoke_cfg.get("motion_diffusion_threshold", 0.12),
            min_region_area_px=smoke_cfg.get("min_region_area_px", 900),
        )
        self._motion_analyzer = MotionAnalyzer(
            frame_diff_threshold=motion_cfg.get("frame_diff_threshold", 25),
            min_motion_pixels_ratio=motion_cfg.get("min_motion_pixels_ratio", 0.002),
            smoke_min_divergence=fp_cfg.get("optical_flow", {}).get("smoke_min_divergence", 0.05),
            rigid_motion_rejection=fp_cfg.get("optical_flow", {}).get("rigid_motion_rejection", True),
        )
        self._zone_mapper = ZoneMapper(decision_cfg.get("zones", []), frame_width, frame_height)

        self._require_all_stages = decision_cfg.get("decision_engine", {}).get("require_all_stages", True)
        self._severity_levels = decision_cfg.get("decision_engine", {}).get(
            "alarm_severity_levels", {"info": 0.0, "warning": 0.6, "critical": 0.85}
        )
        self._cooldown_s = temporal_cfg.get("cooldown_after_alarm_s", 120)
        self._last_alarm_time: dict[str, float] = {}  # keyed by f"{class_name}:{zones}"

    def warm_up(self, frame_bgr: np.ndarray) -> None:
        import cv2

        self._fp_filter.warm_up(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))

    def process(self, detections: list[Detection], frame_bgr: np.ndarray) -> list[AlarmEvent]:
        """Run all five stages for the current frame's detections and return
        any newly-confirmed alarm events (empty list most of the time - that
        is the expected steady state of a well-tuned system)."""
        tracks = self._tracker.update(detections, brightness_by_index=self._region_brightnesses(detections, frame_bgr))

        events: list[AlarmEvent] = []
        for track in tracks:
            stage_results = self._evaluate_track(track, frame_bgr)
            if self._is_confirmed(stage_results):
                event = self._build_event(track, stage_results)
                if self._past_cooldown(event):
                    events.append(event)
                    self._last_alarm_time[self._cooldown_key(event)] = event.timestamp
        return events

    def _region_brightnesses(self, detections: list[Detection], frame_bgr: np.ndarray) -> list[float]:
        import cv2

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        out = []
        for det in detections:
            x1, y1, x2, y2 = (int(max(0, v)) for v in det.box_xyxy)
            x2, y2 = min(x2, gray.shape[1]), min(y2, gray.shape[0])
            region = gray[y1:y2, x1:x2]
            out.append(float(np.mean(region)) if region.size else 0.0)
        return out

    def _evaluate_track(self, track: Track, frame_bgr: np.ndarray) -> dict:
        results: dict = {"stage1_ai_detection": True}  # already true - track exists from detector output

        if track.class_name in SMOKE_CLASSES:
            smoke_result = self._smoke_verifier.verify(frame_bgr, track.box_xyxy)
            results["stage2_smoke_verification"] = smoke_result.verified
            results["smoke_detail"] = smoke_result
        else:
            results["stage2_smoke_verification"] = True  # flame classes skip smoke-specific check

        fp_result = self._fp_filter.evaluate(track, frame_bgr)
        results["stage2_fp_filter"] = fp_result.passed
        results["fp_filter_detail"] = fp_result

        temporal_result = self._temporal_verifier.verify(track)
        results["stage3_temporal_verification"] = temporal_result.verified
        results["temporal_detail"] = temporal_result

        motion_result = self._motion_analyzer.analyze(frame_bgr, track.box_xyxy)
        motion_ok = motion_result.has_motion and not motion_result.is_rigid_translation
        results["stage4_motion_analysis"] = motion_ok
        results["motion_detail"] = motion_result

        return results

    def _is_confirmed(self, stage_results: dict) -> bool:
        gating_keys = [
            "stage1_ai_detection",
            "stage2_smoke_verification",
            "stage2_fp_filter",
            "stage3_temporal_verification",
            "stage4_motion_analysis",
        ]
        if self._require_all_stages:
            return all(stage_results[k] for k in gating_keys)
        # Soft mode: majority of stages must agree (kept for venues that find
        # require_all_stages too conservative after on-site calibration).
        passed = sum(1 for k in gating_keys if stage_results[k])
        return passed >= (len(gating_keys) - 1)

    def _build_event(self, track: Track, stage_results: dict) -> AlarmEvent:
        confidence = track.rolling_confidence_avg(self._temporal_verifier.window)
        severity = "info"
        for level, threshold in sorted(self._severity_levels.items(), key=lambda kv: kv[1]):
            if confidence >= threshold:
                severity = level
        zones = self._zone_mapper.locate(track.box_xyxy)
        return AlarmEvent(
            track_id=track.track_id,
            class_name=track.class_name,
            severity=severity,
            confidence=confidence,
            box_xyxy=track.box_xyxy,
            zones=zones,
            stage_results=stage_results,
        )

    def _cooldown_key(self, event: AlarmEvent) -> str:
        return f"{event.class_name}:{','.join(sorted(event.zones))}"

    def _past_cooldown(self, event: AlarmEvent) -> bool:
        key = self._cooldown_key(event)
        last = self._last_alarm_time.get(key)
        if last is None:
            return True
        return (event.timestamp - last) >= self._cooldown_s
