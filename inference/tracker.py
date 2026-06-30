"""Minimal IoU-based multi-object tracker.

Neither the temporal verifier ("8 consecutive frames of the same fire")
nor the false-positive filter (bbox jitter, flicker frequency) can work on
unlinked per-frame detections - they need a notion of "this is the same
candidate region as last frame". A full ByteTrack/DeepSORT is overkill for
a handful of fire/smoke regions at 416x416, so this is a deliberately
simple greedy IoU tracker, cheap enough to run every frame on CPU.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from inference.detector import Detection


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    class_name: str
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    confidence_history: list[float] = field(default_factory=list)
    box_history: list[tuple[float, float, float, float]] = field(default_factory=list)
    brightness_history: list[float] = field(default_factory=list)
    timestamp_history: list[float] = field(default_factory=list)
    consecutive_hits: int = 1
    missed_frames: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def rolling_confidence_avg(self, window: int) -> float:
        recent = self.confidence_history[-window:]
        return sum(recent) / len(recent) if recent else 0.0


class IoUTracker:
    """Greedy IoU matcher: each new detection is matched to the existing track
    with highest IoU (above threshold), else spawns a new track. Tracks that
    miss too many frames are dropped."""

    def __init__(self, iou_match_threshold: float = 0.3, max_missed_frames: int = 2, history_window: int = 16) -> None:
        self._iou_threshold = iou_match_threshold
        self._max_missed = max_missed_frames
        self._history_window = history_window
        self._tracks: dict[int, Track] = {}
        self._next_id = itertools.count(1)

    @property
    def tracks(self) -> dict[int, Track]:
        return self._tracks

    def update(self, detections: list[Detection], brightness_by_index: list[float] | None = None) -> list[Track]:
        now = time.time()
        unmatched_track_ids = set(self._tracks.keys())
        matched_tracks: list[Track] = []

        for i, det in enumerate(detections):
            best_id, best_iou = None, 0.0
            for tid in unmatched_track_ids:
                track = self._tracks[tid]
                if track.class_name != det.class_name:
                    continue
                iou = _iou(track.box_xyxy, det.box_xyxy)
                if iou > best_iou:
                    best_id, best_iou = tid, iou

            brightness = brightness_by_index[i] if brightness_by_index else 0.0

            if best_id is not None and best_iou >= self._iou_threshold:
                track = self._tracks[best_id]
                track.box_xyxy = det.box_xyxy
                track.confidence = det.confidence
                track.consecutive_hits += 1
                track.missed_frames = 0
                track.last_seen = now
                self._append_bounded(track.confidence_history, det.confidence)
                self._append_bounded(track.box_history, det.box_xyxy)
                self._append_bounded(track.brightness_history, brightness)
                self._append_bounded(track.timestamp_history, now)
                unmatched_track_ids.discard(best_id)
                matched_tracks.append(track)
            else:
                new_id = next(self._next_id)
                track = Track(
                    track_id=new_id,
                    class_name=det.class_name,
                    box_xyxy=det.box_xyxy,
                    confidence=det.confidence,
                    confidence_history=[det.confidence],
                    box_history=[det.box_xyxy],
                    brightness_history=[brightness],
                    timestamp_history=[now],
                )
                self._tracks[new_id] = track
                matched_tracks.append(track)

        for tid in unmatched_track_ids:
            track = self._tracks[tid]
            track.missed_frames += 1
            track.consecutive_hits = 0

        self._tracks = {
            tid: t for tid, t in self._tracks.items() if t.missed_frames <= self._max_missed
        }
        return matched_tracks

    def _append_bounded(self, lst: list, value) -> None:
        lst.append(value)
        if len(lst) > self._history_window:
            del lst[: len(lst) - self._history_window]
