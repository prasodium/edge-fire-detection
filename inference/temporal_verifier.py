"""Stage 3: temporal verification.

The hard requirement from the spec: an alarm must NEVER trigger from a
single frame. A track must accumulate `consecutive_frames_required` hits
(small gaps tolerated up to `max_gap_frames`) AND its rolling confidence
average over `rolling_window_size` frames must exceed `min_average_confidence`.
"""
from __future__ import annotations

from dataclasses import dataclass

from inference.tracker import Track


@dataclass
class TemporalResult:
    verified: bool
    consecutive_hits: int
    rolling_avg_confidence: float
    frames_required: int
    confidence_required: float


class TemporalVerifier:
    def __init__(
        self,
        consecutive_frames_required: int = 8,
        rolling_window_size: int = 16,
        min_average_confidence: float = 0.85,
        max_gap_frames: int = 2,
    ) -> None:
        self._frames_required = consecutive_frames_required
        self._window = rolling_window_size
        self._min_avg_conf = min_average_confidence
        self._max_gap = max_gap_frames

    @property
    def window(self) -> int:
        return self._window

    def verify(self, track: Track) -> TemporalResult:
        # Track.consecutive_hits already accounts for tolerated gaps via the
        # tracker's missed_frames <= max_missed_frames eviction policy, so a
        # track surviving a short gap keeps accumulating rather than resetting
        # to zero - matches "ignore isolated detections" without being overly
        # strict about literal frame-for-frame consecutiveness.
        rolling_avg = track.rolling_confidence_avg(self._window)
        verified = (
            track.consecutive_hits >= self._frames_required
            and rolling_avg >= self._min_avg_conf
        )
        return TemporalResult(
            verified=verified,
            consecutive_hits=track.consecutive_hits,
            rolling_avg_confidence=rolling_avg,
            frames_required=self._frames_required,
            confidence_required=self._min_avg_conf,
        )
