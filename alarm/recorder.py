"""Snapshot + pre/post-event video clip recording.

Keeps a short rolling ring buffer of recent frames in memory so that when an
alarm fires, the saved clip includes footage from *before* the trigger
moment (clip_pre_event_s), not just after - useful for incident review and
fire-marshal reporting.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger("alarm.recorder")


@dataclass
class _RingFrame:
    timestamp: float
    frame: np.ndarray


class ClipRecorder:
    """Maintains a rolling buffer of recent frames; on trigger, writes
    pre-event frames immediately and appends post-event frames as they
    arrive over the next `clip_post_event_s` seconds."""

    def __init__(self, cfg: dict, clips_dir: str | Path, fps: int = 15) -> None:
        self._pre_s = cfg.get("clip_pre_event_s", 5)
        self._post_s = cfg.get("clip_post_event_s", 15)
        self._codec = cfg.get("clip_codec", "mp4v")
        self._target_fps = cfg.get("clip_fps", fps)
        self._clips_dir = Path(clips_dir)
        self._clips_dir.mkdir(parents=True, exist_ok=True)

        buffer_len = int(self._pre_s * fps) + 5
        self._buffer: deque[_RingFrame] = deque(maxlen=buffer_len)

        self._active_writer: cv2.VideoWriter | None = None
        self._active_writer_path: Path | None = None
        self._active_until: float = 0.0

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        now = time.time()
        self._buffer.append(_RingFrame(timestamp=now, frame=frame_bgr))

        if self._active_writer is not None:
            self._active_writer.write(frame_bgr)
            if now >= self._active_until:
                self._finalize_active_clip()

    def start_clip(self, event_id: int) -> Path:
        """Flush the pre-event buffer to a new video file and keep writing
        until clip_post_event_s elapses (handled in subsequent push_frame calls)."""
        if self._active_writer is not None:
            self._finalize_active_clip()

        filename = f"event_{event_id}_{int(time.time())}.mp4"
        path = self._clips_dir / filename
        h, w = self._buffer[-1].frame.shape[:2] if self._buffer else (0, 0)
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        writer = cv2.VideoWriter(str(path), fourcc, self._target_fps, (w, h))

        for ring_frame in self._buffer:
            writer.write(ring_frame.frame)

        self._active_writer = writer
        self._active_writer_path = path
        self._active_until = time.time() + self._post_s
        logger.info("Started alarm clip %s (pre-buffered %s frames)", path, len(self._buffer))
        return path

    def _finalize_active_clip(self) -> None:
        if self._active_writer is not None:
            self._active_writer.release()
            logger.info("Finalized alarm clip %s", self._active_writer_path)
        self._active_writer = None
        self._active_writer_path = None


class SnapshotSaver:
    def __init__(self, snapshots_dir: str | Path, image_format: str = "jpg") -> None:
        self._dir = Path(snapshots_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._format = image_format

    def save(self, frame_bgr: np.ndarray, event_id: int) -> Path:
        path = self._dir / f"event_{event_id}_{int(time.time())}.{self._format}"
        cv2.imwrite(str(path), frame_bgr)
        logger.info("Saved alarm snapshot %s", path)
        return path
