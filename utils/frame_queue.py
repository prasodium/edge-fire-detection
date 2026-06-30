"""Bounded, latest-frame-wins queue used between the async camera capture thread
and the inference thread pool.

A standard FIFO queue would let stale frames pile up under CPU pressure,
growing end-to-end latency unboundedly. This queue always keeps only the
freshest frame(s), trading "process every frame" for "process the latest
frame" - the right tradeoff for a real-time alarm system on constrained CPU.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Condition
from typing import Any


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    data: Any  # numpy.ndarray, kept as Any to avoid importing numpy/cv2 here


class LatestFrameQueue:
    """Thread-safe ring buffer that drops the oldest frame when full."""

    def __init__(self, maxsize: int = 4) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._maxsize = maxsize
        self._buffer: deque[FramePacket] = deque(maxlen=maxsize)
        self._cond = Condition()
        self._closed = False

    def put(self, packet: FramePacket) -> None:
        with self._cond:
            self._buffer.append(packet)  # deque auto-evicts oldest when full
            self._cond.notify_all()

    def get(self, timeout: float | None = 1.0) -> FramePacket | None:
        """Pop the oldest unconsumed frame; returns None on timeout or close."""
        with self._cond:
            if not self._buffer and not self._closed:
                self._cond.wait(timeout=timeout)
            if not self._buffer:
                return None
            return self._buffer.popleft()

    def get_latest_and_clear(self) -> FramePacket | None:
        """Pop only the freshest frame, discarding any backlog. Used when the
        consumer fell behind and wants to resynchronize to real-time."""
        with self._cond:
            if not self._buffer:
                return None
            latest = self._buffer.pop()
            self._buffer.clear()
            return latest

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __len__(self) -> int:
        with self._cond:
            return len(self._buffer)
