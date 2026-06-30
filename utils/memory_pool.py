"""Pre-allocated numpy buffer pool to avoid per-frame allocation/GC churn.

On an 8GB Pi 5 running 24/7, repeated allocation of ~640x640x3 uint8 arrays
(1.2MB each) at 15fps is ~18MB/s of allocator churn. Reusing a small pool of
fixed-shape buffers keeps memory flat and avoids GC pauses that would cause
dropped frames.
"""
from __future__ import annotations

from queue import Empty, Queue
from typing import Tuple

import numpy as np


class FrameBufferPool:
    """Fixed-size pool of pre-allocated np.ndarray buffers of one shape/dtype."""

    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype = np.uint8, pool_size: int = 6) -> None:
        self._shape = shape
        self._dtype = dtype
        self._available: Queue[np.ndarray] = Queue(maxsize=pool_size)
        for _ in range(pool_size):
            self._available.put(np.empty(shape, dtype=dtype))

    def acquire(self, timeout: float = 0.5) -> np.ndarray:
        """Borrow a buffer. Falls back to a fresh allocation if the pool is
        exhausted (better a transient allocation than a dropped frame)."""
        try:
            return self._available.get(timeout=timeout)
        except Empty:
            return np.empty(self._shape, dtype=self._dtype)

    def release(self, buf: np.ndarray) -> None:
        if buf.shape != self._shape or buf.dtype != self._dtype:
            return  # don't pool mismatched buffers, just let GC reclaim it
        try:
            self._available.put_nowait(buf)
        except Exception:
            pass  # pool full - drop it, no harm

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
