"""Lightweight system telemetry: CPU%, RAM, temperature, FPS - feeds the dashboard
and the thermal-throttle guard in the inference pipeline.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import psutil

from utils.logger import get_logger

logger = get_logger(__name__)

_THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")


@dataclass
class SystemSnapshot:
    cpu_percent: float
    ram_percent: float
    ram_used_mb: float
    temperature_c: float | None
    fps: float
    timestamp: float


class FpsTracker:
    """Rolling FPS counter over a fixed window of recent frame timestamps."""

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)
        self._lock = Lock()

    def tick(self) -> None:
        with self._lock:
            self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            span = self._timestamps[-1] - self._timestamps[0]
            if span <= 0:
                return 0.0
            return (len(self._timestamps) - 1) / span


class SystemMonitor:
    """Polls CPU/RAM/temperature on demand. Cheap enough to call every dashboard tick."""

    def __init__(self, fps_tracker: FpsTracker | None = None) -> None:
        self._fps_tracker = fps_tracker or FpsTracker()
        psutil.cpu_percent(interval=None)  # prime the non-blocking baseline

    @property
    def fps_tracker(self) -> FpsTracker:
        return self._fps_tracker

    def read_temperature_c(self) -> float | None:
        try:
            if _THERMAL_ZONE.exists():
                raw = _THERMAL_ZONE.read_text().strip()
                return int(raw) / 1000.0
        except (OSError, ValueError) as exc:
            logger.debug("Could not read thermal zone: %s", exc)
        return None

    def snapshot(self) -> SystemSnapshot:
        vm = psutil.virtual_memory()
        return SystemSnapshot(
            cpu_percent=psutil.cpu_percent(interval=None),
            ram_percent=vm.percent,
            ram_used_mb=vm.used / (1024 * 1024),
            temperature_c=self.read_temperature_c(),
            fps=self._fps_tracker.fps,
            timestamp=time.time(),
        )

    def is_thermal_throttling_risk(self, threshold_c: float) -> bool:
        temp = self.read_temperature_c()
        return temp is not None and temp >= threshold_c
