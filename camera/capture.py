"""Asynchronous camera capture for the Raspberry Pi Camera Module 3 NoIR Wide.

Runs in its own thread, pushing frames into a LatestFrameQueue so the
inference thread pool never blocks the camera and vice versa. Uses
Picamera2 on the Pi; transparently falls back to OpenCV VideoCapture on a
dev machine (laptop webcam / video file) so the rest of the pipeline can be
developed and tested off-Pi.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from utils.config import AppConfig
from utils.frame_queue import FramePacket, LatestFrameQueue
from utils.logger import get_logger
from utils.system_monitor import FpsTracker

logger = get_logger(__name__)

try:
    from picamera2 import Picamera2  # type: ignore

    _HAS_PICAMERA2 = True
except ImportError:  # dev machine, not a Pi
    _HAS_PICAMERA2 = False

import cv2


class CameraBackend:
    """Common interface implemented by the Picamera2 and OpenCV backends."""

    def start(self) -> None:
        raise NotImplementedError

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class Picamera2Backend(CameraBackend):
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._cam: "Picamera2 | None" = None

    def start(self) -> None:
        self._cam = Picamera2()
        w, h = self._cfg["resolution"]["capture"]
        video_config = self._cam.create_video_configuration(
            main={"size": (w, h), "format": self._cfg.get("format", "RGB888")},
            buffer_count=self._cfg.get("buffer_count", 4),
            controls={"FrameRate": self._cfg.get("frame_rate", 15)},
        )
        self._cam.configure(video_config)

        controls = {}
        af = self._cfg.get("autofocus", {})
        if af.get("enabled", True):
            controls["AfMode"] = 2 if af.get("mode") == "continuous" else 1  # 2=Continuous,1=Auto
        else:
            controls["AfMode"] = 0  # Manual

        exp = self._cfg.get("exposure", {})
        if exp.get("mode") == "manual":
            controls["AeEnable"] = False
            controls["ExposureTime"] = exp.get("exposure_time_us", 8000)
            controls["AnalogueGain"] = exp.get("analogue_gain", 2.0)
        else:
            controls["AeEnable"] = True

        wb = self._cfg.get("white_balance", {})
        if wb.get("mode") == "manual":
            controls["AwbEnable"] = False
            controls["ColourGains"] = tuple(wb.get("colour_gains", [1.8, 1.5]))
        else:
            controls["AwbEnable"] = True

        if self._cfg.get("hdr", {}).get("enabled"):
            controls["HdrMode"] = 1  # SingleExposure HDR on IMX708

        self._cam.set_controls(controls)
        self._cam.start()
        logger.info("Picamera2 started: %sx%s @ %sfps", w, h, self._cfg.get("frame_rate"))

    def read(self) -> Optional[np.ndarray]:
        if self._cam is None:
            return None
        return self._cam.capture_array("main")

    def stop(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
            self._cam = None


class OpenCvBackend(CameraBackend):
    """Fallback for development off a Raspberry Pi (webcam or video file)."""

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._cap: "cv2.VideoCapture | None" = None

    def start(self) -> None:
        index = self._cfg.get("device_index", 0)
        self._cap = cv2.VideoCapture(index)
        w, h = self._cfg["resolution"]["capture"]
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.get("frame_rate", 15))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open camera/video source: {index}")
        logger.info("OpenCV camera backend started on source %s", index)

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class AsyncCameraCapture:
    """Runs a CameraBackend on a dedicated thread and feeds a LatestFrameQueue.

    This decouples capture cadence from inference cadence (double buffering):
    if inference falls behind, the queue simply drops stale frames instead of
    backing up the camera driver.
    """

    def __init__(self, config: AppConfig, queue: LatestFrameQueue | None = None) -> None:
        self._cfg = config.camera
        self.queue = queue or LatestFrameQueue(maxsize=config.system.get("frame_queue_size", 4))
        self.fps_tracker = FpsTracker()
        self._backend = self._build_backend()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._frame_id = 0

    def _build_backend(self) -> CameraBackend:
        backend_name = self._cfg.get("backend", "picamera2")
        if backend_name == "picamera2" and _HAS_PICAMERA2:
            return Picamera2Backend(self._cfg)
        if backend_name == "picamera2" and not _HAS_PICAMERA2:
            logger.warning("picamera2 not available on this host - falling back to OpenCV backend")
        return OpenCvBackend(self._cfg)

    def start(self) -> None:
        self._backend.start()
        self._running.set()
        self._thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        consecutive_failures = 0
        while self._running.is_set():
            frame = self._backend.read()
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    logger.error("Camera read failed 30x consecutively - attempting backend restart")
                    self._restart_backend()
                    consecutive_failures = 0
                time.sleep(0.05)
                continue
            consecutive_failures = 0
            self._frame_id += 1
            self.queue.put(FramePacket(frame_id=self._frame_id, timestamp=time.time(), data=frame))
            self.fps_tracker.tick()

    def _restart_backend(self) -> None:
        try:
            self._backend.stop()
        except Exception:
            logger.exception("Error stopping camera backend during restart")
        time.sleep(1.0)
        try:
            self._backend.start()
        except Exception:
            logger.exception("Camera backend restart failed; will retry on next failure threshold")

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._backend.stop()
        self.queue.close()
        logger.info("Camera capture stopped")
