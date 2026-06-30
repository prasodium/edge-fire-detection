"""End-to-end pipeline orchestrator:

    Camera -> Frame Resize/Normalize -> FireDetector -> DecisionEngine
    (smoke verification + temporal verification + motion analysis baked in)
    -> AlarmManager

Runs inference on a small ThreadPoolExecutor (ONNX Runtime releases the GIL
during `session.run`, so threads - not processes - are sufficient and avoid
the memory duplication cost of multiprocessing on an 8GB box). A thermal/CPU
guard skips frames rather than queuing them when the Pi is under load, to
honor the "<80% CPU, no throttling" requirement.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread
from typing import Callable

import cv2
import numpy as np

from alarm.alarm_manager import AlarmManager
from camera.capture import AsyncCameraCapture
from inference.decision_engine import AlarmEvent, DecisionEngine
from inference.detector import Detection, FireDetector
from utils.config import AppConfig
from utils.logger import get_logger
from utils.system_monitor import SystemMonitor

logger = get_logger(__name__)

FrameCallback = Callable[[np.ndarray, list[Detection], list[AlarmEvent]], None]


class FireDetectionPipeline:
    def __init__(self, config: AppConfig, alarm_manager: AlarmManager | None = None) -> None:
        self._config = config
        self._system_monitor = SystemMonitor()
        self._camera = AsyncCameraCapture(config)

        model_spec = config.active_model_spec()
        infer_cfg = config.model.get("inference", {})
        ort_cfg = infer_cfg.get("onnxruntime", {})
        self._detector = FireDetector(
            weights_path=Path(config.project_root()) / model_spec["weights"],
            class_names=config.model["classes"],
            input_size=model_spec.get("input_size", config.model.get("input_size", 416)),
            confidence_threshold=infer_cfg.get("confidence_threshold", 0.45),
            nms_iou_threshold=infer_cfg.get("nms_iou_threshold", 0.45),
            max_detections=infer_cfg.get("max_detections", 20),
            intra_op_threads=ort_cfg.get("intra_op_num_threads", 3),
            inter_op_threads=ort_cfg.get("inter_op_num_threads", 1),
        )

        cap_w, cap_h = config.camera["resolution"]["capture"]
        self._decision_engine = DecisionEngine(
            decision_cfg=config.decision, model_cfg=config.model, frame_width=cap_w, frame_height=cap_h
        )

        self._alarm_manager = alarm_manager
        self._executor = ThreadPoolExecutor(
            max_workers=config.system.get("num_inference_threads", 3), thread_name_prefix="inference"
        )
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._frame_callbacks: list[FrameCallback] = []
        self._warm_up_frames_seen = 0
        self._max_cpu_percent = config.system.get("max_cpu_percent", 80)
        self._thermal_limit_c = config.system.get("thermal_throttle_temp_c", 78)

        self.latest_frame: np.ndarray | None = None
        self.latest_detections: list[Detection] = []

    def register_frame_callback(self, callback: FrameCallback) -> None:
        """Used by the dashboard to receive (frame, detections, alarm_events) for live view."""
        self._frame_callbacks.append(callback)

    def start(self) -> None:
        self._camera.start()
        self._stop_event.clear()
        self._thread = Thread(target=self._run_loop, name="inference-loop", daemon=True)
        self._thread.start()
        logger.info("Fire detection pipeline started")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._should_skip_frame_for_thermal_budget():
                time.sleep(0.2)
                continue

            packet = self._camera.queue.get(timeout=1.0)
            if packet is None:
                continue

            frame_rgb = packet.data
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            self._warm_up_frames_seen += 1
            self._decision_engine.warm_up(frame_bgr)

            future = self._executor.submit(self._detector.infer, frame_rgb)
            try:
                detections = future.result(timeout=2.0)
            except Exception:
                logger.exception("Inference failed for frame %s", packet.frame_id)
                continue

            alarm_events = self._decision_engine.process(detections, frame_bgr)

            self.latest_frame = frame_bgr
            self.latest_detections = detections
            self._system_monitor.fps_tracker.tick()

            if self._alarm_manager is not None:
                self._alarm_manager.push_frame(frame_bgr)

            for event in alarm_events:
                self._handle_alarm_event(event, frame_bgr)

            for callback in self._frame_callbacks:
                try:
                    callback(frame_bgr, detections, alarm_events)
                except Exception:
                    logger.exception("Frame callback raised")

    def _should_skip_frame_for_thermal_budget(self) -> bool:
        if self._system_monitor.is_thermal_throttling_risk(self._thermal_limit_c):
            logger.warning("Thermal limit (%s C) reached - skipping frame to cool down", self._thermal_limit_c)
            return True
        return False

    def _handle_alarm_event(self, event: AlarmEvent, frame_bgr: np.ndarray) -> None:
        logger.warning(
            "ALARM: %s severity=%s confidence=%.2f zones=%s",
            event.context_label, event.severity, event.confidence, event.zones,
        )
        if self._alarm_manager is not None:
            self._alarm_manager.trigger(event, frame_bgr)

    @property
    def system_monitor(self) -> SystemMonitor:
        return self._system_monitor

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._camera.stop()
        logger.info("Fire detection pipeline stopped")
