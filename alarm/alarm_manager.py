"""Stage 5 consumer: turns a confirmed AlarmEvent into physical + digital
actions - GPIO siren/relay/LED, snapshot, video clip, DB log, notifications.

Owns the auto-silence timer and the "require manual reset" acknowledgement
flow described in configs/alarm.yaml.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from alarm.notifier import NotificationDispatcher
from alarm.recorder import ClipRecorder, SnapshotSaver
from gpio.controller import GpioController
from inference.decision_engine import AlarmEvent
from storage.db import EventDatabase, EventRecord
from utils.logger import get_logger

logger = get_logger("alarm.manager")


class AlarmManager:
    def __init__(
        self,
        alarm_cfg: dict,
        db_path: str,
        snapshots_dir: str,
        clips_dir: str,
        camera_fps: int = 15,
    ) -> None:
        """`alarm_cfg` is the full parsed configs/alarm.yaml document, i.e. a
        dict with sibling top-level keys: gpio, alarm, recording, notifications, database."""
        self._cfg = alarm_cfg.get("alarm", {})
        self._gpio = GpioController(alarm_cfg["gpio"])
        self._db = EventDatabase(db_path)
        recording_cfg = alarm_cfg.get("recording", {})
        self._recording_cfg = recording_cfg
        self._snapshot_saver = SnapshotSaver(snapshots_dir, recording_cfg.get("snapshot_format", "jpg"))
        self._clip_recorder = ClipRecorder(recording_cfg, clips_dir, fps=camera_fps)
        self._notifier = NotificationDispatcher(alarm_cfg.get("notifications", {}))

        self._active = False
        self._active_event_id: int | None = None
        self._silence_timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        """Must be called every frame (independent of alarm state) so the
        clip recorder always has a pre-event ring buffer ready."""
        self._clip_recorder.push_frame(frame_bgr)

    def trigger(self, event: AlarmEvent, frame_bgr: np.ndarray) -> None:
        record = EventRecord(
            timestamp=event.timestamp,
            class_name=event.class_name,
            severity=event.severity,
            confidence=event.confidence,
            zones=",".join(event.zones),
            box_x1=event.box_xyxy[0], box_y1=event.box_xyxy[1],
            box_x2=event.box_xyxy[2], box_y2=event.box_xyxy[3],
        )
        event_id = self._db.insert_event(record)

        snapshot_path = None
        clip_path = None
        if self._recording_cfg.get("save_snapshot_on_alarm", True):
            snapshot_path = self._snapshot_saver.save(frame_bgr, event_id)
        if self._recording_cfg.get("save_clip_on_alarm", True):
            clip_path = self._clip_recorder.start_clip(event_id)

        payload = {
            "event_id": event_id,
            "class_name": event.class_name,
            "context_label": event.context_label,
            "severity": event.severity,
            "confidence": event.confidence,
            "zones": event.zones,
            "timestamp": event.timestamp,
            "snapshot_path": str(snapshot_path) if snapshot_path else None,
            "clip_path": str(clip_path) if clip_path else None,
        }
        self._notifier.notify(payload)

        if event.severity == "critical":
            self._activate_physical_alarm(event_id)
        else:
            logger.info("Sub-critical alarm logged without physical siren: %s", payload)

    def _activate_physical_alarm(self, event_id: int) -> None:
        with self._lock:
            self._active = True
            self._active_event_id = event_id
            self._gpio.activate_alarm(blink_hz=self._cfg.get("led_blink_hz", 4))

            auto_silence_s = self._cfg.get("auto_silence_after_s", 300)
            if self._silence_timer is not None:
                self._silence_timer.cancel()
            self._silence_timer = threading.Timer(auto_silence_s, self._auto_silence)
            self._silence_timer.daemon = True
            self._silence_timer.start()

    def _auto_silence(self) -> None:
        logger.warning("Auto-silencing siren after timeout (event_id=%s) - alarm remains logged as active until acknowledged", self._active_event_id)
        with self._lock:
            self._gpio.silence()

    def acknowledge_and_reset(self) -> None:
        """Called from the dashboard's 'Acknowledge & Reset' control."""
        with self._lock:
            if self._active_event_id is not None:
                self._db.acknowledge(self._active_event_id)
            if self._silence_timer is not None:
                self._silence_timer.cancel()
            self._gpio.silence()
            self._active = False
            self._active_event_id = None
        logger.info("Alarm acknowledged and reset by operator")

    @property
    def is_active(self) -> bool:
        return self._active

    def shutdown(self) -> None:
        if self._silence_timer is not None:
            self._silence_timer.cancel()
        self._gpio.close()
