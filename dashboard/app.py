"""FastAPI dashboard: live MJPEG view with bounding boxes, detection history,
system telemetry (CPU/RAM/temp/FPS), and alarm acknowledge control.

Run with: uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from alarm.alarm_manager import AlarmManager
from dashboard.render import draw_detections
from inference.pipeline import FireDetectionPipeline
from storage.db import EventDatabase
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("dashboard")

app = FastAPI(title="Edge Fire & Smoke Detection Dashboard")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

_pipeline: Optional[FireDetectionPipeline] = None
_alarm_manager: Optional[AlarmManager] = None
_db: Optional[EventDatabase] = None


@app.on_event("startup")
def startup() -> None:
    global _pipeline, _alarm_manager, _db
    config = load_config()
    paths = config.system["paths"]
    root = config.project_root()

    _alarm_manager = AlarmManager(
        alarm_cfg=config.alarm,
        db_path=str(root / paths["database_path"]),
        snapshots_dir=str(root / paths["snapshots_dir"]),
        clips_dir=str(root / paths["clips_dir"]),
        camera_fps=config.camera.get("frame_rate", 15),
    )
    _db = EventDatabase(str(root / paths["database_path"]))
    _pipeline = FireDetectionPipeline(config, alarm_manager=_alarm_manager)
    _pipeline.start()
    logger.info("Dashboard startup complete - pipeline running")


@app.on_event("shutdown")
def shutdown() -> None:
    if _pipeline is not None:
        _pipeline.stop()
    if _alarm_manager is not None:
        _alarm_manager.shutdown()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


def _mjpeg_frame_generator():
    boundary = b"--frame"
    while True:
        if _pipeline is None or _pipeline.latest_frame is None:
            time.sleep(0.1)
            continue
        frame = draw_detections(_pipeline.latest_frame, _pipeline.latest_detections)
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        yield (
            boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )
        time.sleep(1 / 15)


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
def status() -> dict:
    if _pipeline is None:
        return {"running": False}
    snap = _pipeline.system_monitor.snapshot()
    return {
        "running": True,
        "cpu_percent": snap.cpu_percent,
        "ram_percent": snap.ram_percent,
        "ram_used_mb": snap.ram_used_mb,
        "temperature_c": snap.temperature_c,
        "fps": snap.fps,
        "alarm_active": _alarm_manager.is_active if _alarm_manager else False,
        "active_detections": [
            {"class_name": d.class_name, "confidence": d.confidence, "box": d.box_xyxy}
            for d in _pipeline.latest_detections
        ],
    }


@app.get("/api/events")
def events(limit: int = 100) -> list[dict]:
    if _db is None:
        return []
    return _db.recent_events(limit=limit)


@app.post("/api/alarm/acknowledge")
def acknowledge() -> dict:
    if _alarm_manager is not None:
        _alarm_manager.acknowledge_and_reset()
    return {"acknowledged": True}


@app.get("/snapshots/{filename}")
def get_snapshot(filename: str) -> FileResponse:
    config = load_config()
    path = config.project_root() / config.system["paths"]["snapshots_dir"] / filename
    return FileResponse(path)


@app.get("/clips/{filename}")
def get_clip(filename: str) -> FileResponse:
    config = load_config()
    path = config.project_root() / config.system["paths"]["clips_dir"] / filename
    return FileResponse(path)


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(status())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.debug("Dashboard websocket client disconnected")
