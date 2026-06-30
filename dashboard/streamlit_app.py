"""Streamlit live dashboard - shows the live camera feed with bounding boxes,
active detections, system telemetry (CPU/RAM/temperature/FPS), alarm state,
and detection history, all backed by the same FireDetectionPipeline /
AlarmManager used by the FastAPI dashboard (dashboard/app.py) - no detection
or alarm logic is duplicated here, this file is presentation only.

Run with: streamlit run dashboard/streamlit_app.py

NOTE on "live": Streamlit reruns its script on a timer (see `REFRESH_SECONDS`
below), it does not push frames like the FastAPI dashboard's MJPEG stream.
At a 1s refresh this is a near-real-time still-frame view, not smooth video -
sufficient for monitoring, not for frame-by-frame review (use the saved
clips in Detection History for that). Run only ONE dashboard process in
production (this OR dashboard/app.py, not both) - each one's own web server
and frame polling adds CPU overhead on top of the inference budget; see
docs/optimization_guide.md.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from alarm.alarm_manager import AlarmManager
from dashboard.render import draw_detections
from inference.pipeline import FireDetectionPipeline
from storage.db import EventDatabase
from utils.config import load_config

st.set_page_config(page_title="Edge Fire & Smoke Detection", layout="wide", page_icon="🔥")

REFRESH_SECONDS = 1.0


@st.cache_resource(show_spinner="Starting camera and inference pipeline...")
def get_system() -> tuple[FireDetectionPipeline, AlarmManager, EventDatabase]:
    """Created once per Streamlit server process and reused across reruns -
    without st.cache_resource, every script rerun (every REFRESH_SECONDS)
    would otherwise spin up a brand new camera + ONNX session."""
    config = load_config()
    paths = config.system["paths"]
    root = config.project_root()

    alarm_manager = AlarmManager(
        alarm_cfg=config.alarm,
        db_path=str(root / paths["database_path"]),
        snapshots_dir=str(root / paths["snapshots_dir"]),
        clips_dir=str(root / paths["clips_dir"]),
        camera_fps=config.camera.get("frame_rate", 15),
    )
    pipeline = FireDetectionPipeline(config, alarm_manager=alarm_manager)
    pipeline.start()
    db = EventDatabase(str(root / paths["database_path"]))
    return pipeline, alarm_manager, db


pipeline, alarm_manager, db = get_system()

st.title("🔥 Edge Fire & Smoke Detection — Live Dashboard")

if alarm_manager.is_active:
    st.error("ALARM ACTIVE — fire/smoke confirmed. Verify the snapshot/clip below before resetting.", icon="🚨")
    if st.button("Acknowledge & Reset Alarm", type="primary"):
        alarm_manager.acknowledge_and_reset()
        st.rerun()


@st.fragment(run_every=REFRESH_SECONDS)
def live_panel() -> None:
    col_video, col_telemetry = st.columns([2, 1])

    with col_video:
        st.subheader("Live Camera Feed")
        if pipeline.latest_frame is not None:
            annotated = draw_detections(pipeline.latest_frame, pipeline.latest_detections)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.info("Waiting for the first camera frame...")

        st.subheader("Active Detections")
        if pipeline.latest_detections:
            detections_df = pd.DataFrame(
                [
                    {
                        "Class": d.class_name,
                        "Confidence": f"{d.confidence:.0%}",
                        "Box (x1,y1,x2,y2)": ", ".join(str(round(v)) for v in d.box_xyxy),
                    }
                    for d in pipeline.latest_detections
                ]
            )
            st.dataframe(detections_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No active detections this frame.")

    with col_telemetry:
        st.subheader("System Telemetry")
        snap = pipeline.system_monitor.snapshot()

        st.metric("CPU Load", f"{snap.cpu_percent:.1f}%")
        st.progress(min(snap.cpu_percent / 100, 1.0))

        st.metric("RAM Used", f"{snap.ram_used_mb:.0f} MB", delta=f"{snap.ram_percent:.1f}% of total")
        st.progress(min(snap.ram_percent / 100, 1.0))

        if snap.temperature_c is not None:
            st.metric("CPU Temperature", f"{snap.temperature_c:.1f} °C")
        else:
            st.metric("CPU Temperature", "N/A (not on Pi)")

        st.metric("Inference FPS", f"{snap.fps:.1f}")
        st.metric("Alarm State", "🚨 ACTIVE" if alarm_manager.is_active else "✅ Normal")


live_panel()

st.divider()
st.subheader("Detection History")

events = db.recent_events(limit=100)
if events:
    history_df = pd.DataFrame(events)
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], unit="s")
    display_cols = ["timestamp", "class_name", "severity", "confidence", "zones", "acknowledged"]
    st.dataframe(history_df[display_cols], use_container_width=True, hide_index=True)

    latest = events[0]
    if latest.get("snapshot_path") and Path(latest["snapshot_path"]).exists():
        with st.expander(f"Latest alarm snapshot (event #{latest['id']})"):
            st.image(latest["snapshot_path"])
else:
    st.caption("No events logged yet.")
