"""Headless entrypoint: runs the detection + alarm pipeline with no dashboard
process. Useful for systemd deployment when the dashboard runs as a separate
service (see scripts/fire-detection.service + scripts/fire-detection-dashboard.service),
or for on-device debugging.

Usage: python scripts/run_headless.py
"""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alarm.alarm_manager import AlarmManager
from inference.pipeline import FireDetectionPipeline
from utils.config import load_config
from utils.logger import configure_logging, get_logger


def main() -> None:
    config = load_config()
    configure_logging(config.system.get("log_dir", "logs"), config.system.get("log_level", "INFO"))
    logger = get_logger("scripts.run_headless")

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

    stop = {"flag": False}

    def _handle_signal(signum, frame) -> None:
        logger.info("Received signal %s - shutting down", signum)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    pipeline.start()
    logger.info("Pipeline running headless. Ctrl+C to stop.")
    try:
        while not stop["flag"]:
            time.sleep(1.0)
    finally:
        pipeline.stop()
        alarm_manager.shutdown()


if __name__ == "__main__":
    main()
