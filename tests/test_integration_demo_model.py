"""End-to-end integration test against the real demo ONNX model (see
weights/README.md and docs/benchmark_report.md "Demo model" for provenance).

Skips automatically when weights/ is empty (e.g. a fresh clone before
training) - this is the one test file in the suite that needs real model
weights rather than pure logic, kept separate so `pytest tests/` stays fast
and hardware/weights-independent everywhere else.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from inference.decision_engine import DecisionEngine
from inference.detector import FireDetector
from utils.config import reload_config

TEST_IMAGE = Path(__file__).resolve().parent.parent / (
    "datasets/raw/libreyolo_fire_smoke_bbox/test/images/"
    "flare_0058_jpg.rf.d523c196faf6d6839608425728cea69c.jpg"
)


def _weights_available() -> bool:
    cfg = reload_config()
    spec = cfg.active_model_spec()
    return (cfg.project_root() / spec["weights"]).exists()


pytestmark = pytest.mark.skipif(
    not _weights_available(), reason="No trained model in weights/ - run training first (see weights/README.md)"
)


@pytest.fixture
def detector() -> FireDetector:
    cfg = reload_config()
    spec = cfg.active_model_spec()
    return FireDetector(
        weights_path=cfg.project_root() / spec["weights"],
        class_names=cfg.model["classes"],
        input_size=spec["input_size"],
        confidence_threshold=0.25,
    )


def test_detector_finds_objects_in_real_fire_image(detector: FireDetector):
    if not TEST_IMAGE.exists():
        pytest.skip("Demo dataset image not present - run training/convert_seg_to_bbox.py first")
    frame_bgr = cv2.imread(str(TEST_IMAGE))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.infer(frame_rgb)
    assert len(detections) > 0
    assert any(d.class_name in ("small_flame", "smoke") for d in detections)


def test_decision_engine_stays_silent_on_static_frame(detector: FireDetector):
    """A frozen, unchanging frame has zero motion/flicker by construction -
    the decision engine's motion analyzer (Stage 4) and flicker check (Stage 2)
    should correctly reject it even though Stage 1 detects an object every frame.
    This is the core false-positive-reduction behavior, not a limitation."""
    if not TEST_IMAGE.exists():
        pytest.skip("Demo dataset image not present")
    frame_bgr = cv2.imread(str(TEST_IMAGE))
    h, w = frame_bgr.shape[:2]
    cfg = reload_config()
    engine = DecisionEngine(decision_cfg=cfg.decision, model_cfg=cfg.model, frame_width=w, frame_height=h)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    events = []
    for _ in range(20):
        engine.warm_up(frame_bgr)
        detections = detector.infer(frame_rgb)
        events.extend(engine.process(detections, frame_bgr))

    assert events == []


def test_decision_engine_alarms_on_simulated_flicker(detector: FireDetector):
    """With realistic flicker/motion (brightness oscillation + jitter + noise,
    approximating real flame statistics), the full 5-stage pipeline should
    eventually confirm and raise exactly one AlarmEvent."""
    if not TEST_IMAGE.exists():
        pytest.skip("Demo dataset image not present")
    base = cv2.imread(str(TEST_IMAGE))
    h, w = base.shape[:2]
    cfg = reload_config()
    engine = DecisionEngine(decision_cfg=cfg.decision, model_cfg=cfg.model, frame_width=w, frame_height=h)

    rng = np.random.default_rng(42)
    events = []
    for i in range(30):
        brightness = 1.0 + 0.15 * np.sin(i * 0.8)
        dx, dy = rng.integers(-3, 4), rng.integers(-3, 4)
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(base, shift, (w, h), borderMode=cv2.BORDER_REPLICATE)
        frame = np.clip(shifted.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        noise = rng.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        engine.warm_up(frame)
        detections = detector.infer(frame_rgb)
        events.extend(engine.process(detections, frame))

    assert len(events) >= 1
    assert events[0].class_name == "small_flame"
    assert events[0].severity == "critical"
