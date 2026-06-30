import numpy as np

from inference.detector import Detection, FireDetector


def _make_postprocessor() -> FireDetector:
    """Build a FireDetector instance without running __init__ (which requires
    a real ONNX file on disk) so _postprocess can be unit tested directly."""
    detector = object.__new__(FireDetector)
    detector._class_names = ["small_flame", "smoke"]
    detector._conf_threshold = 0.45
    detector._iou_threshold = 0.45
    detector._max_detections = 20
    return detector


def test_postprocess_returns_native_floats_not_numpy():
    """Regression test: box_xyxy used to carry numpy.float32 values straight
    out of unletterbox_box's arithmetic. That broke JSON serialization in
    dashboard/app.py's /api/status ("Unable to serialize unknown type:
    numpy.float32") the moment a real detection occurred - caught live while
    running the dashboard against an actual camera feed. _postprocess must
    cast every box coordinate to a native Python float so every downstream
    consumer (dashboard APIs, decision engine, recorder) gets plain floats."""
    detector = _make_postprocessor()

    # Single anchor: cx=100, cy=100, w=50, h=50, class0_score=0.9, class1_score=0.1
    raw = np.array([[[100.0], [100.0], [50.0], [50.0], [0.9], [0.1]]], dtype=np.float32)

    detections = detector._postprocess(raw, scale=1.0, pad=(0, 0))

    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "small_flame"
    for v in det.box_xyxy:
        assert type(v) is float, f"expected native float, got {type(v)}"


def test_detection_box_xyxy_native_float_passthrough():
    det = Detection(
        class_id=0, class_name="small_flame", confidence=0.9,
        box_xyxy=tuple(float(v) for v in np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
    )
    assert all(type(v) is float for v in det.box_xyxy)
