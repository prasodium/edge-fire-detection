import numpy as np

from inference.false_positive_filter import FalsePositiveFilter
from inference.tracker import Track

CFG = {
    "color_consistency": {"enabled": True, "flame_hue_range_deg": [0, 45], "min_saturation": 0.4, "min_value": 0.5},
    "flicker_analysis": {"enabled": True, "min_frequency_hz": 1.0, "max_frequency_hz": 6.0, "min_brightness_variance": 8.0},
    "bounding_box_stability": {"enabled": True, "max_center_jitter_px": 40, "iou_match_threshold": 0.3},
    "known_light_source_suppression": {"enabled": False},  # disabled: needs many warm-up frames
}


def _orange_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = (0, 120, 255)  # BGR orange/flame-like color, high saturation+value
    return frame


def _blue_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)  # BGR blue - outside the flame hue band
    return frame


def test_orange_region_passes_color_check():
    filt = FalsePositiveFilter(CFG)
    track = Track(track_id=1, class_name="small_flame", box_xyxy=(10, 10, 90, 90), confidence=0.9)
    result = filt.evaluate(track, _orange_frame())
    assert result.checks["color_consistency"] is True


def test_blue_region_fails_color_check():
    filt = FalsePositiveFilter(CFG)
    track = Track(track_id=1, class_name="small_flame", box_xyxy=(10, 10, 90, 90), confidence=0.9)
    result = filt.evaluate(track, _blue_frame())
    assert result.checks["color_consistency"] is False
    assert result.passed is False


def test_bbox_jitter_too_large_fails_stability():
    filt = FalsePositiveFilter(CFG)
    track = Track(
        track_id=1, class_name="small_flame", box_xyxy=(200, 200, 240, 240), confidence=0.9,
        box_history=[(0, 0, 40, 40), (100, 100, 140, 140), (200, 200, 240, 240)],
    )
    result = filt.evaluate(track, _orange_frame())
    assert result.checks["bbox_stability"] is False
