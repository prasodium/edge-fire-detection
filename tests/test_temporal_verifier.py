from inference.temporal_verifier import TemporalVerifier
from inference.tracker import Track


def make_track(confidences):
    track = Track(
        track_id=1, class_name="small_flame", box_xyxy=(0, 0, 10, 10), confidence=confidences[-1],
        confidence_history=list(confidences), consecutive_hits=len(confidences),
    )
    return track


def test_single_frame_never_triggers():
    verifier = TemporalVerifier(consecutive_frames_required=8, min_average_confidence=0.85)
    track = make_track([0.99])
    result = verifier.verify(track)
    assert result.verified is False


def test_requires_both_frame_count_and_confidence():
    verifier = TemporalVerifier(consecutive_frames_required=8, min_average_confidence=0.85)

    enough_frames_low_conf = make_track([0.5] * 8)
    assert verifier.verify(enough_frames_low_conf).verified is False

    enough_conf_too_few_frames = make_track([0.99] * 3)
    assert verifier.verify(enough_conf_too_few_frames).verified is False

    confirmed = make_track([0.9] * 8)
    assert verifier.verify(confirmed).verified is True


def test_rolling_average_uses_window():
    verifier = TemporalVerifier(consecutive_frames_required=8, rolling_window_size=4, min_average_confidence=0.85)
    # Old low-confidence frames should fall out of the rolling window.
    track = make_track([0.1, 0.1, 0.1, 0.1, 0.95, 0.95, 0.95, 0.95])
    result = verifier.verify(track)
    assert result.rolling_avg_confidence == 0.95
    assert result.verified is True
