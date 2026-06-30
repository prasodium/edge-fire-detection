from inference.detector import Detection
from inference.tracker import IoUTracker


def det(class_name="small_flame", box=(10, 10, 50, 50), conf=0.9):
    return Detection(class_id=0, class_name=class_name, confidence=conf, box_xyxy=box)


def test_same_region_across_frames_is_one_track():
    tracker = IoUTracker(iou_match_threshold=0.3, max_missed_frames=2)
    tracker.update([det(box=(10, 10, 50, 50))])
    tracker.update([det(box=(12, 11, 52, 51))])
    tracks = tracker.update([det(box=(11, 12, 51, 52))])

    assert len(tracks) == 1
    assert tracks[0].consecutive_hits == 3


def test_far_apart_boxes_are_different_tracks():
    tracker = IoUTracker(iou_match_threshold=0.3)
    tracker.update([det(box=(0, 0, 20, 20))])
    tracks = tracker.update([det(box=(500, 500, 540, 540))])
    assert len(tracker.tracks) == 2


def test_track_dropped_after_max_missed_frames():
    tracker = IoUTracker(iou_match_threshold=0.3, max_missed_frames=1)
    tracker.update([det(box=(10, 10, 50, 50))])
    tracker.update([])  # missed frame 1 - tolerated
    assert len(tracker.tracks) == 1
    tracker.update([])  # missed frame 2 - exceeds max_missed_frames
    assert len(tracker.tracks) == 0


def test_consecutive_hits_resets_on_miss():
    tracker = IoUTracker(iou_match_threshold=0.3, max_missed_frames=2)
    tracker.update([det(box=(10, 10, 50, 50))])
    tracker.update([det(box=(10, 10, 50, 50))])
    tracker.update([])  # miss
    tracks = tracker.update([det(box=(10, 10, 50, 50))])
    assert tracks[0].consecutive_hits == 1
