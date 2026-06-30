from utils.frame_queue import FramePacket, LatestFrameQueue


def test_put_get_round_trip():
    q = LatestFrameQueue(maxsize=4)
    q.put(FramePacket(frame_id=1, timestamp=0.0, data="frame-1"))
    packet = q.get(timeout=0.1)
    assert packet is not None
    assert packet.data == "frame-1"


def test_queue_drops_oldest_when_full():
    q = LatestFrameQueue(maxsize=2)
    for i in range(5):
        q.put(FramePacket(frame_id=i, timestamp=float(i), data=f"frame-{i}"))
    assert len(q) == 2
    first = q.get(timeout=0.1)
    assert first.frame_id == 3  # frames 0,1,2 evicted, only 3 and 4 remain


def test_get_latest_and_clear_drops_backlog():
    q = LatestFrameQueue(maxsize=4)
    for i in range(4):
        q.put(FramePacket(frame_id=i, timestamp=float(i), data=f"frame-{i}"))
    latest = q.get_latest_and_clear()
    assert latest.frame_id == 3
    assert len(q) == 0


def test_get_returns_none_on_timeout_when_empty():
    q = LatestFrameQueue(maxsize=2)
    assert q.get(timeout=0.05) is None
