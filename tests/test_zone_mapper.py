from inference.zone_mapper import ZoneMapper

ZONES = [
    {"name": "stage", "polygon": [[0.1, 0.3], [0.9, 0.3], [0.9, 0.95], [0.1, 0.95]]},
    {"name": "exit_door", "polygon": [[0.85, 0.1], [1.0, 0.1], [1.0, 1.0], [0.85, 1.0]]},
]


def test_box_center_inside_stage_zone():
    mapper = ZoneMapper(ZONES, frame_width=1000, frame_height=1000)
    zones = mapper.locate((400, 400, 500, 500))  # center (450, 450) -> inside stage
    assert "stage" in zones


def test_box_outside_all_zones():
    mapper = ZoneMapper(ZONES, frame_width=1000, frame_height=1000)
    zones = mapper.locate((0, 0, 10, 10))  # top-left corner, outside both zones
    assert zones == []


def test_box_can_match_multiple_overlapping_zones():
    overlapping = ZONES + [{"name": "stage_right_edge", "polygon": [[0.8, 0.2], [1.0, 0.2], [1.0, 1.0], [0.8, 1.0]]}]
    mapper = ZoneMapper(overlapping, frame_width=1000, frame_height=1000)
    zones = mapper.locate((870, 400, 900, 450))  # inside both stage and exit_door/stage_right_edge
    assert "stage" in zones
    assert "stage_right_edge" in zones
