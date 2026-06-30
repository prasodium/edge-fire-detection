"""Maps a detection's bounding-box center to a calibrated venue zone
(stage / projector / podium / exit_door) using normalized polygons from
configs/decision.yaml. This is how "fire near projector", "fire on stage",
etc. are derived without training separate classes for scene location,
which is a property of the installation, not the flame's appearance.
"""
from __future__ import annotations

def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test. Implemented directly
    (no shapely/matplotlib) to avoid pulling a heavy geometry dependency
    into the hot inference path on a CPU/power-constrained device."""
    x, y = point
    inside = False
    n = len(polygon)
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2) and y1 != y2:
            x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            if x1 == x2 or x <= x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


class ZoneMapper:
    def __init__(self, zones_cfg: list[dict], frame_width: int, frame_height: int) -> None:
        self._frame_w = frame_width
        self._frame_h = frame_height
        self._zones: list[tuple[str, list[tuple[float, float]]]] = []
        for zone in zones_cfg:
            polygon = [(px * frame_width, py * frame_height) for px, py in zone["polygon"]]
            self._zones.append((zone["name"], polygon))

    def locate(self, box_xyxy: tuple[float, float, float, float]) -> list[str]:
        x1, y1, x2, y2 = box_xyxy
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return [name for name, polygon in self._zones if _point_in_polygon(center, polygon)]
