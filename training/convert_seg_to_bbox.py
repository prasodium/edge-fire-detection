"""One-off converter: YOLO-segmentation polygon labels -> YOLO-detection
bounding-box labels.

The LibreYOLO/fire-smoke-seg source dataset (Roboflow "fire-and-smoke-
segmentation" project, CC BY 4.0) ships polygon segmentation labels:
    <class_id> x1 y1 x2 y2 x3 y3 ... xn yn   (all normalized 0-1)

This project's decision engine and FireDetector work on axis-aligned
bounding boxes, not polygons, so each polygon is converted to its
axis-aligned bounding box (min/max of the polygon's x/y coordinates):
    <class_id> cx cy w h                      (YOLO detection format)

Usage:
    python training/convert_seg_to_bbox.py \\
        --src datasets/raw/libreyolo_fire_smoke_seg \\
        --dst datasets/raw/libreyolo_fire_smoke_bbox
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def polygon_to_bbox(values: list[float]) -> tuple[float, float, float, float]:
    xs = values[0::2]
    ys = values[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def convert_label_file(src_path: Path, dst_path: Path) -> int:
    lines_out = []
    for line in src_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:  # need class_id + at least 3 (x,y) pairs to be a polygon
            continue
        class_id = parts[0]
        coords = [float(v) for v in parts[1:]]
        cx, cy, w, h = polygon_to_bbox(coords)
        lines_out.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    dst_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return len(lines_out)


def convert_split(src_split_dir: Path, dst_split_dir: Path) -> tuple[int, int]:
    images_out = dst_split_dir / "images"
    labels_out = dst_split_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    n_images, n_boxes = 0, 0
    for img_path in (src_split_dir / "images").glob("*"):
        shutil.copy2(img_path, images_out / img_path.name)
        n_images += 1

        label_path = (src_split_dir / "labels" / f"{img_path.stem}.txt")
        dst_label_path = labels_out / f"{img_path.stem}.txt"
        if label_path.exists():
            n_boxes += convert_label_file(label_path, dst_label_path)
        else:
            dst_label_path.write_text("")  # background image, no objects
    return n_images, n_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    for split in ("train", "valid", "test"):
        split_dir = src / split
        if not split_dir.exists():
            print(f"skip {split}: not found at {split_dir}")
            continue
        n_images, n_boxes = convert_split(split_dir, dst / split)
        print(f"{split}: {n_images} images, {n_boxes} bounding boxes written -> {dst / split}")


if __name__ == "__main__":
    main()
