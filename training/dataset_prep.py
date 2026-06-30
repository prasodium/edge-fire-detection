"""Dataset preparation: normalizes heterogeneous source datasets (see
configs/dataset.yaml) into a single YOLO-format directory tree and emits the
data.yaml that Ultralytics training consumes.

This script defines the *pipeline*; it does not embed copies of third-party
datasets. Each `--source` fetch step shells out to the appropriate
public-dataset tooling (kaggle CLI, OIDv6 toolkit, direct HTTP download) -
run `python training/dataset_prep.py --list` to see what's registered, and
verify each dataset's license before use (see configs/dataset.yaml notes).

Usage:
    python training/dataset_prep.py --source custom_indoor_fire --normalize
    python training/dataset_prep.py --merge-all --output datasets/processed
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path("datasets/raw")
PROCESSED_DIR = Path("datasets/processed")


def list_sources() -> None:
    config = load_config()
    for key, spec in config.dataset["sources"].items():
        print(f"{key:28s} type={spec['type']:24s} license={spec.get('license', '?')}")


def normalize_source(source_key: str) -> Path:
    """Convert one registered source into YOLO-format (images/ + labels/)
    under datasets/raw/<source_key>_yolo/. Classification-only datasets
    (fire/no_fire whole-image labels) are converted into full-frame boxes
    as a weak-label fallback - flag these for manual box refinement before
    using them as primary training signal.
    """
    config = load_config()
    spec = config.dataset["sources"].get(source_key)
    if spec is None:
        raise KeyError(f"Unknown dataset source: {source_key}")

    src_dir = RAW_DIR / source_key
    out_dir = RAW_DIR / f"{source_key}_yolo"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    logger.info("Normalizing %s (%s) -> %s", source_key, spec["type"], out_dir)
    if not src_dir.exists():
        logger.warning(
            "%s not downloaded yet at %s - fetch it per configs/dataset.yaml url/license, "
            "then re-run this command.", source_key, src_dir,
        )
        return out_dir

    if spec["type"] == "classification":
        _normalize_classification_as_weak_boxes(src_dir, out_dir, spec["classes"])
    elif spec["type"] in ("detection", "classification+segmentation"):
        _normalize_detection(src_dir, out_dir)
    else:
        raise ValueError(f"Unsupported dataset type: {spec['type']}")

    return out_dir


def _normalize_classification_as_weak_boxes(src_dir: Path, out_dir: Path, classes: list[str]) -> None:
    """Whole-image label -> single full-frame YOLO box `0 0.5 0.5 1.0 1.0`.
    Weak supervision only: mix with properly-boxed detection datasets, don't
    train solely on this signal or the model learns to ignore localization."""
    positive_class_names = {"fire", "smoke"}
    for class_dir in src_dir.iterdir():
        if not class_dir.is_dir():
            continue
        is_positive = class_dir.name.lower() in positive_class_names
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            dest_img = out_dir / "images" / f"{class_dir.name}_{img_path.name}"
            shutil.copy2(img_path, dest_img)
            label_path = out_dir / "labels" / f"{dest_img.stem}.txt"
            if is_positive:
                label_path.write_text("0 0.5 0.5 1.0 1.0\n")
            else:
                label_path.write_text("")  # background image, no objects


def _normalize_detection(src_dir: Path, out_dir: Path) -> None:
    """Pass-through copy for sources already shipping YOLO-style images/labels
    pairs, or annotated via a converter run upstream (e.g. OIDv6 toolkit
    output, COCO-to-YOLO conversion). Extend this per-source as datasets are
    actually acquired."""
    images_src = src_dir / "images"
    labels_src = src_dir / "labels"
    if not images_src.exists():
        logger.warning("Expected %s - skipping, run the dataset's own conversion tool first", images_src)
        return
    for img_path in images_src.glob("*"):
        shutil.copy2(img_path, out_dir / "images" / img_path.name)
    if labels_src.exists():
        for label_path in labels_src.glob("*.txt"):
            shutil.copy2(label_path, out_dir / "labels" / label_path.name)


def merge_and_split(output_dir: Path = PROCESSED_DIR, seed: int = 42) -> None:
    config = load_config()
    split_cfg = config.dataset["split"]
    classes = config.model["classes"]

    all_pairs: list[tuple[Path, Path]] = []
    for normalized_dir in RAW_DIR.glob("*_yolo"):
        images_dir = normalized_dir / "images"
        labels_dir = normalized_dir / "labels"
        if not images_dir.exists():
            continue
        for img_path in images_dir.glob("*"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                all_pairs.append((img_path, label_path))

    if not all_pairs:
        logger.warning(
            "No normalized (*_yolo) datasets found under %s - run --source <key> --normalize "
            "for each dataset first.", RAW_DIR,
        )

    random.Random(seed).shuffle(all_pairs)
    n = len(all_pairs)
    n_train = int(n * split_cfg["train"])
    n_val = int(n * split_cfg["val"])
    splits = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train : n_train + n_val],
        "test": all_pairs[n_train + n_val :],
    }

    for split_name, pairs in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, label_path in pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(label_path, lbl_out / f"{img_path.stem}.txt")
        logger.info("%s: %d images", split_name, len(pairs))

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(classes)},
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    logger.info("Wrote %s", output_dir / "data.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List registered dataset sources")
    parser.add_argument("--source", type=str, help="Normalize a single registered source")
    parser.add_argument("--merge-all", action="store_true", help="Merge all normalized sources and split")
    parser.add_argument("--output", type=str, default=str(PROCESSED_DIR))
    args = parser.parse_args()

    if args.list:
        list_sources()
        return
    if args.source:
        normalize_source(args.source)
        return
    if args.merge_all:
        merge_and_split(Path(args.output))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
