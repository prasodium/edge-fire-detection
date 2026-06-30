"""Benchmark every exported model variant (FP32/FP16/INT8 x 320/416/640) on
the CPU it's run on, producing the latency/FPS/CPU/memory comparison table
required by the spec. Run this *on the target Raspberry Pi 5* for numbers
that are meaningful for deployment decisions - numbers collected on a dev
laptop are useful for relative comparison only, not absolute SLA validation.

Usage:
    python training/benchmark.py --weights-dir weights/ --images-dir datasets/processed/val/images \\
        --runs 100 --out docs/benchmark_results.csv
"""
from __future__ import annotations

import argparse
import csv
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil

from inference.preprocess import letterbox_resize, to_model_input
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    model_file: str
    precision: str
    input_size: int
    mean_latency_ms: float
    p95_latency_ms: float
    fps: float
    cpu_percent_during: float
    peak_rss_mb: float
    model_size_mb: float
    host_cpu: str = platform.processor() or platform.machine()


def _load_sample_images(images_dir: Path, n: int = 20) -> list[np.ndarray]:
    paths = list(Path(images_dir).glob("*.jpg"))[:n] + list(Path(images_dir).glob("*.png"))[:n]
    images = []
    for p in paths[:n]:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not images:
        # No dataset on hand yet - synthesize neutral frames so the benchmark
        # harness itself can still be exercised/validated end to end.
        logger.warning("No images found in %s - using synthetic frames for a smoke-test benchmark only", images_dir)
        images = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(n)]
    return images


def _infer_precision_from_name(name: str) -> str:
    for p in ("fp32", "fp16", "int8"):
        if p in name:
            return p
    return "unknown"


def _infer_size_from_name(name: str) -> int:
    for size in (320, 416, 640):
        if str(size) in name:
            return size
    return 640


def benchmark_model(onnx_path: Path, sample_images: list[np.ndarray], runs: int = 100, intra_threads: int = 3) -> BenchmarkResult:
    precision = _infer_precision_from_name(onnx_path.name)
    imgsz = _infer_size_from_name(onnx_path.name)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = intra_threads
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    blobs = []
    for img in sample_images:
        canvas, _, _ = letterbox_resize(img, imgsz)
        blobs.append(to_model_input(canvas))

    # Warm-up (excludes first-call graph optimization overhead from timing)
    for blob in blobs[:5]:
        session.run(None, {input_name: blob})

    process = psutil.Process()
    process.cpu_percent(interval=None)
    latencies_ms = []

    start_rss = process.memory_info().rss / (1024 * 1024)
    peak_rss = start_rss

    for i in range(runs):
        blob = blobs[i % len(blobs)]
        t0 = time.perf_counter()
        session.run(None, {input_name: blob})
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        peak_rss = max(peak_rss, process.memory_info().rss / (1024 * 1024))

    cpu_percent = process.cpu_percent(interval=None)
    latencies_ms.sort()
    mean_latency = sum(latencies_ms) / len(latencies_ms)
    p95_latency = latencies_ms[int(len(latencies_ms) * 0.95)]
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
    model_size_mb = onnx_path.stat().st_size / (1024 * 1024)

    return BenchmarkResult(
        model_file=onnx_path.name,
        precision=precision,
        input_size=imgsz,
        mean_latency_ms=round(mean_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        fps=round(fps, 2),
        cpu_percent_during=round(cpu_percent, 1),
        peak_rss_mb=round(peak_rss, 1),
        model_size_mb=round(model_size_mb, 2),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--images-dir", default="datasets/processed/val/images")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--out", default="docs/benchmark_results.csv")
    args = parser.parse_args()

    onnx_files = sorted(Path(args.weights_dir).glob("*.onnx"))
    if not onnx_files:
        logger.error("No .onnx files found in %s - run training/export.py first", args.weights_dir)
        return

    images = _load_sample_images(Path(args.images_dir))
    results = [benchmark_model(p, images, runs=args.runs, intra_threads=args.threads) for p in onnx_files]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    logger.info("Wrote benchmark results -> %s", out_path)
    for r in results:
        print(
            f"{r.model_file:35s} {r.precision:5s} {r.input_size:4d}px  "
            f"{r.mean_latency_ms:7.2f}ms  {r.fps:6.2f}fps  "
            f"cpu={r.cpu_percent_during:5.1f}%  rss={r.peak_rss_mb:7.1f}MB  size={r.model_size_mb:5.2f}MB"
        )


if __name__ == "__main__":
    main()
