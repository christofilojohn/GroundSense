"""
benchmark.py — Measure per-stage inference latency for the GroundSense pipeline.

Loads frames from the NYU Depth v2 dataset (nyu_small.mat) and times each
pipeline stage independently over N warm-up + M measurement runs.

Usage:
    python benchmark.py                         # defaults
    python benchmark.py --frames 30 --warmup 5
    python benchmark.py --device cpu
    python benchmark.py --no-openvocab          # skip Grounding DINO + FastSAM
    python benchmark.py --gemini-key YOUR_KEY   # include Gemini query latency
"""

import argparse
import time
import statistics
import sys
import os

import numpy as np
import cv2
import h5py

# Device auto-detection (same logic as server.py)


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# Timer helper


class Timer:
    """Collect wall-clock samples and report statistics."""

    def __init__(self, name: str):
        self.name = name
        self._samples: list[float] = []

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self._samples.append((time.perf_counter() - self._t0) * 1000)  # ms

    def record(self, ms: float):
        self._samples.append(ms)

    def stats(self) -> dict:
        if not self._samples:
            return {}
        return {
            "n": len(self._samples),
            "mean": statistics.mean(self._samples),
            "median": statistics.median(self._samples),
            "stdev": statistics.stdev(self._samples) if len(self._samples) > 1 else 0.0,
            "min": min(self._samples),
            "max": max(self._samples),
            "p90": float(np.percentile(self._samples, 90)),
        }

    def report(self):
        s = self.stats()
        if not s:
            print(f"  {self.name:<35}  no data")
            return
        print(
            f"  {self.name:<35}"
            f"  mean={s['mean']:6.1f} ms"
            f"  median={s['median']:6.1f} ms"
            f"  p90={s['p90']:6.1f} ms"
            f"  stdev={s['stdev']:5.1f} ms"
            f"  [n={s['n']}]"
        )


# Load NYU frames


def load_nyu_frames(mat_path: str, count: int) -> list[dict]:
    """Return a list of {rgb: ndarray(H,W,3 BGR), depth: ndarray(H,W float32)} dicts."""
    print(f"Loading NYU Depth v2 from {mat_path} …")
    with h5py.File(mat_path, "r") as f:
        images = np.array(f["images"])  # (N, 3, H, W) uint8
        depths = np.array(f["depths"])  # (N, H, W) float32, metres

    n_available = images.shape[0]
    count = min(count, n_available)
    print(f"  Using {count}/{n_available} frames")

    frames = []
    for i in range(count):
        # NYU stores (C, H, W) — transpose to (H, W, C) then convert RGB→BGR
        rgb = np.transpose(images[i], (1, 2, 0))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth = depths[i].astype(np.float32)
        frames.append({"rgb": rgb, "depth": depth})
    return frames


# Build a Frame object compatible with server.py


def make_frame(rgb: np.ndarray, depth: np.ndarray):
    """Construct a server.Frame without importing from server at module level."""
    from server import Frame

    return Frame(
        rgb=rgb,
        depth=depth,
        metadata={
            "rgbWidth": rgb.shape[1],
            "rgbHeight": rgb.shape[0],
            "depthWidth": depth.shape[1],
            "depthHeight": depth.shape[0],
        },
        timestamp=time.time(),
    )


# Stage benchmarks


def bench_depth_preprocessing(frames: list[dict], warmup: int) -> Timer:
    t = Timer("Depth bilateral filter")
    for i, f in enumerate(frames):
        depth = f["depth"]
        if i < warmup:
            cv2.bilateralFilter(depth, d=7, sigmaColor=0.15, sigmaSpace=5)
        else:
            with t:
                cv2.bilateralFilter(depth, d=7, sigmaColor=0.15, sigmaSpace=5)
    return t


def bench_yolo(frames: list[dict], device: str, warmup: int) -> tuple[Timer, Timer, list]:
    """Returns (yolo_timer, depth_fusion_timer, scene_states)."""
    from server import SegmentationPipeline

    pipeline = SegmentationPipeline(model_name="yolo26s-seg.pt", device=device)

    t_yolo = Timer("YOLO26s-seg + track")
    t_fusion = Timer("LiDAR depth fusion")
    scenes = []

    print(f"  Warm-up ({warmup} frames) …", end=" ", flush=True)
    for i, f in enumerate(frames):
        frame = make_frame(f["rgb"], f["depth"])
        h, w = frame.rgb.shape[:2]

        # depth pre-processing (not timed here)
        depth = cv2.bilateralFilter(frame.depth, d=7, sigmaColor=0.15, sigmaSpace=5)
        frame.depth = depth

        # resize for YOLO
        yolo_rgb = cv2.resize(frame.rgb, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)

        is_warmup = i < warmup

        # time YOLO inference only
        t0 = time.perf_counter()
        results = pipeline.model.track(
            yolo_rgb,
            persist=True,
            verbose=False,
            device=device,
            imgsz=640,
            conf=0.3,
        )
        yolo_ms = (time.perf_counter() - t0) * 1000

        # time depth fusion only
        t0 = time.perf_counter()
        scene = pipeline.process_frame(frame)  # full call for scene state
        # subtract yolo time (approximate — process_frame re-runs YOLO internally,
        # so we run the full pipeline and attribute the remainder to fusion)
        full_ms = (time.perf_counter() - t0) * 1000
        fusion_ms = max(0.0, full_ms - yolo_ms)

        if not is_warmup:
            t_yolo.record(yolo_ms)
            t_fusion.record(fusion_ms)
            scenes.append(scene)

    print("done")
    return t_yolo, t_fusion, scenes


def bench_openvocab(frames: list[dict], device: str, warmup: int, targets: list[str]) -> tuple[Timer, Timer]:
    """Returns (gdino_timer, fastsam_timer). Runs every frame (no throttle)."""
    from server import OpenVocabPipeline, Frame

    ov = OpenVocabPipeline(device=device, interval=1)
    ov.set_targets(targets)
    if not ov._ensure_loaded():
        print("  [!] Open-vocab models unavailable — skipping.")
        return Timer("Grounding DINO-tiny (skipped)"), Timer("FastSAM-s (skipped)")

    t_gdino = Timer("Grounding DINO-tiny")
    t_fastsam = Timer("FastSAM-s")

    import torch
    from PIL import Image as PILImage

    print(f"  Warm-up ({warmup} frames) …", end=" ", flush=True)
    for i, f in enumerate(frames):
        frame = make_frame(f["rgb"], f["depth"])
        rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        prompt = " . ".join(targets) + " ."

        inputs = ov._processor(images=pil, text=prompt, return_tensors="pt").to(device)

        # Grounding DINO
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = ov._gdino_model(**inputs)
        gdino_ms = (time.perf_counter() - t0) * 1000

        # FastSAM (if available)
        fastsam_ms = None
        if ov._fastsam_model is not None:
            t0 = time.perf_counter()
            ov._fastsam_model(
                frame.rgb,
                imgsz=640,
                conf=0.4,
                iou=0.9,
                device=device,
                verbose=False,
            )
            fastsam_ms = (time.perf_counter() - t0) * 1000

        if i >= warmup:
            t_gdino.record(gdino_ms)
            if fastsam_ms is not None:
                t_fastsam.record(fastsam_ms)

    print("done")
    return t_gdino, t_fastsam


def bench_gemini(scene, gemini_key: str, n: int) -> Timer:
    """Fire N Gemini API calls with a fixed query and the provided scene."""
    from server import ResponseGenerator

    t = Timer("Gemini query (gemini-2.0-flash)")
    gen = ResponseGenerator(llm="gemini", gemini_key=gemini_key)
    if gen._llm != "gemini":
        print("  [!] Gemini unavailable — skipping.")
        return t

    query = "What is directly in front of me?"
    print(f"  Sending {n} Gemini queries …", end=" ", flush=True)
    for _ in range(n):
        t0 = time.perf_counter()
        gen.answer_query(scene, query)
        t.record((time.perf_counter() - t0) * 1000)
    print("done")
    return t


# Main


def main():
    parser = argparse.ArgumentParser(description="GroundSense latency benchmark")
    parser.add_argument("--mat", default="nyu_small.mat", help="Path to nyu_small.mat")
    parser.add_argument("--frames", type=int, default=20, help="Measurement frames per stage (default 20)")
    parser.add_argument("--warmup", type=int, default=3, help="Warm-up frames discarded before timing (default 3)")
    parser.add_argument("--device", default=None, help="torch device (auto-detected if omitted)")
    parser.add_argument("--no-openvocab", action="store_true", help="Skip Grounding DINO + FastSAM benchmark")
    parser.add_argument(
        "--ov-targets",
        nargs="+",
        default=["chair", "bottle"],
        help="Objects for open-vocab benchmark (default: chair bottle)",
    )
    parser.add_argument("--gemini-key", default="", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--gemini-n", type=int, default=5, help="Number of Gemini API calls to time (default 5)")
    args = parser.parse_args()

    device = args.device or _auto_device()
    total = args.frames + args.warmup

    print("=" * 70)
    print("GroundSense Latency Benchmark")
    print(f"  device  : {device}")
    print(f"  frames  : {args.frames} measurement + {args.warmup} warm-up")
    print("=" * 70)

    # Load data
    if not os.path.exists(args.mat):
        print(f"ERROR: {args.mat} not found. Run from the repo root.", file=sys.stderr)
        sys.exit(1)
    frames = load_nyu_frames(args.mat, total)
    if len(frames) < total:
        print(f"  [!] Only {len(frames)} frames available; adjusting.")
        args.warmup = min(args.warmup, len(frames) // 4)
        args.frames = len(frames) - args.warmup

    results: list[Timer] = []

    # Stage 1: Depth pre-processing
    print("\n[1/4] Depth bilateral filter")
    results.append(bench_depth_preprocessing(frames, args.warmup))

    # Stage 2: YOLO + depth fusion
    print(f"\n[2/4] YOLO segmentation + LiDAR depth fusion  (device={device})")
    t_yolo, t_fusion, scenes = bench_yolo(frames, device, args.warmup)
    results.extend([t_yolo, t_fusion])

    # Stage 3: Open-vocabulary pipeline
    if not args.no_openvocab:
        print(f"\n[3/4] Open-vocabulary pipeline  (targets: {args.ov_targets})")
        t_gdino, t_fastsam = bench_openvocab(frames, device, args.warmup, args.ov_targets)
        results.extend([t_gdino, t_fastsam])
    else:
        print("\n[3/4] Open-vocabulary pipeline  SKIPPED (--no-openvocab)")

    # Stage 4: Gemini query
    gemini_key = args.gemini_key or os.environ.get("GEMINI_API_KEY", "")
    if gemini_key and scenes:
        print(f"\n[4/4] Gemini query latency  (n={args.gemini_n})")
        results.append(bench_gemini(scenes[0], gemini_key, args.gemini_n))
    else:
        print("\n[4/4] Gemini query latency  SKIPPED (no --gemini-key)")

    # Summary table
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    for t in results:
        t.report()

    # End-to-end estimate
    yolo_med = t_yolo.stats().get("median", 0)
    fusion_med = t_fusion.stats().get("median", 0)
    depth_med = results[0].stats().get("median", 0)
    e2e = depth_med + yolo_med + fusion_med
    print(f"\n  {'End-to-end (primary pipeline)':<35}  ~{e2e:.0f} ms  " f"(depth + YOLO + fusion, median)")
    print("=" * 70)


if __name__ == "__main__":
    main()
