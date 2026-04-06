#!/usr/bin/env python3
"""
benchmark_devices.py - Run GroundSense benchmarks across multiple devices.

By default this wrapper runs the existing benchmark suite on CPU and CUDA
with the same forwarded benchmark arguments.

Examples:
    python benchmark_devices.py
    python benchmark_devices.py --frames 30 --warmup 5
    python benchmark_devices.py --devices cuda --no-openvocab
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def detect_available_devices() -> set[str]:
    devices = {"cpu"}
    try:
        import torch
    except ImportError:
        return devices

    if torch.cuda.is_available():
        devices.add("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.add("mps")

    return devices


def validate_forwarded_args(extra_args: list[str]) -> None:
    for arg in extra_args:
        if arg == "--device" or arg.startswith("--device="):
            raise SystemExit(
                "benchmark_devices.py manages --device itself. "
                "Use --devices to choose one or more targets."
            )


def run_for_device(benchmark_script: Path, device: str, extra_args: list[str]) -> int:
    command = [sys.executable, str(benchmark_script), "--device", device, *extra_args]

    print("\n" + "=" * 70)
    print(f"Running benchmark.py on {device}")
    print("=" * 70)
    print("Command:", " ".join(command))

    completed = subprocess.run(command, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark.py on CPU, CUDA, or other supported devices."
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Devices to benchmark (default: cpu cuda)",
    )
    parser.add_argument(
        "--benchmark-script",
        default=str(Path(__file__).with_name("benchmark.py")),
        help="Path to benchmark.py",
    )
    args, extra_args = parser.parse_known_args()

    validate_forwarded_args(extra_args)

    benchmark_script = Path(args.benchmark_script).resolve()
    if not benchmark_script.exists():
        raise SystemExit(f"benchmark script not found: {benchmark_script}")

    requested_devices = args.devices or ["cpu", "cuda"]
    available_devices = detect_available_devices()

    exit_code = 0
    skipped: list[str] = []

    for device in requested_devices:
        if device != "cpu" and device not in available_devices:
            print(f"Skipping {device}: not available in this Python environment.")
            skipped.append(device)
            continue

        rc = run_for_device(benchmark_script, device, extra_args)
        if rc != 0 and exit_code == 0:
            exit_code = rc

    if skipped:
        print("\nSkipped devices:", ", ".join(skipped))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
