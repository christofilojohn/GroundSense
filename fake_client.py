"""
fake_client.py — Stream NYU Depth v2 frames to the GroundSense server,
mimicking the iPhone app wire protocol.

Usage:
    pip install h5py numpy opencv-python websockets

    # Stream 30 frames starting at index 0
    python fake_client.py

    # Stream frames 100-200 at 10 fps, then ask a query
    python fake_client.py --start 100 --count 100 --fps 10 --query "what is to my left?"

    # Save a preview PNG to verify the data looks right 
    python fake_client.py --preview --count 0

Wire format (must match Frame.from_bytes in server.py):
    [4B jpeg_size uint32LE][jpeg_bytes]
    [4B depth_size uint32LE][depth_float16_bytes]
    [4B meta_size uint32LE][meta_json_utf8]
"""

import asyncio
import argparse
import json
import struct
import time
import logging
import sys

import numpy as np
import cv2
import h5py
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("fake_client")

# NYU Depth v2 Kinect camera intrinsics (standard published values)
NYU_INTRINSICS = {"fx": 518.858, "fy": 519.470, "cx": 325.582, "cy": 253.736}


# ── MAT file loading ─────────────────────────────────────────────────

class NyuDataset:
    """
    Lazy loader for nyu_depth_v2_labeled.mat.

    h5py reads MATLAB v7.3 HDF5 files with reversed dimension order
    (MATLAB is column-major; h5py is row-major).

    MATLAB layout in file:
        images  (H, W, 3, N) = (480, 640, 3, 1449)
        depths  (H, W, N)    = (480, 640, 1449)

    h5py presents them as:
        images  (N, 3, W, H) = (1449, 3, 640, 480)
        depths  (N, W, H)    = (1449, 640, 480)

    After transposing to standard (N, H, W, C) / (N, H, W):
        images  transpose(0, 3, 2, 1) → (N, H, W, 3)
        depths  transpose(0, 2, 1)    → (N, H, W)
    """

    def __init__(self, path: str):
        self._f = h5py.File(path, "r")
        self._images = self._f["images"]  # lazy Dataset
        self._depths = self._f["depths"]  # lazy Dataset

        log.info(f"Opened {path}")
        log.info(f"  h5py images shape : {self._images.shape}")
        log.info(f"  h5py depths shape : {self._depths.shape}")

        # Detect layout by inspecting the last two spatial dimensions.
        # Valid frames are 480×640 (H×W); if the shapes are wrong here
        # the log will tell you and you can flip the transpose below.
        n = self._images.shape[0]
        log.info(f"  Dataset size      : {n} frames")

    def __len__(self):
        return self._images.shape[0]

    def __getitem__(self, idx: int):
        """Return (rgb_hwc_uint8, depth_hw_float32) for frame idx."""
        # Load single frame (avoids pulling the whole 2.8 GB into RAM)
        raw_img   = self._images[idx]   # (3, W, H) — see class docstring
        raw_depth = self._depths[idx]   # (W, H)

        # (3, W, H) → (H, W, 3)
        rgb = raw_img.transpose(2, 1, 0).astype(np.uint8)

        # (W, H) → (H, W)
        depth = raw_depth.T.astype(np.float32)

        return rgb, depth

    def close(self):
        self._f.close()


# ── Wire-format packing ──────────────────────────────────────────────

def pack_frame(bgr: np.ndarray, depth: np.ndarray, frame_idx: int) -> bytes:
    """
    Pack one frame into the GroundSense binary wire format.
    bgr   : (H, W, 3) uint8  — OpenCV BGR
    depth : (H, W)    float32 metres
    """
    h, w = bgr.shape[:2]
    dh, dw = depth.shape

    # 1. JPEG-encode the BGR image (server decodes with cv2.IMREAD_COLOR → BGR)
    ok, jpeg_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    jpeg_bytes = jpeg_buf.tobytes()

    # 2. Depth as raw float16 (halves bandwidth, matches iPhone behaviour)
    depth_bytes = depth.astype(np.float16).tobytes()

    # 3. Metadata JSON (server reads depthWidth/depthHeight to reshape depth)
    meta = {
        "timestamp":  time.time(),
        "rgbWidth":   w,
        "rgbHeight":  h,
        "depthWidth":  dw,
        "depthHeight": dh,
        "intrinsics": [
            NYU_INTRINSICS["fx"],
            NYU_INTRINSICS["fy"],
            NYU_INTRINSICS["cx"],
            NYU_INTRINSICS["cy"],
        ],
        "frameIndex": frame_idx,
        "source":     "nyu_depth_v2",
        "orientation": "landscape",
    }
    meta_bytes = json.dumps(meta).encode("utf-8")

    # 4. Concatenate: [size][data] × 3
    return (
        struct.pack("<I", len(jpeg_bytes)) + jpeg_bytes
        + struct.pack("<I", len(depth_bytes)) + depth_bytes
        + struct.pack("<I", len(meta_bytes)) + meta_bytes
    )


# ── Display window ───────────────────────────────────────────────────

_ALERT = 1.0
_WARN  = 2.0

# Stored so interactive_query can log to terminal (no longer drawn on screen)
_last_query  = ""
_last_answer = ""
_last_source = ""


def render_frame(bgr: np.ndarray, scene: dict | None = None):
    """
    Show the RGB image with YOLO bounding boxes only.
    Press 'q' to quit. Returns False when the user presses 'q'.
    """
    canvas = bgr.copy()
    h, w = canvas.shape[:2]

    if scene:
        for obj in scene.get("objects", []):
            if "bbox" not in obj:
                continue
            nx1, ny1, nx2, ny2 = obj["bbox"]
            x1, y1 = int(nx1 * w), int(ny1 * h)
            x2, y2 = int(nx2 * w), int(ny2 * h)
            d = obj["distance_m"]
            colour = (
                (0, 0, 220)   if d < _ALERT else
                (0, 140, 255) if d < _WARN  else
                (50, 200, 50)
            )
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

    cv2.imshow("GroundSense", canvas)
    cv2.waitKey(1)


# ── Preview helper ───────────────────────────────────────────────────

def save_preview(dataset: NyuDataset, idx: int = 0):
    """Save RGB + depth-heatmap PNGs for visual sanity-check."""
    rgb, depth = dataset[idx]

    # RGB preview
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("preview_rgb.png", bgr)

    # Depth heatmap (0–10 m → plasma colormap)
    d_norm = (np.clip(depth, 0, 10) / 10 * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)
    cv2.imwrite("preview_depth.png", heatmap)

    log.info(f"Saved preview_rgb.png and preview_depth.png  (frame {idx})")
    log.info(f"  RGB   shape: {rgb.shape}   dtype: {rgb.dtype}   "
             f"range [{rgb.min()}, {rgb.max()}]")
    log.info(f"  Depth shape: {depth.shape}  dtype: {depth.dtype}  "
             f"range [{depth.min():.2f}, {depth.max():.2f}] m")


# ── WebSocket streaming ──────────────────────────────────────────────

async def stream(
    server: str,
    dataset: NyuDataset,
    fps: float,
    start: int,
    count: int,
):
    end = min(start + count, len(dataset))
    interval = 1.0 / fps

    log.info(f"Connecting to {server} …")
    async with websockets.connect(server, max_size=10 * 1024 * 1024) as ws:
        log.info(f"Connected. Streaming frames {start}–{end - 1} at {fps:.1f} fps")

        for i in range(start, end):
            rgb, depth = dataset[i]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # server expects BGR

            t0 = time.time()
            packet = pack_frame(bgr, depth, i)
            await ws.send(packet)

            # Read server response
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=8.0)
                resp = json.loads(raw)

                if resp.get("type") == "scene_update":
                    scene = resp["scene"]
                    objs  = scene["objects"]
                    free  = scene["free_direction"]
                    close = scene["closest_obstacle_m"]

                    obj_str = "  ".join(
                        f"{o['class']} {o['distance_m']}m {o['direction']}"
                        for o in objs[:4]
                    ) or "(none)"
                    log.info(
                        f"  [{i:4d}] {len(objs)} obj | free={free} | "
                        f"closest={close:.1f}m | {obj_str}"
                    )
                    if "warning" in resp:
                        log.warning(f"  ⚠  {resp['warning']}")

                else:
                    log.info(f"  [{i:4d}] server: {raw[:120]}")

            except asyncio.TimeoutError:
                log.warning(f"  [{i:4d}] no response within 8 s")

            # Throttle to target FPS
            wait = interval - (time.time() - t0)
            if wait > 0:
                await asyncio.sleep(wait)

        log.info("Stream finished.")
        return ws   # caller may reuse for interactive queries


async def interactive_query(ws, initial_query: str = "", display: bool = False,
                            last_bgr=None, last_scene=None):
    """Send one or more text queries and print the answers."""
    global _last_query, _last_answer, _last_source

    async def ask(q: str):
        global _last_query, _last_answer, _last_source
        _last_query = q
        _last_answer = "…"
        _last_source = ""
        await ws.send(json.dumps({"query": q}))
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            resp = json.loads(raw)
            if resp.get("type") == "query_response":
                _last_answer = resp["answer"]
                _last_source = resp.get("source", "unknown")
                print(f"\n  Q: {resp['query']}\n  A [{_last_source}]: {resp['answer']}\n")
            else:
                print(f"  raw: {raw[:200]}")
        except asyncio.TimeoutError:
            _last_answer = "(no answer within 10 s)"
            print("  (no answer within 10 s)")
        # Refresh display with updated Q/A overlay
        if display and last_bgr is not None:
            render_frame(last_bgr, last_scene)

    if initial_query:
        await ask(initial_query)

    # Drop into REPL if stdin is a terminal
    if sys.stdin.isatty():
        print("\nInteractive query mode — type a question and press Enter (blank to quit):")
        while True:
            try:
                q = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            await ask(q)


# ── Entry point ───────────────────────────────────────────────────────

async def main_async(args):
    dataset = NyuDataset(args.mat)

    try:
        if args.preview:
            save_preview(dataset, idx=args.start)

        if args.count == 0:
            log.info("--count 0: skipping stream (preview only).")
            return

        # Stream frames, get back the open WebSocket
        end = min(args.start + args.count, len(dataset))
        interval = 1.0 / args.fps

        if args.display:
            cv2.namedWindow("GroundSense", cv2.WINDOW_NORMAL)

        log.info(f"Connecting to {args.server} …")
        async with websockets.connect(
            args.server, max_size=10 * 1024 * 1024
        ) as ws:
            log.info(
                f"Connected. Streaming frames {args.start}–{end - 1} "
                f"at {args.fps:.1f} fps"
            )

            last_bgr   = None
            last_scene = None

            for i in range(args.start, end):
                rgb, depth = dataset[i]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                last_bgr = bgr

                t0 = time.time()
                packet = pack_frame(bgr, depth, i)
                await ws.send(packet)

                scene = None
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=8.0)
                    resp = json.loads(raw)

                    if resp.get("type") == "scene_update":
                        scene = resp["scene"]
                        last_scene = scene
                        objs  = scene["objects"]
                        free  = scene["free_direction"]
                        close = scene["closest_obstacle_m"]
                        obj_str = "  ".join(
                            f"{o['class']} {o['distance_m']}m {o['direction']}"
                            for o in objs[:4]
                        ) or "(none)"
                        log.info(
                            f"  [{i:4d}] {len(objs)} obj | free={free} | "
                            f"closest={close:.1f}m | {obj_str}"
                        )
                        if "warning" in resp:
                            log.warning(f"  ⚠  {resp['warning']}")
                    else:
                        log.info(f"  [{i:4d}] {raw[:120]}")

                except asyncio.TimeoutError:
                    log.warning(f"  [{i:4d}] no response within 8 s")

                if args.display:
                    render_frame(bgr, scene)

                # Pump the OpenCV event loop every ~16 ms while waiting for
                # the next frame — keeps the window draggable and responsive.
                remaining = interval - (time.time() - t0)
                while remaining > 0:
                    if args.display:
                        if (cv2.waitKey(16) & 0xFF) == ord("q"):
                            break
                    else:
                        await asyncio.sleep(min(remaining, 0.016))
                    remaining = interval - (time.time() - t0)
                    await asyncio.sleep(0)  # yield to asyncio each iteration

            log.info("Stream finished.")
            await interactive_query(
                ws, args.query,
                display=args.display,
                last_bgr=last_bgr,
                last_scene=last_scene,
            )

        if args.display:
            cv2.destroyAllWindows()

    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(
        description="Stream NYU Depth v2 frames to the GroundSense server"
    )
    parser.add_argument(
        "--mat",
        default="nyu_small.mat",
        help="Path to dataset .mat file (default: nyu_small.mat)",
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:8765",
        help="WebSocket server URL (default: ws://localhost:8765)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Playback speed in frames per second (default: 5)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of first frame to send (default: 0)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=60,
        help="Number of frames to send; 0 = preview only (default: 60)",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Voice query to send after streaming (e.g. 'what is ahead of me?')",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save preview_rgb.png + preview_depth.png before streaming",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Open a live window showing RGB + depth + server detections (press q to quit)",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
