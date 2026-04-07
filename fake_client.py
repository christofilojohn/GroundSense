"""
fake_client.py — Stream frames to the GroundSense server for offline testing.

Supports two sources:
  1. NYU Depth v2 .mat dataset  (default)
  2. .gsrecording files captured on iPhone with the GroundSense app

Usage:
    pip install h5py numpy opencv-python websockets

    # Stream 30 NYU frames starting at index 0
    python fake_client.py

    # Stream frames 100-200 at 10 fps, then ask a query
    python fake_client.py --start 100 --count 100 --fps 10 --query "what is to my left?"

    # Save a preview PNG to verify NYU data looks right
    python fake_client.py --preview --count 0

    # Replay a real iPhone recording captured with GroundSense
    python fake_client.py --recording my_walk.gsrecording

    # Replay at a different speed (default: use the recorded fps)
    python fake_client.py --recording my_walk.gsrecording --replay-fps 5

    # Replay with open-vocabulary detection enabled alongside YOLO
    python fake_client.py --recording my_walk.gsrecording --pipeline both --targets "wheelchair,wet floor sign"

Wire format (must match Frame.from_bytes in server.py):
    [4B jpeg_size uint32LE][jpeg_bytes]
    [4B depth_size uint32LE][depth_float16_bytes]
    [4B meta_size uint32LE][meta_json_utf8]

.gsrecording format (written by iOS RecordingManager.swift):
    Fixed 32-byte header:
      [0:8]   magic    b"GSREC\\x01\\x00\\n"
      [8:12]  version  UInt32 LE = 1
      [12:16] fps_mHz  UInt32 LE  (fps * 1000)
      [16:20] frameCount UInt32 LE
      [20:28] createdAt Float64 LE (Unix timestamp)
      [28:32] flags    UInt32 LE = 0
    Followed by frameCount wire-format frame packets (same layout as above).
"""

from __future__ import annotations

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

# NYU Depth v2 camera intrinsics
NYU_INTRINSICS = {"fx": 518.858, "fy": 519.470, "cx": 325.582, "cy": 253.736}


# MAT file loading


class NyuDataset:
    """
    Lazy loader for nyu_depth_v2_labeled.mat.

    h5py presents MATLAB arrays with reversed axes (MATLAB is column-major),
    so images come out as (N, 3, W, H) and depths as (N, W, H).
    We transpose back to standard (N, H, W, C) / (N, H, W) on access.
    """

    def __init__(self, path: str):
        self._f = h5py.File(path, "r")
        self._images = self._f["images"]  # lazy Dataset
        self._depths = self._f["depths"]  # lazy Dataset

        log.info(f"Opened {path}")
        log.info(f"  h5py images shape : {self._images.shape}")
        log.info(f"  h5py depths shape : {self._depths.shape}")

        n = self._images.shape[0]
        log.info(f"  Dataset size      : {n} frames")

    def __len__(self):
        return self._images.shape[0]

    def __getitem__(self, idx: int):
        """Return (rgb_hwc_uint8, depth_hw_float32) for frame idx."""
        # Load one frame at a time — the full dataset is too large for RAM
        raw_img = self._images[idx]  # (3, W, H) — see class docstring
        raw_depth = self._depths[idx]  # (W, H)

        # (3, W, H) → (H, W, 3)
        rgb = raw_img.transpose(2, 1, 0).astype(np.uint8)

        # (W, H) → (H, W)
        depth = raw_depth.T.astype(np.float32)

        return rgb, depth

    def close(self):
        self._f.close()


# Wire-format packing


def pack_frame(bgr: np.ndarray, depth: np.ndarray, frame_idx: int) -> bytes:
    """
    Pack one frame into the GroundSense binary wire format.
    bgr   : (H, W, 3) uint8  — OpenCV BGR
    depth : (H, W)    float32 metres
    """
    h, w = bgr.shape[:2]
    dh, dw = depth.shape

    # 1. JPEG-encode the BGR image
    ok, jpeg_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    jpeg_bytes = jpeg_buf.tobytes()

    # 2. Depth as float16 (matches iPhone wire format)
    depth_bytes = depth.astype(np.float16).tobytes()

    # 3. Metadata JSON
    meta = {
        "timestamp": time.time(),
        "rgbWidth": w,
        "rgbHeight": h,
        "depthWidth": dw,
        "depthHeight": dh,
        "intrinsics": [
            NYU_INTRINSICS["fx"],
            NYU_INTRINSICS["fy"],
            NYU_INTRINSICS["cx"],
            NYU_INTRINSICS["cy"],
        ],
        "frameIndex": frame_idx,
        "source": "nyu_depth_v2",
        "orientation": "landscape",
    }
    meta_bytes = json.dumps(meta).encode("utf-8")

    # 4. Concatenate: [size][data] × 3
    return (
        struct.pack("<I", len(jpeg_bytes))
        + jpeg_bytes
        + struct.pack("<I", len(depth_bytes))
        + depth_bytes
        + struct.pack("<I", len(meta_bytes))
        + meta_bytes
    )


# Display window

_ALERT = 1.0
_WARN = 2.0

# Stored so interactive_query can log to terminal (no longer drawn on screen)
_last_query = ""
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
            colour = (0, 0, 220) if d < _ALERT else (0, 140, 255) if d < _WARN else (50, 200, 50)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

    cv2.imshow("GroundSense", canvas)
    cv2.waitKey(1)


# Preview helper


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
    log.info(f"  RGB   shape: {rgb.shape}   dtype: {rgb.dtype}   " f"range [{rgb.min()}, {rgb.max()}]")
    log.info(f"  Depth shape: {depth.shape}  dtype: {depth.dtype}  " f"range [{depth.min():.2f}, {depth.max():.2f}] m")


# WebSocket streaming


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
                    objs = scene["objects"]
                    free = scene["free_direction"]
                    close = scene["closest_obstacle_m"]

                    obj_str = (
                        "  ".join(f"{o['class']} {o['distance_m']}m {o['direction']}" for o in objs[:4]) or "(none)"
                    )
                    log.info(f"  [{i:4d}] {len(objs)} obj | free={free} | " f"closest={close:.1f}m | {obj_str}")
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
        return ws  # caller may reuse for interactive queries


async def interactive_query(ws, initial_query: str = "", display: bool = False, last_bgr=None, last_scene=None):
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


# .gsrecording playback

# Magic bytes written by iOS RecordingManager.swift
_GSREC_MAGIC = b"GSREC\x01\x00\n"
_GSREC_HEADER = 32  # fixed header size in bytes


def read_gsrecording(path: str):
    """
    Parse a .gsrecording file and return (fps, list_of_raw_packets).

    Each element of list_of_raw_packets is a bytes object containing one
    complete wire-format frame — it can be sent directly to the server via
    ws.send() without any re-encoding.

    Header layout (all little-endian):
      [0:8]   magic     b"GSREC\\x01\\x00\\n"
      [8:12]  version   uint32
      [12:16] fps_mHz   uint32  (fps * 1000)
      [16:20] frameCount uint32
      [20:28] createdAt float64 (Unix timestamp)
      [28:32] flags     uint32  (reserved)
    """
    with open(path, "rb") as f:
        header = f.read(_GSREC_HEADER)
        if len(header) < _GSREC_HEADER:
            raise ValueError(f"{path}: file too short to contain a valid header")

        magic = header[0:8]
        if magic != _GSREC_MAGIC:
            raise ValueError(f"{path}: bad magic bytes {magic!r}. " "Is this a GroundSense .gsrecording file?")

        fps_mhz = struct.unpack_from("<I", header, 12)[0]
        frame_count = struct.unpack_from("<I", header, 16)[0]
        created_at = struct.unpack_from("<d", header, 20)[0]
        fps = fps_mhz / 1000.0

        import datetime

        ts_str = datetime.datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M:%S")
        log.info(f"Recording: {frame_count} frames @ {fps:.1f} fps  created {ts_str}")

        packets: list[bytes] = []
        for i in range(frame_count):
            # Each frame is three TLV blocks: jpeg | depth_f16 | meta_json
            pieces: list[bytes] = []
            ok = True
            for _ in range(3):
                sz_bytes = f.read(4)
                if len(sz_bytes) < 4:
                    log.warning(f"  Truncated at frame {i}, block {_}")
                    ok = False
                    break
                sz = struct.unpack("<I", sz_bytes)[0]
                data = f.read(sz)
                if len(data) < sz:
                    log.warning(f"  Short read at frame {i}: expected {sz}, got {len(data)}")
                    ok = False
                    break
                pieces.append(sz_bytes + data)
            if ok and len(pieces) == 3:
                packets.append(b"".join(pieces))

        log.info(f"Loaded {len(packets)} / {frame_count} complete frames from {path}")
        return fps, packets


async def _send_set_targets(ws, targets_str: str) -> None:
    """
    Parse a comma-separated targets string and send a set_targets message.
    Waits for the server's set_targets_ack.  No-op if targets_str is empty.
    """
    if not targets_str.strip():
        return
    objects = [t.strip() for t in targets_str.split(",") if t.strip()]
    msg = json.dumps({"type": "set_targets", "objects": objects})
    log.info(f"[OpenVocab] Sending set_targets: {objects}")
    await ws.send(msg)
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        ack = json.loads(raw)
        if ack.get("type") == "set_targets_ack":
            if ack.get("ok"):
                log.info(f"[OpenVocab] Targets confirmed by server: {ack.get('targets')}")
            else:
                log.warning(f"[OpenVocab] Server rejected targets: {ack.get('error')}")
        else:
            log.warning(f"[OpenVocab] Unexpected response to set_targets: {raw[:120]}")
    except asyncio.TimeoutError:
        log.warning("[OpenVocab] No ack received within 5 s — continuing anyway")


async def _send_set_pipeline(ws, pipeline_mode: str) -> None:
    """
    Switch the server pipeline mode before streaming starts.
    No-op when pipeline_mode is "keep".
    """
    mode = pipeline_mode.strip().lower()
    if mode == "keep":
        return

    msg = json.dumps({"type": "set_pipeline", "mode": mode})
    log.info(f"[Pipeline] Requesting mode: {mode}")
    await ws.send(msg)
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        ack = json.loads(raw)
        if ack.get("type") == "set_pipeline_ack":
            if ack.get("ok"):
                log.info(f"[Pipeline] Server mode confirmed: {ack.get('mode')}")
            else:
                log.warning(f"[Pipeline] Server rejected mode '{mode}': {ack.get('error')}")
        else:
            log.warning(f"[Pipeline] Unexpected response to set_pipeline: {raw[:120]}")
    except asyncio.TimeoutError:
        log.warning("[Pipeline] No ack received within 5 s — continuing anyway")


async def stream_gsrecording(
    server: str,
    recording_path: str,
    fps_override: float | None = None,
    query: str = "",
    targets: str = "",
    pipeline: str = "keep",
):
    """Replay a .gsrecording file to the GroundSense server."""
    fps, packets = read_gsrecording(recording_path)
    if fps_override:
        fps = fps_override
        log.info(f"FPS overridden to {fps:.1f}")
    interval = 1.0 / fps

    log.info(f"Connecting to {server} …")
    async with websockets.connect(server, max_size=10 * 1024 * 1024) as ws:
        await _send_set_pipeline(ws, pipeline)
        # Activate open-vocab targets before the first frame
        await _send_set_targets(ws, targets)
        log.info(f"Connected. Replaying {len(packets)} frames at {fps:.1f} fps")

        last_scene = None

        for i, packet in enumerate(packets):
            t0 = time.time()
            await ws.send(packet)

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=8.0)
                resp = json.loads(raw)

                if resp.get("type") == "scene_update":
                    scene = resp["scene"]
                    last_scene = scene
                    objs = scene["objects"]
                    free = scene["free_direction"]
                    close = scene["closest_obstacle_m"]
                    obj_str = (
                        "  ".join(f"{o['class']} {o['distance_m']}m {o['direction']}" for o in objs[:4]) or "(none)"
                    )
                    log.info(
                        f"  [{i:4d}/{len(packets)}] {len(objs)} obj | "
                        f"free={free} | closest={close:.1f}m | {obj_str}"
                    )
                    if "warning" in resp:
                        log.warning(f"  ⚠  {resp['warning']}")
                else:
                    log.info(f"  [{i:4d}] server: {raw[:120]}")

            except asyncio.TimeoutError:
                log.warning(f"  [{i:4d}] no response within 8 s")

            # Throttle to recorded (or overridden) FPS
            wait = interval - (time.time() - t0)
            if wait > 0:
                await asyncio.sleep(wait)

        log.info("Playback finished.")

        # Optional post-playback query
        if query or sys.stdin.isatty():
            await interactive_query(ws, query)


# Entry point


async def main_async(args):

    # Recording playback mode
    if args.recording:
        await stream_gsrecording(
            server=args.server,
            recording_path=args.recording,
            fps_override=args.replay_fps if args.replay_fps > 0 else None,
            query=args.query,
            targets=args.targets,
            pipeline=args.pipeline,
        )
        return

    # NYU dataset streaming mode
    dataset = NyuDataset(args.mat)

    try:
        if args.preview:
            save_preview(dataset, idx=args.start)

        if args.count == 0:
            log.info("--count 0: skipping stream (preview only).")
            return

        end = min(args.start + args.count, len(dataset))
        interval = 1.0 / args.fps

        if args.display:
            cv2.namedWindow("GroundSense", cv2.WINDOW_NORMAL)

        log.info(f"Connecting to {args.server} …")
        async with websockets.connect(args.server, max_size=10 * 1024 * 1024) as ws:
            await _send_set_pipeline(ws, args.pipeline)
            # Activate open-vocab targets before the first frame
            await _send_set_targets(ws, args.targets)
            log.info(f"Connected. Streaming frames {args.start}–{end - 1} " f"at {args.fps:.1f} fps")

            last_bgr = None
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
                        objs = scene["objects"]
                        free = scene["free_direction"]
                        close = scene["closest_obstacle_m"]
                        obj_str = (
                            "  ".join(f"{o['class']} {o['distance_m']}m {o['direction']}" for o in objs[:4]) or "(none)"
                        )
                        log.info(f"  [{i:4d}] {len(objs)} obj | free={free} | " f"closest={close:.1f}m | {obj_str}")
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
                ws,
                args.query,
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
        description=(
            "Stream frames to the GroundSense server.\n"
            "Use --recording to replay a .gsrecording captured on iPhone,\n"
            "or omit it to stream from the NYU Depth v2 dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Recording playback
    rec_group = parser.add_argument_group("Recording playback (iPhone captures)")
    rec_group.add_argument(
        "--recording",
        default="",
        metavar="FILE.gsrecording",
        help=(
            "Path to a .gsrecording file captured with the GroundSense iOS app. "
            "Replays it to the server instead of the NYU dataset. "
            "No dataset dependency required."
        ),
    )
    rec_group.add_argument(
        "--replay-fps",
        type=float,
        default=0.0,
        metavar="FPS",
        help=("Override playback speed for --recording mode. " "0 = use the fps stored in the recording (default)."),
    )

    # NYU dataset streaming
    nyu_group = parser.add_argument_group("NYU dataset streaming (default mode)")
    nyu_group.add_argument(
        "--mat",
        default="nyu_small.mat",
        help="Path to dataset .mat file (default: nyu_small.mat)",
    )
    nyu_group.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Playback speed in frames per second (default: 5)",
    )
    nyu_group.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of first frame to send (default: 0)",
    )
    nyu_group.add_argument(
        "--count",
        type=int,
        default=60,
        help="Number of frames to send; 0 = preview only (default: 60)",
    )
    nyu_group.add_argument(
        "--preview",
        action="store_true",
        help="Save preview_rgb.png + preview_depth.png before streaming",
    )
    nyu_group.add_argument(
        "--display",
        action="store_true",
        help="Open a live window showing RGB + detections (press q to quit)",
    )

    # Shared
    parser.add_argument(
        "--server",
        default="ws://localhost:8765",
        help="WebSocket server URL (default: ws://localhost:8765)",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Voice query to send after streaming finishes",
    )
    parser.add_argument(
        "--targets",
        default="",
        metavar="OBJ1,OBJ2,...",
        help=(
            "Comma-separated open-vocabulary objects to detect via Grounding DINO "
            "(e.g. --targets 'wheelchair,service dog,wet floor sign'). "
            "Sent to the server before streaming starts. "
            "Requires --open-vocab on the server (enabled by default). "
            "Leave empty for YOLO-only mode."
        ),
    )
    parser.add_argument(
        "--pipeline",
        choices=["keep", "yolo", "gdino", "both"],
        default="keep",
        help=(
            "Detection pipeline to request from the server before streaming. "
            "'keep' leaves the current server mode unchanged (default). "
            "'yolo' = YOLO-only, 'gdino' = open-vocab only, "
            "'both' = YOLO + open-vocab."
        ),
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
