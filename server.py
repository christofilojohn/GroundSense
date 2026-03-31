"""
GroundSense Backend Server
==========================
WebSocket server that receives synchronized RGB + LiDAR depth frames
from the iPhone app, runs segmentation, and returns scene descriptions.

Usage:
    pip install websockets numpy opencv-python-headless pillow ultralytics
    python server.py --host 0.0.0.0 --port 8765

    # With Gemini LLM query engine (default):
    pip install google-genai
    export GEMINI_API_KEY=your_key   # or pass --gemini-key
    python server.py

    # Rule-based only (no API key needed):
    python server.py --llm none
"""

import asyncio
import json
import struct
import argparse
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import os

import numpy as np
import cv2
from PIL import Image
import io
import websockets

# Optional Gemini dependency — imported lazily so the server still starts
# without it when --llm none is passed.
try:
    from google import genai as _genai_module
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("groundsense")


# ── Frame unpacking ──────────────────────────────────────────────────

@dataclass
class Frame:
    """A single captured frame from the iPhone."""
    rgb: np.ndarray              # (H, W, 3) uint8 BGR
    depth: Optional[np.ndarray]  # (Hd, Wd) float32 meters, or None
    metadata: dict
    timestamp: float

    @staticmethod
    def from_bytes(data: bytes) -> "Frame":
        """
        Unpack the binary frame packet sent by the Swift app.
        
        Wire format:
            [4B jpeg_size][jpeg_bytes][4B depth_size][depth_bytes][4B meta_size][meta_json]
        """
        offset = 0

        # 1. JPEG RGB
        jpeg_size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        jpeg_bytes = data[offset : offset + jpeg_size]
        offset += jpeg_size

        rgb = cv2.imdecode(
            np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        # 2. Depth (float16 raw)
        depth_size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        depth = None

        if depth_size > 0:
            depth_bytes = data[offset : offset + depth_size]
            offset += depth_size

        # 3. Metadata JSON
        meta_size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        meta_json = data[offset : offset + meta_size]
        metadata = json.loads(meta_json)

        # Reconstruct depth from float16
        if depth_size > 0 and metadata.get("depthWidth", 0) > 0:
            depth_f16 = np.frombuffer(depth_bytes, dtype=np.float16)
            depth = depth_f16.astype(np.float32).reshape(
                metadata["depthHeight"], metadata["depthWidth"]
            )

        return Frame(
            rgb=rgb,
            depth=depth,
            metadata=metadata,
            timestamp=metadata.get("timestamp", time.time()),
        )


# ── Scene Representation ────────────────────────────────────────────

@dataclass
class DetectedObject:
    """An object detected in the scene with spatial info."""
    class_name: str
    track_id: int
    confidence: float
    distance_m: float          # median depth within mask
    direction: str             # "left", "center", "right"
    bbox: tuple                # (x1, y1, x2, y2) normalized
    mask_area_ratio: float     # fraction of frame covered

@dataclass 
class SceneState:
    """Current understanding of the scene."""
    objects: list = field(default_factory=list)
    free_direction: str = "center"  # safest direction to walk
    closest_obstacle_m: float = float("inf")
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "objects": [
                {
                    "class": o.class_name,
                    "track_id": o.track_id,
                    "distance_m": round(o.distance_m, 2),
                    "direction": o.direction,
                    "confidence": round(o.confidence, 2),
                }
                for o in self.objects
            ],
            "free_direction": self.free_direction,
            "closest_obstacle_m": round(self.closest_obstacle_m, 2),
            "timestamp": self.timestamp,
        }


# ── Segmentation Pipeline ───────────────────────────────────────────

class SegmentationPipeline:
    """
    Primary pipeline: YOLO-seg for instance segmentation + tracking.
    Fuses with LiDAR depth for distance estimation.
    """

    def __init__(self, model_name: str = "yolo11n-seg.pt", device: str = "cpu"):
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model: {model_name} on {device}")
        self.model = YOLO(model_name)
        self.device = device
        logger.info("Model loaded successfully")

    def process_frame(self, frame: Frame) -> SceneState:
        """Run segmentation + depth fusion on a single frame."""
        h, w = frame.rgb.shape[:2]

        # ── Depth pre-processing ──────────────────────────────────────
        # Bilateral filter: edge-preserving denoise on the LiDAR depth map.
        # sigmaColor=0.15 → blur depth values within 15 cm of each other.
        # sigmaSpace=5    → consider pixels within a 5-pixel spatial radius.
        depth = frame.depth
        if depth is not None:
            depth = cv2.bilateralFilter(depth, d=7, sigmaColor=0.15, sigmaSpace=5)

        # ── RGB pre-resize for YOLO ───────────────────────────────────
        # YOLO internally rescales to imgsz=640 anyway; pre-resizing avoids
        # passing a multi-megapixel array through the model's preprocessing.
        yolo_w = 640
        yolo_h = int(h * yolo_w / w)
        yolo_rgb = cv2.resize(frame.rgb, (yolo_w, yolo_h), interpolation=cv2.INTER_AREA)

        # Run YOLO segmentation with tracking
        results = self.model.track(
            yolo_rgb,
            persist=True,       # Keep track IDs across frames
            verbose=False,
            device=self.device,
            imgsz=640,
            conf=0.3,
        )

        scene = SceneState(timestamp=frame.timestamp)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            masks = result.masks

            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_id]
                track_id = int(box.id[0]) if box.id is not None else -1

                # Bounding box (xyxy normalized)
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                center_x = (x1 + x2) / 2

                # Direction from horizontal position
                if center_x < 0.33:
                    direction = "left"
                elif center_x > 0.66:
                    direction = "right"
                else:
                    direction = "center"

                # Distance from LiDAR depth (using the bilaterally-filtered map)
                distance = float("inf")
                if depth is not None and masks is not None:
                    mask = masks[i].data[0].cpu().numpy()  # (H_mask, W_mask)
                    # Resize mask to depth map dimensions
                    dh, dw = depth.shape
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST
                    )
                    # Get depth values within the mask
                    depth_values = depth[mask_resized > 0.5]
                    depth_values = depth_values[(depth_values > 0.1) & (depth_values < 10)]
                    if len(depth_values) > 0:
                        distance = float(np.median(depth_values))

                obj = DetectedObject(
                    class_name=class_name,
                    track_id=track_id,
                    confidence=conf,
                    distance_m=distance,
                    direction=direction,
                    bbox=(x1, y1, x2, y2),
                    mask_area_ratio=(x2 - x1) * (y2 - y1),
                )
                scene.objects.append(obj)

            # Sort objects closest first
            scene.objects.sort(key=lambda o: o.distance_m)
            if scene.objects:
                scene.closest_obstacle_m = scene.objects[0].distance_m

        # ── Free-space estimation (LiDAR-grid primary, object-based fallback) ──
        scene.free_direction = self._estimate_free_direction_lidar(
            depth, scene.objects
        )
        return scene

    def _estimate_free_direction_lidar(
        self,
        depth: Optional[np.ndarray],
        objects: list,
    ) -> str:
        """
        Divide the depth map into left / centre / right thirds and measure
        the 10th-percentile depth in each sector (lower portion of frame only,
        where ground-level obstacles matter most).

        Falls back to the object-based heuristic when no depth map is available.
        """
        if depth is not None:
            dh, dw = depth.shape
            # Focus on the lower two-thirds of the frame — that's where obstacles
            # the user would walk into are most likely to appear.
            roi = depth[dh // 3 :, :]

            third = dw // 3
            sectors = {
                "left":   roi[:, :third],
                "center": roi[:, third : 2 * third],
                "right":  roi[:, 2 * third :],
            }

            clearances: dict[str, float] = {}
            for name, sector in sectors.items():
                valid = sector[(sector > 0.1) & (sector < 8.0)]
                if len(valid) < 10:
                    # Too few valid readings — assume clear
                    clearances[name] = 8.0
                else:
                    # 10th percentile is robust to noise without being as aggressive
                    # as the minimum, which is easily skewed by a single bad pixel.
                    clearances[name] = float(np.percentile(valid, 10))

            return max(clearances, key=clearances.get)

        # ── Fallback: use detected objects when no LiDAR depth is available ──
        sector_min_dist = {"left": 10.0, "center": 10.0, "right": 10.0}
        for obj in objects:
            if obj.distance_m < sector_min_dist[obj.direction]:
                sector_min_dist[obj.direction] = obj.distance_m
        return max(sector_min_dist, key=sector_min_dist.get)


# ── Obstacle Avoidance & Response Generation ─────────────────────────

class ResponseGenerator:
    """Generates spoken navigation instructions from scene state."""

    WARN_DISTANCE = 2.0   # meters — warn about objects closer than this
    ALERT_DISTANCE = 1.0  # meters — urgent alert

    # Gemini model used for query answering
    GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

    def __init__(self, llm: str = "gemini", gemini_key: str = ""):
        """
        llm       : "gemini" | "none"
        gemini_key: API key (falls back to GEMINI_API_KEY env var)
        """
        self._llm = "none"
        self._gemini_client = None

        if llm == "gemini":
            if not _GENAI_AVAILABLE:
                logger.warning(
                    "google-genai not installed — falling back to rule-based engine. "
                    "Run: pip install google-genai"
                )
            else:
                key = gemini_key or os.environ.get("GEMINI_API_KEY", "")
                if not key:
                    logger.warning(
                        "No Gemini API key found. Pass --gemini-key or set GEMINI_API_KEY. "
                        "Falling back to rule-based engine."
                    )
                else:
                    self._gemini_client = _genai_module.Client(api_key=key)
                    self._llm = "gemini"
                    logger.info(f"Gemini query engine active ({self.GEMINI_MODEL})")

        if self._llm == "none":
            logger.info("Using rule-based query engine.")

    def generate_obstacle_warning(self, scene: SceneState) -> Optional[str]:
        """Generate a spoken warning if obstacles are dangerously close."""
        close_objects = [
            o for o in scene.objects if o.distance_m < self.WARN_DISTANCE
        ]

        if not close_objects:
            return None

        # Sort by distance (closest first)
        close_objects.sort(key=lambda o: o.distance_m)
        nearest = close_objects[0]

        if nearest.distance_m < self.ALERT_DISTANCE:
            return (
                f"Warning! {nearest.class_name} directly {nearest.direction}, "
                f"{nearest.distance_m:.1f} metres. Move {scene.free_direction}."
            )
        else:
            return (
                f"{nearest.class_name} {nearest.direction} "
                f"in {nearest.distance_m:.1f} metres."
            )

    def answer_query(self, scene: SceneState, query: str) -> tuple[str, str]:
        """Answer a spatial query. Returns (answer, source) where source is 'gemini' or 'rule-based'."""
        if self._llm == "gemini":
            try:
                return self._gemini_answer(scene, query), "gemini"
            except Exception as e:
                logger.warning(f"Gemini call failed ({e}), falling back to rule-based.")

        return self._rule_based_answer(scene, query), "rule-based"

    def _gemini_answer(self, scene: SceneState, query: str) -> str:
        """Call Gemini with the current scene state as context."""
        scene_dict = scene.to_dict()

        # Compact scene description to keep the prompt short and latency low
        if scene_dict["objects"]:
            obj_lines = "\n".join(
                f"  - {o['class']} | {o['distance_m']} m | {o['direction']}"
                for o in scene_dict["objects"][:8]
            )
        else:
            obj_lines = "  (none detected)"

        system_prompt = (
            "You are a real-time navigation assistant for a visually impaired person. "
            "Answer in plain spoken English, maximum 2 short sentences. "
            "Be direct and actionable — the response will be read aloud immediately.\n\n"
            "Current scene:\n"
            f"  Free direction to walk: {scene_dict['free_direction']}\n"
            f"  Closest obstacle: {scene_dict['closest_obstacle_m']} m\n"
            f"Detected objects:\n{obj_lines}"
        )

        response = self._gemini_client.models.generate_content(
            model=self.GEMINI_MODEL,
            contents=f"{system_prompt}\n\nUser question: {query}",
        )
        return response.text.strip()

    def _rule_based_answer(self, scene: SceneState, query: str) -> str:
        """
        Fallback rule-based engine.

        Handles patterns like:
          • "What is to my left / right / ahead?"
          • "What can you see?" / "Describe the scene"
          • "How far is the <object>?" / "Where is the <object>?"
          • "Is it safe to go <direction>?"
          • "Which way should I go?"
        """
        q = query.lower().strip(" ?")

        # ── 1. Direction queries ──────────────────────────────────────
        direction_map = {
            "left":   "left",
            "right":  "right",
            "ahead":  "center",
            "front":  "center",
            "center": "center",
            "forward": "center",
            "straight": "center",
        }
        target_direction: Optional[str] = None
        direction_label: str = ""
        for keyword, dir_val in direction_map.items():
            if keyword in q:
                target_direction = dir_val
                direction_label = keyword
                break

        if target_direction is not None and any(
            w in q for w in ("what", "is there", "see", "there", "show", "tell")
        ):
            filtered = [o for o in scene.objects if o.direction == target_direction]
            if not filtered:
                return f"Nothing detected to your {direction_label}. The path looks clear."
            items = ", ".join(
                f"{o.class_name} {o.distance_m:.1f} metres away"
                for o in sorted(filtered, key=lambda o: o.distance_m)
            )
            return f"To your {direction_label} I can see: {items}."

        # ── 2. Safety / navigation queries ────────────────────────────
        if any(w in q for w in ("safe", "go", "walk", "move", "direction", "which way")):
            free = scene.free_direction
            closest = scene.closest_obstacle_m
            if closest > self.WARN_DISTANCE:
                return f"The path looks clear. You can proceed {free}."
            else:
                return (
                    f"Caution — closest obstacle is {closest:.1f} metres. "
                    f"The clearest direction is {free}."
                )

        # ── 3. Object-specific queries ("how far is the chair") ───────
        for obj in scene.objects:
            if obj.class_name.lower() in q:
                return (
                    f"The {obj.class_name} is {obj.distance_m:.1f} metres "
                    f"to your {obj.direction}."
                )

        # ── 4. General scene description ──────────────────────────────
        if any(w in q for w in ("see", "around", "scene", "describe", "what is", "what's")):
            if not scene.objects:
                return "I don't see any objects right now."
            items = ", ".join(
                f"{o.class_name} {o.distance_m:.1f} m {o.direction}"
                for o in scene.objects[:5]  # cap at 5 to keep response brief
            )
            return f"I can see: {items}."

        # ── 5. Closest obstacle ───────────────────────────────────────
        if any(w in q for w in ("closest", "nearest", "danger", "obstacle")):
            if not scene.objects:
                return "No obstacles detected nearby."
            nearest = scene.objects[0]
            return (
                f"The nearest obstacle is a {nearest.class_name} "
                f"{nearest.distance_m:.1f} metres to your {nearest.direction}."
            )

        # ── 6. Fallback ───────────────────────────────────────────────
        if not scene.objects:
            return "The scene looks clear — no objects detected."
        items = ", ".join(
            f"{o.class_name} {o.direction}"
            for o in scene.objects[:4]
        )
        return f"I can see: {items}."


# ── Live Visualizer ───────────────────────────────────────────────────

class Visualizer:
    """
    Opens a macOS window showing the live camera feed with YOLO overlays
    on the left and a false-colour depth heatmap on the right.

    Requires opencv-python (not headless):
        pip uninstall opencv-python-headless -y && pip install opencv-python
    """

    # Colour thresholds (green → orange → red as distance decreases)
    _WARN  = 2.0   # metres
    _ALERT = 1.0

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None          # (frame, scene) set by asyncio thread
        self._fps_time = time.time()
        self._fps = 0.0
        self._frame_idx = 0
        self._window_ready = False

    def update(self, frame: "Frame", scene: "SceneState") -> None:
        """Called from the asyncio thread — just stores the latest data."""
        with self._lock:
            self._pending = (frame, scene)

    def render(self) -> bool:
        """
        Called from the MAIN thread only.
        Draws the latest frame and pumps the OpenCV event loop.
        Returns False when the user presses 'q'.
        """
        with self._lock:
            data = self._pending
            self._pending = None

        if not self._window_ready:
            cv2.namedWindow("GroundSense", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("GroundSense", 1280, 480)
            self._window_ready = True

        if data is not None:
            frame, scene = data

            # ── FPS counter ──────────────────────────────────────────
            self._frame_idx += 1
            now = time.time()
            elapsed = now - self._fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_idx / elapsed
                self._frame_idx = 0
                self._fps_time = now

            rgb = frame.rgb.copy()
            h, w = rgb.shape[:2]

            # ── YOLO bounding boxes + labels ─────────────────────────
            # bbox is stored as normalised (0-1); scale to pixel coords here.
            for obj in scene.objects:
                nx1, ny1, nx2, ny2 = obj.bbox
                x1, y1 = int(nx1 * w), int(ny1 * h)
                x2, y2 = int(nx2 * w), int(ny2 * h)
                if obj.distance_m < self._ALERT:
                    colour = (0, 0, 220)
                elif obj.distance_m < self._WARN:
                    colour = (0, 140, 255)
                else:
                    colour = (50, 200, 50)
                cv2.rectangle(rgb, (x1, y1), (x2, y2), colour, 2)
                label = f"{obj.class_name}  {obj.distance_m:.1f}m  {obj.direction}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(rgb, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(rgb, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # ── HUD overlay ──────────────────────────────────────────
            hud = (f"FPS {self._fps:.1f}   objects {len(scene.objects)}"
                   f"   closest {scene.closest_obstacle_m:.1f}m   go {scene.free_direction}")
            cv2.rectangle(rgb, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(rgb, hud, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # ── Depth heatmap ────────────────────────────────────────
            if frame.depth is not None:
                d = np.clip(frame.depth, 0.0, 10.0)
                d_norm = (d / 10.0 * 255).astype(np.uint8)
                depth_colour = cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)
                # INTER_CUBIC gives smoother upsampling from LiDAR's low native res
                depth_colour = cv2.resize(depth_colour, (w, h),
                                          interpolation=cv2.INTER_CUBIC)
                cv2.putText(depth_colour, "depth  (0 m \u2192 10 m)", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                combined = np.hstack([rgb, depth_colour])
            else:
                combined = rgb

            cv2.imshow("GroundSense", combined)

        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')

    def close(self) -> None:
        cv2.destroyAllWindows()


# ── WebSocket Server ─────────────────────────────────────────────────

class GroundSenseServer:
    """WebSocket server that processes iPhone frames and returns guidance."""

    def __init__(self, model_name: str = "yolo11n-seg.pt", device: str = "cpu",
                 visualize: bool = False, llm: str = "gemini", gemini_key: str = ""):
        self.pipeline = SegmentationPipeline(model_name=model_name, device=device)
        self.response_gen = ResponseGenerator(llm=llm, gemini_key=gemini_key)
        self.frame_count = 0
        self.last_warning_time = 0
        self.warning_cooldown = 1.5  # seconds between spoken warnings
        self.visualizer = Visualizer() if visualize else None
        # Persistent scene state — updated every frame, read by query handler
        self.last_scene: Optional[SceneState] = None

    async def handle_client(self, websocket):
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._process_frame(websocket, message)
                elif isinstance(message, str):
                    # Text message = voice query
                    await self._handle_query(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")

    async def _process_frame(self, websocket, data: bytes):
        """Process a binary frame packet."""
        self.frame_count += 1

        try:
            frame = Frame.from_bytes(data)
        except Exception as e:
            logger.error(f"Frame decode error: {e}")
            return

        # Run segmentation pipeline in a thread so YOLO's synchronous inference
        # (~50-300 ms) doesn't block the asyncio event loop and stall WebSocket I/O.
        loop = asyncio.get_event_loop()
        scene = await loop.run_in_executor(None, self.pipeline.process_frame, frame)

        # Persist scene for query handler
        self.last_scene = scene

        # Generate obstacle warning (with cooldown)
        now = time.time()
        response = {"type": "scene_update", "scene": scene.to_dict()}

        if now - self.last_warning_time > self.warning_cooldown:
            warning = self.response_gen.generate_obstacle_warning(scene)
            if warning:
                response["warning"] = warning
                self.last_warning_time = now

        await websocket.send(json.dumps(response))

        # Update live window (if --visualize was passed)
        if self.visualizer is not None:
            self.visualizer.update(frame, scene)

        if self.frame_count % 30 == 0:
            logger.info(
                f"Processed {self.frame_count} frames | "
                f"Objects: {len(scene.objects)} | "
                f"Closest: {scene.closest_obstacle_m:.1f}m | "
                f"Free: {scene.free_direction}"
            )

    async def _handle_query(self, websocket, query_json: str):
        """Handle a voice query from the user using the last known scene state."""
        try:
            query_data = json.loads(query_json)
            query_text = query_data.get("query", "")
        except json.JSONDecodeError:
            query_text = query_json

        query_text = query_text.strip()
        logger.info(f"Voice query: '{query_text}'")

        if not query_text:
            return

        if self.last_scene is None:
            answer, source = "I haven't processed any frames yet. Please start the camera stream first.", "rule-based"
        else:
            answer, source = self.response_gen.answer_query(self.last_scene, query_text)

        logger.info(f"Query answer [{source}]: '{answer}'")
        response = {
            "type": "query_response",
            "query": query_text,
            "answer": answer,
            "source": source,
        }
        await websocket.send(json.dumps(response))



# ── Network interface detection ───────────────────────────────────────

def _get_local_ips() -> dict:
    """
    Return a dict of interface-name → IP for every active non-loopback interface.
    Used at startup so you can see exactly which address to type into the app.
    """
    import subprocess, re
    ips = {}
    try:
        out = subprocess.check_output(["ifconfig"], text=True, stderr=subprocess.DEVNULL)
        # Match blocks like "en0: ... inet 192.168.x.x"
        for block in re.split(r'\n(?=\S)', out):
            iface = re.match(r'^(\S+):', block)
            addr  = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', block)
            if iface and addr and not addr.group(1).startswith("127."):
                ips[iface.group(1)] = addr.group(1)
    except Exception:
        pass
    return ips


def _print_connection_guide(port: int):
    """Print a startup banner showing every reachable address and how to use them."""
    ips = _get_local_ips()

    # Classify each interface
    usb_entries    = []
    sharing_entries = []
    wifi_entries   = []
    other_entries  = []

    for iface, ip in ips.items():
        if iface.startswith("utun") or iface.startswith("lo"):
            continue
        if "bridge" in iface or iface in ("en5", "en6", "en7", "en8"):
            usb_entries.append((iface, ip))
        elif ip.startswith("192.168.2."):
            # macOS Internet Sharing always uses the 192.168.2.x subnet
            sharing_entries.append((iface, ip))
        elif iface.startswith("en0") or iface.startswith("en1"):
            wifi_entries.append((iface, ip))
        else:
            other_entries.append((iface, ip))

    logger.info("═" * 62)
    logger.info("  GroundSense — connection options")
    logger.info("═" * 62)

    if sharing_entries:
        for iface, ip in sharing_entries:
            logger.info(f"  ★  ws://{ip}:{port}   [{iface}]  ← Mac hotspot (RECOMMENDED)")
        logger.info("     No router, no data, no cable — use this!")

    if usb_entries:
        for iface, ip in usb_entries:
            logger.info(f"  ⚡  ws://{ip}:{port}   [{iface}]  ← USB cable")

    if wifi_entries:
        for iface, ip in wifi_entries:
            logger.info(f"  📶  ws://{ip}:{port}   [{iface}]  ← Wi-Fi")

    for iface, ip in other_entries:
        logger.info(f"      ws://{ip}:{port}   [{iface}]")

    if not ips:
        logger.info(f"  (no active interfaces found — check your network)")

    logger.info("")
    logger.info("  ── Option A: Mac hotspot  (no Wi-Fi, no data, no cable) ──")
    logger.info("     Mac → System Settings → General → Sharing")
    logger.info("     → Internet Sharing → share via Wi-Fi → turn ON")
    logger.info("     iPhone joins the Mac's Wi-Fi network")
    logger.info(f"     Server address: ws://192.168.2.1:{port}")
    logger.info("")
    logger.info("  ── Option B: USB cable  (fastest, zero data) ──")
    logger.info("     iPhone → Settings → Personal Hotspot → ON")
    logger.info("     Connect cable — use the ⚡ address above")
    logger.info("     Note: the WebSocket is local; no cellular data is used")
    logger.info("═" * 62)


# ── Entry point ───────────────────────────────────────────────────────

async def _serve(host: str, port: int, server: "GroundSenseServer",
                 stop_event: asyncio.Event):
    async with websockets.serve(
        server.handle_client,
        host,
        port,
        max_size=10 * 1024 * 1024,
    ):
        _print_connection_guide(port)
        logger.info("Server ready — waiting for iPhone connection...")
        await stop_event.wait()


def main(host: str, port: int, model: str, device: str, visualize: bool,
         llm: str, gemini_key: str):
    server = GroundSenseServer(
        model_name=model, device=device, visualize=visualize,
        llm=llm, gemini_key=gemini_key,
    )
    logger.info(f"Starting GroundSense server on ws://{host}:{port}")
    logger.info(f"Model: {model} | Device: {device} | LLM: {llm}"
                + (" | Visualizer: ON  (press q to quit)" if visualize else ""))

    if visualize:
        # ── asyncio runs in a background thread; main thread owns OpenCV ──
        loop = asyncio.new_event_loop()
        stop_event = asyncio.Event()

        def _run_loop():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_serve(host, port, server, stop_event))

        t = threading.Thread(target=_run_loop, daemon=True)
        t.start()

        # Main thread: OpenCV render loop (~60 Hz)
        vis = server.visualizer
        try:
            while True:
                if not vis.render():          # returns False when 'q' pressed
                    break
                time.sleep(0.016)
        finally:
            loop.call_soon_threadsafe(stop_event.set)
            vis.close()
            t.join(timeout=3)
    else:
        asyncio.run(_serve_forever(host, port, server))


async def _serve_forever(host: str, port: int, server: "GroundSenseServer"):
    async with websockets.serve(
        server.handle_client,
        host,
        port,
        max_size=10 * 1024 * 1024,
    ):
        _print_connection_guide(port)
        logger.info("Server ready — waiting for iPhone connection...")
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GroundSense Backend Server")
    parser.add_argument("--host",   default="0.0.0.0",       help="Bind address")
    parser.add_argument("--port",   type=int, default=8765,   help="WebSocket port")
    parser.add_argument("--model",  default="yolo11n-seg.pt", help="YOLO model name")
    parser.add_argument("--device", default=None,
                        help="Inference device: cuda | mps | cpu (auto-detected if omitted)")
    parser.add_argument("--visualize", action="store_true",
                        help="Open a live OpenCV window (requires opencv-python, not headless)")
    parser.add_argument("--llm", default="gemini", choices=["gemini", "none"],
                        help="Query engine: gemini (default) | none (rule-based)")
    parser.add_argument("--gemini-key", default="",
                        help="Gemini API key (overrides GEMINI_API_KEY env var)")
    args = parser.parse_args()

    if args.device is None:
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
        logger.info(f"Auto-selected device: {args.device}")

    main(args.host, args.port, args.model, args.device, args.visualize,
         args.llm, args.gemini_key)
