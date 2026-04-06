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
import errno
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
try:
    import qrcode as _qrcode_mod
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("groundsense")


class ServerStartupError(RuntimeError):
    """Raised when the WebSocket server cannot start cleanly."""


def _wrap_server_startup_error(host: str, port: int, exc: OSError) -> Exception:
    """Convert low-level bind errors into actionable startup messages."""
    if exc.errno == errno.EADDRINUSE:
        return ServerStartupError(
            f"Port {port} is already in use on {host}. "
            f"Stop the existing listener or rerun with --port <free-port>."
        )
    return exc


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
    source: str = "yolo"       # "yolo" | "gdino"
    contour: Optional[list] = None  # [(nx, ny), ...] normalised portrait points

@dataclass
class PathArrow:
    """A candidate free-space corridor inferred from LiDAR depth only."""
    direction: str
    center_x: float            # 0..1, user-facing left→right guidance axis
    angle_deg: float
    clearance_m: float
    width_m: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "center_x": round(self.center_x, 3),
            "angle_deg": round(self.angle_deg, 1),
            "clearance_m": round(self.clearance_m, 2),
            "width_m": round(self.width_m, 2),
            "confidence": round(self.confidence, 2),
        }


@dataclass
class PathPlan:
    """Primary walking recommendation plus a few alternative free-space arrows."""
    direction: str = "center"
    center_x: float = 0.5
    angle_deg: float = 0.0
    clearance_m: float = 0.0
    width_m: float = 0.0
    confidence: float = 0.0
    arrows: list[PathArrow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "center_x": round(self.center_x, 3),
            "angle_deg": round(self.angle_deg, 1),
            "clearance_m": round(self.clearance_m, 2),
            "width_m": round(self.width_m, 2),
            "confidence": round(self.confidence, 2),
            "arrows": [arrow.to_dict() for arrow in self.arrows],
        }


@dataclass 
class SceneState:
    """Current understanding of the scene."""
    objects: list = field(default_factory=list)
    free_direction: str = "center"  # safest direction to walk
    closest_obstacle_m: float = float("inf")
    timestamp: float = 0.0
    path: PathPlan = field(default_factory=PathPlan)

    def to_dict(self) -> dict:
        return {
            "objects": [
                {
                    "class": o.class_name,
                    "track_id": o.track_id,
                    "distance_m": round(o.distance_m, 2),
                    "direction": o.direction,
                    "confidence": round(o.confidence, 2),
                    "source": o.source,
                    "bbox": list(o.bbox),
                }
                for o in self.objects
            ],
            "free_direction": self.free_direction,
            "closest_obstacle_m": round(self.closest_obstacle_m, 2),
            "timestamp": self.timestamp,
            "path": self.path.to_dict(),
        }


# ── Segmentation Pipeline ───────────────────────────────────────────

class SegmentationPipeline:
    """
    Primary pipeline: YOLO-seg for instance segmentation + tracking.
    Fuses with LiDAR depth for distance estimation.
    """

    MIN_WALK_WIDTH_M = 0.75
    MAX_PATH_DEPTH_M = 5.0
    MIN_PATH_DEPTH_M = 0.35
    PATH_DISPLAY_COUNT = 3
    MAX_OBSTACLE_PROJECTION = 0.18

    def __init__(self, model_name: str = "yolo26s-seg.pt", device: str = "cpu"):
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model: {model_name} on {device}")
        self.model = YOLO(model_name)
        self.device = device
        logger.info("Model loaded successfully")

    @staticmethod
    def _direction_from_center(center_x: float) -> str:
        if center_x < 0.38:
            return "left"
        if center_x > 0.62:
            return "right"
        return "center"

    @staticmethod
    def _fallback_center(direction: str) -> float:
        return {"left": 0.24, "center": 0.5, "right": 0.76}.get(direction, 0.5)

    @staticmethod
    def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start = None
        for idx, is_on in enumerate(np.r_[mask.astype(np.uint8), 0]):
            if is_on and start is None:
                start = idx
                continue
            if start is None or is_on:
                continue
            runs.append((start, idx))
            start = None
        return runs

    def _depth_intrinsics(
        self,
        metadata: Optional[dict],
        depth_width: int,
        planner_width: int,
    ) -> tuple[float, float]:
        intrinsics = metadata.get("intrinsics") if metadata else None
        rgb_width = float(metadata.get("rgbWidth", depth_width)) if metadata else float(depth_width)

        if isinstance(intrinsics, list) and len(intrinsics) >= 3 and rgb_width > 0:
            fx = float(intrinsics[0]) * (depth_width / rgb_width)
            cx = float(intrinsics[2]) * (depth_width / rgb_width)
        else:
            # Reasonable fallback when running offline without camera intrinsics.
            fx = depth_width * 1.1
            cx = depth_width / 2.0

        scale = planner_width / max(depth_width, 1)
        return fx * scale, cx * scale

    def _path_arrow(
        self,
        center_col: float,
        planner_width: int,
        fx_small: float,
        cx_small: float,
        clearance_m: float,
        width_m: float,
        confidence: float,
    ) -> PathArrow:
        denom = max(planner_width - 1, 1)
        center_x = float(np.clip(center_col / denom, 0.0, 1.0))
        angle_deg = float(np.degrees(np.arctan2(center_col - cx_small, max(fx_small, 1e-6))))
        return PathArrow(
            direction=self._direction_from_center(center_x),
            center_x=center_x,
            angle_deg=angle_deg,
            clearance_m=float(clearance_m),
            width_m=float(width_m),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

    def plan_path(
        self,
        depth: Optional[np.ndarray],
        metadata: Optional[dict] = None,
        objects: Optional[list] = None,
    ) -> PathPlan:
        """
        Lightweight LiDAR-only corridor planner.

        The depth map is reduced into:
          1. a forward-clearance profile, and
          2. a near-field obstacle projection built row-by-row relative to the
             local "horizon" depth at each scanline.

        This avoids steering toward columns that look far away only because the
        obstacle occupies the lower part of the image while open space exists
        above it.
        """
        if depth is None:
            fallback_dir = self._fallback_direction_from_objects(objects or [])
            return PathPlan(
                direction=fallback_dir,
                center_x=self._fallback_center(fallback_dir),
                confidence=0.0,
            )

        dh, dw = depth.shape
        planner_w = min(96, max(48, dw))
        planner_h = max(48, int(round(dh * planner_w / max(dw, 1))))
        depth_small = cv2.resize(depth, (planner_w, planner_h), interpolation=cv2.INTER_AREA)

        row_start = int(planner_h * 0.34)
        row_end = max(row_start + 16, int(planner_h * 0.96))
        roi = depth_small[row_start:row_end, :]
        valid = (roi > self.MIN_PATH_DEPTH_M) & (roi < self.MAX_PATH_DEPTH_M)

        if int(valid.sum()) < planner_w:
            return PathPlan(direction="center", center_x=0.5, confidence=0.0)

        min_valid_per_col = max(4, roi.shape[0] // 6)
        clearance = np.zeros(planner_w, dtype=np.float32)
        for col in range(planner_w):
            vals = roi[:, col][valid[:, col]]
            if len(vals) >= min_valid_per_col:
                clearance[col] = float(np.percentile(vals, 35))

        if np.count_nonzero(clearance) == 0:
            return PathPlan(direction="center", center_x=0.5, confidence=0.0)

        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        clearance = np.convolve(clearance, kernel / kernel.sum(), mode="same")

        # Row-relative obstacle projection:
        # free ground / floor tends to follow the far envelope of each row,
        # while obstacles appear as columns that are significantly closer than
        # the row's local horizon depth.
        obstacle_projection = np.zeros(planner_w, dtype=np.float32)
        row_weights = np.linspace(0.65, 1.75, roi.shape[0], dtype=np.float32)
        min_valid_per_row = max(8, planner_w // 10)
        for ridx, (row, row_valid) in enumerate(zip(roi, valid)):
            vals = row[row_valid]
            if len(vals) < min_valid_per_row:
                continue
            row_horizon = float(np.percentile(vals, 78))
            block_cutoff = max(self.MIN_PATH_DEPTH_M + 0.1, row_horizon * 0.76)
            obstacle_row = row_valid & (row < block_cutoff)
            obstacle_projection[obstacle_row] += row_weights[ridx]

        total_weight = float(row_weights.sum())
        if total_weight > 0:
            obstacle_projection /= total_weight
        obstacle_projection = np.convolve(
            obstacle_projection,
            kernel / kernel.sum(),
            mode="same",
        )

        fx_small, cx_small = self._depth_intrinsics(metadata, dw, planner_w)

        base_free_mask = obstacle_projection < self.MAX_OBSTACLE_PROJECTION
        if base_free_mask.any():
            base_free_mask = cv2.morphologyEx(
                base_free_mask.astype(np.uint8)[np.newaxis, :],
                cv2.MORPH_CLOSE,
                np.ones((1, 5), dtype=np.uint8),
            )[0].astype(bool)

        thresholds = (1.0, 1.35, 1.8, 2.3, 3.0)
        candidates: list[tuple[float, PathArrow]] = []

        for target_depth in thresholds:
            min_cols = max(
                4,
                int(np.ceil(self.MIN_WALK_WIDTH_M * max(fx_small, 1e-6) / max(target_depth, 0.5))),
            )

            free_mask = (base_free_mask & (clearance >= target_depth)).astype(np.uint8)
            if free_mask.any():
                free_mask = cv2.morphologyEx(
                    free_mask[np.newaxis, :],
                    cv2.MORPH_CLOSE,
                    np.ones((1, 3), dtype=np.uint8),
                )[0]

            for start, end in self._mask_runs(free_mask):
                run_len = end - start
                if run_len >= min_cols:
                    run_clearance = float(np.percentile(clearance[start:end], 25))
                    run_obstacle = float(np.percentile(obstacle_projection[start:end], 75))
                    effective_depth = max(min(run_clearance, target_depth + 0.8), 0.8)
                    width_m = float(run_len * effective_depth / max(fx_small, 1e-6))
                    center_col = (start + end - 1) / 2.0
                    center_x = center_col / max(planner_w - 1, 1)
                    center_bias = 1.0 - min(abs(center_x - 0.5) / 0.5, 1.0)
                    width_bonus = min(width_m / (self.MIN_WALK_WIDTH_M * 1.8), 1.0)
                    depth_bonus = min(run_clearance / 3.0, 1.0)
                    obstacle_bonus = 1.0 - min(run_obstacle / self.MAX_OBSTACLE_PROJECTION, 1.0)
                    score = (
                        obstacle_bonus * 1.45
                        + width_bonus * 0.80
                        + depth_bonus * 0.55
                        + center_bias * 0.20
                    )
                    confidence = (
                        0.12
                        + obstacle_bonus * 0.53
                        + width_bonus * 0.20
                        + depth_bonus * 0.15
                    )
                    candidates.append((
                        score,
                        self._path_arrow(
                            center_col=center_col,
                            planner_width=planner_w,
                            fx_small=fx_small,
                            cx_small=cx_small,
                            clearance_m=run_clearance,
                            width_m=width_m,
                            confidence=confidence,
                        ),
                    ))

        # Deduplicate near-identical corridors found at different thresholds.
        deduped: list[tuple[float, PathArrow]] = []
        for score, arrow in sorted(candidates, key=lambda item: item[0], reverse=True):
            if any(abs(arrow.center_x - kept.center_x) < 0.08 for _, kept in deduped):
                continue
            deduped.append((score, arrow))
            if len(deduped) >= self.PATH_DISPLAY_COUNT:
                break

        if not deduped:
            fallback_runs = self._mask_runs(base_free_mask.astype(np.uint8))
            if fallback_runs:
                start, end = max(fallback_runs, key=lambda run: run[1] - run[0])
                best_col = int((start + end - 1) / 2.0)
                best_depth = float(np.percentile(clearance[start:end], 25))
                run_len = end - start
            else:
                best_col = int(np.argmin(obstacle_projection))
                best_depth = float(clearance[best_col])
                side_threshold = obstacle_projection[best_col] + 0.03
                left = best_col
                while left > 0 and obstacle_projection[left - 1] <= side_threshold:
                    left -= 1
                right = best_col
                while right + 1 < planner_w and obstacle_projection[right + 1] <= side_threshold:
                    right += 1
                run_len = right - left + 1
            width_m = float(run_len * max(best_depth, 0.75) / max(fx_small, 1e-6))
            best_arrow = self._path_arrow(
                center_col=float(best_col),
                planner_width=planner_w,
                fx_small=fx_small,
                cx_small=cx_small,
                clearance_m=best_depth,
                width_m=width_m,
                confidence=min(max((1.0 - obstacle_projection[best_col]) * 0.35, 0.05), 0.42),
            )
            return PathPlan(
                direction=best_arrow.direction,
                center_x=best_arrow.center_x,
                angle_deg=best_arrow.angle_deg,
                clearance_m=best_arrow.clearance_m,
                width_m=best_arrow.width_m,
                confidence=best_arrow.confidence,
                arrows=[best_arrow],
            )

        best_arrow = deduped[0][1]
        return PathPlan(
            direction=best_arrow.direction,
            center_x=best_arrow.center_x,
            angle_deg=best_arrow.angle_deg,
            clearance_m=best_arrow.clearance_m,
            width_m=best_arrow.width_m,
            confidence=best_arrow.confidence,
            arrows=[arrow for _, arrow in deduped],
        )

    def _fallback_direction_from_objects(self, objects: list) -> str:
        sector_min_dist = {"left": 10.0, "center": 10.0, "right": 10.0}
        for obj in objects:
            if obj.distance_m < sector_min_dist[obj.direction]:
                sector_min_dist[obj.direction] = obj.distance_m
        return max(sector_min_dist, key=sector_min_dist.get)

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

                # ── Extract mask contour for visualizer overlay ──────
                contour_pts = None
                if masks is not None:
                    raw_mask = masks[i].data[0].cpu().numpy()  # float32 [0,1]
                    mh, mw = raw_mask.shape
                    bin_mask = (raw_mask > 0.5).astype(np.uint8)
                    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        biggest = max(cnts, key=cv2.contourArea)
                        eps = 0.02 * cv2.arcLength(biggest, True)
                        approx = cv2.approxPolyDP(biggest, eps, True)
                        # Normalise to [0,1] in landscape frame
                        contour_pts = [(float(p[0][0]) / mw,
                                        float(p[0][1]) / mh)
                                       for p in approx]

                obj = DetectedObject(
                    class_name=class_name,
                    track_id=track_id,
                    confidence=conf,
                    distance_m=distance,
                    direction=direction,
                    bbox=(x1, y1, x2, y2),
                    mask_area_ratio=(x2 - x1) * (y2 - y1),
                    contour=contour_pts,
                )
                scene.objects.append(obj)

            # Sort objects closest first
            scene.objects.sort(key=lambda o: o.distance_m)
            if scene.objects:
                scene.closest_obstacle_m = scene.objects[0].distance_m

        # ── Free-space estimation (LiDAR corridor planner primary) ────────────
        scene.path = self.plan_path(depth, frame.metadata, scene.objects)
        scene.free_direction = scene.path.direction
        return scene

    def _estimate_free_direction_lidar(
        self,
        depth: Optional[np.ndarray],
        objects: list,
    ) -> str:
        return self.plan_path(depth, None, objects).direction


# ── Open-Vocabulary Pipeline (Grounding DINO + MobileSAM) ────────────

class OpenVocabPipeline:
    """
    Open-vocabulary object detection via Grounding DINO + MobileSAM.

    Grounding DINO detects any object described in plain text (e.g.
    "wheelchair", "service dog", "wet floor sign") — not limited to COCO
    classes.  MobileSAM refines each detection to a precise segmentation
    mask so depth fusion is as accurate as the YOLO path.

    This pipeline is **throttled**: it runs every `interval` frames and
    caches its result, so it blends into the 20-fps YOLO stream without
    stalling the event loop.

    Installation (one-time):
        pip install transformers torch          # Grounding DINO (auto-downloads ~300 MB)
        pip install mobile-sam                  # Optional — precise masks
        # MobileSAM weights (place next to server.py):
        #   wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

    Activation:
        Send the WebSocket text message {"type": "set_targets", "objects": ["wheelchair"]}
        The pipeline is dormant (zero overhead) when target_objects is empty.
    """

    GDINO_MODEL     = "IDEA-Research/grounding-dino-tiny"
    BOX_THRESHOLD   = 0.42   # raised — GDINO hallucinates below ~0.40
    TEXT_THRESHOLD  = 0.25
    SAM_WEIGHTS     = "mobile_sam.pt"
    FASTSAM_MODEL   = "FastSAM-s.pt"  # auto-downloaded by ultralytics on first use

    def __init__(self, device: str = "cpu", interval: int = 5,
                 sam_interval: int = 1):
        """
        device       : "cpu" | "cuda" | "mps"
        interval     : run Grounding DINO every N frames (default 5 ≈ 4 fps
                       at a 20-fps YOLO stream).  Cached results fill the gaps.
        sam_interval : run FastSAM/SAM every N *GDINO calls* (default 1 = every
                       call).  Set to 2-4 to reduce GPU load; bbox depth is used
                       on skipped calls.
        """
        self.device       = device
        self.interval     = interval
        self.sam_interval = sam_interval

        self.target_objects: list[str] = []
        self._cached: list[DetectedObject] = []
        self._frame_count = 0
        self._sam_call_count = 0   # counts GDINO calls, gates SAM runs

        # Lazy-loaded — avoid paying the import cost unless the feature is used
        self._processor     = None
        self._gdino_model   = None
        self._fastsam_model = None   # FastSAM (primary segmenter — ultralytics)
        self._sam_pred      = None   # MobileSAM (fallback — needs mobile_sam.pt)
        self._loaded          = False
        self._load_err: Optional[str] = None
        self._load_err_logged = False   # print the missing-dep warning only once

    # ── Public API ────────────────────────────────────────────────────

    def set_targets(self, objects: list[str]) -> None:
        """Update detection targets at runtime (thread-safe for simple list replace)."""
        self.target_objects = [o.strip().lower() for o in objects if o.strip()]
        self._cached = []          # invalidate cache so next frame re-detects
        logger.info(f"[OpenVocab] targets → {self.target_objects}")

    def process_frame(self, frame: "Frame") -> list[DetectedObject]:
        """
        Run Grounding DINO on this frame (throttled), return DetectedObjects.
        Returns the cached result on off-frames.  Returns [] when no targets
        are set so there is zero overhead in the default YOLO-only mode.
        """
        if not self.target_objects:
            return []

        self._frame_count += 1

        # Off-frame: return stale cache rather than running inference
        if self._frame_count % self.interval != 0:
            return self._cached

        if not self._ensure_loaded():
            return []

        try:
            result = self._run_gdino(frame)
            self._cached = result
            return result
        except Exception as exc:
            logger.warning(f"[OpenVocab] inference error: {exc}")
            return self._cached     # serve stale on error

    # ── Model loading ─────────────────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._load_err:
            if not self._load_err_logged:
                logger.warning(
                    f"[OpenVocab] model unavailable ({self._load_err}). "
                    "Install with:  pip install transformers torch\n"
                    "  All open-vocab inference will be skipped until the server restarts."
                )
                self._load_err_logged = True
            return False
        try:
            import torch
            from transformers import (
                AutoProcessor,
                AutoModelForZeroShotObjectDetection,
            )

            logger.info(
                "[OpenVocab] Loading Grounding DINO "
                f"({self.GDINO_MODEL}) — first use, may take a moment …"
            )
            self._processor = AutoProcessor.from_pretrained(self.GDINO_MODEL)
            self._gdino_model = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(self.GDINO_MODEL)
                .to(self.device)
                .eval()
            )
            logger.info("[OpenVocab] Grounding DINO ready")

            # MobileSAM — optional, gracefully skipped if missing
            self._try_load_sam()

            self._loaded = True
            return True

        except ImportError as exc:
            self._load_err = str(exc)
            logger.warning(
                f"[OpenVocab] Missing dependency: {exc}. "
                "Install with: pip install transformers torch"
            )
            return False
        except Exception as exc:
            self._load_err = str(exc)
            logger.error(f"[OpenVocab] Model load failed: {exc}")
            return False

    def _try_load_sam(self) -> None:
        """
        Load a segmentation model for precise per-object masks (used for better
        depth estimation from LiDAR).  Priority order:

        1. FastSAM-s  — ultralytics is already installed for YOLO; FastSAM-s.pt
                        (~23 MB) auto-downloads on first use.  ~60 ms on CPU,
                        ~8 ms on MPS/CUDA.
        2. MobileSAM  — needs `pip install mobile-sam` + mobile_sam.pt weights.
        3. Bbox-only  — falls back to the bounding-box region average (no mask).
        """
        # ── 1. FastSAM (preferred — ultralytics already present) ─────
        try:
            from ultralytics import FastSAM as _FastSAM
            logger.info(
                f"[OpenVocab] Loading FastSAM ({self.FASTSAM_MODEL}) "
                "— downloads ~23 MB on first run …"
            )
            self._fastsam_model = _FastSAM(self.FASTSAM_MODEL)
            logger.info("[OpenVocab] FastSAM ready — precise mask depth fusion active")
            return
        except Exception as exc:
            logger.info(f"[OpenVocab] FastSAM unavailable ({exc}), trying MobileSAM …")

        # ── 2. MobileSAM (fallback) ───────────────────────────────────
        if not os.path.exists(self.SAM_WEIGHTS):
            logger.info(
                f"[OpenVocab] {self.SAM_WEIGHTS} not found — "
                "falling back to bbox-based depth estimation. "
                "For precise masks: pip install mobile-sam + download mobile_sam.pt"
            )
            return
        try:
            from mobile_sam import sam_model_registry, SamPredictor
            sam = sam_model_registry["vit_t"](checkpoint=self.SAM_WEIGHTS)
            sam.to(self.device)
            self._sam_pred = SamPredictor(sam)
            logger.info("[OpenVocab] MobileSAM ready — precise mask depth fusion active")
        except ImportError:
            logger.info(
                "[OpenVocab] mobile-sam not installed — using bbox depth estimation. "
                "Install with: pip install mobile-sam"
            )

    # ── Inference ─────────────────────────────────────────────────────

    def _run_gdino(self, frame: "Frame") -> list[DetectedObject]:
        import torch
        from PIL import Image as PILImage

        # BGR → RGB PIL for the processor
        rgb     = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2RGB)
        h, w    = rgb.shape[:2]
        pil_img = PILImage.fromarray(rgb)

        # Grounding DINO prompt format: "cat . dog . wheelchair ."
        prompt = " . ".join(self.target_objects) + " ."

        inputs = self._processor(
            images=pil_img, text=prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._gdino_model(**inputs)

        # transformers < ~4.44 used `box_threshold`; newer versions renamed it to
        # `threshold`.  Try both so the server works across versions.
        try:
            raw = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD,
                target_sizes=[(h, w)],
            )
        except TypeError:
            raw = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.BOX_THRESHOLD,
                target_sizes=[(h, w)],
            )
        detections = raw[0]

        boxes  = detections["boxes"].cpu().numpy()   # (N, 4) xyxy pixel
        scores = detections["scores"].cpu().numpy()  # (N,)

        # `labels` can be a list of strings (newer transformers) or a list of
        # ints / tensors (older) that need batch-decoding.
        raw_labels = detections.get("labels", [])
        if raw_labels and isinstance(raw_labels[0], str):
            labels = raw_labels
        else:
            # Decode token IDs → text
            labels = [
                self._processor.decode(torch.tensor([tok]), skip_special_tokens=True).strip()
                for tok in raw_labels
            ]

        depth = frame.depth

        # ── FastSAM: run ONCE per GDINO call, not once per box ───────
        # Gated by sam_interval so SAM can run less often than GDINO.
        self._sam_call_count += 1
        run_sam = (self._sam_call_count % self.sam_interval == 0)
        frame_masks_np = None   # (N_masks, Hm, Wm) from FastSAM, reused per box
        if run_sam and self._fastsam_model is not None and len(boxes) > 0:
            frame_masks_np = self._fastsam_run_frame(rgb)
        objects: list[DetectedObject] = []

        for i in range(len(boxes)):
            x1p, y1p, x2p, y2p = boxes[i]

            # Normalised coords
            nx1 = float(x1p) / w
            ny1 = float(y1p) / h
            nx2 = float(x2p) / w
            ny2 = float(y2p) / h
            cx  = (nx1 + nx2) / 2

            direction = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")

            # ── Depth estimation (mask → LiDAR sampling) ─────────────
            distance = float("inf")
            seg_mask = None   # reused below for contour — avoids double inference
            if depth is not None:
                dh, dw = depth.shape

                # FastSAM: match best pre-run mask to this box (pure numpy, cheap)
                # MobileSAM: per-box GPU call (fallback if FastSAM not loaded)
                if frame_masks_np is not None:
                    seg_mask = self._fastsam_match_box(
                        frame_masks_np, boxes[i], rgb.shape[:2]
                    )
                elif self._fastsam_model is not None and run_sam:
                    # FastSAM loaded but frame_masks_np is None only if 0 boxes
                    pass
                elif self._sam_pred is not None and run_sam:
                    seg_mask = self._sam_segment(rgb, boxes[i])

                if seg_mask is not None:
                    mask_r = cv2.resize(
                        seg_mask.astype(np.uint8), (dw, dh),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    vals = depth[mask_r > 0]
                    vals = vals[(vals > 0.1) & (vals < 10.0)]
                    if len(vals) > 0:
                        distance = float(np.median(vals))

                if distance == float("inf"):
                    # Bbox-based fallback — used when SAM is skipped or unavailable
                    ix1 = max(0, int(nx1 * dw))
                    iy1 = max(0, int(ny1 * dh))
                    ix2 = min(dw, int(nx2 * dw))
                    iy2 = min(dh, int(ny2 * dh))
                    if ix2 > ix1 and iy2 > iy1:
                        roi  = depth[iy1:iy2, ix1:ix2]
                        vals = roi[(roi > 0.1) & (roi < 10.0)]
                        if len(vals) > 0:
                            distance = float(np.median(vals))

            # ── Extract contour from SAM mask for visualizer ─────────
            # seg_mask already computed above — no second inference needed
            gdino_contour = None
            if seg_mask is not None:
                bin_mask = seg_mask.astype(np.uint8)
                mh, mw = bin_mask.shape
                cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    biggest = max(cnts, key=cv2.contourArea)
                    eps = 0.02 * cv2.arcLength(biggest, True)
                    approx = cv2.approxPolyDP(biggest, eps, True)
                    gdino_contour = [(float(p[0][0]) / mw,
                                     float(p[0][1]) / mh)
                                    for p in approx]

            objects.append(DetectedObject(
                class_name    = str(labels[i]),
                track_id      = -1,           # no tracker on open-vocab path
                confidence    = float(scores[i]),
                distance_m    = distance,
                direction     = direction,
                bbox          = (nx1, ny1, nx2, ny2),
                mask_area_ratio = (nx2 - nx1) * (ny2 - ny1),
                source        = "gdino",
                contour       = gdino_contour,
            ))

        objects.sort(key=lambda o: o.distance_m)
        if objects:
            logger.info(
                "[OpenVocab] detected: "
                + ", ".join(
                    f"{o.class_name} {o.distance_m:.1f}m {o.direction}"
                    for o in objects
                )
            )
        return objects

    def _fastsam_run_frame(
        self, rgb: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Run FastSAM **once** on the full frame and return all raw masks as a
        numpy array (N, Hm, Wm) float32 [0,1].  Called once per GDINO
        inference; individual boxes are matched cheaply in `_fastsam_match_box`.
        """
        try:
            results = self._fastsam_model(
                rgb,
                device=self.device,
                retina_masks=True,
                imgsz=640,
                conf=0.3,
                iou=0.9,
                verbose=False,
            )
            if not results or results[0].masks is None:
                return None
            return results[0].masks.data.cpu().numpy()  # (N, Hm, Wm)
        except Exception as exc:
            logger.warning(f"[OpenVocab] FastSAM error: {exc}")
            return None

    def _fastsam_match_box(
        self,
        frame_masks_np: np.ndarray,  # (N, Hm, Wm)
        box_xyxy: np.ndarray,        # pixel coords in original frame
        frame_hw: tuple,             # (H, W) of original RGB frame
    ) -> Optional[np.ndarray]:
        """
        Pure-numpy IoU matching: pick the pre-run FastSAM mask that best
        overlaps a GDINO box.  No GPU call — runs in microseconds per box.
        Returns a boolean mask (H, W) at original frame resolution, or None.
        """
        h, w = frame_hw
        x1p, y1p, x2p, y2p = box_xyxy
        best_mask, best_iou = None, 0.0

        for raw_mask in frame_masks_np:
            mh, mw = raw_mask.shape
            bx1 = max(0, int(x1p * mw / w))
            by1 = max(0, int(y1p * mh / h))
            bx2 = min(mw, int(x2p * mw / w))
            by2 = min(mh, int(y2p * mh / h))

            bbox_region = np.zeros_like(raw_mask, dtype=bool)
            bbox_region[by1:by2, bx1:bx2] = True
            mask_bool = raw_mask > 0.5

            intersection = float((mask_bool & bbox_region).sum())
            union        = float((mask_bool | bbox_region).sum())
            iou = intersection / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou  = iou
                best_mask = mask_bool

        if best_mask is not None and best_iou > 0.1:
            return cv2.resize(
                best_mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return None

    def _sam_segment(
        self, rgb: np.ndarray, box_xyxy: np.ndarray
    ) -> Optional[np.ndarray]:
        """Run MobileSAM with a bbox prompt → boolean mask (H, W)."""
        try:
            self._sam_pred.set_image(rgb)
            masks, _, _ = self._sam_pred.predict(
                box=box_xyxy[None],      # (1, 4)
                multimask_output=False,
            )
            return masks[0]              # (H, W) bool
        except Exception as exc:
            logger.warning(f"[OpenVocab] MobileSAM error: {exc}")
            return None


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
            path_width = scene.path.width_m
            path_clearance = scene.path.clearance_m
            if scene.path.confidence < 0.18 or path_width < SegmentationPipeline.MIN_WALK_WIDTH_M * 0.85:
                return "I don't see a confidently walkable gap yet. Pause and scan again."
            if closest > self.WARN_DISTANCE:
                return (
                    f"The clearest path is {free}, about {path_width:.1f} metres wide "
                    f"with {path_clearance:.1f} metres of clearance."
                )
            else:
                return (
                    f"Caution — closest obstacle is {closest:.1f} metres. "
                    f"The best walkable path is {free}, about {path_width:.1f} metres wide."
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

    _DISPLAY_H = 640   # target display height per panel (px)

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None          # (frame, scene) set by asyncio thread
        self._fps_time = time.time()
        self._fps = 0.0
        self._frame_idx = 0
        self._window_ready = False

    def _path_colour(self, confidence: float, width_m: float) -> tuple[int, int, int]:
        if confidence < 0.18 or width_m < SegmentationPipeline.MIN_WALK_WIDTH_M * 0.85:
            return (0, 140, 255)
        return (80, 255, 120)

    def _draw_path_guidance(self, canvas: np.ndarray, path: PathPlan) -> None:
        if not path.arrows:
            return

        h, w = canvas.shape[:2]
        origin = (w // 2, h - 54)

        for idx, arrow in enumerate(reversed(path.arrows)):
            scale = 0.58 if idx > 0 else 0.72
            tip_x = int(np.clip(arrow.center_x, 0.08, 0.92) * w)
            tip_y = int(h * (0.64 - idx * 0.05))
            colour = self._path_colour(arrow.confidence, arrow.width_m)
            thickness = 4 if idx == len(path.arrows) - 1 else 2
            cv2.arrowedLine(
                canvas,
                origin,
                (tip_x, tip_y),
                colour,
                thickness,
                tipLength=0.22 * scale,
            )
            if idx == len(path.arrows) - 1:
                label = (
                    f"path {arrow.direction}  "
                    f"{arrow.clearance_m:.1f}m clear  "
                    f"{arrow.width_m:.1f}m wide"
                )
                cv2.rectangle(canvas, (12, h - 68), (min(w - 12, 340), h - 34), (0, 0, 0), -1)
                cv2.putText(
                    canvas,
                    label,
                    (18, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.53,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    def update(self, frame: "Frame", scene: "SceneState") -> None:
        """Called from the asyncio thread — just stores the latest data."""
        with self._lock:
            self._pending = (frame, scene)

    def render(self) -> bool:
        """
        Called from the MAIN thread only.
        Draws the latest frame and pumps the OpenCV event loop.
        Returns False when the user presses 'q'.

        Rotates the frame only if it's not marked as 'landscape' (default for iPhone
        which delivers landscape pixels that need a 90° CW turn for portrait display).
        """
        with self._lock:
            data = self._pending
            self._pending = None

        if not self._window_ready:
            cv2.namedWindow("GroundSense", cv2.WINDOW_NORMAL)
            # Portrait panel × 2 side-by-side
            cv2.resizeWindow("GroundSense", self._DISPLAY_H * 2, int(self._DISPLAY_H * 1.5))
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

            # ── Handle rotation based on metadata ────────────────────
            orientation = frame.metadata.get("orientation", "portrait")
            if orientation == "landscape":
                rgb = frame.rgb
            else:
                # Default iPhone behaviour: rotate 90° CW to match screen
                rgb = cv2.rotate(frame.rgb, cv2.ROTATE_90_CLOCKWISE)

            # Scale to a fixed display height so the window stays manageable
            h_orig, w_orig = rgb.shape[:2]
            dh = self._DISPLAY_H
            dw = int(w_orig * dh / h_orig)
            rgb = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
            h, w = rgb.shape[:2]   # now equals (dh, dw)

            # ── Segmentation mask overlays ───────────────────────────
            # Draw filled semi-transparent polygons before the bbox lines so the
            # boxes are always readable on top.
            # Coordinate transform (portrait display after 90° CW rotation):
            #   landscape (nx, ny) → portrait (px, py) = (1-ny, nx)
            overlay = rgb.copy()
            for obj in scene.objects:
                if not obj.contour:
                    continue
                if orientation == "landscape":
                    pts = [(int(nx * w), int(ny * h)) for nx, ny in obj.contour]
                else:
                    pts = [(int((1.0 - ny) * w), int(nx * h))
                           for nx, ny in obj.contour]
                poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                if obj.source == "gdino":
                    fill_colour = (180, 0, 180)   # purple for open-vocab
                elif obj.distance_m < self._ALERT:
                    fill_colour = (0, 0, 200)
                elif obj.distance_m < self._WARN:
                    fill_colour = (0, 120, 255)
                else:
                    fill_colour = (30, 160, 30)
                cv2.fillPoly(overlay, [poly], fill_colour)
            cv2.addWeighted(overlay, 0.35, rgb, 0.65, 0, rgb)

            # ── YOLO bounding boxes + labels ─────────────────────────
            # Original bbox is in landscape-normalised coords (nx1,ny1,nx2,ny2).
            # After ROTATE_90_CLOCKWISE: (nx,ny) → (1-ny, nx).
            # New bbox corners: (1-ny2, nx1, 1-ny1, nx2).
            for obj in scene.objects:
                nx1, ny1, nx2, ny2 = obj.bbox
                if orientation == "landscape":
                    rx1, ry1, rx2, ry2 = nx1, ny1, nx2, ny2
                else:
                    rx1, ry1, rx2, ry2 = 1.0 - ny2, nx1, 1.0 - ny1, nx2
                x1, y1 = int(rx1 * w), int(ry1 * h)
                x2, y2 = int(rx2 * w), int(ry2 * h)
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

            # ── Path planner overlay ────────────────────────────────
            self._draw_path_guidance(rgb, scene.path)

            # ── HUD overlay ──────────────────────────────────────────
            hud = (f"FPS {self._fps:.1f}   objects {len(scene.objects)}"
                   f"   closest {scene.closest_obstacle_m:.1f}m"
                   f"   go {scene.free_direction}"
                   f"   width {scene.path.width_m:.1f}m")
            cv2.rectangle(rgb, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(rgb, hud, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # ── Depth heatmap (also rotated to match) ────────────────
            if frame.depth is not None:
                d = np.clip(frame.depth, 0.0, 10.0)
                d_norm = (d / 10.0 * 255).astype(np.uint8)
                depth_colour = cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)
                if orientation != "landscape":
                    depth_colour = cv2.rotate(depth_colour, cv2.ROTATE_90_CLOCKWISE)
                depth_colour = cv2.resize(depth_colour, (dw, dh),
                                          interpolation=cv2.INTER_CUBIC)
                self._draw_path_guidance(depth_colour, scene.path)
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

    def __init__(self, model_name: str = "yolo26s-seg.pt", device: str = "cpu",
                 visualize: bool = False, llm: str = "gemini", gemini_key: str = "",
                 open_vocab: bool = True, gdino_interval: int = 5, sam_interval: int = 1):
        self.pipeline     = SegmentationPipeline(model_name=model_name, device=device)
        self.open_vocab   = OpenVocabPipeline(device=device, interval=gdino_interval,
                                              sam_interval=sam_interval)
        self.response_gen = ResponseGenerator(llm=llm, gemini_key=gemini_key)
        self._open_vocab_enabled = open_vocab
        # Active detection mode: "yolo" | "gdino" | "both"
        # Changed at runtime via {"type": "set_pipeline", "mode": "..."}
        self._pipeline_mode: str = "yolo"
        # Guard: prevents queuing multiple concurrent GDINO background tasks.
        # True while a background inference task is in flight.
        self._gdino_running: bool = False
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
                    # Peek at the JSON type to route correctly
                    try:
                        msg = json.loads(message)
                        if msg.get("type") == "set_targets":
                            await self._handle_set_targets(websocket, msg)
                        elif msg.get("type") == "set_pipeline":
                            await self._handle_set_pipeline(websocket, msg)
                        else:
                            await self._handle_query(websocket, message)
                    except json.JSONDecodeError:
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

        loop = asyncio.get_event_loop()
        mode = self._pipeline_mode   # snapshot — may change between awaits

        # ── Primary pipeline: YOLO ────────────────────────────────────
        # Runs synchronous inference in a thread so the event loop stays free.
        # Skipped in "gdino" mode to reduce CPU/GPU load.
        if mode in ("yolo", "both"):
            scene = await loop.run_in_executor(None, self.pipeline.process_frame, frame)
        else:
            # GDINO-only: build a blank scene and still estimate free direction
            # from the raw LiDAR depth map.
            scene = SceneState(timestamp=frame.timestamp)
            scene.path = self.pipeline.plan_path(frame.depth, frame.metadata, [])
            scene.free_direction = scene.path.direction

        # ── Secondary pipeline: Grounding DINO (non-blocking) ────────
        # GDINO inference takes ~500–2000 ms on CPU, far too slow to await
        # on every frame.  Instead:
        #   • Merge the *cached* result from the last completed inference
        #     into this frame's scene immediately (always fast — a list copy).
        #   • Every `interval` frames, if no inference is already running,
        #     fire a background asyncio Task that calls the executor and
        #     updates the cache when done.  The current frame never waits.
        if (self._open_vocab_enabled
                and mode in ("gdino", "both")
                and self.open_vocab.target_objects):

            # 1. Use last known result now (stale by at most interval frames)
            ov_objects = list(self.open_vocab._cached)
            if ov_objects:
                scene.objects.extend(ov_objects)
                scene.objects.sort(key=lambda o: o.distance_m)
                scene.closest_obstacle_m = scene.objects[0].distance_m

            # 2. Kick off fresh inference in the background if it's due
            # NOTE: _ensure_loaded() is intentionally NOT called here — it may
            # block for many seconds on first use (model download).  It is
            # called inside _run_gdino_background which runs in an executor
            # thread, so it never freezes the event loop.
            self.open_vocab._frame_count += 1
            if (self.open_vocab._frame_count % self.open_vocab.interval == 0
                    and not self._gdino_running
                    and not self.open_vocab._load_err):
                self._gdino_running = True
                asyncio.create_task(self._run_gdino_background(frame))

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
                f"Free: {scene.free_direction} | "
                f"Path: {scene.path.clearance_m:.1f}m clear / {scene.path.width_m:.1f}m wide"
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

    async def _handle_set_targets(self, websocket, msg: dict):
        """
        Handle {"type": "set_targets", "objects": ["wheelchair", "service dog"]}

        Updates the open-vocabulary pipeline's target list at runtime.
        Send an empty list to disable open-vocab detection and return to
        YOLO-only mode with zero overhead.
        """
        objects = msg.get("objects", [])
        if not isinstance(objects, list):
            await websocket.send(json.dumps({
                "type": "set_targets_ack",
                "ok": False,
                "error": "'objects' must be a JSON array of strings",
            }))
            return

        if not self._open_vocab_enabled:
            logger.warning(
                "[OpenVocab] set_targets received but open-vocab pipeline is disabled "
                "(start server without --no-open-vocab to enable)"
            )
            await websocket.send(json.dumps({
                "type": "set_targets_ack",
                "ok": False,
                "error": "open-vocab pipeline is disabled on this server",
                "targets": [],
            }))
            return

        self.open_vocab.set_targets(objects)
        logger.info(
            f"[OpenVocab] targets updated → {self.open_vocab.target_objects} "
            f"({'active' if objects else 'disabled — YOLO-only mode'})"
        )
        await websocket.send(json.dumps({
            "type": "set_targets_ack",
            "ok": True,
            "targets": self.open_vocab.target_objects,
        }))

    async def _handle_set_pipeline(self, websocket, msg: dict):
        """
        Handle {"type": "set_pipeline", "mode": "yolo" | "gdino" | "both"}

        Switches the active detection pipeline at runtime — no restart needed.

        "yolo"  — YOLO-only (default). Fast COCO-class detection + LiDAR depth.
        "gdino" — Grounding DINO only. Open-vocabulary user-specified targets,
                  no YOLO overhead.  Requires targets to be set via set_targets.
        "both"  — YOLO always-on plus GDINO for any set targets.
                  Highest coverage; highest CPU load.
        """
        mode = msg.get("mode", "yolo").lower().strip()
        valid = {"yolo", "gdino", "both"}

        if mode not in valid:
            await websocket.send(json.dumps({
                "type": "set_pipeline_ack",
                "ok": False,
                "error": f"Invalid mode '{mode}'. Must be one of: {sorted(valid)}",
            }))
            return

        if mode in ("gdino", "both") and not self._open_vocab_enabled:
            await websocket.send(json.dumps({
                "type": "set_pipeline_ack",
                "ok": False,
                "error": (
                    f"Mode '{mode}' requires the open-vocab pipeline, but it is "
                    "disabled on this server. Restart without --no-open-vocab."
                ),
            }))
            return

        prev = self._pipeline_mode
        self._pipeline_mode = mode
        logger.info(
            f"[Pipeline] mode {prev} → {mode}"
            + (f" | targets: {self.open_vocab.target_objects}" if mode != "yolo" else "")
        )
        await websocket.send(json.dumps({
            "type": "set_pipeline_ack",
            "ok": True,
            "mode": mode,
        }))

    async def _run_gdino_background(self, frame: "Frame") -> None:
        """
        Fire-and-forget background task for Grounding DINO inference.

        Runs `_ensure_loaded` + `_run_gdino` in a thread-pool executor so the
        event loop stays free — including the first call which may download
        the ~300 MB GDINO model.  When inference completes, the shared
        `open_vocab._cached` list is updated atomically (GIL-protected single
        assignment in CPython).  The `_gdino_running` guard prevents queueing
        multiple overlapping tasks.
        """
        loop = asyncio.get_event_loop()

        def _load_and_run(frame: "Frame"):
            """Called in the executor thread — safe to block."""
            if not self.open_vocab._ensure_loaded():
                return None
            return self.open_vocab._run_gdino(frame)

        try:
            logger.info(
                f"[OpenVocab] running GDINO inference "
                f"(frame {self.open_vocab._frame_count}, "
                f"targets={self.open_vocab.target_objects})"
            )
            result = await loop.run_in_executor(None, _load_and_run, frame)
            if result is None:
                return
            self.open_vocab._cached = result
            if result:
                logger.info(
                    "[OpenVocab] detected: "
                    + ", ".join(
                        f"{o.class_name} {o.distance_m:.1f}m {o.direction}"
                        for o in result
                    )
                )
            else:
                logger.info(
                    f"[OpenVocab] no detections for {self.open_vocab.target_objects} "
                    f"(box_thresh={self.open_vocab.BOX_THRESHOLD}) — "
                    "try a longer description, e.g. 'dog' not 'canine'"
                )
        except Exception as exc:
            logger.warning(f"[OpenVocab] background inference error: {exc}")
        finally:
            self._gdino_running = False


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


def _get_best_url(port: int) -> str:
    """Return the single best WebSocket URL to advertise (hotspot > USB > Wi-Fi)."""
    ips = _get_local_ips()
    best_ip: Optional[str] = None
    for iface, ip in ips.items():
        if ip.startswith("192.168.2."):          # Mac hotspot
            best_ip = ip; break
    if best_ip is None:
        for iface, ip in ips.items():
            if "bridge" in iface or iface in ("en5","en6","en7","en8"):  # USB
                best_ip = ip; break
    if best_ip is None:
        for iface, ip in ips.items():
            if iface.startswith("en0") or iface.startswith("en1"):  # Wi-Fi
                best_ip = ip; break
    if best_ip is None and ips:
        best_ip = next(iter(ips.values()))
    return f"ws://{best_ip}:{port}" if best_ip else ""


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

    # ── ASCII QR code for the best available address ─────────────────
    best_url = _get_best_url(port)
    if best_url and HAS_QRCODE:
        try:
            import io as _io
            qr = _qrcode_mod.QRCode(box_size=1, border=2)
            qr.add_data(best_url)
            qr.make(fit=True)
            buf = _io.StringIO()
            try:
                qr.print_ascii(out=buf, invert=True)
            except TypeError:
                # older qrcode versions don't have invert=
                qr.print_ascii(out=buf)
            lines = buf.getvalue().splitlines()
            print("", flush=True)
            print(f"  ┌─ Scan to connect ─────────────────────────────────┐", flush=True)
            print(f"  │  {best_url}", flush=True)
            print(f"  └───────────────────────────────────────────────────┘", flush=True)
            for line in lines:
                print("  " + line, flush=True)
            print("", flush=True)
        except Exception as e:
            print(f"\n  QR URL: {best_url}\n  (QR render failed: {e})\n", flush=True)
    elif best_url:
        print(f"\n  ┌──────────────────────────────────────────────────────┐", flush=True)
        print(f"  │  Connect URL: {best_url}", flush=True)
        print(f"  │  Install qrcode for a scannable QR: pip install 'qrcode[pil]'", flush=True)
        print(f"  └──────────────────────────────────────────────────────┘\n", flush=True)


# ── Entry point ───────────────────────────────────────────────────────

async def _serve(host: str, port: int, server: "GroundSenseServer",
                 stop_event: asyncio.Event, startup_ready=None):
    try:
        async with websockets.serve(
            server.handle_client,
            host,
            port,
            max_size=10 * 1024 * 1024,
        ):
            if startup_ready is not None:
                startup_ready.set()
            _print_connection_guide(port)
            logger.info("Server ready — waiting for iPhone connection...")
            await stop_event.wait()
    except OSError as exc:
        if startup_ready is not None:
            startup_ready.set()
        raise _wrap_server_startup_error(host, port, exc) from exc


def main(host: str, port: int, model: str, device: str, visualize: bool,
         llm: str, gemini_key: str,
         open_vocab: bool = True, gdino_interval: int = 5, sam_interval: int = 1):
    server = GroundSenseServer(
        model_name=model, device=device, visualize=visualize,
        llm=llm, gemini_key=gemini_key,
        open_vocab=open_vocab, gdino_interval=gdino_interval, sam_interval=sam_interval,
    )
    logger.info(f"Starting GroundSense server on ws://{host}:{port}")
    ov_status = (
        f"enabled (interval={gdino_interval} frames)" if open_vocab else "disabled"
    )
    logger.info(
        f"Model: {model} | Device: {device} | LLM: {llm} | OpenVocab: {ov_status}"
        + (" | Visualizer: ON  (press q to quit)" if visualize else "")
    )

    if visualize:
        # ── asyncio runs in a background thread; main thread owns OpenCV ──
        #
        # Python 3.9 bug: asyncio.Event() binds to the *current* loop at
        # construction time.  Creating it in the main thread (before the
        # background loop is set) attaches it to the wrong loop and raises
        # "Future attached to a different loop" at runtime.
        #
        # Fix: create stop_event *inside* the background thread after
        # set_event_loop(), then hand it back via a mutable list + a
        # threading.Event that gates the main thread until it's ready.
        loop = asyncio.new_event_loop()
        _ready      = threading.Event()   # set once startup succeeds or fails
        _stop_holder: list = []           # [asyncio.Event] — filled by thread
        _startup_error: list = []         # [Exception] — set if bind/startup fails

        def _run_loop():
            asyncio.set_event_loop(loop)
            stop_event = asyncio.Event()  # now bound to the correct loop
            _stop_holder.append(stop_event)
            try:
                loop.run_until_complete(
                    _serve(host, port, server, stop_event, startup_ready=_ready)
                )
            except Exception as exc:
                _startup_error.append(exc)
                _ready.set()              # unblock main thread on startup failure

        t = threading.Thread(target=_run_loop, daemon=True)
        t.start()
        _ready.wait()

        if _startup_error:
            raise SystemExit(str(_startup_error[0]))

        # Main thread: OpenCV render loop (~60 Hz)
        vis = server.visualizer
        try:
            while True:
                if not vis.render():      # returns False when 'q' pressed
                    break
                time.sleep(0.016)
        finally:
            loop.call_soon_threadsafe(_stop_holder[0].set)
            vis.close()
            t.join(timeout=3)
    else:
        try:
            asyncio.run(_serve_forever(host, port, server))
        except ServerStartupError as exc:
            raise SystemExit(str(exc)) from exc


async def _serve_forever(host: str, port: int, server: "GroundSenseServer"):
    try:
        async with websockets.serve(
            server.handle_client,
            host,
            port,
            max_size=10 * 1024 * 1024,
        ):
            _print_connection_guide(port)
            logger.info("Server ready — waiting for iPhone connection...")
            await asyncio.Future()
    except OSError as exc:
        raise _wrap_server_startup_error(host, port, exc) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GroundSense Backend Server")
    parser.add_argument("--host",   default="0.0.0.0",       help="Bind address")
    parser.add_argument("--port",   type=int, default=8765,   help="WebSocket port")
    parser.add_argument("--model",  default="yolo26s-seg.pt", help="YOLO model name")
    parser.add_argument("--device", default=None,
                        help="Inference device: cuda | mps | cpu (auto-detected if omitted)")
    parser.add_argument("--visualize", action="store_true",
                        help="Open a live OpenCV window (requires opencv-python, not headless)")
    parser.add_argument("--llm", default="gemini", choices=["gemini", "none"],
                        help="Query engine: gemini (default) | none (rule-based)")
    parser.add_argument("--gemini-key", default="",
                        help="Gemini API key (overrides GEMINI_API_KEY env var)")

    # ── Open-vocabulary detection (Grounding DINO + MobileSAM) ──
    ov_group = parser.add_mutually_exclusive_group()
    ov_group.add_argument(
        "--open-vocab", dest="open_vocab", action="store_true", default=True,
        help="Enable open-vocabulary detection via Grounding DINO (default ON). "
             "Activate at runtime with: {\"type\":\"set_targets\",\"objects\":[\"wheelchair\"]}",
    )
    ov_group.add_argument(
        "--no-open-vocab", dest="open_vocab", action="store_false",
        help="Disable the Grounding DINO pipeline entirely (YOLO-only mode).",
    )
    parser.add_argument(
        "--gdino-interval", type=int, default=5, metavar="N",
        help="Run Grounding DINO every N frames (default 5 ≈ 4 fps at 20-fps stream). "
             "Lower = more responsive but higher CPU/GPU load.",
    )
    parser.add_argument(
        "--sam-interval", type=int, default=1, metavar="N",
        help="Run FastSAM/SAM every N Grounding-DINO calls (default 1 = every call). "
             "Set to 2-4 to cut GPU load; bbox depth used on skipped calls.",
    )

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
         args.llm, args.gemini_key,
         open_vocab=args.open_vocab, gdino_interval=args.gdino_interval,
         sam_interval=args.sam_interval)
