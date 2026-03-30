# GroundSense

Open-vocabulary 3D scene segmentation and querying for visually impaired users.

## Architecture

```
iPhone (LiDAR Pro)          Backend Server
┌──────────────────┐       ┌──────────────────────────┐
│  ARKit Session    │       │  WebSocket Server        │
│  ├─ RGB Camera    │──WS──▶│  ├─ Frame Unpacker       │
│  ├─ LiDAR Depth   │       │  ├─ YOLO-Seg Pipeline    │
│  └─ Streaming     │       │  ├─ Depth Fusion          │
│                   │◀──WS──│  ├─ Obstacle Avoidance    │
│  TTS / STT        │       │  └─ Response Generator    │
└──────────────────┘       └──────────────────────────┘
```

## Quick Start

### 1. Backend Server

```bash
cd backend
pip install -r requirements.txt
python server.py --device mps   # use 'cuda' for NVIDIA, 'cpu' for fallback
```

The server starts on `ws://0.0.0.0:8765` by default.

### 2. iPhone App

1. Open `GroundSense/` in Xcode
2. Set your development team in Signing & Capabilities
3. Build and run on an iPhone Pro (12 Pro or later — needs LiDAR)
4. Enter your Mac's local IP in the server address field
5. Tap **Start** → **Stream**

### Requirements

- **iPhone**: iPhone 12 Pro or later (LiDAR required)
- **iOS**: 16.0+
- **Backend**: Python 3.10+, macOS/Linux
- **Model**: YOLO11n-seg (auto-downloaded on first run)

## Wire Protocol

Binary WebSocket messages:

| Segment     | Size    | Format              |
|-------------|---------|---------------------|
| JPEG size   | 4 bytes | uint32 LE           |
| JPEG data   | N bytes | RGB image           |
| Depth size  | 4 bytes | uint32 LE           |
| Depth data  | M bytes | float16 raw pixels  |
| Meta size   | 4 bytes | uint32 LE           |
| Meta JSON   | K bytes | UTF-8 JSON          |

Metadata includes: `timestamp`, `rgbWidth`, `rgbHeight`, `depthWidth`, `depthHeight`, `intrinsics [fx, fy, cx, cy]`.

## Team

- **Ioannis**: iPhone capture, ARKit, LiDAR streaming, system integration
- **Parth**: Segmentation pipeline, depth estimation, MobileSAM + Grounding DINO
- **Antoni**: Query engine, obstacle avoidance, voice interface, evaluation
