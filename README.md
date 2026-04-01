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
│  TTS / STT        │       │  └─ Response Generator - Gemini/rule-based   │
└──────────────────┘       └──────────────────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
pip install websockets numpy opencv-python pillow ultralytics torch h5py google-genai
```

### 2. Backend Server

```bash
# Device is auto-detected: CUDA → MPS → CPU
python server.py --gemini-key YOUR_API_KEY
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

The server starts on `ws://0.0.0.0:8765` by default.

```bash
# Rule-based query engine (no API key needed):
python server.py --llm none

# Override device manually:
python server.py --device cpu --gemini-key YOUR_KEY

# If port 8765 is already in use:
python server.py --port 8766
```

### 3. iPhone App

1. Open `GroundSense/` in Xcode
2. Set your development team in Signing & Capabilities
3. Build and run on an iPhone Pro (12 Pro or later — needs LiDAR)
4. Enter your Mac's local IP in the server address field
5. Tap **Start** → **Stream**

### 4. Testing without a LiDAR iPhone (fake client)

A fake WebSocket client streams frames from the included NYU Depth v2 subset (`nyu_small.mat`, 60 indoor scenes) directly to the server — no iPhone needed.

```bash
# Stream all 60 frames with live display, then drop into query REPL
python fake_client.py --display --fps 3

# Ask a query automatically after streaming
python fake_client.py --query "what is in front of me?"
```

## Requirements

- **iPhone**: iPhone 12 Pro or later (LiDAR required)
- **iOS**: 16.0+
- **Backend**: Python 3.10+, Windows/macOS/Linux
- **Model**: YOLO26s-seg (auto-downloaded on first run)
- **Query engine**: Gemini API key (free tier) — falls back to rule-based if absent

## Wire Protocol

Binary WebSocket messages:

| Segment     | Size    | Format              |
|-------------|---------|---------------------|
| JPEG size   | 4 bytes | uint32 LE           |
| JPEG data   | N bytes | RGB image (in BGR)     |
| Depth size  | 4 bytes | uint32 LE           |
| Depth data  | M bytes | float16 raw pixels  |
| Meta size   | 4 bytes | uint32 LE           |
| Meta JSON   | K bytes | UTF-8 JSON          |

Metadata includes: `timestamp`, `rgbWidth`, `rgbHeight`, `depthWidth`, `depthHeight`, `intrinsics [fx, fy, cx, cy]`.

## Team

- **Ioannis**: iPhone capture, ARKit, LiDAR streaming, system integration
- **Parth**: Segmentation pipeline, depth estimation, MobileSAM + Grounding DINO
- **Antoni**: Query engine (Gemini LLM), obstacle avoidance, voice interface, evaluation
