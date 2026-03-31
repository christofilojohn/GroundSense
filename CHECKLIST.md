# GroundSense — Implementation Checklist

## ✅ Done

### Week 1 — Capture & Stream
- [x] ARKit session setup with `ARWorldTrackingConfiguration`
- [x] LiDAR `smoothedSceneDepth` (+ `sceneDepth` fallback) capture
- [x] RGB → JPEG encoding (shared `CIContext`, 0.6 quality)
- [x] Depth → Float16 encoding (halves bandwidth)
- [x] Binary wire-format packet: `[jpeg_size][jpeg][depth_size][depth][meta_size][meta_json]`
- [x] WebSocket client in Swift (`URLSessionWebSocketTask`)
- [x] Backpressure: skip frame if previous send still in flight
- [x] FPS throttle (10 fps target, configurable)
- [x] Connection presets: Mac Hotspot / USB / Wi-Fi
- [x] Python WebSocket server (`websockets`)
- [x] Frame unpacking (Float16 → Float32, JPEG decode via OpenCV)
- [x] YOLO11n-seg instance segmentation + persistent tracking (`model.track`)
- [x] Bilateral filter denoising on LiDAR depth map
- [x] LiDAR-mask fusion: per-object median depth from mask region
- [x] `SceneState` / `DetectedObject` dataclasses
- [x] Live OpenCV visualiser: RGB + YOLO overlays + depth heatmap side-by-side
- [x] Startup banner showing all reachable network addresses

### Week 2 — Depth Fusion, Obstacle Avoidance & Voice Interface
- [x] **LiDAR-grid free-space estimation** — raw depth map divided into left/centre/right thirds (lower ⅔ of frame), 10th-percentile clearance per sector; robust to noise and catches obstacles YOLO misses
- [x] Object-based free-direction fallback (used when no depth map available)
- [x] `ResponseGenerator.generate_obstacle_warning` with WARN (2 m) / ALERT (1 m) thresholds and cooldown
- [x] **TTS** — `AVSpeechSynthesizer` in `ARCaptureManager`; obstacle warnings via cooldown, query answers forced-speak; audio session managed correctly
- [x] **STT** — `SFSpeechRecognizer` + `AVAudioEngine`; tap mic to start/stop; partial results shown in UI; final transcript sent to server as `{"query": "..."}`
- [x] Mic button in `ContentView` (disabled until streaming; cyan when active)
- [x] Live transcript strip (shows partial STT result while listening)
- [x] Warning banner overlay (shows last spoken obstacle alert)
- [x] `NSMicrophoneUsageDescription` + `NSSpeechRecognitionUsageDescription` in `Info.plist`
- [x] **Persistent scene state** on server (`self.last_scene` updated every frame)
- [x] **Query engine** — `_handle_query` uses real scene state; handles directional queries, safety/navigation, object-specific distance, and general scene description

---

## 🔲 To Do

### Week 3 — Open-Vocabulary Extension & End-to-End Polish
- [ ] **MobileSAM + Grounding DINO pipeline** (optional extension)
  - [ ] Integrate `groundingdino` for text-prompted object detection
  - [ ] Integrate `mobile_sam` for mask generation on detected boxes
  - [ ] Route open-vocabulary queries (e.g. "find the exit sign") through this pipeline
  - [ ] Server flag `--open-vocab` to enable/disable at startup
- [ ] End-to-end demo recording
- [ ] Evaluation on a set of test scenes

### Voice Interface Improvements
- [ ] Auto-stop listening after silence timeout (VAD — e.g. `webrtcvad`) so user doesn't need to tap twice
- [ ] Interrupt TTS gracefully if user starts speaking mid-warning
- [ ] Internationalisation: configurable `SFSpeechRecognizer` locale

### Obstacle Avoidance Improvements
- [ ] Floor-plane detection (ARKit `ARPlaneAnchor`) to exclude the ground from obstacle depth readings
- [ ] Temporal smoothing of `free_direction` (e.g. rolling majority vote over last 5 frames) to avoid rapid flip-flopping

### Server / Backend
- [ ] Persist `SceneState` history (last N frames) for richer temporal queries ("has anything moved?")
- [ ] `--model` hot-swap endpoint so you can switch YOLO variant at runtime without restarting
- [ ] Basic auth / token check on the WebSocket handshake for safety

### 🚀 Future — Replace Rule-Based Query Engine with LLM
- [ ] **Swap `ResponseGenerator.answer_query` for an LLM call**
  - Serialise `SceneState.to_dict()` into a compact system prompt
  - Pass the user's transcribed query as the user message
  - Return the model's natural-language answer directly to TTS
  - Suggested options (pick one):
    - **Anthropic Claude API** (`claude-haiku-*` for low latency) — best quality
    - **OpenAI GPT-4o-mini** — good balance of speed and cost
    - **Local Ollama / llama.cpp** (`llama3.2:3b` or `phi3-mini`) — offline, no API key
  - [ ] Add `--llm` flag to `server.py` (`none` | `claude` | `openai` | `ollama`)
  - [ ] Graceful fallback to rule-based engine if API call fails or times out
  - [ ] Cap response length in the prompt to keep TTS utterances short (≤ 2 sentences)
