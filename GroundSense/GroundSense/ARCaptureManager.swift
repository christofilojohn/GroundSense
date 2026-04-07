import Foundation
import ARKit
import Combine
import AVFoundation
import Speech
import Accelerate

/// Manages ARKit session for synchronized RGB + LiDAR depth capture,
/// WebSocket streaming to the backend, TTS spoken output, and STT voice queries.
/// Also owns the RecordingManager for local .gsrecording capture.
class ARCaptureManager: NSObject, ObservableObject, ARSessionDelegate {

    let session = ARSession()

    // MARK: - Published State
    @Published var isRunning = false
    @Published var isStreaming = false
    @Published var fps: Double = 0
    @Published var lastDepthRange: (min: Float, max: Float) = (0, 0)
    @Published var statusMessage: String = "Ready"

    // Voice interface state
    @Published var isListening = false
    @Published var lastSpokenText: String = ""
    @Published var transcribedQuery: String = ""

    // MARK: - Scene objects (from server scene_update)
    /// Last detected objects, used to draw bounding-box overlay on the camera feed.
    @Published var sceneObjects: [SceneObject] = []

    // MARK: - Audio alerts mute
    /// When true the spoken obstacle warnings are suppressed; visual banner still shows.
    @Published var alertsMuted: Bool = false

    // MARK: - Detection pipeline mode
    /// Active server-side pipeline.  Mirrors the server's _pipeline_mode.
    /// "yolo" | "gdino" | "both"
    @Published var pipelineMode: String = "yolo"
    /// Comma-separated open-vocabulary targets, e.g. "wheelchair, service dog".
    /// Sent to the server via set_targets when non-empty and mode != "yolo".
    @Published var detectionTargets: String = ""

    // MARK: - Recording (owned here so ContentView can observe it)
    let recordingManager = RecordingManager()

    // MARK: - Configuration
    var serverURL: URL?
    var targetFPS: Int = 20  // 20 fps gives smooth obstacle detection
    var jpegQuality: CGFloat = 0.6
    /// When true, uses ARKit's temporally smoothed depth map; false = raw per-frame depth.
    @Published var depthSmoothing: Bool = true

    // MARK: - Internals (streaming)
    private var webSocket: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var lastFrameTime: CFTimeInterval = 0
    private let frameInterval: CFTimeInterval  // computed from targetFPS
    private var frameCount = 0
    private var fpsTimer: Timer?
    // Reuse CIContext across frames — creating one per frame is expensive
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    // Backpressure: skip encoding for streaming if a send is already in flight
    private var isSending = false
    // Dedicated queue for JPEG/depth encoding — keeps ARKit delegate queue free
    private let encodeQueue = DispatchQueue(label: "com.groundsense.encode", qos: .userInteractive)

    // MARK: - TTS
    private let speechSynthesizer = AVSpeechSynthesizer()
    /// Timestamp when the last warning utterance finished playing.
    private var lastSpeechEndTime: CFTimeInterval = 0
    /// Minimum gap between successive obstacle warnings (seconds), measured from
    /// the *end* of the previous utterance so long alerts are fully audible.
    private let speakCooldown: CFTimeInterval = 5.0
    /// Last warning text that was spoken — suppresses exact duplicates.
    private var lastSpokenWarning: String = ""

    // MARK: - STT
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()

    override init() {
        self.frameInterval = 1.0 / Double(targetFPS)
        super.init()
        session.delegate = self
        speechSynthesizer.delegate = self
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    }

    // MARK: - Permission Requests

    /// Call once from the UI (e.g. .onAppear) to prompt the user for mic + speech permissions.
    func requestPermissions() {
        SFSpeechRecognizer.requestAuthorization { status in
            DispatchQueue.main.async {
                if status != .authorized {
                    print("Speech recognition not authorized: \(status.rawValue)")
                }
            }
        }
        if #available(iOS 17.0, *) {
            AVAudioApplication.requestRecordPermission { granted in
                if !granted { print("Microphone permission denied") }
            }
        }
    }

    // MARK: - Session Lifecycle

    func startSession() {
        guard ARWorldTrackingConfiguration.isSupported else {
            statusMessage = "ARKit World Tracking not supported on this device"
            return
        }

        let config = ARWorldTrackingConfiguration()

        // Enable LiDAR scene depth (requires iPhone Pro with LiDAR)
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
            statusMessage = "LiDAR depth enabled (smoothed)"
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
            statusMessage = "LiDAR depth enabled"
        } else {
            statusMessage = "⚠️ LiDAR not available — RGB only mode"
        }

        // Target ~1280px wide — the server downscales to 640px anyway.
        let backFormats = ARWorldTrackingConfiguration.supportedVideoFormats
            .filter { $0.captureDevicePosition == .back }
        let targetWidth: CGFloat = 1280
        if let bestFormat = backFormats
            .sorted(by: { abs($0.imageResolution.width - targetWidth) < abs($1.imageResolution.width - targetWidth) })
            .first {
            config.videoFormat = bestFormat
        }

        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
        startFPSCounter()
    }

    func stopSession() {
        // Stop any active recording before killing the AR session
        if recordingManager.isRecording {
            recordingManager.stopRecording()
        }
        session.pause()
        isRunning = false
        disconnectWebSocket()
        fpsTimer?.invalidate()
        stopListening()
    }

    // MARK: - WebSocket Connection

    func connectToServer(url: URL) {
        serverURL = url
        urlSession = URLSession(configuration: .default)
        webSocket = urlSession?.webSocketTask(with: url)
        webSocket?.resume()
        isStreaming = true
        statusMessage = "Connected to \(url.host ?? "server")"
        listenForMessages()

        // On (re-)connect: if the user had a non-default mode or targets configured,
        // send them once after the WebSocket handshake settles.
        // Guard: only send if actually non-default, and send mode+targets as a single
        // coordinated pair so the server never sees them out of order.
        let savedMode    = pipelineMode
        let savedTargets = detectionTargets
        guard savedMode != "yolo" || !savedTargets.isEmpty else { return }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
            guard let self, self.isStreaming else { return }
            // Send mode first, then targets — setPipelineMode would re-send targets
            // internally, so call the lower-level sender directly to avoid doubling up.
            if savedMode != "yolo" {
                if let data = try? JSONSerialization.data(withJSONObject: [
                    "type": "set_pipeline", "mode": savedMode
                ]), let text = String(data: data, encoding: .utf8) {
                    self.sendTextMessage(text)
                }
            }
            if !savedTargets.isEmpty {
                self.setDetectionTargets(savedTargets)
            }
        }
    }

    func disconnectWebSocket() {
        webSocket?.cancel(with: .goingAway, reason: nil)
        webSocket = nil
        isStreaming = false
    }

    private func listenForMessages() {
        webSocket?.receive { [weak self] result in
            switch result {
            case .success(let message):
                self?.handleServerMessage(message)
                self?.listenForMessages()  // Continue listening
            case .failure(let error):
                print("WebSocket receive error: \(error)")
                DispatchQueue.main.async {
                    self?.isStreaming = false
                    self?.statusMessage = "Connection lost"
                }
            }
        }
    }

    private func handleServerMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
            guard let data = text.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return }

            let msgType = json["type"] as? String

            // ── Scene objects → bounding box overlay ─────────────────
            if msgType == "scene_update",
               let scene = json["scene"] as? [String: Any],
               let rawObjs = scene["objects"] as? [[String: Any]] {
                let parsed: [SceneObject] = rawObjs.compactMap { obj in
                    guard let bbox = obj["bbox"] as? [Double], bbox.count == 4
                    else { return nil }
                    return SceneObject(
                        className:  obj["class"]       as? String ?? "?",
                        distanceM:  obj["distance_m"]  as? Double ?? 0,
                        direction:  obj["direction"]   as? String ?? "",
                        confidence: obj["confidence"]  as? Double ?? 0,
                        source:     obj["source"]      as? String ?? "yolo",
                        // Stored in landscape-normalized coords (0–1).
                        // The overlay view applies the 90° CW rotation to portrait.
                        bboxX1: bbox[0], bboxY1: bbox[1],
                        bboxX2: bbox[2], bboxY2: bbox[3]
                    )
                }
                DispatchQueue.main.async { self.sceneObjects = parsed }
            }

            // ── Warnings & query responses ────────────────────────────
            if let warning = json["warning"] as? String {
                // Visual banner always updates; speech is suppressed when muted.
                DispatchQueue.main.async { self.lastSpokenText = warning }
                if !alertsMuted {
                    speakWithCooldown(warning)
                }
            }
            if msgType == "query_response", let answer = json["answer"] as? String {
                // Query responses always speak (they're user-initiated).
                speakForced(answer)
            }

        case .data(let data):
            print("Server binary data: \(data.count) bytes")
        @unknown default:
            break
        }
    }

    // MARK: - TTS

    func speakWithCooldown(_ text: String) {
        // Don't interrupt speech already in progress — let the user hear it fully.
        guard !speechSynthesizer.isSpeaking else { return }

        // Enforce cooldown measured from *end* of the previous utterance.
        let now = CACurrentMediaTime()
        guard now - lastSpeechEndTime >= speakCooldown else { return }

        // Suppress exact duplicate warnings (e.g. same obstacle at same distance).
        guard text != lastSpokenWarning else { return }

        lastSpokenWarning = text
        speakForced(text)
    }

    func speakForced(_ text: String) {
        guard !isListening else { return }

        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .word)
        }

        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.playback, mode: .default, options: [])
        try? audioSession.setActive(true)

        let utterance = AVSpeechUtterance(string: text)
        utterance.rate  = AVSpeechUtteranceDefaultSpeechRate * 1.05
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0
        speechSynthesizer.speak(utterance)

        DispatchQueue.main.async { self.lastSpokenText = text }
    }

    // MARK: - STT

    func startListening() {
        guard !isListening else { return }

        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }

        recognitionTask?.cancel()
        recognitionTask = nil

        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Audio session error: \(error)")
            return
        }

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest,
              let speechRecognizer = speechRecognizer,
              speechRecognizer.isAvailable
        else {
            print("Speech recognizer unavailable")
            return
        }

        recognitionRequest.shouldReportPartialResults = true

        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }

            if let result = result {
                let query = result.bestTranscription.formattedString
                DispatchQueue.main.async { self.transcribedQuery = query }

                if result.isFinal {
                    self.sendVoiceQuery(query)
                    self.stopListening()
                }
            }
            if let error = error {
                print("Recognition error: \(error)")
                self.stopListening()
            }
        }

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            print("AudioEngine start error: \(error)")
            stopListening()
            return
        }

        DispatchQueue.main.async {
            self.isListening = true
            self.transcribedQuery = ""
        }
    }

    func stopListening() {
        guard audioEngine.isRunning else {
            DispatchQueue.main.async { self.isListening = false }
            return
        }

        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask?.cancel()
        recognitionTask = nil

        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)

        DispatchQueue.main.async { self.isListening = false }
    }

    private func sendVoiceQuery(_ query: String) {
        guard !query.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        let payload: [String: String] = ["query": query]
        guard let jsonData = try? JSONEncoder().encode(payload),
              let jsonString = String(data: jsonData, encoding: .utf8)
        else { return }

        let socket = webSocket
        Task { try? await socket?.send(.string(jsonString)) }
        print("Sent voice query: \(query)")
    }

    // MARK: - Pipeline / Target control

    /// Send any JSON string to the server (fire-and-forget).
    /// Silently no-ops if the WebSocket is not connected.
    func sendTextMessage(_ text: String) {
        guard let socket = webSocket else { return }
        Task { try? await socket.send(.string(text)) }
    }

    /// Switch the server-side detection pipeline.
    /// - "yolo"  — YOLO only (fast, COCO classes)
    /// - "gdino" — Grounding DINO only (open-vocabulary targets, no YOLO)
    /// - "both"  — YOLO always + GDINO for user targets (highest coverage)
    func setPipelineMode(_ mode: String) {
        let previous = pipelineMode
        pipelineMode = mode
        guard isStreaming else { return }

        // Only send set_pipeline when the mode actually changes.
        if mode != previous {
            guard let data = try? JSONSerialization.data(withJSONObject: [
                "type": "set_pipeline",
                "mode": mode,
            ]), let text = String(data: data, encoding: .utf8) else { return }
            sendTextMessage(text)
        }

        // Send targets when switching INTO a GDINO-capable mode for the first time,
        // but not if the mode didn't change (avoids redundant cache resets on server).
        if mode != "yolo" && !detectionTargets.isEmpty && mode != previous {
            setDetectionTargets(detectionTargets)
        }
    }

    /// Push the current detection targets to the server.
    /// Parses the comma-separated `targetsString`, trims whitespace,
    /// and sends {"type":"set_targets","objects":[…]}.
    func setDetectionTargets(_ targetsString: String) {
        detectionTargets = targetsString
        guard isStreaming else { return }

        let objects = targetsString
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        guard let data = try? JSONSerialization.data(withJSONObject: [
            "type": "set_targets",
            "objects": objects,
        ] as [String: Any]), let text = String(data: data, encoding: .utf8) else { return }
        sendTextMessage(text)
        print("[Pipeline] targets → \(objects)")
    }

    // MARK: - ARSessionDelegate — Frame Processing

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = frame.timestamp

        // Throttle to target FPS
        guard now - lastFrameTime >= frameInterval else { return }
        lastFrameTime = now
        frameCount += 1

        // Update depth stats for UI
        let depthMap: CVPixelBuffer? = depthSmoothing
            ? frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap
            : frame.sceneDepth?.depthMap ?? frame.smoothedSceneDepth?.depthMap

        if let dm = depthMap { updateDepthStats(dm) }

        // Decide whether encoding is needed this frame.
        // Streaming uses backpressure (isSending) to avoid socket overrun.
        // Recording is independent — it captures every throttled frame.
        let shouldStream  = isStreaming && !isSending
        let shouldRecord  = recordingManager.isRecording

        guard shouldStream || shouldRecord else { return }

        if shouldStream { isSending = true }

        // Retain the ARFrame (not the CVPixelBuffer directly) so we don't
        // exhaust ARKit's internal buffer pool before encoding finishes.
        let capturedFrame = frame
        let socket        = webSocket
        let willStream    = shouldStream
        let willRecord    = shouldRecord

        encodeQueue.async { [weak self] in
            guard let self else { return }

            guard let packet = self.buildPacket(
                rgb:        capturedFrame.capturedImage,
                depth:      capturedFrame.smoothedSceneDepth?.depthMap ?? capturedFrame.sceneDepth?.depthMap,
                timestamp:  now,
                intrinsics: capturedFrame.camera.intrinsics
            ) else {
                if willStream { self.isSending = false }
                return
            }

            // ── Write to recording file ──────────────────────────────
            if willRecord {
                self.recordingManager.writeFrame(packet)
            }

            // ── Send over WebSocket ──────────────────────────────────
            if willStream {
                Task {
                    defer { self.isSending = false }
                    try? await socket?.send(.data(packet))
                }
            }
        }
    }

    // MARK: - Frame Encoding (runs on encodeQueue, not the ARKit delegate thread)

    /// Encode one ARFrame worth of data into a wire-format packet.
    /// Returns nil if encoding fails. Wire format:
    ///   [4B jpeg_size][jpeg][4B depth_size][depth_f16][4B meta_size][meta_json]
    private func buildPacket(
        rgb: CVPixelBuffer,
        depth: CVPixelBuffer?,
        timestamp: CFTimeInterval,
        intrinsics: simd_float3x3
    ) -> Data? {
        // 1. Encode RGB as JPEG
        guard let jpegData = encodeRGBAsJPEG(rgb) else { return nil }

        // 2. Encode depth as float16 array (halves bandwidth vs float32)
        let depthData = encodeDepthMap(depth)

        // 3. Build metadata JSON
        let metadata = FrameMetadata(
            timestamp:   timestamp,
            rgbWidth:    CVPixelBufferGetWidth(rgb),
            rgbHeight:   CVPixelBufferGetHeight(rgb),
            depthWidth:  depth.map { CVPixelBufferGetWidth($0) }  ?? 0,
            depthHeight: depth.map { CVPixelBufferGetHeight($0) } ?? 0,
            intrinsics:  [
                intrinsics[0][0], intrinsics[1][1],  // fx, fy
                intrinsics[2][0], intrinsics[2][1]   // cx, cy
            ]
        )
        guard let metadataJSON = try? JSONEncoder().encode(metadata) else { return nil }

        // 4. Pack: [4B size][data] × 3
        var packet = Data()
        packet.appendUInt32LE(UInt32(jpegData.count))
        packet.append(jpegData)
        packet.appendUInt32LE(UInt32(depthData?.count ?? 0))
        if let d = depthData { packet.append(d) }
        packet.appendUInt32LE(UInt32(metadataJSON.count))
        packet.append(metadataJSON)

        return packet
    }

    // MARK: - Encoding Helpers

    private func encodeRGBAsJPEG(_ pixelBuffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        return UIImage(cgImage: cgImage).jpegData(compressionQuality: jpegQuality)
    }

    private func encodeDepthMap(_ depthMap: CVPixelBuffer?) -> Data? {
        guard let depthMap = depthMap else { return nil }

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width  = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

        let floatBuffer = baseAddress.assumingMemoryBound(to: Float.self)
        let count = width * height

        // Convert Float32 → Float16 via Accelerate (SIMD-vectorized, ~10× faster than scalar)
        var float16Data = Data(count: count * MemoryLayout<UInt16>.size)
        float16Data.withUnsafeMutableBytes { rawDst in
            var src = vImage_Buffer(
                data:     UnsafeMutableRawPointer(mutating: floatBuffer),
                height:   vImagePixelCount(height),
                width:    vImagePixelCount(width),
                rowBytes: width * MemoryLayout<Float>.size
            )
            var dst = vImage_Buffer(
                data:     rawDst.baseAddress!,
                height:   vImagePixelCount(height),
                width:    vImagePixelCount(width),
                rowBytes: width * MemoryLayout<UInt16>.size
            )
            vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
        }
        return float16Data
    }

    private func updateDepthStats(_ depthMap: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width  = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else { return }

        let floatBuffer = baseAddress.assumingMemoryBound(to: Float32.self)
        let count = width * height

        var minDepth: Float = .infinity
        var maxDepth: Float = -.infinity

        for i in 0..<count {
            let d = floatBuffer[i]
            if d > 0 && d < 10 {  // Valid range: 0–10 metres
                minDepth = min(minDepth, d)
                maxDepth = max(maxDepth, d)
            }
        }

        DispatchQueue.main.async {
            self.lastDepthRange = (minDepth, maxDepth)
        }
    }

    // MARK: - FPS Counter

    private func startFPSCounter() {
        frameCount = 0
        fpsTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            DispatchQueue.main.async {
                self.fps = Double(self.frameCount)
                self.frameCount = 0
            }
        }
    }
}

// MARK: - Frame Metadata

private extension Data {
    mutating func appendUInt32LE(_ value: UInt32) {
        var v = value.littleEndian
        append(Data(bytes: &v, count: 4))
    }
}

// MARK: - Scene Object (for bounding-box overlay)

/// A single detected object received from the server's scene_update response.
/// `bbox*` fields are in **landscape-normalised** coordinates (0–1), matching
/// the frame the server processed.  The overlay view converts them to portrait.
struct SceneObject: Identifiable {
    let id = UUID()
    let className: String
    let distanceM: Double
    let direction: String
    let confidence: Double
    let source: String          // "yolo" | "gdino"
    let bboxX1: Double          // landscape left   (0–1)
    let bboxY1: Double          // landscape top    (0–1)
    let bboxX2: Double          // landscape right  (0–1)
    let bboxY2: Double          // landscape bottom (0–1)

    /// Box colour: GDINO detections are purple; YOLO uses distance-based traffic-light.
    var overlayColor: UIColor {
        if source == "gdino" { return UIColor(red: 0.7, green: 0.2, blue: 1.0, alpha: 1) }
        if distanceM < 1.0   { return .systemRed }
        if distanceM < 2.0   { return .systemOrange }
        return .systemGreen
    }
}

// MARK: - AVSpeechSynthesizerDelegate

extension ARCaptureManager: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didFinish utterance: AVSpeechUtterance) {
        lastSpeechEndTime = CACurrentMediaTime()
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didCancel utterance: AVSpeechUtterance) {
        lastSpeechEndTime = CACurrentMediaTime()
    }
}

struct FrameMetadata: Codable {
    let timestamp: Double
    let rgbWidth:  Int
    let rgbHeight: Int
    let depthWidth:  Int
    let depthHeight: Int
    let intrinsics: [Float]  // [fx, fy, cx, cy]
}
