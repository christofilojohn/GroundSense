import Foundation
import ARKit
import Combine
import AVFoundation
import Speech

/// Manages ARKit session for synchronized RGB + LiDAR depth capture,
/// WebSocket streaming to the backend, TTS spoken output, and STT voice queries.
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

    // MARK: - Configuration
    var serverURL: URL?
    var targetFPS: Int = 10  // Don't need 60fps for assistive use — 10-15 is fine
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
    private let ciContext = CIContext()
    // Backpressure: skip encoding if a send is already in flight
    private var isSending = false

    // MARK: - TTS
    private let speechSynthesizer = AVSpeechSynthesizer()
    /// Last time a warning was spoken (obstacle-avoidance cooldown).
    private var lastSpeakTime: CFTimeInterval = 0
    /// Don't repeat obstacle warnings more than once per this interval (seconds).
    private let speakCooldown: CFTimeInterval = 2.5

    // MARK: - STT
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()

    override init() {
        self.frameInterval = 1.0 / Double(10)
        super.init()
        session.delegate = self
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
        } else {
            // Fallback on earlier versions
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

        // We want high-res RGB for segmentation
        if let hiResFormat = ARWorldTrackingConfiguration.supportedVideoFormats
            .filter({ $0.captureDevicePosition == .back })
            .sorted(by: { $0.imageResolution.width > $1.imageResolution.width })
            .first {
            config.videoFormat = hiResFormat
        }

        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
        startFPSCounter()
    }

    func stopSession() {
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

            if let warning = json["warning"] as? String {
                // Obstacle warning — goes through cooldown
                speakWithCooldown(warning)
            }
            if msgType == "query_response", let answer = json["answer"] as? String {
                // Query answers always speak (bypass cooldown)
                speakForced(answer)
            }

        case .data(let data):
            print("Server binary data: \(data.count) bytes")
        @unknown default:
            break
        }
    }

    // MARK: - TTS

    /// Speak with obstacle-avoidance cooldown (won't fire more often than speakCooldown).
    func speakWithCooldown(_ text: String) {
        let now = CACurrentMediaTime()
        guard now - lastSpeakTime >= speakCooldown else { return }
        lastSpeakTime = now
        speakForced(text)
    }

    /// Speak immediately, bypassing cooldown (for query responses).
    func speakForced(_ text: String) {
        // Don't speak over an active STT session
        guard !isListening else { return }

        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .word)
        }

        // Configure audio session for playback
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

    /// Start listening for a voice query. Stops TTS if speaking.
    func startListening() {
        guard !isListening else { return }

        // Silence TTS before starting mic
        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }

        // Cancel any ongoing recognition task
        recognitionTask?.cancel()
        recognitionTask = nil

        // Set up audio session for recording
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

        // Tap the mic input
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

    /// Stop listening and clean up the audio engine + recognition session.
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

        // Restore audio session to default so TTS can work again
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)

        DispatchQueue.main.async { self.isListening = false }
    }

    /// Send a recognised query string to the backend as a JSON text message.
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

    // MARK: - ARSessionDelegate — Frame Processing

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = frame.timestamp

        // Throttle to target FPS
        guard now - lastFrameTime >= frameInterval else { return }
        lastFrameTime = now
        frameCount += 1

        // Extract RGB image
        let pixelBuffer = frame.capturedImage

        // Extract LiDAR depth — use smoothed or raw depending on user toggle.
        // Smoothed applies temporal filtering across frames (cleaner but adds ~1 frame lag).
        // Raw gives the latest unfiltered LiDAR reading.
        let depthMap: CVPixelBuffer?
        if depthSmoothing {
            depthMap = frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap
        } else {
            depthMap = frame.sceneDepth?.depthMap ?? frame.smoothedSceneDepth?.depthMap
        }

        // Update depth stats for UI
        if let depthMap = depthMap {
            updateDepthStats(depthMap)
        }

        // Stream if connected and not already mid-send (backpressure)
        if isStreaming, !isSending {
            guard let packet = buildPacket(
                rgb: pixelBuffer,
                depth: depthMap,
                timestamp: now,
                intrinsics: frame.camera.intrinsics
            ) else { return }

            isSending = true
            let socket = webSocket
            Task {
                defer { isSending = false }
                try? await socket?.send(.data(packet))
            }
        }
    }

    // MARK: - Frame Encoding (synchronous — must be called on the AR delegate thread)

    /// Encode one ARFrame worth of data into a wire-format packet.
    /// Returns nil if encoding fails. Called synchronously so that CVPixelBuffers
    /// are never captured by an async closure — this eliminates the ARFrame
    /// retention warnings.
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
            timestamp: timestamp,
            rgbWidth: CVPixelBufferGetWidth(rgb),
            rgbHeight: CVPixelBufferGetHeight(rgb),
            depthWidth: depth.map { CVPixelBufferGetWidth($0) } ?? 0,
            depthHeight: depth.map { CVPixelBufferGetHeight($0) } ?? 0,
            intrinsics: [
                intrinsics[0][0], intrinsics[1][1],  // fx, fy
                intrinsics[2][0], intrinsics[2][1]    // cx, cy
            ]
        )
        guard let metadataJSON = try? JSONEncoder().encode(metadata) else { return nil }

        // 4. Pack: [4B jpeg_size][jpeg][4B depth_size][depth][4B meta_size][meta]
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
        // Use the shared CIContext — creating one per frame costs ~10ms
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        let uiImage = UIImage(cgImage: cgImage)
        return uiImage.jpegData(compressionQuality: jpegQuality)
    }

    private func encodeDepthMap(_ depthMap: CVPixelBuffer?) -> Data? {
        guard let depthMap = depthMap else { return nil }

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

        let floatBuffer = baseAddress.assumingMemoryBound(to: Float32.self)
        let count = width * height

        // Convert Float32 -> Float16 to halve bandwidth
        var float16Data = Data(count: count * 2)
        float16Data.withUnsafeMutableBytes { rawBuffer in
            let f16Buffer = rawBuffer.bindMemory(to: UInt16.self)
            for i in 0..<count {
                f16Buffer[i] = floatToFloat16(floatBuffer[i])
            }
        }

        return float16Data
    }

    /// IEEE 754 Float32 -> Float16 conversion
    private func floatToFloat16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = (bits >> 16) & 0x8000
        let exponent = Int((bits >> 23) & 0xFF) - 127 + 15
        let mantissa = bits & 0x007FFFFF

        if exponent <= 0 {
            return UInt16(sign)  // Flush to zero for very small values
        } else if exponent >= 31 {
            return UInt16(sign | 0x7C00)  // Infinity
        }
        return UInt16(sign | UInt32(exponent << 10) | (mantissa >> 13))
    }

    private func updateDepthStats(_ depthMap: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else { return }

        let floatBuffer = baseAddress.assumingMemoryBound(to: Float32.self)
        let count = width * height

        var minDepth: Float = .infinity
        var maxDepth: Float = -.infinity

        for i in 0..<count {
            let d = floatBuffer[i]
            if d > 0 && d < 10 {  // Valid range: 0-10 meters
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
            guard let self = self else { return }
            DispatchQueue.main.async {
                self.fps = Double(self.frameCount)
                self.frameCount = 0
            }
        }
    }
}

// MARK: - Frame Metadata

private extension Data {
    /// Append a UInt32 in little-endian byte order.
    mutating func appendUInt32LE(_ value: UInt32) {
        var v = value.littleEndian
        append(Data(bytes: &v, count: 4))
    }
}

struct FrameMetadata: Codable {
    let timestamp: Double
    let rgbWidth: Int
    let rgbHeight: Int
    let depthWidth: Int
    let depthHeight: Int
    let intrinsics: [Float]  // [fx, fy, cx, cy]
}
