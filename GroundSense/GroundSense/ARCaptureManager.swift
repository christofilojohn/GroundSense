import Foundation
import ARKit
import Combine

/// Manages ARKit session for synchronized RGB + LiDAR depth capture
/// and streams frames to a backend server for processing.
class ARCaptureManager: NSObject, ObservableObject, ARSessionDelegate {
    
    let session = ARSession()
    
    // MARK: - Published State
    @Published var isRunning = false
    @Published var isStreaming = false
    @Published var fps: Double = 0
    @Published var lastDepthRange: (min: Float, max: Float) = (0, 0)
    @Published var statusMessage: String = "Ready"

    // MARK: - Configuration
    var serverURL: URL?
    var targetFPS: Int = 10  // Don't need 60fps for assistive use — 10-15 is fine
    var jpegQuality: CGFloat = 0.6

    // MARK: - Internals
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
    
    override init() {
        self.frameInterval = 1.0 / Double(10)
        super.init()
        session.delegate = self
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
        // Server sends back scene descriptions, obstacle warnings, query responses
        switch message {
        case .string(let text):
            // Parse JSON response from backend
            print("Server response: \(text)")
            // TODO: Route to TTS engine for spoken output
        case .data(let data):
            print("Server binary data: \(data.count) bytes")
        @unknown default:
            break
        }
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
        
        // Extract LiDAR depth — prefer smoothedSceneDepth which applies temporal
        // filtering across frames, giving significantly cleaner measurements.
        // Falls back to raw sceneDepth on devices/configs that don't provide it.
        let depthMap = frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap
        
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
