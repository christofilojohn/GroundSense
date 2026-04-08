import SwiftUI
import ARKit
import AVFoundation

struct ContentView: View {
    @StateObject private var captureManager = ARCaptureManager()
    @State private var serverAddress: String = ""
    @State private var showQRScanner: Bool = false
    @State private var showGallery: Bool = false
    @State private var showDetectionBoxes: Bool = false

    var body: some View {
        ZStack {
            // AR Camera Preview
            ARViewContainer(session: captureManager.session)
                .ignoresSafeArea()

            // Logo idle screen — fades out when AR session starts
            LogoIdleScreen(isRunning: captureManager.isRunning)

            // Detection bounding-box overlay (toggled from top bar)
            if showDetectionBoxes && captureManager.isStreaming {
                DetectionOverlayView(objects: captureManager.sceneObjects)
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
            }

            // Overlay UI
            VStack(spacing: 0) {

                // ── Top bar: status + FPS + recording indicator ──────
                TopBar(captureManager: captureManager,
                       showDetectionBoxes: $showDetectionBoxes)

                Spacer()

                // ── Warning banner (last spoken obstacle warning) ────
                if !captureManager.lastSpokenText.isEmpty {
                    WarningBanner(text: captureManager.lastSpokenText)
                        .padding(.horizontal)
                        .padding(.bottom, 8)
                }

                // ── Depth info ───────────────────────────────────────
                if captureManager.isRunning {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Depth Range")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.7))
                            Text(String(format: "%.1fm – %.1fm",
                                        captureManager.lastDepthRange.min,
                                        captureManager.lastDepthRange.max))
                                .font(.system(.body, design: .monospaced))
                                .foregroundColor(.white)
                        }
                        Spacer()
                        // Depth smoothing toggle
                        Button(action: { captureManager.depthSmoothing.toggle() }) {
                            HStack(spacing: 5) {
                                Image(systemName: captureManager.depthSmoothing
                                      ? "waveform.path.ecg" : "waveform")
                                    .font(.caption)
                                Text(captureManager.depthSmoothing ? "Smoothed" : "Raw")
                                    .font(.caption.weight(.medium))
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(captureManager.depthSmoothing
                                        ? Color.blue.opacity(0.7)
                                        : Color.white.opacity(0.15))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 8)
                }

                // ── Bottom controls ──────────────────────────────────
                BottomControls(
                    captureManager: captureManager,
                    recordingManager: captureManager.recordingManager,
                    serverAddress: $serverAddress,
                    showQRScanner: $showQRScanner,
                    showGallery: $showGallery
                )
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            captureManager.requestPermissions()
        }
        .sheet(isPresented: $showQRScanner) {
            QRScannerSheet(isPresented: $showQRScanner, scannedAddress: $serverAddress)
        }
        .sheet(isPresented: $showGallery) {
            RecordingGalleryView(recordingManager: captureManager.recordingManager)
        }
    }
}

// MARK: - Top Bar

private struct TopBar: View {
    @ObservedObject var captureManager: ARCaptureManager
    @Binding var showDetectionBoxes: Bool

    var body: some View {
        HStack {
            // Connection indicator
            Circle()
                .fill(captureManager.isStreaming ? Color.green : Color.red)
                .frame(width: 12, height: 12)

            Text(captureManager.statusMessage)
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.white)
                .lineLimit(1)

            Spacer()

            // Recording indicator — shown when a recording is active
            if captureManager.recordingManager.isRecording {
                RecordingIndicator(
                    frameCount: captureManager.recordingManager.currentFrameCount,
                    startDate: captureManager.recordingManager.recordingStartTime ?? Date()
                )
            }

            // Detection box overlay toggle
            Button(action: { showDetectionBoxes.toggle() }) {
                Image(systemName: showDetectionBoxes
                      ? "rectangle.3.group.fill"
                      : "rectangle.3.group")
                    .font(.caption)
                    .foregroundColor(showDetectionBoxes ? .cyan : .white.opacity(0.8))
                    .padding(6)
                    .background(showDetectionBoxes
                                ? Color.cyan.opacity(0.2)
                                : Color.black.opacity(0.35))
                    .cornerRadius(7)
            }
            .accessibilityLabel(showDetectionBoxes ? "Hide detection boxes" : "Show detection boxes")

            // Mute toggle — silences obstacle warning speech; visual banner stays on
            Button(action: { captureManager.alertsMuted.toggle() }) {
                Image(systemName: captureManager.alertsMuted
                      ? "speaker.slash.fill"
                      : "speaker.wave.2.fill")
                    .font(.caption)
                    .foregroundColor(captureManager.alertsMuted
                                     ? .orange
                                     : .white.opacity(0.8))
                    .padding(6)
                    .background(captureManager.alertsMuted
                                ? Color.orange.opacity(0.2)
                                : Color.black.opacity(0.35))
                    .cornerRadius(7)
            }
            .accessibilityLabel(captureManager.alertsMuted ? "Unmute alerts" : "Mute alerts")

            Text("\(Int(captureManager.fps)) FPS")
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.black.opacity(0.6))
                .cornerRadius(8)
        }
        .padding()
        .background(
            LinearGradient(
                colors: [Color.black.opacity(0.7), Color.clear],
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }
}

// MARK: - Recording Indicator (top bar badge)

private struct RecordingIndicator: View {
    let frameCount: Int
    let startDate: Date

    var body: some View {
        HStack(spacing: 5) {
            // Pulsing red dot
            Circle()
                .fill(Color.red)
                .frame(width: 8, height: 8)
                .modifier(PulseAnimation())

            ElapsedTimerView(startDate: startDate)

            Text("• \(frameCount) fr")
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(.white.opacity(0.7))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.red.opacity(0.2))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.red.opacity(0.6), lineWidth: 1)
        )
    }
}

private struct PulseAnimation: ViewModifier {
    @State private var scale: CGFloat = 1.0

    func body(content: Content) -> some View {
        content
            .scaleEffect(scale)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                    scale = 1.5
                }
            }
    }
}

// MARK: - Warning Banner

private struct WarningBanner: View {
    let text: String

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.yellow)
            Text(text)
                .font(.system(.subheadline, design: .rounded).weight(.medium))
                .foregroundColor(.white)
                .multilineTextAlignment(.leading)
            Spacer()
        }
        .padding(12)
        .background(Color.black.opacity(0.75))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.yellow.opacity(0.6), lineWidth: 1)
        )
    }
}

// MARK: - Bottom Controls

private struct BottomControls: View {
    @ObservedObject var captureManager: ARCaptureManager
    @ObservedObject var recordingManager: RecordingManager
    @Binding var serverAddress: String
    @Binding var showQRScanner: Bool
    @Binding var showGallery: Bool

    var body: some View {
        VStack(spacing: 12) {

            // STT transcript feedback
            if captureManager.isListening {
                HStack {
                    if #available(iOS 17.0, *) {
                        Image(systemName: "waveform")
                            .foregroundColor(.cyan)
                            .symbolEffect(.pulse)
                    } else {
                        Image(systemName: "waveform")
                            .foregroundColor(.cyan)
                    }
                    Text(captureManager.transcribedQuery.isEmpty
                         ? "Listening…"
                         : captureManager.transcribedQuery)
                        .font(.system(.subheadline, design: .rounded))
                        .foregroundColor(.white)
                        .lineLimit(2)
                    Spacer()
                }
                .padding(10)
                .background(Color.cyan.opacity(0.15))
                .cornerRadius(10)
            }

            // QR scan button
            Button(action: { showQRScanner = true }) {
                Label("Scan Server QR Code", systemImage: "qrcode.viewfinder")
                    .font(.subheadline.weight(.medium))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(Color.white.opacity(0.15))
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }

            // Server address field
            HStack {
                Image(systemName: "network")
                    .foregroundColor(.white.opacity(0.7))
                TextField("ws://host:8765", text: $serverAddress)
                    .font(.system(.body, design: .monospaced))
                    .foregroundColor(.white)
                    .autocapitalization(.none)
                    .disableAutocorrection(true)
                    .keyboardType(.URL)
            }
            .padding()
            .background(Color.white.opacity(0.15))
            .cornerRadius(12)

            // ── Row 1: Start / Stop  |  Stream / Disconnect  |  Mic ──
            HStack(spacing: 12) {

                // Start / Stop AR
                Button(action: {
                    if captureManager.isRunning {
                        captureManager.stopSession()
                    } else {
                        captureManager.startSession()
                    }
                }) {
                    Label(captureManager.isRunning ? "Stop" : "Start",
                          systemImage: captureManager.isRunning ? "stop.circle.fill" : "play.circle.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(captureManager.isRunning ? Color.red : Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }

                // Connect / Disconnect streaming
                Button(action: {
                    if captureManager.isStreaming {
                        captureManager.disconnectWebSocket()
                    } else if let url = URL(string: serverAddress) {
                        captureManager.connectToServer(url: url)
                    }
                }) {
                    Label(captureManager.isStreaming ? "Disconnect" : "Stream",
                          systemImage: captureManager.isStreaming ? "wifi.slash" : "wifi")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(captureManager.isStreaming ? Color.orange : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
                .disabled(!captureManager.isRunning)

                // Mic button
                Button(action: {
                    if captureManager.isListening {
                        captureManager.stopListening()
                    } else {
                        captureManager.startListening()
                    }
                }) {
                    Image(systemName: captureManager.isListening ? "mic.fill" : "mic")
                        .font(.title2)
                        .frame(width: 56, height: 56)
                        .background(captureManager.isListening ? Color.cyan : Color.white.opacity(0.2))
                        .foregroundColor(.white)
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(captureManager.isListening
                                        ? Color.cyan.opacity(0.8)
                                        : Color.clear, lineWidth: 2)
                        )
                }
                .disabled(!captureManager.isStreaming)
            }

            // ── Row 2: Gallery  |  Record / Stop ────────────────────
            HStack(spacing: 12) {

                // Gallery button
                Button(action: { showGallery = true }) {
                    VStack(spacing: 4) {
                        Image(systemName: "film.stack")
                            .font(.title2)
                        Text("\(recordingManager.recordings.count)")
                            .font(.caption2.weight(.bold))
                    }
                    .frame(width: 64, height: 56)
                    .background(Color.white.opacity(0.15))
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }

                // Record / Stop button (the big prominent one)
                RecordButton(recordingManager: captureManager.recordingManager,
                             fps: Double(captureManager.targetFPS))
                    .disabled(!captureManager.isRunning)
            }

            // ── Row 3: Detection pipeline (shown when running) ───────
            if captureManager.isRunning {
                PipelineControls(captureManager: captureManager)
            }
        }
        .padding()
        .background(
            LinearGradient(
                colors: [Color.clear, Color.black.opacity(0.85)],
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }
}

// MARK: - Detection Bounding-Box Overlay

/// Draws server-detected bounding boxes over the live camera feed.
///
/// **Coordinate system note**
/// The server processes landscape frames from ARKit's `capturedImage`.
/// YOLO/GDINO return normalised landscape bbox coords (x left→right, y top→bottom).
/// The iPhone AR view is portrait, which is a 90° CW rotation of that landscape frame.
///
/// Rotation mapping  landscape(nx, ny) → portrait(1−ny, nx):
///   portrait_x = 1 − landscape_y
///   portrait_y = landscape_x
///
/// For a bbox (lx1,ly1,lx2,ly2):
///   portrait left  = 1 − ly2
///   portrait top   = lx1
///   portrait right = 1 − ly1
///   portrait btm   = lx2
private struct DetectionOverlayView: View {
    let objects: [SceneObject]

    var body: some View {
        GeometryReader { geo in
            Canvas { ctx, size in
                for obj in objects {
                    // ── Portrait coordinate transform ─────────────────
                    let px1 = CGFloat(1.0 - obj.bboxY2) * size.width
                    let py1 = CGFloat(obj.bboxX1)       * size.height
                    let px2 = CGFloat(1.0 - obj.bboxY1) * size.width
                    let py2 = CGFloat(obj.bboxX2)       * size.height

                    guard px2 > px1, py2 > py1 else { continue }
                    let rect = CGRect(x: px1, y: py1,
                                      width: px2 - px1, height: py2 - py1)

                    // ── Box colour ────────────────────────────────────
                    // GDINO detections: purple  |  YOLO: green/orange/red by distance
                    let swiftColor: Color = obj.source == "gdino"
                        ? Color(red: 0.7, green: 0.2, blue: 1.0)
                        : (obj.distanceM < 1.0 ? .red
                           : obj.distanceM < 2.0 ? .orange
                           : .green)

                    // ── Draw box ──────────────────────────────────────
                    ctx.stroke(
                        Path(rect),
                        with: .color(swiftColor.opacity(0.9)),
                        lineWidth: 2
                    )

                    // ── Corner accent (top-left L) ────────────────────
                    let cornerLen: CGFloat = min(12, (px2 - px1) * 0.2)
                    var corner = Path()
                    corner.move(to: CGPoint(x: px1, y: py1 + cornerLen))
                    corner.addLine(to: CGPoint(x: px1, y: py1))
                    corner.addLine(to: CGPoint(x: px1 + cornerLen, y: py1))
                    ctx.stroke(corner, with: .color(swiftColor), lineWidth: 3)

                    // ── Label pill ────────────────────────────────────
                    let label = "\(obj.className)  \(String(format: "%.1f", obj.distanceM))m"
                    let tag   = Text(label)
                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                        .foregroundColor(.white)
                    let tagSize = CGSize(width: CGFloat(label.count) * 6.5 + 8, height: 18)
                    let pillOrigin = CGPoint(x: px1, y: max(0, py1 - 20))
                    ctx.fill(
                        Path(CGRect(origin: pillOrigin, size: tagSize)
                             .insetBy(dx: -2, dy: -1)),
                        with: .color(swiftColor.opacity(0.8))
                    )
                    ctx.draw(tag,
                             at: CGPoint(x: pillOrigin.x + 4, y: pillOrigin.y + 1),
                             anchor: .topLeading)
                }
            }
        }
    }
}

// MARK: - Pipeline Controls

/// Segmented picker to switch between YOLO / GDINO / Both detection modes,
/// with an optional targets text field shown for GDINO-capable modes.
private struct PipelineControls: View {
    @ObservedObject var captureManager: ARCaptureManager

    // Local draft of the targets string; committed on Return or the arrow button.
    @State private var targetsDraft: String = ""
    @FocusState private var targetsFieldFocused: Bool

    var body: some View {
        VStack(spacing: 8) {

            // ── Mode picker ───────────────────────────────────────────
            HStack(spacing: 0) {
                PipelineModeButton(label: "YOLO",  tag: "yolo",  current: captureManager.pipelineMode) {
                    captureManager.setPipelineMode("yolo")
                }
                PipelineModeButton(label: "GDINO", tag: "gdino", current: captureManager.pipelineMode) {
                    captureManager.setPipelineMode("gdino")
                    targetsDraft = captureManager.detectionTargets
                }
                PipelineModeButton(label: "Both",  tag: "both",  current: captureManager.pipelineMode) {
                    captureManager.setPipelineMode("both")
                    targetsDraft = captureManager.detectionTargets
                }
            }
            .background(Color.white.opacity(0.1))
            .cornerRadius(10)
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.2), lineWidth: 1))

            // ── Target objects field (GDINO / Both only) ──────────────
            if captureManager.pipelineMode != "yolo" {
                HStack(spacing: 8) {
                    Image(systemName: "text.magnifyingglass")
                        .foregroundColor(.white.opacity(0.6))
                        .font(.caption)

                    TextField("wheelchair, service dog, fire hydrant…",
                              text: $targetsDraft)
                        .font(.system(.caption, design: .rounded))
                        .foregroundColor(.white)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                        .focused($targetsFieldFocused)
                        .onSubmit { commitTargets() }

                    // Send button
                    Button(action: { commitTargets(); targetsFieldFocused = false }) {
                        Image(systemName: "arrow.up.circle.fill")
                            .foregroundColor(targetsDraft.isEmpty ? .white.opacity(0.3) : .blue)
                            .font(.title3)
                    }
                    .disabled(targetsDraft.isEmpty)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.white.opacity(0.1))
                .cornerRadius(10)
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.2), lineWidth: 1))
                .onAppear { targetsDraft = captureManager.detectionTargets }
            }
        }
    }

    private func commitTargets() {
        captureManager.setDetectionTargets(targetsDraft)
    }
}

private struct PipelineModeButton: View {
    let label: String
    let tag: String
    let current: String
    let action: () -> Void

    var isSelected: Bool { current == tag }

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.caption.weight(isSelected ? .semibold : .regular))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
                .background(isSelected ? Color.blue.opacity(0.75) : Color.clear)
                .foregroundColor(isSelected ? .white : .white.opacity(0.65))
        }
        .cornerRadius(isSelected ? 10 : 0)
        .animation(.easeInOut(duration: 0.15), value: isSelected)
    }
}

// MARK: - Record Button

private struct RecordButton: View {
    @ObservedObject var recordingManager: RecordingManager
    let fps: Double

    var isRecording: Bool { recordingManager.isRecording }
    var isFull: Bool {
        !isRecording && recordingManager.recordings.count >= RecordingManager.maxSlots
    }

    var body: some View {
        Button(action: toggle) {
            HStack(spacing: 10) {
                // Icon: filled circle when idle, filled square when recording
                Image(systemName: isRecording ? "stop.circle.fill" : "record.circle")
                    .font(.title2)
                    .foregroundColor(isRecording ? .white : .red)

                if isRecording, let start = recordingManager.recordingStartTime {
                    VStack(alignment: .leading, spacing: 2) {
                        ElapsedTimerView(startDate: start)
                        Text("\(recordingManager.currentFrameCount) frames")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.white.opacity(0.7))
                    }
                } else if isFull {
                    Text("Gallery Full")
                        .font(.headline)
                        .foregroundColor(.white)
                } else {
                    Text("Record")
                        .font(.headline)
                        .foregroundColor(.white)
                }

                Spacer()
            }
            .padding(.horizontal, 16)
            .frame(maxWidth: .infinity, minHeight: 56)
            .background(recordingBackground)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(isRecording ? Color.red.opacity(0.8) : Color.clear, lineWidth: 2)
            )
        }
        .disabled(isFull)
    }

    @ViewBuilder
    private var recordingBackground: some View {
        if isRecording {
            Color.red.opacity(0.85)
        } else if isFull {
            Color.gray.opacity(0.4)
        } else {
            Color.red.opacity(0.25)
        }
    }

    private func toggle() {
        if isRecording {
            recordingManager.stopRecording()
        } else {
            recordingManager.startRecording(fps: fps)
        }
    }
}

// MARK: - AR View Container (UIKit bridge)

struct ARViewContainer: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let arView = ARSCNView()
        arView.session = session
        arView.automaticallyUpdatesLighting = true
        arView.debugOptions = []
        return arView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

// MARK: - QR Scanner Sheet

struct QRScannerSheet: View {
    @Binding var isPresented: Bool
    @Binding var scannedAddress: String

    var body: some View {
        NavigationView {
            ZStack {
                QRScannerView { scanned in
                    if scanned.hasPrefix("ws://") || scanned.hasPrefix("wss://") {
                        scannedAddress = scanned
                        isPresented = false
                    }
                }
                .ignoresSafeArea()

                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.white.opacity(0.8), lineWidth: 2)
                    .frame(width: 220, height: 220)

                VStack {
                    Spacer()
                    Text("Point at the QR code shown in the server terminal")
                        .font(.caption)
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                        .padding(10)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(8)
                        .padding(.bottom, 40)
                }
            }
            .navigationTitle("Scan Server QR")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Cancel") { isPresented = false }
                        .foregroundColor(.white)
                }
            }
        }
        .preferredColorScheme(.dark)
    }
}

// MARK: - QR Scanner View (AVFoundation bridge)

struct QRScannerView: UIViewControllerRepresentable {
    let onScan: (String) -> Void

    func makeUIViewController(context: Context) -> QRScannerViewController {
        let vc = QRScannerViewController()
        vc.onScan = onScan
        return vc
    }

    func updateUIViewController(_ uiViewController: QRScannerViewController, context: Context) {}
}

final class QRScannerViewController: UIViewController,
                                     AVCaptureMetadataOutputObjectsDelegate {

    var onScan: ((String) -> Void)?

    private var captureSession: AVCaptureSession?
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var didScan = false

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        setupSession()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        didScan = false
        if captureSession?.isRunning == false {
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.captureSession?.startRunning()
            }
        }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        if captureSession?.isRunning == true {
            captureSession?.stopRunning()
        }
    }

    private func setupSession() {
        let session = AVCaptureSession()
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else { return }

        session.addInput(input)

        let output = AVCaptureMetadataOutput()
        guard session.canAddOutput(output) else { return }
        session.addOutput(output)
        output.setMetadataObjectsDelegate(self, queue: .main)
        output.metadataObjectTypes = [.qr]

        let preview = AVCaptureVideoPreviewLayer(session: session)
        preview.frame = view.bounds
        preview.videoGravity = .resizeAspectFill
        view.layer.addSublayer(preview)
        self.previewLayer = preview
        self.captureSession = session

        DispatchQueue.global(qos: .userInitiated).async { session.startRunning() }
    }

    func metadataOutput(_ output: AVCaptureMetadataOutput,
                        didOutput metadataObjects: [AVMetadataObject],
                        from connection: AVCaptureConnection) {
        guard !didScan,
              let obj = metadataObjects.first as? AVMetadataMachineReadableCodeObject,
              let str = obj.stringValue else { return }
        didScan = true
        captureSession?.stopRunning()
        onScan?(str)
    }
}

#Preview {
    ContentView()
}
