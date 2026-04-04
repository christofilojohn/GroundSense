import SwiftUI
import ARKit
import AVFoundation

struct ContentView: View {
    @StateObject private var captureManager = ARCaptureManager()
    @State private var serverAddress: String = ""
    @State private var showQRScanner: Bool = false
    @State private var showGallery: Bool = false

    var body: some View {
        ZStack {
            // AR Camera Preview
            ARViewContainer(session: captureManager.session)
                .ignoresSafeArea()

            // Logo idle screen — fades out when AR session starts
            LogoIdleScreen(isRunning: captureManager.isRunning)

            // Overlay UI
            VStack(spacing: 0) {

                // ── Top bar: status + FPS + recording indicator ──────
                TopBar(captureManager: captureManager)

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
                        Text("\(captureManager.recordingManager.recordings.count)")
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
