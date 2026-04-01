import SwiftUI
import ARKit
import AVFoundation

struct ContentView: View {
    @StateObject private var captureManager = ARCaptureManager()
    @State private var serverAddress: String = ""
    @State private var showQRScanner: Bool = false

    var body: some View {
        ZStack {
            // AR Camera Preview
            ARViewContainer(session: captureManager.session)
                .ignoresSafeArea()

            // Overlay UI
            VStack(spacing: 0) {

                // ── Top bar: status + FPS ────────────────────────────
                TopBar(captureManager: captureManager)

                Spacer()

                // ── Warning banner (shows last spoken obstacle warning) ──
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
                    showQRScanner: $showQRScanner
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
    }
}

// MARK: - Top Bar

private struct TopBar: View {
    @ObservedObject var captureManager: ARCaptureManager

    var body: some View {
        HStack {
            Circle()
                .fill(captureManager.isStreaming ? Color.green : Color.red)
                .frame(width: 12, height: 12)

            Text(captureManager.statusMessage)
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.white)
                .lineLimit(1)

            Spacer()

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

    var body: some View {
        VStack(spacing: 12) {

            // STT transcript feedback (shows while listening)
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

            // Action buttons row
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

                // Mic button — tap to start, tap again to stop
                Button(action: {
                    if captureManager.isListening {
                        captureManager.stopListening()
                    } else {
                        captureManager.startListening()
                    }
                }) {
                    Image(systemName: captureManager.isListening
                          ? "mic.fill"
                          : "mic")
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

// MARK: - AR View Container (UIKit bridge)

struct ARViewContainer: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let arView = ARSCNView()
        arView.session = session
        arView.automaticallyUpdatesLighting = true
        // Show camera feed only, no debug overlays
        arView.debugOptions = []
        return arView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

// MARK: - QR Scanner Sheet

/// A modal sheet that wraps the camera-based QR scanner.
struct QRScannerSheet: View {
    @Binding var isPresented: Bool
    @Binding var scannedAddress: String

    var body: some View {
        NavigationView {
            ZStack {
                QRScannerView { scanned in
                    // Only accept ws:// URLs
                    if scanned.hasPrefix("ws://") || scanned.hasPrefix("wss://") {
                        scannedAddress = scanned
                        isPresented = false
                    }
                }
                .ignoresSafeArea()

                // Viewfinder guide overlay
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

/// UIViewControllerRepresentable that uses AVCaptureMetadataOutput to scan QR codes.
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
    private var didScan = false   // fire onScan only once per presentation

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

    // AVCaptureMetadataOutputObjectsDelegate
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
