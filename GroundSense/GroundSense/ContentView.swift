import SwiftUI
import ARKit

// ── Server address presets ───────────────────────────────────────────
//
// Hotspot: Mac creates its own Wi-Fi in System Settings → Sharing → Internet Sharing.
//          iPhone joins it. No router, no data, no cable. Server IP is always 192.168.2.1.
//
// USB:     iPhone Personal Hotspot ON + cable. Server IP is always 172.20.10.2.
//          The WebSocket is local — zero cellular data is used either way.
//
private let kHotspotAddress = "ws://192.168.2.1:8765"
private let kUSBAddress     = "ws://172.20.10.2:8765"
private let kWiFiAddress    = "ws://192.168.1.100:8765"  // replace with Mac's Wi-Fi IP

struct ContentView: View {
    @StateObject private var captureManager = ARCaptureManager()
    @State private var serverAddress: String = kHotspotAddress
    @State private var showSettings = false
    
    var body: some View {
        ZStack {
            // AR Camera Preview
            ARViewContainer(session: captureManager.session)
                .ignoresSafeArea()
            
            // Overlay UI
            VStack {
                // Top bar: status
                HStack {
                    // Connection indicator
                    Circle()
                        .fill(captureManager.isStreaming ? Color.green : Color.red)
                        .frame(width: 12, height: 12)
                    
                    Text(captureManager.statusMessage)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.white)
                    
                    Spacer()
                    
                    // FPS counter
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
                
                Spacer()
                
                // Depth info overlay
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
                    }
                    .padding()
                    .background(Color.black.opacity(0.5))
                    .cornerRadius(12)
                    .padding(.horizontal)
                }
                
                // Bottom controls
                VStack(spacing: 12) {
                    // Quick-pick transport
                    HStack(spacing: 8) {
                        PresetButton(
                            label: "Hotspot",
                            icon: "personalhotspot",
                            address: kHotspotAddress,
                            current: $serverAddress
                        )
                        PresetButton(
                            label: "USB",
                            icon: "cable.connector",
                            address: kUSBAddress,
                            current: $serverAddress
                        )
                        PresetButton(
                            label: "Wi-Fi",
                            icon: "wifi",
                            address: kWiFiAddress,
                            current: $serverAddress
                        )
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
                    
                    // Action buttons
                    HStack(spacing: 16) {
                        // Start/Stop AR Session
                        Button(action: {
                            if captureManager.isRunning {
                                captureManager.stopSession()
                            } else {
                                captureManager.startSession()
                            }
                        }) {
                            Label(
                                captureManager.isRunning ? "Stop" : "Start",
                                systemImage: captureManager.isRunning ? "stop.circle.fill" : "play.circle.fill"
                            )
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(captureManager.isRunning ? Color.red : Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        
                        // Connect/Disconnect streaming
                        Button(action: {
                            if captureManager.isStreaming {
                                captureManager.disconnectWebSocket()
                            } else if let url = URL(string: serverAddress) {
                                captureManager.connectToServer(url: url)
                            }
                        }) {
                            Label(
                                captureManager.isStreaming ? "Disconnect" : "Stream",
                                systemImage: captureManager.isStreaming ? "wifi.slash" : "wifi"
                            )
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(captureManager.isStreaming ? Color.orange : Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .disabled(!captureManager.isRunning)
                    }
                }
                .padding()
                .background(
                    LinearGradient(
                        colors: [Color.clear, Color.black.opacity(0.8)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
            }
        }
        .preferredColorScheme(.dark)
    }
}

// MARK: - Preset address button

struct PresetButton: View {
    let label: String
    let icon: String
    let address: String
    @Binding var current: String

    var isActive: Bool { current == address }

    var body: some View {
        Button(action: { current = address }) {
            Label(label, systemImage: icon)
                .font(.subheadline.weight(.medium))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
                .background(isActive ? Color.blue : Color.white.opacity(0.15))
                .foregroundColor(.white)
                .cornerRadius(10)
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
        // Show camera feed only, no debug overlays
        arView.debugOptions = []
        return arView
    }
    
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

#Preview {
    ContentView()
}
