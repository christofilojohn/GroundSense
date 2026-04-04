import SwiftUI

/// Full-screen idle / splash overlay shown when the AR session is not running.
/// Fades out smoothly when the session starts, and back in when it stops.
struct LogoIdleScreen: View {

    /// Whether the AR session is live. Drives the fade-out.
    let isRunning: Bool

    // MARK: - Animation state
    @State private var arcScale: CGFloat    = 0.82
    @State private var arcOpacity: Double   = 0.55
    @State private var dotGlow: CGFloat     = 0.7
    @State private var dotScale: CGFloat    = 1.0
    @State private var gridOpacity: Double  = 0.5
    @State private var subtitleOpacity: Double = 0.0

    var body: some View {
        ZStack {
            // ── Navy background ──────────────────────────────────────
            Color(red: 0.008, green: 0.043, blue: 0.114)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()

                // ── Logo image ───────────────────────────────────────
                Image("GroundSenseLogo")
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 340)
                    // Subtle breathing scale on the arcs layer
                    .scaleEffect(arcScale)
                    .opacity(arcOpacity)
                    .animation(
                        .easeInOut(duration: 2.4).repeatForever(autoreverses: true),
                        value: arcScale
                    )

                Spacer().frame(height: 28)

                // ── Tagline ──────────────────────────────────────────
                Text("LiDAR-powered scene awareness")
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundColor(.white.opacity(0.45))
                    .opacity(subtitleOpacity)

                Spacer()

                // ── Start hint ───────────────────────────────────────
                StartHint()
                    .padding(.bottom, 48)
            }
        }
        // Entire overlay fades out once AR session is running
        .opacity(isRunning ? 0 : 1)
        .animation(.easeInOut(duration: 0.55), value: isRunning)
        // Prevent touches passing through while visible
        .allowsHitTesting(!isRunning)
        .onAppear { beginAnimations() }
    }

    private func beginAnimations() {
        // Breathing arcs
        withAnimation(.easeInOut(duration: 2.4).repeatForever(autoreverses: true)) {
            arcScale   = 1.0
            arcOpacity = 1.0
        }
        // Dot pulse
        withAnimation(.easeInOut(duration: 1.6).repeatForever(autoreverses: true)) {
            dotGlow  = 1.0
            dotScale = 1.12
        }
        // Grid shimmer
        withAnimation(.easeInOut(duration: 3.0).repeatForever(autoreverses: true)) {
            gridOpacity = 0.85
        }
        // Tagline fade-in after a short delay
        withAnimation(.easeIn(duration: 0.8).delay(0.5)) {
            subtitleOpacity = 1.0
        }
    }
}

// MARK: - Start Hint

private struct StartHint: View {
    @State private var arrowOffset: CGFloat = 0

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: "chevron.up")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.white.opacity(0.3))
                .offset(y: arrowOffset)
                .animation(
                    .easeInOut(duration: 1.1).repeatForever(autoreverses: true),
                    value: arrowOffset
                )

            Text("Tap  Start  to begin")
                .font(.system(.caption, design: .rounded))
                .foregroundColor(.white.opacity(0.3))
                .kerning(0.5)
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.1).repeatForever(autoreverses: true)) {
                arrowOffset = -6
            }
        }
    }
}

#Preview {
    LogoIdleScreen(isRunning: false)
}
