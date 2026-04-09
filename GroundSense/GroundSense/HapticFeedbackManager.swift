import CoreHaptics
import UIKit

/// Manages spatial-haptic feedback for nearby detected scene objects.
///
/// ## Perceptual encoding
///
/// | Haptic dimension | Scene property                              |
/// |------------------|---------------------------------------------|
/// | Intensity        | Proximity: 1.0 at 0 m → 0.0 at 3 m         |
/// | Sharpness        | Urgency: crisp (0.9) for <1 m, soft for >2 m|
/// | Direction        | 3-tap micro-burst with Gaussian envelope     |
/// | Pulse rate       | Adaptive: faster when closer                |
/// | Multi-object     | Up to 3 staggered bursts (180 ms apart)     |
///
/// ## Direction encoding (single-actuator spatial cue)
/// Each object produces a burst of 3 transient taps spaced 70 ms apart.
/// Their intensities follow a Gaussian (σ=0.40) centered on the object's
/// portrait-x position (0 = left, 1 = right):
///
///   Left  (x≈0.0): [strong, medium, faint ]  → leading edge heavy
///   Center(x≈0.5): [medium, strong, medium]  → symmetric peak
///   Right (x≈1.0): [faint,  medium, strong]  → trailing edge heavy
///
/// This exploits the brain's ability to perceive the "sweep" direction of a
/// brief vibration envelope — the same principle as a binaural panning cue.
final class HapticFeedbackManager {

    // MARK: – Device capability

    static let deviceSupportsHaptics =
        CHHapticEngine.capabilitiesForHardware().supportsHaptics

    // MARK: – Configuration

    /// Objects further than this distance are ignored entirely.
    private let dangerRadius: Double = 3.0

    /// Closest pulse interval (seconds) — prevents overwhelming the user.
    private let minInterval: CFTimeInterval = 0.10

    /// Furthest pulse interval. At dangerRadius, pulses are this far apart.
    private let maxInterval: CFTimeInterval = 0.45

    // MARK: – State

    private var engine: CHHapticEngine?
    private var lastPlayTime: CFTimeInterval = 0
    private let engineQueue = DispatchQueue(label: "com.groundsense.haptic",
                                           qos: .userInteractive)

    // MARK: – Init / deinit

    init() {
        guard Self.deviceSupportsHaptics else { return }
        startEngine()
    }

    deinit {
        engine?.stop()
    }

    // MARK: – Engine lifecycle

    private func startEngine() {
        engineQueue.async { [weak self] in
            guard let self else { return }
            do {
                let eng = try CHHapticEngine()
                eng.playsHapticsOnly = true

                // Auto-restart on unexpected stop (e.g. phone call, system interruption)
                eng.stoppedHandler = { [weak self] reason in
                    print("[Haptic] Engine stopped (\(reason)); restarting…")
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        self?.startEngine()
                    }
                }
                // Re-apply start after engine reset (e.g. route change)
                eng.resetHandler = { [weak self] in
                    guard let self else { return }
                    self.engineQueue.async {
                        try? self.engine?.start()
                    }
                }

                try eng.start()
                self.engine = eng
            } catch {
                print("[Haptic] Engine start error: \(error)")
            }
        }
    }

    // MARK: – Public API

    /// Update haptics based on the latest detected scene objects.
    ///
    /// Call this every time `sceneObjects` changes (i.e. on each server scene_update).
    /// Thread-safe — dispatches all heavy work internally.
    func update(objects: [SceneObject]) {
        guard Self.deviceSupportsHaptics else { return }

        // Filter to objects within the danger radius, sorted closest-first
        let close = objects
            .filter { $0.distanceM < dangerRadius && $0.distanceM > 0 }
            .sorted { $0.distanceM < $1.distanceM }

        guard !close.isEmpty else { return }

        // Adaptive pulse rate: closer objects → faster feedback
        let closestDist = close[0].distanceM
        let t = closestDist / dangerRadius                          // 0…1
        let adaptiveInterval = minInterval + t * (maxInterval - minInterval)

        let now = CACurrentMediaTime()
        guard now - lastPlayTime >= adaptiveInterval else { return }
        lastPlayTime = now

        // Capture at most 3 objects so the pattern stays readable
        let snapshot = Array(close.prefix(3))

        engineQueue.async { [weak self] in
            self?.playBurst(objects: snapshot)
        }
    }

    // MARK: – Pattern generation

    private func playBurst(objects: [SceneObject]) {
        guard let engine else { return }

        var events: [CHHapticEvent] = []

        for (slot, obj) in objects.enumerated() {

            // ── Portrait-x position from landscape bbox (mirrors DetectionOverlayView) ──
            // Landscape (bboxY1…bboxY2) → portrait x:  px = 1 − landscapeY
            let bboxCenterY = (obj.bboxY1 + obj.bboxY2) / 2.0   // landscape 0–1
            let portraitX   = Float(1.0 - bboxCenterY)           // 0=left … 1=right

            // ── Master intensity: proximity-based ──────────────────────────────────────
            // 1.0 at contact, smoothly falls to 0.0 at dangerRadius
            let masterIntensity = Float(max(0, 1.0 - obj.distanceM / dangerRadius))

            // ── Sharpness: encodes urgency level ──────────────────────────────────────
            // High sharpness = crisp, alarming; low = smooth, ambient
            let sharpness: Float
            switch obj.distanceM {
            case ..<1.0: sharpness = 0.90   // danger (red box)
            case ..<2.0: sharpness = 0.55   // caution (orange box)
            default:     sharpness = 0.28   // awareness (green box)
            }

            // ── Stagger multiple objects in time ───────────────────────────────────────
            // Each additional object's burst is delayed by 180 ms so they feel distinct.
            let slotOffset: Double = Double(slot) * 0.18

            // ── 3-tap directional micro-burst ─────────────────────────────────────────
            // Three taps at t+0, t+70 ms, t+140 ms.
            // Their intensities follow a Gaussian envelope centered at portraitX,
            // so the "peak" tap encodes the object's lateral position.
            let tapTimes:     [Double] = [0.000, 0.070, 0.140]
            let tapPositions: [Float]  = [0.0,   0.5,   1.0]
            let sigma: Float = 0.40     // controls how sharply the envelope peaks

            for (ti, tapX) in tapPositions.enumerated() {
                let diff      = portraitX - tapX
                let envelope  = expf(-(diff * diff) / (2 * sigma * sigma))
                let intensity = envelope * masterIntensity

                guard intensity > 0.04 else { continue }   // skip imperceptible taps

                events.append(CHHapticEvent(
                    eventType: .hapticTransient,
                    parameters: [
                        CHHapticEventParameter(parameterID: .hapticIntensity, value: intensity),
                        CHHapticEventParameter(parameterID: .hapticSharpness, value: sharpness),
                    ],
                    relativeTime: slotOffset + tapTimes[ti]
                ))
            }

            // ── Urgency underlay: very close objects get a short continuous growl ──────
            // This adds a felt texture beneath the taps for objects inside 0.8 m.
            if obj.distanceM < 0.8 {
                let growlIntensity = masterIntensity * 0.45
                let growlSharpness = sharpness * 0.60
                events.append(CHHapticEvent(
                    eventType:    .hapticContinuous,
                    parameters: [
                        CHHapticEventParameter(parameterID: .hapticIntensity, value: growlIntensity),
                        CHHapticEventParameter(parameterID: .hapticSharpness, value: growlSharpness),
                    ],
                    relativeTime: slotOffset,
                    duration:     0.18          // just long enough to be felt
                ))
            }
        }

        guard !events.isEmpty else { return }

        do {
            let pattern = try CHHapticPattern(events: events, parameters: [])
            let player  = try engine.makePlayer(with: pattern)
            try player.start(atTime: CHHapticTimeImmediate)
        } catch {
            print("[Haptic] Playback error: \(error)")
            // Try to recover the engine on the next update tick
            try? engine.start()
        }
    }
}
