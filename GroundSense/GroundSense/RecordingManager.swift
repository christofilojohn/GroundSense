import Foundation
import UIKit

// MARK: - Recording Entry

/// Represents a single .gsrecording file in the gallery.
struct RecordingEntry: Identifiable {
    let id: UUID
    let url: URL
    let createdAt: Date
    let frameCount: Int
    let fps: Double
    let thumbnailData: Data?      // JPEG bytes of the first captured frame

    var durationSeconds: Double { Double(frameCount) / max(fps, 1) }

    var formattedDuration: String {
        let total = Int(durationSeconds.rounded())
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    var thumbnail: UIImage? {
        guard let data = thumbnailData else { return nil }
        return UIImage(data: data)
    }
}

// MARK: - Recording Manager

/// Manages on-device GroundSense recording files (.gsrecording).
///
/// ## File Layout
/// ```
/// Fixed 32-byte header:
///   [0:8]   magic     b"GSREC\x01\x00\n"
///   [8:12]  version   UInt32 LE = 1
///   [12:16] fps_mHz   UInt32 LE  (fps * 1000, e.g. 20 000 for 20 fps)
///   [16:20] frameCount UInt32 LE ← patched to the real count on stopRecording()
///   [20:28] createdAt Float64 LE (Unix timestamp)
///   [28:32] flags     UInt32 LE = 0 (reserved)
///
/// Followed by frameCount wire-format frames, each:
///   [4B] jpeg_size  UInt32 LE
///   [jpeg_size B]   JPEG image
///   [4B] depth_size UInt32 LE
///   [depth_size B]  float-16 depth map
///   [4B] meta_size  UInt32 LE
///   [meta_size B]   JSON metadata (UTF-8)
/// ```
///
/// This is the identical packet layout the live-stream uses, so `fake_client.py --recording`
/// can forward frames verbatim to the server without any re-encoding.
final class RecordingManager: ObservableObject {

    // MARK: - Public Config
    static let maxSlots = 10

    // MARK: - Binary constants
    static let magic: [UInt8] = [0x47, 0x53, 0x52, 0x45, 0x43, 0x01, 0x00, 0x0A]
    // G    S    R    E    C   \x01  \x00   \n
    static let headerSize = 32

    // Byte offset of the UInt32 frame_count field inside the header.
    private static let frameCountOffset: UInt64 = 16

    // MARK: - Published State
    @Published var isRecording = false
    @Published var recordings: [RecordingEntry] = []
    @Published var currentFrameCount: Int = 0
    @Published var recordingStartTime: Date?

    // MARK: - Private
    private var fileHandle: FileHandle?
    private var firstFrameJPEG: Data?   // captured for thumbnail
    private let ioQueue = DispatchQueue(label: "com.groundsense.recio", qos: .utility)

    // MARK: - Init
    init() { refreshList() }

    // MARK: - Recordings Directory

    private var recordingsDir: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("GroundSense/Recordings", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Gallery Refresh

    func refreshList() {
        ioQueue.async { [weak self] in
            guard let self else { return }
            guard let urls = try? FileManager.default.contentsOfDirectory(
                at: self.recordingsDir,
                includingPropertiesForKeys: [.creationDateKey],
                options: .skipsHiddenFiles
            ) else { return }

            let entries = urls
                .filter { $0.pathExtension == "gsrecording" }
                .sorted {
                    let a = (try? $0.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                    let b = (try? $1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                    return a > b   // newest first
                }
                .compactMap { self.parseEntry(url: $0) }

            DispatchQueue.main.async { self.recordings = entries }
        }
    }

    // MARK: - Start Recording

    func startRecording(fps: Double) {
        guard !isRecording else { return }
        enforceSlotLimit()

        let nowTS = Date().timeIntervalSince1970
        let nowDate = Date(timeIntervalSince1970: nowTS)
        let name = ISO8601DateFormatter()
            .string(from: nowDate)
            .replacingOccurrences(of: ":", with: "-")

        let url = recordingsDir.appendingPathComponent("\(name).gsrecording")
        FileManager.default.createFile(atPath: url.path, contents: nil)

        guard let fh = try? FileHandle(forWritingTo: url) else {
            print("[RecordingManager] Cannot open file for writing: \(url.path)")
            return
        }

        // Write fixed 32-byte header
        var hdr = Data(capacity: Self.headerSize)
        hdr.append(contentsOf: Self.magic)        // [0:8]
        hdr.appendUInt32LE(1)                      // [8:12]  version
        hdr.appendUInt32LE(UInt32(fps * 1000))     // [12:16] fps_mHz
        hdr.appendUInt32LE(0)                      // [16:20] frame_count placeholder
        hdr.appendFloat64LE(nowTS)                 // [20:28] created_at
        hdr.appendUInt32LE(0)                      // [28:32] flags
        fh.write(hdr)

        fileHandle = fh
        firstFrameJPEG = nil

        DispatchQueue.main.async {
            self.isRecording = true
            self.currentFrameCount = 0
            self.recordingStartTime = nowDate
        }
    }

    // MARK: - Write Frame

    /// Append one wire-format packet to the active recording.
    /// Safe to call from any queue (typically ARCaptureManager's encodeQueue).
    func writeFrame(_ packet: Data) {
        guard isRecording, let fh = fileHandle else { return }

        // Grab the JPEG bytes from the first frame for gallery thumbnail
        if firstFrameJPEG == nil, packet.count > 4 {
            let jpegSize = packet.withUnsafeBytes {
                Int(UInt32(littleEndian: $0.load(as: UInt32.self)))
            }
            if jpegSize > 0, 4 + jpegSize <= packet.count {
                firstFrameJPEG = packet.subdata(in: 4 ..< 4 + jpegSize)
            }
        }

        let frameSnapshot = packet
        ioQueue.async { fh.write(frameSnapshot) }

        DispatchQueue.main.async { self.currentFrameCount += 1 }
    }

    // MARK: - Stop Recording

    func stopRecording() {
        guard isRecording, let fh = fileHandle else { return }
        let finalCount = currentFrameCount

        // Flip state immediately so no more frames are written
        isRecording = false

        ioQueue.async { [weak self] in
            guard let self else { return }

            // Patch frame_count at the fixed offset in the header
            fh.seek(toFileOffset: Self.frameCountOffset)
            var le = UInt32(finalCount).littleEndian
            fh.write(Data(bytes: &le, count: 4))

            try? fh.close()
            self.fileHandle = nil

            DispatchQueue.main.async {
                self.recordingStartTime = nil
                self.refreshList()
            }
        }
    }

    // MARK: - Delete

    func delete(_ entry: RecordingEntry) {
        try? FileManager.default.removeItem(at: entry.url)
        DispatchQueue.main.async {
            self.recordings.removeAll { $0.id == entry.id }
        }
    }

    // MARK: - Private Helpers

    private func enforceSlotLimit() {
        // Sort oldest-first, remove enough entries so starting a new one
        // keeps us within maxSlots.
        let oldest = recordings.sorted { $0.createdAt < $1.createdAt }
        let excess = oldest.count - (Self.maxSlots - 1)
        guard excess > 0 else { return }
        oldest.prefix(excess).forEach { delete($0) }
    }

    private func parseEntry(url: URL) -> RecordingEntry? {
        guard let fh = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? fh.close() }

        // Read the fixed 32-byte header
        let hdr = fh.readData(ofLength: Self.headerSize)
        guard hdr.count == Self.headerSize else { return nil }

        // Verify magic
        guard [UInt8](hdr[0..<8]) == Self.magic else { return nil }

        let fpsMHz    = hdr.uint32LE(at: 12)
        let frameCount = hdr.uint32LE(at: 16)
        let createdAt  = hdr.float64LE(at: 20)

        let fps = Double(fpsMHz) / 1000.0

        // Read the first frame's JPEG for thumbnail display
        var thumbnailData: Data?
        let szData = fh.readData(ofLength: 4)
        if szData.count == 4 {
            let jpegSize = Int(szData.uint32LE(at: 0))
            if jpegSize > 0 {
                thumbnailData = fh.readData(ofLength: jpegSize)
            }
        }

        return RecordingEntry(
            id: UUID(),
            url: url,
            createdAt: Date(timeIntervalSince1970: createdAt),
            frameCount: Int(frameCount),
            fps: fps,
            thumbnailData: thumbnailData
        )
    }
}

// MARK: - Data Helpers

private extension Data {
    mutating func appendUInt32LE(_ value: UInt32) {
        var v = value.littleEndian
        append(Data(bytes: &v, count: 4))
    }

    mutating func appendFloat64LE(_ value: Double) {
        var bits = value.bitPattern.littleEndian
        append(Data(bytes: &bits, count: 8))
    }

    // Read helpers (offset is in bytes from start of this Data)
    func uint32LE(at offset: Int) -> UInt32 {
        guard offset + 4 <= count else { return 0 }
        return subdata(in: offset ..< offset + 4).withUnsafeBytes {
            UInt32(littleEndian: $0.load(as: UInt32.self))
        }
    }

    func float64LE(at offset: Int) -> Double {
        guard offset + 8 <= count else { return 0 }
        let bits = subdata(in: offset ..< offset + 8).withUnsafeBytes {
            UInt64(littleEndian: $0.load(as: UInt64.self))
        }
        return Double(bitPattern: bits)
    }
}
