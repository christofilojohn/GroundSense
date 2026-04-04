import SwiftUI
import UIKit

// MARK: - Gallery Sheet

struct RecordingGalleryView: View {
    @ObservedObject var recordingManager: RecordingManager
    @Environment(\.dismiss) private var dismiss

    // sheet(item:) guarantees the value is non-nil when the sheet builds,
    // fixing the blank-sheet bug that sheet(isPresented:) + if-let causes.
    @State private var exportEntry: RecordingEntry?

    var body: some View {
        NavigationView {
            Group {
                if recordingManager.recordings.isEmpty {
                    emptyState
                } else {
                    recordingList
                }
            }
            .navigationTitle("Recordings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                        .foregroundColor(.white)
                }
            }
        }
        .preferredColorScheme(.dark)
        // sheet(item:) — triggers as soon as exportEntry becomes non-nil,
        // and clears it back to nil when the sheet is dismissed.
        .sheet(item: $exportEntry) { entry in
            ShareSheet(url: entry.url)
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 20) {
            Image(systemName: "film.stack")
                .font(.system(size: 64))
                .foregroundColor(.gray)

            Text("No Recordings Yet")
                .font(.title2.weight(.semibold))
                .foregroundColor(.white)

            Text("Tap the red record button on the main screen\nto capture synchronized RGB + LiDAR data.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            Divider().padding(.horizontal, 48)

            VStack(alignment: .leading, spacing: 8) {
                Label("Up to \(RecordingManager.maxSlots) recordings stored on device",
                      systemImage: "internaldrive")
                Label("Export .gsrecording files via AirDrop or Files",
                      systemImage: "square.and.arrow.up")
                Label("Replay on server: python fake_client.py --recording <file>",
                      systemImage: "terminal")
            }
            .font(.caption)
            .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
    }

    // MARK: - List

    private var recordingList: some View {
        List {
            Section {
                ForEach(recordingManager.recordings) { entry in
                    RecordingRow(entry: entry) {
                        exportEntry = entry
                    }
                    // Native iOS red trash swipe — no EditButton needed.
                    // allowsFullSwipe: full swipe-to-delete gesture supported.
                    .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                        Button(role: .destructive) {
                            recordingManager.delete(entry)
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                    // Share on leading swipe too, for discoverability
                    .swipeActions(edge: .leading, allowsFullSwipe: false) {
                        Button {
                            exportEntry = entry
                        } label: {
                            Label("Share", systemImage: "square.and.arrow.up")
                        }
                        .tint(.blue)
                    }
                }
            } header: {
                Text("\(recordingManager.recordings.count) of \(RecordingManager.maxSlots) slots used")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } footer: {
                Text("Replay on server:\npython fake_client.py --recording <file.gsrecording>")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
        .background(Color.black)
    }
}

// MARK: - Row

private struct RecordingRow: View {
    let entry: RecordingEntry
    let onExport: () -> Void

    var body: some View {
        HStack(spacing: 12) {

            // Thumbnail
            thumbnailView
                .frame(width: 76, height: 50)
                .cornerRadius(6)
                .clipped()

            // Metadata
            VStack(alignment: .leading, spacing: 5) {
                Text(entry.createdAt, style: .date)
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.primary)

                Text(entry.createdAt, style: .time)
                    .font(.caption)
                    .foregroundColor(.secondary)

                HStack(spacing: 10) {
                    Label(entry.formattedDuration, systemImage: "clock")
                    Label("\(entry.frameCount) frames", systemImage: "film")
                    Label(String(format: "%.0f fps", entry.fps), systemImage: "gauge.with.dots.needle.33percent")
                }
                .font(.caption2)
                .foregroundColor(.secondary)
            }

            Spacer()

            // Export button
            Button(action: onExport) {
                Image(systemName: "square.and.arrow.up")
                    .font(.body.weight(.medium))
                    .foregroundColor(.blue)
                    .frame(width: 36, height: 36)
                    .background(Color.blue.opacity(0.15))
                    .cornerRadius(8)
            }
            .buttonStyle(.plain)
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var thumbnailView: some View {
        if let img = entry.thumbnail {
            Image(uiImage: img)
                .resizable()
                .scaledToFill()
        } else {
            ZStack {
                Color.gray.opacity(0.3)
                Image(systemName: "camera.fill")
                    .foregroundColor(.gray)
                    .font(.title3)
            }
        }
    }
}

// MARK: - UIKit Share Sheet Bridge

struct ShareSheet: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let vc = UIActivityViewController(activityItems: [url], applicationActivities: nil)
        // Prevents a crash / blank popover on iPad by clearing the popover anchor.
        // On iPhone this is a no-op.
        vc.popoverPresentationController?.sourceView = UIView()
        return vc
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - Recording Elapsed Timer

/// A live "mm:ss" counter that ticks every second from a given start date.
struct ElapsedTimerView: View {
    let startDate: Date
    @State private var elapsed: TimeInterval = 0
    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        Text(formattedElapsed)
            .font(.system(.caption, design: .monospaced).weight(.semibold))
            .foregroundColor(.white)
            .onReceive(timer) { _ in
                elapsed = Date().timeIntervalSince(startDate)
            }
            .onAppear {
                elapsed = Date().timeIntervalSince(startDate)
            }
    }

    private var formattedElapsed: String {
        let s = Int(elapsed)
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}
