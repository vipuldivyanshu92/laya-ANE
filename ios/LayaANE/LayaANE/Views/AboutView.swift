import SwiftUI

struct AboutView: View {
    @EnvironmentObject var engine: LayaEngine

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("LAYA on ANE")
                        .font(.system(size: 32, weight: .black, design: .serif))
                    Text("Typed decisions (choice / score / noul) running locally through CoreML so Apple can schedule work on the Neural Engine.")
                        .foregroundStyle(LayaTheme.muted)
                    Group {
                        labeled("Status", engine.status)
                        labeled("Compute", engine.computeLabel)
                        labeled("Model", engine.modelLabel)
                        if let err = engine.lastError {
                            Text("Error")
                                .font(.caption)
                                .foregroundStyle(LayaTheme.muted)
                            Text(err)
                                .font(.footnote.monospaced())
                                .foregroundStyle(LayaTheme.danger)
                                .textSelection(.enabled)
                        }
                    }
                    Divider().overlay(Color.white.opacity(0.1))
                    Text("Prepare the model bundle")
                        .font(.headline)
                    Text("""
                    From the repo root:

                      source .venv/bin/activate
                      python scripts/prepare_ios_bundle.py

                    Then open ios/LayaANE/LayaANE.xcodeproj in Xcode, select an iPhone simulator or device, and Run.
                    """)
                    .font(.system(.footnote, design: .monospaced))
                    .foregroundStyle(LayaTheme.muted)
                }
                .padding(20)
            }
            .background(LayaTheme.ink.ignoresSafeArea())
            .navigationTitle("About")
        }
    }

    private func labeled(_ k: String, _ v: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(k).font(.caption).foregroundStyle(LayaTheme.muted)
            Text(v).foregroundStyle(.white)
        }
    }
}
