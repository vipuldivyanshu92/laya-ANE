import SwiftUI

/// Live as-you-type guardrail scoring — interesting demo of low-latency on-device decisions.
struct GuardView: View {
    @EnvironmentObject var engine: LayaEngine
    @State private var text = ""
    @State private var report: TriageReport?
    @State private var task: Task<Void, Never>?
    @FocusState private var editorFocused: Bool

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Type a prompt. Laya scores jailbreak / harm / toxicity on-device as you pause.")
                        .font(.callout)
                        .foregroundStyle(LayaTheme.muted)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .contentShape(Rectangle())
                        .onTapGesture { dismissKeyboard() }

                    TextEditor(text: $text)
                        .focused($editorFocused)
                        .frame(minHeight: 140)
                        .scrollContentBackground(.hidden)
                        .padding(12)
                        .background(LayaTheme.panel)
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .foregroundStyle(.white)
                        .onChange(of: text) { _, _ in schedule() }

                    if let report {
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                Label(
                                    String(format: "%.0f ms", report.totalLatencyMs),
                                    systemImage: "bolt.fill"
                                )
                                Spacer()
                                riskBadge(report)
                            }
                            .font(.caption.monospaced())
                            .foregroundStyle(LayaTheme.accent)

                            ForEach(report.answers) { AnswerCard(answer: $0) }
                        }
                        .contentShape(Rectangle())
                        .onTapGesture { dismissKeyboard() }
                    }
                }
                .padding(20)
                .frame(maxWidth: .infinity, minHeight: 500, alignment: .topLeading)
            }
            .scrollDismissesKeyboard(.interactively)
            .background {
                LayaTheme.ink
                    .ignoresSafeArea()
                    .onTapGesture { dismissKeyboard() }
            }
            .navigationTitle("Live Guard")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismissKeyboard() }
                        .fontWeight(.semibold)
                        .foregroundStyle(LayaTheme.accent)
                        .opacity(editorFocused ? 1 : 0)
                        .disabled(!editorFocused)
                        .accessibilityHidden(!editorFocused)
                }
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { dismissKeyboard() }
                        .fontWeight(.semibold)
                }
            }
        }
    }

    private func dismissKeyboard() {
        editorFocused = false
        Keyboard.dismiss()
    }

    private func schedule() {
        task?.cancel()
        task = Task {
            try? await Task.sleep(nanoseconds: 350_000_000)
            guard !Task.isCancelled else { return }
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard engine.ready, trimmed.count >= 8 else {
                report = nil
                return
            }
            report = await engine.predict(state: trimmed, questions: Presets.guardrail)
        }
    }

    private func riskBadge(_ report: TriageReport) -> some View {
        let jail = report.answers.first { $0.id == "is_jailbreak" }?.noul ?? 0
        let harm = report.answers.first { $0.id == "is_harmful" }?.noul ?? 0
        let risk = max(jail, harm)
        let label = risk > 0.65 ? "HIGH RISK" : risk > 0.35 ? "WATCH" : "CLEAR"
        let color = risk > 0.65 ? LayaTheme.danger : risk > 0.35 ? LayaTheme.warn : LayaTheme.accent
        return Text(label)
            .font(.caption.weight(.bold))
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(color.opacity(0.2))
            .foregroundStyle(color)
            .clipShape(Capsule())
    }
}
