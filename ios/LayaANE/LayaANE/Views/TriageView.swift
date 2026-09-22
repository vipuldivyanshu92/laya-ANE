import SwiftUI

struct TriageView: View {
    @EnvironmentObject var engine: LayaEngine
    @State private var message = SampleMail.refund
    @State private var report: TriageReport?
    @State private var running = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    header
                    messageEditor
                    runButton
                    if let report {
                        results(report)
                    }
                }
                .padding(20)
            }
            .background(LayaTheme.ink.ignoresSafeArea())
            .navigationTitle("Inbox Copilot")
            .dismissesKeyboard()
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Text(engine.status)
                        .font(.caption.monospaced())
                        .foregroundStyle(engine.ready ? LayaTheme.accent : LayaTheme.warn)
                }
            }
            .safeAreaInset(edge: .bottom) {
                if let err = engine.lastError, !engine.ready {
                    Text(err)
                        .font(.caption2.monospaced())
                        .foregroundStyle(LayaTheme.danger)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(LayaTheme.panel)
                }
            }
            .simultaneousGesture(
                TapGesture().onEnded { Keyboard.dismiss() }
            )
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("LAYA")
                .font(.system(size: 44, weight: .black, design: .serif))
                .foregroundStyle(.white)
            Text("On-device typed decisions for support mail — department, urgency, spam & phishing. No cloud.")
                .font(.callout)
                .foregroundStyle(LayaTheme.muted)
        }
    }

    private var messageEditor: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Message")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(LayaTheme.muted)
                Spacer()
                Menu("Samples") {
                    Button("Refund dispute") { message = SampleMail.refund }
                    Button("Outage panic") { message = SampleMail.outage }
                    Button("Phishing bait") { message = SampleMail.phishing }
                    Button("Sales inquiry") { message = SampleMail.sales }
                }
                .font(.caption)
            }
            TextEditor(text: $message)
                .frame(minHeight: 160)
                .scrollContentBackground(.hidden)
                .padding(12)
                .background(LayaTheme.panel)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                .foregroundStyle(.white)
        }
    }

    private var runButton: some View {
        Button {
            Keyboard.dismiss()
            Task { await run() }
        } label: {
            HStack {
                if running { ProgressView().tint(.black) }
                Text(running ? "Thinking on ANE…" : "Triage on device")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(engine.ready ? LayaTheme.accent : Color.gray)
            .foregroundStyle(.black)
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .disabled(!engine.ready || running || message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    private func results(_ report: TriageReport) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label(String(format: "%.0f ms total", report.totalLatencyMs), systemImage: "bolt.fill")
                Spacer()
                Text("\(report.inputTokens) tokens · 0 out")
            }
            .font(.caption.monospaced())
            .foregroundStyle(LayaTheme.accent)

            ForEach(report.answers) { answer in
                AnswerCard(answer: answer)
            }
        }
    }

    private func run() async {
        running = true
        defer { running = false }
        report = await engine.predict(state: message, questions: Presets.inbox)
    }
}

struct AnswerCard: View {
    let answer: AnswerResult

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(answer.id.uppercased())
                    .font(.caption.weight(.bold))
                    .foregroundStyle(LayaTheme.muted)
                Spacer()
                Text(String(format: "%.1f ms", answer.latencyMs))
                    .font(.caption2.monospaced())
                    .foregroundStyle(LayaTheme.muted)
            }

            switch answer.type {
            case .choice:
                Text(answer.choice ?? "—")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.white)
                ForEach(answer.probabilities.sorted(by: { $0.key < $1.key }), id: \.key) { item in
                    ProbBar(label: item.key, value: item.value)
                }
            case .score:
                Text(String(format: "score %.2f", answer.score ?? 0))
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.white)
                ForEach(answer.probabilities.sorted(by: { ($0.key as NSString).integerValue < ($1.key as NSString).integerValue }), id: \.key) { item in
                    ProbBar(label: "L\(item.key)", value: item.value)
                }
            case .noul:
                let p = answer.noul ?? 0
                Text(p >= 0.5 ? "YES" : "NO")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(p >= 0.5 ? LayaTheme.warn : LayaTheme.accent)
                ProbBar(label: "P(true)", value: p)
            }

            Text(String(format: "confidence %.0f%%", (answer.confidence) * 100))
                .font(.caption)
                .foregroundStyle(LayaTheme.muted)
        }
        .padding(14)
        .background(LayaTheme.panel)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

struct ProbBar: View {
    let label: String
    let value: Float

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label).font(.caption).foregroundStyle(.white.opacity(0.8))
                Spacer()
                Text(String(format: "%.0f%%", value * 100))
                    .font(.caption.monospaced())
                    .foregroundStyle(LayaTheme.muted)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.08))
                    Capsule()
                        .fill(LayaTheme.accent)
                        .frame(width: max(4, geo.size.width * CGFloat(value)))
                        .animation(.easeOut(duration: 0.35), value: value)
                }
            }
            .frame(height: 8)
        }
    }
}

enum SampleMail {
    static let refund = """
    Hi — I was billed twice for invoice #4412 last Tuesday. Please refund the duplicate charge today. This is blocking payroll.
    """
    static let outage = """
    Production API is returning 503 since 09:10 UTC. Customers can't check out. Need someone from technical ASAP.
    """
    static let phishing = """
    URGENT: Your mailbox will be deleted in 2 hours. Click http://secure-mail-login.xyz to verify your password immediately.
    """
    static let sales = """
    We're evaluating vendors for Q3 and would love a demo plus volume pricing for ~200 seats.
    """
}
