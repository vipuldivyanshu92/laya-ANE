import Combine
import Foundation

@MainActor
final class LayaEngine: ObservableObject {
    @Published var status: String = "Loading…"
    @Published var ready = false
    @Published var lastError: String?
    @Published var computeLabel = "Neural Engine + CPU/GPU"
    @Published var modelLabel = "—"

    private var tokenizer: LayaTokenizer?
    private var runner: CoreMLRunner?
    private var temperature: [Float] = [1, 1, 1]
    private var maxLen = 64
    private var headMaxLen = 24
    private var padId: Int32 = 0

    func loadIfNeeded() async {
        guard !ready else { return }
        do {
            let root = try Self.resourceRoot()
            status = "Compiling CoreML…"
            let tok = try LayaTokenizer(resourceDirectory: root)
            let run = try CoreMLRunner(modelDirectory: root, computeUnits: .all)
            if let cfgData = try? Data(contentsOf: root.appendingPathComponent("rl_agent_config.json")),
               let cfg = try? JSONSerialization.jsonObject(with: cfgData) as? [String: Any] {
                maxLen = cfg["max_len"] as? Int ?? run.maxLen
                headMaxLen = cfg["head_max_len"] as? Int ?? 192
                if let t = cfg["temperature"] as? [Double] {
                    temperature = t.map { Float($0) }
                }
            }
            if let metaData = try? Data(contentsOf: root.appendingPathComponent("ane_config.json")),
               let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any] {
                let source = meta["source"] as? String ?? "bundled"
                let len = meta["max_len"] as? Int ?? maxLen
                let quant = meta["quantize"] as? String ?? "none"
                modelLabel = "\(source) · max_len \(len) · \(quant)"
            } else {
                modelLabel = root.lastPathComponent
            }
            tokenizer = tok
            runner = run
            padId = tok.padTokenId
            ready = true
            status = "Ready · on-device CoreML"
            lastError = nil
        } catch {
            ready = false
            status = "Load failed"
            lastError = error.localizedDescription
        }
    }

    func predict(state: String, questions: [QuestionDef]) async -> TriageReport {
        guard let tokenizer, let runner else {
            return TriageReport(answers: [], totalLatencyMs: 0, inputTokens: 0)
        }
        var answers: [AnswerResult] = []
        var totalMs = 0.0
        var tokens = 0
        for q in questions {
            let (ids, markers) = PromptBuilder.buildSequence(
                tokenizer: tokenizer,
                state: state,
                question: q,
                maxLen: maxLen,
                headMaxLen: headMaxLen
            )
            tokens += ids.count
            var inputIds = [Int32](repeating: padId, count: maxLen)
            var attn = [Float](repeating: 0, count: maxLen)
            for (i, id) in ids.prefix(maxLen).enumerated() {
                inputIds[i] = id
                attn[i] = 1
            }
            var markerPos = [Int32](repeating: 0, count: runner.maxMarkers)
            var markerMask = [Float](repeating: 0, count: runner.maxMarkers)
            for (i, m) in markers.prefix(runner.maxMarkers).enumerated() {
                markerPos[i] = Int32(m)
                markerMask[i] = 1
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            do {
                let out = try runner.predict(
                    inputIds: inputIds,
                    attentionMask: attn,
                    markerPos: markerPos,
                    markerMask: markerMask,
                    qtype: q.type.index
                )
                let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                totalMs += ms
                let k = markers.count
                let scale = max(1e-3, temperature[Int(q.type.index)])
                let sliced = Array(out.logits.prefix(k)).map { $0 / scale }
                let probs = Softmax.probs(sliced)
                var result = AnswerResult(
                    id: q.id,
                    type: q.type,
                    confidence: Softmax.confidence(probs),
                    choice: nil,
                    probabilities: [:],
                    score: nil,
                    noul: nil,
                    latencyMs: ms
                )
                switch q.type {
                case .choice:
                    let labels = q.choiceLabels()
                    var map: [String: Float] = [:]
                    for (i, label) in labels.enumerated() where i < probs.count {
                        map[label] = probs[i]
                    }
                    result.probabilities = map
                    if let best = map.max(by: { $0.value < $1.value })?.key {
                        result.choice = best
                    }
                case .score:
                    var map: [String: Float] = [:]
                    var expected: Float = 0
                    for (i, p) in probs.enumerated() {
                        map[String(i)] = p
                        expected += Float(i) * p
                    }
                    result.probabilities = map
                    result.score = expected
                case .noul:
                    let pTrue = probs.count > 1 ? probs[1] : 0
                    result.noul = pTrue
                    result.confidence = max(pTrue, 1 - pTrue)
                    result.probabilities = ["false": probs.first ?? 0, "true": pTrue]
                }
                answers.append(result)
            } catch {
                lastError = error.localizedDescription
            }
        }
        return TriageReport(answers: answers, totalLatencyMs: totalMs, inputTokens: tokens)
    }

    /// Find the on-device model directory. Tries several bundle layouts Xcode may produce.
    private static func resourceRoot() throws -> URL {
        let fm = FileManager.default
        var candidates: [URL] = []

        if let bundleRes = Bundle.main.resourceURL {
            candidates.append(bundleRes.appendingPathComponent("Model", isDirectory: true))
            candidates.append(bundleRes.appendingPathComponent("Resources/Model", isDirectory: true))
            candidates.append(bundleRes) // configs copied flat into .app
        }
        if let url = Bundle.main.url(
            forResource: "ane_config", withExtension: "json", subdirectory: "Model"
        ) {
            candidates.insert(url.deletingLastPathComponent(), at: 0)
        }
        if let url = Bundle.main.url(forResource: "ane_config", withExtension: "json") {
            candidates.insert(url.deletingLastPathComponent(), at: 0)
        }
        if let pkg = Bundle.main.url(
            forResource: "DecisionModel", withExtension: "mlpackage", subdirectory: "Model"
        ) {
            candidates.insert(pkg.deletingLastPathComponent(), at: 0)
        }
        if let pkg = Bundle.main.url(forResource: "DecisionModel", withExtension: "mlpackage") {
            candidates.insert(pkg.deletingLastPathComponent(), at: 0)
        }

        // Dev-only: source tree when running from a desktop build without Copy Resources.
        let sourceTree = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Resources/Model", isDirectory: true)
        candidates.append(sourceTree)

        var tried: [String] = []
        for root in candidates {
            tried.append(root.path)
            let cfg = root.appendingPathComponent("ane_config.json")
            let pkg = root.appendingPathComponent("DecisionModel.mlpackage")
            let tok = root.appendingPathComponent("tokenizer/tokenizer.json")
            if fm.fileExists(atPath: cfg.path),
               fm.fileExists(atPath: pkg.path),
               fm.fileExists(atPath: tok.path) {
                return root
            }
        }

        let listing = (Bundle.main.resourceURL.flatMap { url -> String? in
            (try? fm.contentsOfDirectory(atPath: url.path))?.joined(separator: ", ")
        }) ?? "(no resourceURL)"

        throw EngineError.modelNotFound(
            "Could not find Model/ane_config.json + DecisionModel.mlpackage in the app bundle. "
                + "Bundle resources: \(listing). Tried: \(tried.prefix(6).joined(separator: " | "))"
        )
    }
}

enum EngineError: LocalizedError {
    case modelNotFound(String)
    var errorDescription: String? {
        switch self {
        case .modelNotFound(let detail): return detail
        }
    }
}

enum Softmax {
    static func probs(_ logits: [Float]) -> [Float] {
        guard let m = logits.max() else { return [] }
        let exps = logits.map { exp($0 - m) }
        let s = exps.reduce(0, +)
        return exps.map { $0 / s }
    }

    static func confidence(_ p: [Float]) -> Float {
        let k = p.count
        guard k >= 2 else { return 1 }
        let ent = -p.map { v -> Float in
            let x = max(v, 1e-12)
            return x * log(x)
        }.reduce(0, +)
        return max(0, min(1, 1 - ent / log(Float(k))))
    }
}
