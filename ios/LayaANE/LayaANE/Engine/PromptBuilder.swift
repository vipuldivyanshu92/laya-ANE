import Foundation

/// Prompt construction matching `laya_ane.common.build_sequence`.
enum PromptBuilder {
    static func buildSequence(
        tokenizer: LayaTokenizer,
        state: String,
        question: QuestionDef,
        maxLen: Int = 512,
        headMaxLen: Int = 192
    ) -> (ids: [Int32], markers: [Int]) {
        let (prefix, markers) = buildPrefix(tokenizer: tokenizer, question: question, headMaxLen: headMaxLen)
        let room = max(0, maxLen - prefix.count - 1)
        let cleaned = state.replacingOccurrences(of: tokenizer.maskToken, with: " ")
        var stateIds = tokenizer.encode(cleaned, addSpecialTokens: false)
        if stateIds.count > room {
            stateIds = Array(stateIds.prefix(room))
        }
        var ids = prefix + stateIds + [tokenizer.sepTokenId]
        if ids.count > maxLen {
            ids = Array(ids.prefix(maxLen))
        }
        let kept = markers.filter { $0 < ids.count }
        return (ids, kept)
    }

    private static func buildPrefix(
        tokenizer: LayaTokenizer,
        question: QuestionDef,
        headMaxLen: Int
    ) -> (ids: [Int32], markers: [Int]) {
        let maskTok = tokenizer.maskToken
        let opts = question.renderedOptions()
        let ins = question.instructions.replacingOccurrences(of: maskTok, with: " ")
        var headIds = tokenizer.encode("\(question.type.rawValue) question: \(ins)", addSpecialTokens: false)
        var optIds: [[Int32]] = opts.map { opt in
            let text = " " + opt.replacingOccurrences(of: maskTok, with: " ")
            let ids = tokenizer.encode(text, addSpecialTokens: false)
            return [tokenizer.maskTokenId] + Array(ids.prefix(48))
        }
        var optBudget = headMaxLen - optIds.reduce(0) { $0 + $1.count }
        if optBudget < 16 {
            let per = max(4, (headMaxLen - 16) / max(1, optIds.count))
            optIds = optIds.map { Array($0.prefix(per)) }
            optBudget = headMaxLen - optIds.reduce(0) { $0 + $1.count }
        }
        headIds = Array(headIds.prefix(max(8, optBudget)))
        var ids: [Int32] = [tokenizer.clsTokenId] + headIds + [tokenizer.sepTokenId]
        var markers: [Int] = []
        for o in optIds {
            markers.append(ids.count)
            ids.append(contentsOf: o)
        }
        ids.append(tokenizer.sepTokenId)
        return (ids, markers)
    }
}
