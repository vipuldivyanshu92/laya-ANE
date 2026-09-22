import Foundation

/// HuggingFace `tokenizers` JSON loader supporting ByteLevel BPE (ModernBERT) and WordPiece/WordLevel.
final class LayaTokenizer {
    let clsTokenId: Int32
    let sepTokenId: Int32
    let padTokenId: Int32
    let maskTokenId: Int32
    let clsToken: String
    let sepToken: String
    let padToken: String
    let maskToken: String

    private let vocab: [String: Int]
    private let unkId: Int
    private let kind: Kind
    private let lowercase: Bool
    private let merges: [String: Int] // "a b" -> rank
    private let byteEncoder: [UInt8: Character]
    private let addedTokens: [String: Int]
    private let splitRegex: NSRegularExpression?

    private enum Kind { case bpe, wordPiece, wordLevel }

    init(resourceDirectory: URL) throws {
        let tokURL = resourceDirectory.appendingPathComponent("tokenizer/tokenizer.json")
        let cfgURL = resourceDirectory.appendingPathComponent("tokenizer/tokenizer_config.json")
        let tokData = try Data(contentsOf: tokURL)
        let cfgData = try Data(contentsOf: cfgURL)
        guard let tokJSON = try JSONSerialization.jsonObject(with: tokData) as? [String: Any],
              let cfgJSON = try JSONSerialization.jsonObject(with: cfgData) as? [String: Any]
        else { throw TokenizerError.invalidJSON }

        func tokenString(_ key: String) -> String {
            if let s = cfgJSON[key] as? String { return s }
            if let d = cfgJSON[key] as? [String: Any], let c = d["content"] as? String { return c }
            return key
        }

        clsToken = tokenString("cls_token")
        sepToken = tokenString("sep_token")
        padToken = tokenString("pad_token")
        maskToken = tokenString("mask_token")

        let model = tokJSON["model"] as? [String: Any] ?? [:]
        let type = (model["type"] as? String ?? "").lowercased()

        var map: [String: Int] = [:]
        if let v = model["vocab"] as? [String: Int] {
            map = v
        } else if let v = model["vocab"] as? [String: Any] {
            for (k, val) in v {
                if let i = val as? Int { map[k] = i }
                else if let n = val as? NSNumber { map[k] = n.intValue }
            }
        }

        var added: [String: Int] = [:]
        if let arr = tokJSON["added_tokens"] as? [[String: Any]] {
            for item in arr {
                guard let content = item["content"] as? String else { continue }
                let id: Int?
                if let i = item["id"] as? Int { id = i }
                else if let n = item["id"] as? NSNumber { id = n.intValue }
                else { id = nil }
                if let id {
                    added[content] = id
                    map[content] = id
                }
            }
        }
        addedTokens = added

        guard !map.isEmpty else { throw TokenizerError.missingVocab }
        vocab = map

        if type.contains("bpe") {
            kind = .bpe
        } else if type.contains("wordpiece") || model["continuing_subword_prefix"] != nil {
            kind = .wordPiece
        } else {
            kind = .wordLevel
        }

        lowercase = ((tokJSON["normalizer"] as? [String: Any])?["type"] as? String) == "Lowercase"
            || ((tokJSON["normalizer"] as? [String: Any])?["lowercase"] as? Bool) == true

        var mergeRank: [String: Int] = [:]
        if let rawMerges = model["merges"] as? [Any] {
            for (i, item) in rawMerges.enumerated() {
                if let pair = item as? [String], pair.count == 2 {
                    mergeRank["\(pair[0]) \(pair[1])"] = i
                } else if let s = item as? String {
                    mergeRank[s] = i
                }
            }
        }
        merges = mergeRank
        byteEncoder = Self.bytesToUnicode()

        let pre = tokJSON["pre_tokenizer"] as? [String: Any]
        if kind == .bpe, (pre?["use_regex"] as? Bool) ?? true {
            // GPT-2 / ModernBERT ByteLevel regex
            let pattern = #"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"#
            splitRegex = try NSRegularExpression(pattern: pattern, options: [])
        } else {
            splitRegex = nil
        }

        func resolveID(_ token: String) throws -> Int32 {
            if let v = map[token] { return Int32(v) }
            if let v = added[token] { return Int32(v) }
            throw TokenizerError.missingSpecial(token)
        }

        clsTokenId = try resolveID(clsToken)
        sepTokenId = try resolveID(sepToken)
        padTokenId = try resolveID(padToken)
        maskTokenId = try resolveID(maskToken)
        unkId = map["[UNK]"] ?? added["[UNK]"] ?? map["<unk>"] ?? 1
    }

    func encode(_ text: String, addSpecialTokens: Bool = false) -> [Int32] {
        var ids: [Int32]
        switch kind {
        case .bpe:
            ids = encodeBPE(text)
        case .wordPiece:
            ids = wordPiece(text).map { Int32(vocab[$0] ?? unkId) }
        case .wordLevel:
            let normalized = lowercase ? text.lowercased() : text
            ids = normalized.split { $0.isWhitespace }.map { Int32(vocab[String($0)] ?? unkId) }
        }
        if addSpecialTokens {
            ids = [clsTokenId] + ids + [sepTokenId]
        }
        return ids
    }

    // MARK: - ByteLevel BPE

    private func encodeBPE(_ text: String) -> [Int32] {
        let pieces = splitByteLevel(text)
        var ids: [Int32] = []
        for piece in pieces {
            let unicode = byteLevelEncode(piece)
            let tokens = bpe(unicode)
            for t in tokens {
                ids.append(Int32(vocab[t] ?? unkId))
            }
        }
        return ids
    }

    private func splitByteLevel(_ text: String) -> [String] {
        guard let splitRegex else { return [text] }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        return splitRegex.matches(in: text, options: [], range: range).map { ns.substring(with: $0.range) }
    }

    private func byteLevelEncode(_ text: String) -> String {
        var out = ""
        for b in text.utf8 {
            if let ch = byteEncoder[b] {
                out.append(ch)
            }
        }
        return out
    }

    private func bpe(_ token: String) -> [String] {
        if token.isEmpty { return [] }
        var word = token.map { String($0) }
        if word.count == 1 { return word }

        while word.count > 1 {
            var minRank = Int.max
            var minIndex: Int?
            for i in 0 ..< (word.count - 1) {
                let key = "\(word[i]) \(word[i + 1])"
                if let rank = merges[key], rank < minRank {
                    minRank = rank
                    minIndex = i
                }
            }
            guard let i = minIndex else { break }
            let merged = word[i] + word[i + 1]
            var next: [String] = []
            var j = 0
            while j < word.count {
                if j == i {
                    next.append(merged)
                    j += 2
                } else {
                    next.append(word[j])
                    j += 1
                }
            }
            word = next
        }
        return word
    }

    /// GPT-2 bytes_to_unicode mapping.
    private static func bytesToUnicode() -> [UInt8: Character] {
        var bs = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var cs = bs
        var n = 0
        for b in 0 ..< 256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var map: [UInt8: Character] = [:]
        for i in bs.indices {
            map[UInt8(bs[i])] = Character(UnicodeScalar(cs[i])!)
        }
        return map
    }

    // MARK: - WordPiece (legacy tiny checkpoints)

    private func wordPiece(_ text: String) -> [String] {
        let normalized = lowercase ? text.lowercased() : text
        var chars: [Character] = []
        for ch in normalized {
            if ch.isPunctuation || ch.isSymbol {
                chars.append(" ")
                chars.append(ch)
                chars.append(" ")
            } else {
                chars.append(ch)
            }
        }
        let words = String(chars).split { $0.isWhitespace }.map(String.init)
        var output: [String] = []
        for word in words {
            if vocab[word] != nil {
                output.append(word)
                continue
            }
            var start = word.startIndex
            var subtokens: [String] = []
            var bad = false
            while start < word.endIndex {
                var end = word.endIndex
                var found: String?
                while start < end {
                    var substr = String(word[start ..< end])
                    if start != word.startIndex { substr = "##" + substr }
                    if vocab[substr] != nil {
                        found = substr
                        break
                    }
                    end = word.index(before: end)
                }
                if let found {
                    subtokens.append(found)
                    start = end
                } else {
                    bad = true
                    break
                }
            }
            if bad || subtokens.isEmpty {
                output.append("[UNK]")
            } else {
                output.append(contentsOf: subtokens)
            }
        }
        return output
    }
}

enum TokenizerError: LocalizedError {
    case invalidJSON
    case missingVocab
    case missingSpecial(String)

    var errorDescription: String? {
        switch self {
        case .invalidJSON: return "tokenizer.json is invalid"
        case .missingVocab: return "tokenizer vocab is empty"
        case .missingSpecial(let t): return "tokenizer is missing special token \(t)"
        }
    }
}
