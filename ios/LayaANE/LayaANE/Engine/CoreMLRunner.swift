import CoreML
import Foundation

/// Runs a converted `DecisionModel.mlpackage` (fixed batch=1, fixed max_len).
final class CoreMLRunner {
    private let model: MLModel
    let maxLen: Int
    let maxMarkers: Int

    struct Outputs {
        let logits: [Float]
        let action: [Float]
    }

    init(modelDirectory: URL, computeUnits: MLComputeUnits = .all) throws {
        let package = modelDirectory.appendingPathComponent("DecisionModel.mlpackage")
        let compiled: URL
        if FileManager.default.fileExists(atPath: package.path) {
            compiled = try MLModel.compileModel(at: package)
        } else {
            // Already-compiled .mlmodelc
            let mlmodelc = modelDirectory.appendingPathComponent("DecisionModel.mlmodelc")
            compiled = mlmodelc
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: compiled, configuration: config)

        let metaURL = modelDirectory.appendingPathComponent("ane_config.json")
        if let data = try? Data(contentsOf: metaURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            maxLen = json["max_len"] as? Int ?? 64
            maxMarkers = json["max_markers"] as? Int ?? 8
        } else {
            maxLen = 64
            maxMarkers = 8
        }
    }

    func predict(
        inputIds: [Int32],
        attentionMask: [Float],
        markerPos: [Int32],
        markerMask: [Float],
        qtype: Int32
    ) throws -> Outputs {
        let idsArr = try MLMultiArray(shape: [1, NSNumber(value: maxLen)], dataType: .int32)
        let maskArr = try MLMultiArray(shape: [1, NSNumber(value: maxLen)], dataType: .float32)
        let posArr = try MLMultiArray(shape: [1, NSNumber(value: maxMarkers)], dataType: .int32)
        let mMaskArr = try MLMultiArray(shape: [1, NSNumber(value: maxMarkers)], dataType: .float32)
        let qArr = try MLMultiArray(shape: [1], dataType: .int32)

        for i in 0..<maxLen {
            let idx = [0, i] as [NSNumber]
            idsArr[idx] = NSNumber(value: i < inputIds.count ? inputIds[i] : 0)
            maskArr[idx] = NSNumber(value: i < attentionMask.count ? attentionMask[i] : Float(0))
        }
        for i in 0..<maxMarkers {
            let idx = [0, i] as [NSNumber]
            posArr[idx] = NSNumber(value: i < markerPos.count ? markerPos[i] : 0)
            mMaskArr[idx] = NSNumber(value: i < markerMask.count ? markerMask[i] : Float(0))
        }
        qArr[[0] as [NSNumber]] = NSNumber(value: qtype)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArr),
            "attention_mask": MLFeatureValue(multiArray: maskArr),
            "marker_pos": MLFeatureValue(multiArray: posArr),
            "marker_mask": MLFeatureValue(multiArray: mMaskArr),
            "qtype": MLFeatureValue(multiArray: qArr),
        ])
        let out = try model.prediction(from: provider)
        let logits = multiArrayToFloats(out.featureValue(for: "logits")?.multiArrayValue)
        let action = multiArrayToFloats(out.featureValue(for: "action")?.multiArrayValue)
        return Outputs(logits: logits, action: action)
    }

    private func multiArrayToFloats(_ array: MLMultiArray?) -> [Float] {
        guard let array else { return [] }
        let count = array.count
        var result = [Float](repeating: 0, count: count)
        for i in 0..<count {
            result[i] = array[i].floatValue
        }
        return result
    }
}
