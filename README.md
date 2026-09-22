# Laya-ANE

**Open-weight typed decisions on Apple Silicon via CoreML / Neural Engine.**

Same Laya API as [laya-mlx](https://github.com/mizorewww/laya-mlx): `load()`, `predict()`, presets, router, snake — forward pass as a CoreML `mlprogram` so Apple can schedule work on the **Neural Engine**.

Inference only. On-device training / fine-tuning is future work.

## Quick start (macOS)

```bash
pip install 'laya-ane[convert]'
laya-ane convert --model aac6fef/laya-typed-decisions-mlx --output models/laya-coreml --dtype float16
```

```python
import laya_ane as laya

agent = laya.load("models/laya-coreml")
print(agent.predict(
    "I was billed twice. Please refund the duplicate.",
    {"department": {
        "type": "choice",
        "instructions": "Who should handle this?",
        "criteria": ["billing", "technical", "sales"],
    }},
)["answers"]["department"])
```

## iOS app (iPhone)

SwiftUI demo: **Inbox Copilot**, **Live Guard**, **Snake × Laya**.

```bash
pip install -e '.[convert]'
python scripts/prepare_ios_bundle.py   # typed-decisions + FP16 + 6-bit palettize
open ios/LayaANE/LayaANE.xcodeproj
```

See [`ios/README.md`](ios/README.md). Model weights are **not** in git — `prepare_ios_bundle.py` downloads and converts them locally.

## Why CoreML / ANE?

| | laya-mlx | laya-ane |
|---|---|---|
| Runtime | MLX (Metal) | CoreML → ANE / GPU / CPU |
| API | `Agent` / `load` / `predict` | Same surface |
| Mobile | — | SwiftUI iPhone demo |

## License

Apache-2.0. See [NOTICE](NOTICE).
