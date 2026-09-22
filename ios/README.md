# Laya ANE — iOS app

SwiftUI demo that runs **Laya typed decisions on-device via CoreML** (Neural Engine when available).

Note: [ANE](https://github.com/vipuldivyanshu92/ANE) is an ANE *training* research stack, not an iOS UI. This app is the inference counterpart for Laya — local, private, zero cloud.

## What it does

| Tab | Demo |
|---|---|
| **Inbox Copilot** | Paste support mail → category, spam, phishing, urgency, needs-reply with probability bars + ms latency |
| **Live Guard** | As-you-type jailbreak / harm / toxicity scoring |
| **Snake × Laya** | Classic snake; each move is a Laya `choice` over UP/DOWN/LEFT/RIGHT |

## Setup

### 1. Bundle the iPhone 15 model (default)

```bash
source .venv/bin/activate
python scripts/prepare_ios_bundle.py
```

This converts **`aac6fef/laya-typed-decisions-mlx`** (421M ModernBERT-large, fine-tuned for typed decisions) to CoreML with:

- `float16` + **6-bit weight palettization** (ANE-friendly)
- `max_len=160` (enough for inbox/snake/guard on phone)
- iOS 17 deployment target

Rebuild the Xcode app afterward.

| Checkpoint | Why |
|---|---|
| **typed-decisions (default)** | Best quality for triage / choice / score / noul workflows |
| `aac6fef/laya-mlx` | Same 421M English backbone, general (not task-tuned) |
| `aac6fef/laya-multilingual-mlx` | Smaller/faster, better non-English; weaker on English |

```bash
# Full float16 (no palettize) if you want max fidelity
python scripts/prepare_ios_bundle.py --no-quantize

# Multilingual instead
python scripts/prepare_ios_bundle.py --hub aac6fef/laya-multilingual-mlx --max-len 128
```

### 2. Open in Xcode

```bash
open ios/LayaANE/LayaANE.xcodeproj
```

1. Select your **Team** under Signing (left empty in git — set yours locally)
2. Pick an **iPhone** (Neural Engine) or simulator
3. Product → Clean Build Folder, then Run

Requires **Xcode 15+**, **iOS 17+**. Full Xcode (not Command Line Tools only).

## Architecture

```
SwiftUI tabs
    ↓
LayaEngine  (prompt build + calibration)
    ↓
CoreML DecisionModel.mlpackage  →  Apple Neural Engine / GPU / CPU
```

Prompt formatting matches `laya_ane.common.build_sequence`. Tokenizer loads the checkpoint's `tokenizer.json` (ByteLevel BPE for ModernBERT; WordPiece/WordLevel for tiny test stubs).

## Tips

- Simulator runs CoreML on CPU — good for UI; timing looks best on a real iPhone.
- Tiny bundle answers are random (untrained weights). Use `--hub` for meaningful triage.
- If the About tab says “Model missing”, re-run `prepare_ios_bundle.py`.
