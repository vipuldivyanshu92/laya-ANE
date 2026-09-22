#!/usr/bin/env python3
"""Build a CoreML + tokenizer bundle for the LayaANE iOS app.

Default profile targets iPhone 15:
  aac6fef/laya-typed-decisions-mlx  (421M ModernBERT-large, fine-tuned for typed decisions)
  float16 + 6-bit weight palettization, max_len=160

That checkpoint is the best quality match for Inbox / Guard / Snake, and still fits
comfortably in iPhone 15's ~6 GB RAM when exported this way.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "ios" / "LayaANE" / "LayaANE" / "Resources" / "Model"

# Best on-device default for English typed-decision demos on iPhone 15.
IPHONE15_HUB = "aac6fef/laya-typed-decisions-mlx"


def make_tiny_safetensors(path: Path) -> None:
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    from laya_ane.model import EncoderConfig
    from laya_ane.torch_model import DecisionModel

    cfg = {
        "model_type": "modernbert",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 1,
        "local_attention": 16,
        "max_position_embeddings": 256,
    }
    agent_cfg = {
        "encoder": "ios/tiny",
        "head_layers": 1,
        "max_len": 64,
        "head_max_len": 24,
        "act_costs": {"escalate": 0.5},
        "temperature": [1.0, 1.0, 1.0],
    }
    path.mkdir(parents=True)
    (path / "encoder").mkdir()
    (path / "tokenizer").mkdir()
    (path / "encoder/config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (path / "rl_agent_config.json").write_text(json.dumps(agent_cfg, indent=2) + "\n")
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "refund", "billing"]
    while len(words) < 128:
        words.append(f"tok{len(words)}")
    vocab = {t: i for i, t in enumerate(words[:128])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer/tokenizer.json"))
    (path / "tokenizer/tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            },
            indent=2,
        )
        + "\n"
    )
    torch.manual_seed(7)
    model = DecisionModel(EncoderConfig.from_dict(cfg), {**agent_cfg, "_export_max_markers": 8}).eval()
    save_file(
        {k: v.detach().contiguous() for k, v in model.state_dict().items()},
        str(path / "model.safetensors"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true", help="Untrained stub (debug only)")
    parser.add_argument(
        "--hub",
        default=None,
        help=f"HF repo (default iPhone 15 profile: {IPHONE15_HUB})",
    )
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--max-markers", type=int, default=16)
    parser.add_argument(
        "--max-len",
        type=int,
        default=160,
        help="Fixed sequence length (160 fits email/snake on iPhone 15 well)",
    )
    parser.add_argument(
        "--quantize",
        choices=("none", "palettize6"),
        default="palettize6",
        help="ANE-oriented weight compression (default: 6-bit palettize)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Keep full float16 weights (larger, slightly higher fidelity)",
    )
    args = parser.parse_args()
    quantize = None if args.no_quantize or args.quantize == "none" else args.quantize

    sys.path.insert(0, str(ROOT))
    from laya_ane.convert import convert

    staging = ROOT / ".cache" / "ios-convert"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    if args.tiny:
        tiny = staging / "tiny"
        print("Building tiny untrained stub (not for quality demos) …")
        make_tiny_safetensors(tiny)
        convert(
            tiny,
            staging / "coreml",
            dtype="float32",
            max_batch=1,
            max_markers=args.max_markers,
            max_len=64,
            quantize=None,
        )
    else:
        source = args.hub or IPHONE15_HUB
        print(f"iPhone 15 profile → {source}")
        print(f"  max_len={args.max_len}  dtype={args.dtype}  quantize={quantize}")
        convert(
            source,
            staging / "coreml",
            dtype=args.dtype,
            max_batch=1,
            max_markers=args.max_markers,
            max_len=args.max_len,
            quantize=quantize,
        )

    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(staging / "coreml", DEST)
    print(f"Wrote iOS model bundle → {DEST}")
    print("Rebuild the Xcode app (Product → Clean Build Folder, then Run).")


if __name__ == "__main__":
    main()
