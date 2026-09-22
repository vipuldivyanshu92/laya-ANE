"""Export a Laya checkpoint to CoreML for Apple Neural Engine inference."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from safetensors.torch import load_file

from .model import EncoderConfig, sanitize_weights
from .torch_model import DecisionModel, load_state_dict_from_numpy


def _load_numpy_weights(path: Path) -> dict:
    raw = load_file(str(path))
    return sanitize_weights({k: v.cpu().numpy() for k, v in raw.items()})


def convert(
    model_id_or_path,
    output,
    *,
    dtype="float16",
    revision=None,
    subfolder=None,
    token=None,
    max_batch=8,
    max_markers=32,
    max_len=None,
    quantize=None,
):
    """Convert safetensors (+ configs/tokenizer) into a CoreML DecisionModel package.

    Sequence length is fixed to ``max_len`` (checkpoint default, or override for on-device).
    """
    from .agent import resolve_model

    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    source = resolve_model(
        model_id_or_path, token=token, subfolder=subfolder, revision=revision, require_coreml=False
    )
    cfg = json.loads((source / "rl_agent_config.json").read_text())
    encoder_cfg = EncoderConfig.from_dict(json.loads((source / "encoder/config.json").read_text()))
    if max_len is not None:
        cfg = {**cfg, "max_len": int(max_len)}
    max_len = int(cfg.get("max_len", 512))
    head_max = int(cfg.get("head_max_len", 192))
    if not (4 < head_max < max_len):
        cfg["head_max_len"] = max(16, min(head_max, max_len // 2 - 1))
    if max_markers < 2:
        raise ValueError("max_markers must be at least 2")
    export_cfg = {**cfg, "_export_max_markers": max_markers}

    torch_dtype = torch.float32  # CoreML LayerNorm requires fp32 eps; quantize via compute_precision.
    model = DecisionModel(encoder_cfg, export_cfg).eval()
    weights = _load_numpy_weights(source / "model.safetensors")
    load_state_dict_from_numpy(model, weights)
    model = model.to(dtype=torch_dtype)

    # Trace at a representative batch; CoreML RangeDim covers other batch sizes.
    example = (
        torch.zeros(1, max_len, dtype=torch.int32),
        torch.ones(1, max_len, dtype=torch.float32),
        torch.zeros(1, max_markers, dtype=torch.int32),
        torch.zeros(1, max_markers, dtype=torch.float32),
        torch.zeros(1, dtype=torch.int32),
    )
    example[3][0, :2] = 1.0

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            logits, action = self.inner(input_ids, attention_mask, marker_pos, marker_mask, qtype)
            return logits, action

    wrapped = _Wrapper(model).eval()
    traced = torch.jit.trace(wrapped, example, strict=False)

    inputs = [
        ct.TensorType(name="input_ids", shape=(1, max_len), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, max_len), dtype=np.float32),
        ct.TensorType(name="marker_pos", shape=(1, max_markers), dtype=np.int32),
        ct.TensorType(name="marker_mask", shape=(1, max_markers), dtype=np.float32),
        ct.TensorType(name="qtype", shape=(1,), dtype=np.int32),
    ]
    outputs = [ct.TensorType(name="logits"), ct.TensorType(name="action")]
    precision = ct.precision.FLOAT16 if dtype == "float16" else ct.precision.FLOAT32
    # iOS 17 / macOS 14 mlprogram — ANE-capable on iPhone 15 (A16).
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=precision,
    )

    if quantize == "palettize6":
        # 6-bit weight palettization: smaller + typically faster on ANE with small quality loss.
        op_config = ct.optimize.coreml.OpPalettizerConfig(mode="kmeans", nbits=6)
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config)

    output.mkdir(parents=True)
    try:
        shutil.copytree(source / "tokenizer", output / "tokenizer")
        (output / "encoder").mkdir()
        (output / "encoder/config.json").write_text(
            json.dumps(json.loads((source / "encoder/config.json").read_text()), indent=2) + "\n"
        )
        (output / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        mlmodel.save(str(output / "DecisionModel.mlpackage"))
        # Do not ship safetensors in the iOS bundle — CoreML package is enough.
        metadata = {
            "format": "laya-ane",
            "format_version": 1,
            "dtype": dtype,
            "backend": "coreml",
            "compute_units": "ALL",
            "quantize": quantize,
            "max_batch": max_batch,
            "max_markers": max_markers,
            "max_len": max_len,
            "fixed_sequence_length": True,
            "source": str(model_id_or_path),
            "revision": revision,
            "subfolder": subfolder,
            "mlpackage": "DecisionModel.mlpackage",
            "profile": "iphone15" if max_len <= 192 else "default",
        }
        (output / "ane_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return output
