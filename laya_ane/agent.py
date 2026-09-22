"""Public CoreML inference runtime; prompt and result formats follow upstream Laya."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath

import numpy as np
from huggingface_hub import snapshot_download

from .common import QTYPES, build_sequence, confidence_from_probs, render_options, temp_bucket
from .coreml_runtime import CoreMLDecisionModel
from .prepared import PrefixCache
from .tokenizer import Tokenizer

DTYPES = {"float32": "float32", "float16": "float16", "bfloat16": "float16"}


def resolve_model(
    model_id_or_path, *, token=None, subfolder=None, revision=None, require_coreml=True
):
    if subfolder:
        part = PurePosixPath(subfolder)
        if part.is_absolute() or ".." in part.parts:
            raise ValueError("subfolder must be a relative path inside the model repository")
    path = Path(model_id_or_path).expanduser()
    if not path.exists():
        value = str(model_id_or_path)
        if value.startswith(("/", "./", "../", "~")) or isinstance(model_id_or_path, Path):
            raise FileNotFoundError(f"Local model directory does not exist: {value}")
        prefix = subfolder.rstrip("/") + "/" if subfolder else ""
        patterns = [
            prefix + name
            for name in (
                "model.safetensors",
                "DecisionModel.mlpackage/*",
                "DecisionModel.mlpackage/**",
                "rl_agent_config.json",
                "encoder/config.json",
                "tokenizer/*",
                "ane_config.json",
                "mlx_config.json",
            )
        ]
        path = Path(
            snapshot_download(value, token=token, revision=revision, allow_patterns=patterns)
        )
    if subfolder:
        path /= subfolder
    for name in ("rl_agent_config.json", "encoder/config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Not a complete Laya checkpoint: {path / name} is missing")
    if require_coreml:
        package = path / "DecisionModel.mlpackage"
        if not package.exists():
            raise FileNotFoundError(
                f"CoreML package missing at {package}. Convert first:\n"
                f"  laya-ane convert --model {model_id_or_path} --output /path/to/coreml-ckpt"
            )
    elif (
        not (path / "model.safetensors").is_file()
        and not (path / "DecisionModel.mlpackage").exists()
    ):
        raise FileNotFoundError(
            f"Checkpoint needs model.safetensors or DecisionModel.mlpackage under {path}"
        )
    return path


def collate_items(items, pad_id, *, pad_to_multiple=None, max_length=None, max_markers=None):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    n = len(items)
    # CoreML export uses a fixed sequence length (= checkpoint max_len).
    if max_length is None:
        length = max(len(item["ids"]) for item in items)
        if pad_to_multiple:
            length = ((length + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    else:
        length = int(max_length)
    count = max(2, max(len(item["markers"]) for item in items))
    if max_markers is not None:
        if count > max_markers:
            raise ValueError(
                f"Batch needs {count} marker slots but CoreML export max is {max_markers}"
            )
        count = max_markers
    batch = {
        "input_ids": np.full((n, length), pad_id, dtype=np.int32),
        "attention_mask": np.zeros((n, length), dtype=np.float32),
        "marker_pos": np.zeros((n, count), dtype=np.int32),
        "marker_mask": np.zeros((n, count), dtype=np.float32),
        "qtype": np.array([item["qtype"] for item in items], dtype=np.int32),
    }
    for i, item in enumerate(items):
        seq = item["ids"][:length]
        seq_len, marker_count = len(seq), len(item["markers"])
        batch["input_ids"][i, :seq_len] = seq
        batch["attention_mask"][i, :seq_len] = 1.0
        batch["marker_pos"][i, :marker_count] = item["markers"]
        batch["marker_mask"][i, :marker_count] = 1.0
    return batch


class Agent:
    def __init__(
        self,
        model_id_or_path="convaiinnovations/laya",
        device=None,
        token=None,
        subfolder=None,
        *,
        dtype="float16",
        revision=None,
        batch_size=16,
        compile=False,
        pad_to_multiple=None,
        cache_prompts=False,
    ):
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {list(DTYPES)}")
        if device not in (None, "all", "ane", "gpu", "metal", "cpu"):
            raise ValueError("device must be 'all', 'ane', 'gpu', 'metal', or 'cpu'")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        # compile is accepted for API parity with laya-mlx; CoreML graphs are already compiled.
        self.compile = bool(compile)
        self.device = device
        self.dtype = DTYPES[dtype]
        self.batch_size = batch_size
        if pad_to_multiple is not None and (
            not isinstance(pad_to_multiple, int)
            or isinstance(pad_to_multiple, bool)
            or pad_to_multiple < 1
        ):
            raise ValueError("pad_to_multiple must be a positive integer or None")
        self.pad_to_multiple = pad_to_multiple
        self._prefix_cache = PrefixCache() if cache_prompts else None
        self.model_id = str(model_id_or_path)
        self.revision = revision
        self.model_dir = resolve_model(
            model_id_or_path, token=token, subfolder=subfolder, revision=revision
        )
        self.cfg = json.loads((self.model_dir / "rl_agent_config.json").read_text())
        self.encoder_cfg = json.loads((self.model_dir / "encoder/config.json").read_text())
        if "encoder" not in self.cfg or "head_layers" not in self.cfg:
            raise ValueError("Laya config must specify encoder and head_layers")
        from .model import EncoderConfig

        enc_cfg = EncoderConfig.from_dict(self.encoder_cfg)
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        if not 4 < head_max_len < max_len <= enc_cfg.max_position_embeddings:
            raise ValueError("Expected 4 < head_max_len < max_len <= max_position_embeddings")
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        if len(self.temperature) != 3 or any(
            not math.isfinite(float(t)) or float(t) <= 0
            for t in [*self.temperature, *self.temperature_by_options.values()]
        ):
            raise ValueError("Calibration temperatures must be finite and positive")
        self.tok = Tokenizer(self.model_dir / "tokenizer")
        ane_meta = {}
        meta_path = self.model_dir / "ane_config.json"
        if meta_path.is_file():
            ane_meta = json.loads(meta_path.read_text())
        self.max_markers = int(ane_meta.get("max_markers", 32))
        self.model = CoreMLDecisionModel(self.model_dir / "DecisionModel.mlpackage", device=device)

    @staticmethod
    def _to_internal(qdef):
        if not isinstance(qdef, dict):
            raise ValueError("Each question must be a dictionary")
        kind = qdef.get("type")
        if kind not in QTYPES:
            raise ValueError(f"Unknown question type {kind!r}; expected choice, score, or noul")
        if "instructions" not in qdef:
            raise ValueError("Question is missing instructions")
        criteria = qdef.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                if not all(isinstance(c, str) for c in criteria):
                    raise ValueError("Choice labels must be strings")
                if len(set(criteria)) != len(criteria):
                    raise ValueError("Choice labels must be unique")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("Choice criteria must be a nonempty dictionary or list")
            if not all(isinstance(k, str) for k in criteria):
                raise ValueError("Choice labels must be strings")
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("Score criteria must be a nonempty list")
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError("Noul criteria must be a dictionary with false/true descriptions")
        instructions = qdef["instructions"]
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions)
        return {"t": kind, "ins": instructions, "crit": criteria}

    def prepare(self, state, questions):
        """Construct upstream-compatible CPU inputs, useful for parity and profiling."""
        if self._prefix_cache is not None:
            return self._prefix_cache.prepare(self, state, questions)
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id")
        items, internal = [], []
        for qid, definition in questions.items():
            q = self._to_internal(definition)
            ids, markers = build_sequence(
                self.tok, state, q, self.cfg.get("max_len", 512), self.cfg.get("head_max_len", 192)
            )
            if len(markers) != len(render_options(q)):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            internal.append(q)
        return items, internal

    def forward(self, batch):
        """Run one prepared batch and return numpy (logits, action)."""
        return self.model.predict(batch)

    def system_one(self, state, questions):
        items, internal = self.prepare(state, questions)
        answers = {}
        question_ids = list(questions)
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            batch = collate_items(
                chunk,
                self.tok.pad_token_id,
                pad_to_multiple=self.pad_to_multiple,
                max_length=self.cfg.get("max_len", 512),
                max_markers=self.max_markers,
            )
            logits, act = self.forward(batch)
            logits, act = np.asarray(logits), np.asarray(act)
            if not np.isfinite(logits).all() or not np.isfinite(act).all():
                raise FloatingPointError("Non-finite model outputs; retry with dtype='float32'")
            act = np.exp(act - act.max(axis=-1, keepdims=True))
            act /= act.sum(axis=-1, keepdims=True)
            for row, item in enumerate(chunk):
                qid, q = question_ids[start + row], internal[start + row]
                k, qt = len(item["markers"]), item["qtype"]
                scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
                z = logits[row, :k] / max(1e-3, float(scale))
                p = np.exp(z - z.max())
                p /= p.sum()
                answer = {
                    "type": q["t"],
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "action": {"act_probability": round(float(act[row, 0]), 4)},
                }
                if q["t"] == "choice":
                    labels = list(q["crit"])
                    answer.update(
                        choice=labels[int(p.argmax())],
                        probabilities={label: round(float(v), 4) for label, v in zip(labels, p)},
                    )
                elif q["t"] == "score":
                    answer.update(
                        score=round(float((np.arange(k) * p).sum()), 4),
                        legend={str(i): value for i, value in enumerate(q["crit"])},
                        probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    )
                else:
                    answer.update(
                        noul=round(float(p[1]), 4),
                        confidence=round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    )
                answers[qid] = answer
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": sum(len(item["ids"]) for item in items), "output_tokens": 0},
        }

    predict = system_one


RLAgent = Agent


def load(
    model_id_or_path="convaiinnovations/laya", device=None, token=None, subfolder=None, **kwargs
):
    return Agent(model_id_or_path, device=device, token=token, subfolder=subfolder, **kwargs)
