import json
import os

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers

from laya_ane.convert import convert
from laya_ane.model import EncoderConfig
from laya_ane.torch_model import DecisionModel


@pytest.fixture
def tiny_safetensors_checkpoint(tmp_path):
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
        "encoder": "test/tiny",
        "head_layers": 1,
        "max_len": 64,
        "head_max_len": 24,
        "act_costs": {"escalate": 0.5},
        "temperature": [1.3, 1.1, 2.0],
        "temperature_by_options": {"choice:2": 1.7},
    }
    path = tmp_path / "safetensors"
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir()
    (path / "encoder/config.json").write_text(json.dumps(cfg))
    (path / "rl_agent_config.json").write_text(json.dumps(agent_cfg))
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello"])}
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
            }
        )
    )
    torch.manual_seed(7)
    model = DecisionModel(EncoderConfig.from_dict(cfg), agent_cfg).eval()
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path / "model.safetensors"))
    return path


@pytest.fixture
def tiny_checkpoint(tiny_safetensors_checkpoint, tmp_path):
    """CoreML-converted tiny checkpoint used by most runtime tests."""
    if os.environ.get("LAYA_ANE_SKIP_COREML") == "1":
        pytest.skip("LAYA_ANE_SKIP_COREML=1")
    out = tmp_path / "coreml"
    convert(
        tiny_safetensors_checkpoint,
        out,
        dtype="float32",
        max_batch=4,
        max_markers=8,
    )
    return out


@pytest.fixture
def questions():
    return {
        "topic": {"type": "choice", "instructions": "Choose", "criteria": ["a", "b", "c"]},
        "level": {"type": "score", "instructions": "Level", "criteria": ["low", "high"]},
        "yes": {"type": "noul", "instructions": "Is this true?"},
    }
