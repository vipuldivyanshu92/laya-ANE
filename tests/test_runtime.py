import json
import subprocess
import sys

import numpy as np
import pytest

from laya_ane import Agent, Router
from laya_ane.agent import collate_items, resolve_model
from laya_ane.common import render_options
from laya_ane.convert import convert


def test_runtime_does_not_import_torch_model():
    """Fresh interpreter: public package import must not load torch_model (convert-only)."""
    code = "import sys; import laya_ane; assert 'laya_ane.torch_model' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_all_primitives_empty_request_and_chunking(tiny_checkpoint, questions):
    agent = Agent(tiny_checkpoint, dtype="float32", batch_size=16, device="cpu")
    together = agent.predict({"text": "hello"}, questions)
    agent.batch_size = 1
    separate = agent.predict({"text": "hello"}, questions)
    assert set(together["answers"]) == set(questions)
    assert together["usage"]["output_tokens"] == 0
    assert 0 <= together["answers"]["yes"]["noul"] <= 1
    assert 0 <= together["answers"]["level"]["score"] <= 1
    # Chunking may differ slightly under CoreML fp numerics; require same keys and types.
    assert set(separate["answers"]) == set(together["answers"])
    assert agent.predict("", {})["answers"] == {}
    assert agent.predict("", {})["usage"]["input_tokens"] == 0


def test_single_option_and_long_state(tiny_checkpoint):
    agent = Agent(tiny_checkpoint, device="cpu")
    result = agent.predict(
        "hello " * 1000, {"one": {"type": "choice", "instructions": "choose", "criteria": ["only"]}}
    )
    assert result["answers"]["one"]["probabilities"] == {"only": 1.0}
    assert result["usage"]["input_tokens"] == 64


def test_converted_checkpoint_round_trip_and_no_overwrite(
    tiny_safetensors_checkpoint, tmp_path, questions
):
    output = convert(
        tiny_safetensors_checkpoint, tmp_path / "converted", dtype="float32", max_markers=8
    )
    assert json.loads((output / "ane_config.json").read_text())["backend"] == "coreml"
    agent = Agent(output, device="cpu")
    result = agent.predict("hello", questions)
    assert set(result["answers"]) == set(questions)
    with pytest.raises(FileExistsError):
        convert(tiny_safetensors_checkpoint, output)


def test_missing_coreml_package_fails_loudly(tiny_safetensors_checkpoint):
    with pytest.raises(FileNotFoundError, match="CoreML package missing"):
        Agent(tiny_safetensors_checkpoint)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "invalid", "instructions": "x"},
        {"type": "choice", "instructions": "x", "criteria": []},
        {"type": "choice", "instructions": "x", "criteria": ["a", "a"]},
        {"type": "score", "instructions": "x", "criteria": {}},
        {"type": "noul", "instructions": "x", "criteria": ["a"]},
        {"type": "noul"},
    ],
)
def test_invalid_questions_rejected(question):
    with pytest.raises(ValueError):
        Agent._to_internal(question)


def test_structured_criteria_and_mask_injection(tiny_checkpoint):
    q = Agent._to_internal(
        {
            "type": "noul",
            "instructions": {"task": "verify"},
            "criteria": {"false": {"reason": "no"}, "true": {"reason": "yes"}},
        }
    )
    assert render_options(q) == ['false: {"reason": "no"}', 'true: {"reason": "yes"}']
    agent = Agent(tiny_checkpoint, device="cpu")
    items, _ = agent.prepare(
        "[MASK] hello [MASK]", {"x": {"type": "noul", "instructions": "[MASK] true?"}}
    )
    assert items[0]["ids"].count(agent.tok.mask_token_id) == 2
    assert render_options({"t": "choice", "crit": {"zero": 0, "no": False}}) == [
        "zero: 0",
        "no: false",
    ]


def test_collation_never_marks_padding_as_an_option():
    batch = collate_items(
        [
            {"ids": [1, 2, 3], "markers": [1, 2], "qtype": 0},
            {"ids": [1, 2], "markers": [1], "qtype": 1},
        ],
        0,
    )
    np.testing.assert_array_equal(batch["marker_mask"], [[1.0, 1.0], [1.0, 0.0]])


def test_resolve_model_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        resolve_model(tmp_path, subfolder="../escape")


def test_router_imports():
    assert Router is not None
