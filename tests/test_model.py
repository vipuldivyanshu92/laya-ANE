import numpy as np
import pytest
import torch

from laya_ane.model import EncoderConfig, sanitize_weights
from laya_ane.torch_model import HeadLayer, ModernBert, attention_masks

transformers = pytest.importorskip("transformers")


def torch_config(**kwargs):
    return transformers.ModernBertConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=1,
        local_attention=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        cls_token_id=2,
        sep_token_id=3,
        **kwargs,
    )


@pytest.mark.parametrize("local_theta", [10000.0, 160000.0])
def test_encoder_matches_transformers_across_window_and_padding(local_theta):
    torch.manual_seed(0)
    cfg = torch_config(
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": local_theta},
        }
    )
    reference = transformers.ModernBertModel(cfg).eval()
    model = ModernBert(EncoderConfig.from_dict(cfg.to_dict()), max_len=145).eval()
    state = {k: v.detach() for k, v in reference.state_dict().items()}
    model.load_state_dict(state, strict=True)
    ids = np.random.default_rng(1).integers(1, 128, size=(2, 145)).astype(np.int64)
    mask = np.ones((2, 145), np.float32)
    mask[1, 13:] = 0
    with torch.inference_mode():
        expected = reference(
            torch.tensor(ids), attention_mask=torch.tensor(mask.astype(np.int64))
        ).last_hidden_state.numpy()
        actual = model(torch.tensor(ids), torch.tensor(mask)).numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual[mask.astype(bool)], expected[mask.astype(bool)], atol=2e-4, rtol=2e-4
    )


def test_decision_head_uses_pre_norm_relu_and_biases():
    torch.manual_seed(1)
    reference = torch.nn.TransformerEncoderLayer(
        64, 1, 256, batch_first=True, norm_first=True
    ).eval()
    model = HeadLayer(64, seq_len=19).eval()
    weights = sanitize_weights({k: v.detach() for k, v in reference.state_dict().items()})
    model.load_state_dict(weights, strict=True)
    x = np.random.default_rng(2).normal(size=(2, 19, 64)).astype(np.float32)
    mask = np.ones((2, 19), bool)
    mask[1, 6:] = False
    with torch.inference_mode():
        expected = reference(torch.tensor(x), src_key_padding_mask=torch.tensor(~mask)).numpy()
        keep = torch.tensor(mask)[:, None, None, :].to(dtype=torch.float32)
        additive = (1.0 - keep) * (-1e4)
        actual = model(torch.tensor(x), additive).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)


def test_local_attention_includes_both_boundary_tokens():
    mask = torch.ones(1, 140, dtype=torch.float32)
    local = attention_masks(mask, 128, seq_len=140)["sliding_attention"][0, 0]
    assert float(local[70, 6]) == 0.0 and float(local[70, 134]) == 0.0
    assert float(local[70, 5]) < 0 and float(local[70, 135]) < 0
