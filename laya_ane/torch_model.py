"""PyTorch DecisionModel used only for CoreML export and reference tests.

Runtime inference loads CoreML; this module is not imported by Agent.
Shapes are fixed to max_len at export time for reliable CoreML conversion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import EncoderConfig


def build_rope_cache(max_len: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    positions = torch.arange(max_len, dtype=torch.float32)
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    angles = positions[:, None] * freqs[None, :]
    return angles.cos(), angles.sin()


def apply_rope_cached(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, half_dim: int
) -> torch.Tensor:
    # half_dim is a Python int so CoreML never sees dynamic aten::Int casts.
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def attention_masks(
    attention_mask: torch.Tensor, window: int, seq_len: int
) -> dict[str, torch.Tensor]:
    """Additive attention masks (0 keep, -1e4 block). CoreML-safe float ops only."""
    valid_f = attention_mask.to(dtype=torch.float32)
    positions = torch.arange(seq_len, device=valid_f.device, dtype=torch.float32)
    within = ((positions.unsqueeze(1) - positions.unsqueeze(0)).abs() <= float(window // 2)).to(
        dtype=torch.float32
    )
    valid_key = valid_f.unsqueeze(1).unsqueeze(1)
    pad_query = (1.0 - valid_f).unsqueeze(1).unsqueeze(3)
    local_keep = torch.clamp(within.unsqueeze(0).unsqueeze(0) + pad_query, max=1.0) * valid_key
    full_keep = valid_key.expand(-1, 1, seq_len, seq_len)
    neg = -1e4
    full = (1.0 - full_keep) * neg
    local = (1.0 - local_keep) * neg
    return {"full_attention": full, "sliding_attention": local}


def sdpa(q, k, v, mask, scale: float):
    """Explicit attention — more CoreML-portable than fused SDPA."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores + mask.to(dtype=scores.dtype)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


class Embeddings(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def forward(self, ids):
        return self.norm(self.tok_embeddings(ids))


class EncoderAttention(nn.Module):
    def __init__(self, cfg: EncoderConfig, kind: str, max_len: int):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.half_dim = cfg.head_dim // 2
        self.seq_len = int(max_len)
        self.hidden = cfg.hidden_size
        self.scale = self.head_dim**-0.5
        self.Wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=cfg.attention_bias)
        self.Wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.attention_bias)
        cos, sin = build_rope_cache(max_len, self.head_dim, cfg.rope_base(kind))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, mask):
        # seq_len is a Python int so CoreML never sees dynamic int() casts on shapes.
        qkv = self.Wqkv(x).view(-1, self.seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        q = apply_rope_cached(q, self.rope_cos, self.rope_sin, self.half_dim)
        k = apply_rope_cached(k, self.rope_cos, self.rope_sin, self.half_dim)
        out = sdpa(q, k, v, mask, self.scale)
        out = out.permute(0, 2, 1, 3).contiguous().view(-1, self.seq_len, self.hidden)
        return self.Wo(out)


class EncoderMLP(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.Wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=cfg.mlp_bias)
        self.Wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x):
        value, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(value) * gate)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: EncoderConfig, index: int, max_len: int):
        super().__init__()
        self.attention_type = cfg.layer_types[index]
        self.attn_norm = (
            nn.Identity()
            if index == 0
            else nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        )
        self.attn = EncoderAttention(cfg, self.attention_type, max_len)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.mlp = EncoderMLP(cfg)

    def forward(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))


class ModernBert(nn.Module):
    def __init__(self, cfg: EncoderConfig, max_len: int):
        super().__init__()
        self.config = cfg
        self.seq_len = int(max_len)
        self.embeddings = Embeddings(cfg)
        self.layers = nn.ModuleList(
            [EncoderLayer(cfg, i, max_len) for i in range(cfg.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        masks = attention_masks(attention_mask, self.config.local_attention, self.seq_len)
        for layer in self.layers:
            x = layer(x, masks[layer.attention_type])
        return self.final_norm(x)


class HeadAttention(nn.Module):
    def __init__(self, dims: int, seq_len: int):
        super().__init__()
        self.num_heads = max(1, dims // 64)
        if dims % self.num_heads:
            raise ValueError("Decision head dimensions must be divisible by its head count")
        self.head_dim = dims // self.num_heads
        self.seq_len = int(seq_len)
        self.dims = dims
        self.scale = self.head_dim**-0.5
        self.in_proj = nn.Linear(dims, 3 * dims)
        self.out_proj = nn.Linear(dims, dims)

    def forward(self, x, mask):
        qkv = self.in_proj(x).view(-1, self.seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        out = sdpa(q, k, v, mask, self.scale)
        out = out.permute(0, 2, 1, 3).contiguous().view(-1, self.seq_len, self.dims)
        return self.out_proj(out)


class HeadLayer(nn.Module):
    def __init__(self, dims: int, seq_len: int):
        super().__init__()
        self.self_attn = HeadAttention(dims, seq_len)
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.linear1 = nn.Linear(dims, 4 * dims)
        self.linear2 = nn.Linear(4 * dims, dims)

    def forward(self, x, mask):
        x = x + self.self_attn(self.norm1(x), mask)
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class DecisionHead(nn.Module):
    def __init__(self, dims: int, count: int, seq_len: int):
        super().__init__()
        self.layers = nn.ModuleList([HeadLayer(dims, seq_len) for _ in range(count)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecisionModel(nn.Module):
    def __init__(self, encoder_config: EncoderConfig, agent_config: dict):
        super().__init__()
        dims = encoder_config.hidden_size
        max_len = int(agent_config.get("max_len", 512))
        self.max_len = max_len
        self.max_markers = int(agent_config.get("_export_max_markers", 32))
        self.encoder = ModernBert(encoder_config, max_len=max_len)
        self.head = DecisionHead(dims, agent_config.get("head_layers", 2), max_len)
        self.type_emb = nn.Embedding(3, dims)
        self.scorer = nn.Sequential(
            nn.LayerNorm(dims), nn.Linear(dims, dims), nn.GELU(), nn.Linear(dims, 1)
        )
        n_actions = len(agent_config.get("act_costs", {})) + 1
        self.act_head = nn.Sequential(
            nn.Linear(dims + 4, 256),
            nn.GELU(),
            nn.Linear(256, n_actions),
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids, attention_mask)
        h = h + self.type_emb(qtype).unsqueeze(1)
        key_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(1).to(dtype=h.dtype)) * (-1e4)
        h = self.head(h, key_mask)
        # Fixed dims for one_hot / matmul gather.
        pos = marker_pos.clamp(0, self.max_len - 1)
        one_hot = F.one_hot(pos.long(), num_classes=self.max_len).to(dtype=h.dtype)
        markers = torch.matmul(one_hot, h)
        logits = self.scorer(markers).squeeze(-1)
        keep = marker_mask.to(dtype=logits.dtype)
        logits = logits * keep + (-1e4) * (1.0 - keep)
        p = torch.softmax(logits, dim=-1)
        k = marker_mask.to(dtype=torch.float32).sum(dim=-1).clamp_min(2.0)
        entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(dim=-1) / torch.log(k)
        top2 = torch.topk(p, k=2, dim=-1).values
        features = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], entropy, k / 255.0], dim=-1)
        pooled = torch.cat([h[:, 0], features.to(dtype=h.dtype)], dim=-1)
        action = self.act_head(pooled)
        return logits.float(), action.float()


def load_state_dict_from_numpy(model: DecisionModel, weights: dict) -> None:
    """Load sanitized numpy/torch tensors into the model."""
    state = {}
    for name, value in weights.items():
        if not hasattr(value, "detach"):
            value = torch.as_tensor(value)
        state[name] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if m != "temperature" and not m.startswith("rope_")]
    # rope buffers are regenerated; ignore missing non-persistent buffers
    missing = [m for m in missing if "rope_cos" not in m and "rope_sin" not in m]
    if missing or unexpected:
        raise ValueError(f"Weight mismatch: missing={missing!r} unexpected={unexpected!r}")
