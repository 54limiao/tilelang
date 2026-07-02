from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from safetensors import safe_open

from examples.qwen3_int_only.utils.quarot import (
    ROTATE_SEED,
    random_hadamard_rotation,
    hadamard_rotation,
    fast_hadamard,
    rotate_block_input,
    rotate_head_input,
    rotate_head_output,
    rotate_input,
    rotate_norm_input,
    rotate_output,
)

Q15_16 = 1 << 16
QT_SHIFT = 25
QT_WIDTH = 26


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    vocab_size: int = 151936

    @property
    def q_size(self):
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self):
        return self.num_key_value_heads * self.head_dim

    @staticmethod
    def from_model_dir(model_dir):
        data = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        return Qwen3Config(
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=int(data.get("head_dim", data["hidden_size"] // data["num_attention_heads"])),
            rope_theta=float(data.get("rope_theta", 1_000_000.0)),
            vocab_size=int(data["vocab_size"]),
        )


QWEN3_0_6B = Qwen3Config()


class SafeTensorReader:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        index = self.model_dir / "model.safetensors.index.json"
        self.weight_map = None
        self.files = {}
        if index.exists():
            self.weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        else:
            self.single_file = self.model_dir / "model.safetensors"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        for f in self.files.values():
            f.__exit__(exc_type, exc, tb)

    def _file_for(self, name):
        if self.weight_map is None:
            return self.single_file.name
        return self.weight_map[name]

    def get_tensor(self, name, device="cpu"):
        file_name = self._file_for(name)
        if file_name not in self.files:
            f = safe_open(str(self.model_dir / file_name), framework="pt", device="cpu")
            self.files[file_name] = f.__enter__()
        return self.files[file_name].get_tensor(name).to(device)


def q15_16(x):
    return torch.clamp(torch.round(x * Q15_16), -(1 << 31), (1 << 31) - 1).to(torch.int32)


def ratio_qt(numer, denom):
    n = numer.to(torch.int64).clamp(min=1)
    d = denom.to(torch.int64).clamp(min=1)
    return pack_qt(n.to(torch.float64) / d.to(torch.float64))


def pack_qt(ratio):
    ratio = ratio.to(torch.float64) if isinstance(ratio, torch.Tensor) else torch.tensor(ratio, dtype=torch.float64)
    ratio = ratio.clamp(min=2.0**-63)
    best_mul = torch.zeros(ratio.shape, dtype=torch.int64, device=ratio.device)
    best_shift = torch.zeros(ratio.shape, dtype=torch.int64, device=ratio.device)
    best_err = torch.full_like(ratio, float("inf"), dtype=torch.float64)
    for shift in range(64):
        mul_f = torch.round(ratio * float(1 << shift))
        valid = (mul_f >= 1.0) & (mul_f < float(1 << QT_WIDTH))
        mul = torch.where(valid, mul_f, torch.zeros_like(mul_f)).to(torch.int64)
        err = torch.abs(ratio - (mul_f / float(1 << shift)))
        take = valid & (err <= best_err)
        best_mul = torch.where(take, mul, best_mul)
        best_shift = torch.where(take, torch.full_like(best_shift, shift), best_shift)
        best_err = torch.where(take, err, best_err)
    return (((best_shift << QT_WIDTH) | best_mul).to(torch.uint32)).contiguous()


def scale_to_q15(scale):
    if scale is None:
        return None
    if scale.dtype.is_floating_point:
        return torch.clamp(torch.round(scale * Q15_16), 1, (1 << 31) - 1).to(torch.uint32).contiguous()
    return scale.to(torch.uint32).contiguous()


def scale_to_fp32(scale):
    if scale is None:
        return None
    if scale.dtype.is_floating_point:
        return scale.to(torch.float32).contiguous()
    return (scale.to(torch.float32) / float(Q15_16)).contiguous()


def reciprocal_qt(scale):
    return ratio_qt(torch.ones_like(scale_to_q15(scale)), scale_to_q15(scale))


def reciprocal_sqrt_qt(scale, dim):
    denom = torch.clamp(torch.round(scale_to_q15(scale).float() * (dim**0.5)).to(torch.int64), min=1)
    return ratio_qt(torch.ones_like(denom), denom)


def per_channel_i8_weight(w):
    scale = w.abs().amax(dim=1).clamp(min=1e-6) / 127.0
    return torch.round(w / scale[:, None]).clamp(-128, 127).to(torch.int8), scale.to(torch.float32)


def rope_tables(seq_len, head_dim, rope_theta=1_000_000.0, device="cuda"):
    pos = torch.arange(seq_len, device=device).float()[:, None]
    dim = torch.arange(head_dim // 2, device=device).float()[None, :]
    theta = pos * (rope_theta ** (-(2.0 * dim) / head_dim))
    return torch.cos(theta).contiguous(), torch.sin(theta).contiguous(), theta


def rope_tables_q15_16(seq_len, head_dim, rope_theta=1_000_000.0, device="cuda"):
    cos, sin, theta = rope_tables(seq_len, head_dim, rope_theta, device)
    return q15_16(cos), q15_16(sin), theta


@dataclass
class Int8LinearWeight:
    weight: torch.Tensor
    scale: torch.Tensor

    @staticmethod
    def pack(w):
        qw, qs = per_channel_i8_weight(w)
        return Int8LinearWeight(qw, qs)


@dataclass
class Qwen3BlockWeights:
    q_proj: Int8LinearWeight
    k_proj: Int8LinearWeight
    v_proj: Int8LinearWeight
    qkv_proj: Int8LinearWeight
    o_proj: Int8LinearWeight
    gate_proj: Int8LinearWeight
    up_proj: Int8LinearWeight
    gate_up_proj: Int8LinearWeight
    down_proj: Int8LinearWeight
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    input_qkv_i8_scale: torch.Tensor | None = None
    input_qkv_i8_qt: torch.Tensor | None = None
    q_post_rope_i8_scale: torch.Tensor | None = None
    q_post_rope_i8_qt: torch.Tensor | None = None
    k_post_rope_i8_scale: torch.Tensor | None = None
    k_post_rope_i8_qt: torch.Tensor | None = None
    v_i8_scale: torch.Tensor | None = None
    v_i8_qt: torch.Tensor | None = None
    attn_i8_scale: torch.Tensor | None = None
    attn_i8_qt: torch.Tensor | None = None
    attn_score_qt: torch.Tensor | None = None
    attn_score_scale: torch.Tensor | None = None
    attn_out_qt: torch.Tensor | None = None
    qkv_out_qt: torch.Tensor | None = None
    o_out_qt: torch.Tensor | None = None
    gate_up_out_qt: torch.Tensor | None = None
    down_out_qt: torch.Tensor | None = None
    post_mlp_i8_scale: torch.Tensor | None = None
    post_mlp_i8_qt: torch.Tensor | None = None
    gated_mlp_i8_scale: torch.Tensor | None = None
    gated_mlp_i8_qt: torch.Tensor | None = None


# Used by small standalone tests and by prepack-compatible float loading.
def pack_block_weights(q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, input_layernorm, post_attention_layernorm, q_norm, k_norm):
    return Qwen3BlockWeights(
        q_proj=Int8LinearWeight.pack(q_proj),
        k_proj=Int8LinearWeight.pack(k_proj),
        v_proj=Int8LinearWeight.pack(v_proj),
        qkv_proj=Int8LinearWeight.pack(torch.cat((q_proj, k_proj, v_proj), dim=0)),
        o_proj=Int8LinearWeight.pack(o_proj),
        gate_proj=Int8LinearWeight.pack(gate_proj),
        up_proj=Int8LinearWeight.pack(up_proj),
        gate_up_proj=Int8LinearWeight.pack(torch.cat((gate_proj, up_proj), dim=0)),
        down_proj=Int8LinearWeight.pack(down_proj),
        input_layernorm=q15_16(input_layernorm),
        post_attention_layernorm=q15_16(post_attention_layernorm),
        q_norm=q15_16(q_norm),
        k_norm=q15_16(k_norm),
    )


def load_packed_qwen3(packed_dir, config=QWEN3_0_6B, device="cuda", layers=None):
    path = f"{packed_dir}/qwen3_int_only.safetensors"
    blocks = []
    n_layers = config.num_hidden_layers if layers is None else layers

    with safe_open(path, framework="pt", device=device) as f:
        keys = set(f.keys())

        def tensor(name):
            return f.get_tensor(name)

        def optional(name):
            return tensor(name) if name in keys else None

        def optional_scale(name):
            return scale_to_fp32(optional(name))

        def optional_qt(name):
            s = optional(name)
            return None if s is None else reciprocal_qt(s)

        def linear(name):
            return Int8LinearWeight(tensor(f"{name}.weight"), tensor(f"{name}.scale"))

        def cat_linear(prefix, names):
            if f"{prefix}.weight" in keys:
                return linear(prefix)
            return Int8LinearWeight(torch.cat([tensor(f"{name}.weight") for name in names], dim=0), torch.cat([tensor(f"{name}.scale") for name in names], dim=0))

        def linear_i8_qt(xs_name, linear_weight):
            return pack_qt(scale_to_q15(optional(xs_name)).to(torch.float64) * linear_weight.scale.to(torch.float64))

        for layer_idx in range(n_layers):
            p = f"layers.{layer_idx}"
            q_rope_scale = optional_scale(f"{p}.q_post_rope_i8.scale")
            k_rope_scale = optional_scale(f"{p}.k_post_rope_i8.scale")
            group = config.num_attention_heads // config.num_key_value_heads
            q_rope_q15 = scale_to_q15(q_rope_scale).to(torch.int64)
            k_rope_q15 = scale_to_q15(k_rope_scale).repeat_interleave(group).to(torch.int64)
            attn_score_ratio = (((q_rope_q15 >> 4) * (k_rope_q15 >> 4)) >> 8).to(torch.float64) * 5793.0 / float(1 << 18)
            attn_score_qt = pack_qt(attn_score_ratio)
            attn_score_scale = (q_rope_scale.to(torch.float64) * k_rope_scale.repeat_interleave(group).to(torch.float64) / (config.head_dim**0.5)).to(torch.float32).contiguous()
            qkv_proj = cat_linear(f"{p}.qkv_proj", (f"{p}.q_proj", f"{p}.k_proj", f"{p}.v_proj"))
            o_proj = linear(f"{p}.o_proj")
            gate_up_proj = cat_linear(f"{p}.gate_up_proj", (f"{p}.gate_proj", f"{p}.up_proj"))
            down_proj = linear(f"{p}.down_proj")
            blocks.append(
                Qwen3BlockWeights(
                    q_proj=linear(f"{p}.q_proj"),
                    k_proj=linear(f"{p}.k_proj"),
                    v_proj=linear(f"{p}.v_proj"),
                    qkv_proj=qkv_proj,
                    o_proj=o_proj,
                    gate_proj=linear(f"{p}.gate_proj"),
                    up_proj=linear(f"{p}.up_proj"),
                    gate_up_proj=gate_up_proj,
                    down_proj=down_proj,
                    input_layernorm=tensor(f"{p}.input_layernorm"),
                    post_attention_layernorm=tensor(f"{p}.post_attention_layernorm"),
                    q_norm=tensor(f"{p}.q_norm"),
                    k_norm=tensor(f"{p}.k_norm"),
                    input_qkv_i8_scale=optional_scale(f"{p}.input_qkv_i8.scale"),
                    input_qkv_i8_qt=optional_qt(f"{p}.input_qkv_i8.scale"),
                    q_post_rope_i8_scale=optional_scale(f"{p}.q_post_rope_i8.scale"),
                    q_post_rope_i8_qt=optional_qt(f"{p}.q_post_rope_i8.scale"),
                    k_post_rope_i8_scale=optional_scale(f"{p}.k_post_rope_i8.scale"),
                    k_post_rope_i8_qt=optional_qt(f"{p}.k_post_rope_i8.scale"),
                    v_i8_scale=optional_scale(f"{p}.v_i8.scale"),
                    v_i8_qt=optional_qt(f"{p}.v_i8.scale"),
                    attn_i8_scale=optional_scale(f"{p}.attn_i8.scale"),
                    attn_i8_qt=optional_qt(f"{p}.attn_i8.scale"),
                    attn_score_qt=attn_score_qt,
                    attn_score_scale=attn_score_scale,
                    attn_out_qt=ratio_qt(scale_to_q15(optional(f"{p}.v_i8.scale")), scale_to_q15(optional(f"{p}.attn_i8.scale")).to(torch.int64) * 32767),
                    qkv_out_qt=linear_i8_qt(f"{p}.input_qkv_i8.scale", qkv_proj),
                    o_out_qt=linear_i8_qt(f"{p}.attn_i8.scale", o_proj),
                    gate_up_out_qt=linear_i8_qt(f"{p}.post_mlp_i8.scale", gate_up_proj),
                    down_out_qt=linear_i8_qt(f"{p}.gated_mlp_i8.scale", down_proj),
                    gated_mlp_i8_scale=optional_scale(f"{p}.gated_mlp_i8.scale"),
                    gated_mlp_i8_qt=None if optional(f"{p}.gated_mlp_i8.scale") is None else reciprocal_sqrt_qt(optional(f"{p}.gated_mlp_i8.scale"), config.head_dim),
                    post_mlp_i8_scale=optional_scale(f"{p}.post_mlp_i8.scale"),
                    post_mlp_i8_qt=optional_qt(f"{p}.post_mlp_i8.scale"),
                )
            )
        return tensor("model.embed_tokens.weight"), tensor("lm_head.weight"), tensor("model.norm.weight"), blocks


def load_embed_tokens(model_dir="/publicdata/huggingface.co/Qwen/Qwen3-0.6B", device="cuda"):
    with SafeTensorReader(model_dir) as reader:
        return reader.get_tensor("model.embed_tokens.weight").to(torch.float32).to(device)


def load_lm_head(model_dir="/publicdata/huggingface.co/Qwen/Qwen3-0.6B", device="cuda"):
    with SafeTensorReader(model_dir) as reader:
        return reader.get_tensor("lm_head.weight").to(torch.float32).to(device)


def rmsnorm_torch(x, weight):
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6) * weight


def rope_torch(x, cos, sin, heads, head_dim):
    y = x.reshape(x.shape[0], heads, head_dim)
    a, b = y[..., : head_dim // 2], y[..., head_dim // 2 :]
    out = torch.cat((a * cos[:, None, :] - b * sin[:, None, :], a * sin[:, None, :] + b * cos[:, None, :]), dim=-1)
    return out.reshape(x.shape[0], heads * head_dim)
