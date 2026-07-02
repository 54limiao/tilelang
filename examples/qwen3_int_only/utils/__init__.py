from __future__ import annotations

from dataclasses import dataclass

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from examples.qwen3_int_only.utils.quarot import (
    ROTATE_SEED,
    random_hadamard_rotation,
    rotate_head_input,
    rotate_head_output,
    rotate_input,
    rotate_norm_input,
    rotate_output,
)

Q15_16 = 1 << 16


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


QWEN3_0_6B = Qwen3Config()


def q15_16(x):
    return torch.clamp(torch.round(x * Q15_16), -(1 << 31), (1 << 31) - 1).to(torch.int32)


def per_channel_i8_weight(w):
    scale = w.abs().amax(dim=1).clamp(min=1e-6) / 127.0
    return torch.round(w / scale[:, None]).clamp(-128, 127).to(torch.int8), q15_16(scale).to(torch.uint32)


def rope_tables_q15_16(seq_len, head_dim, rope_theta=1_000_000.0, device="cuda"):
    pos = torch.arange(seq_len, device=device).float()[:, None]
    dim = torch.arange(head_dim // 2, device=device).float()[None, :]
    theta = pos * (rope_theta ** (-(2.0 * dim) / head_dim))
    return q15_16(torch.cos(theta)), q15_16(torch.sin(theta)), theta


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
    q_post_rope_i8_scale: torch.Tensor | None = None
    k_post_rope_i8_scale: torch.Tensor | None = None
    v_i8_scale: torch.Tensor | None = None
    attn_i8_scale: torch.Tensor | None = None
    post_mlp_i8_scale: torch.Tensor | None = None
    gated_mlp_i16_scale: torch.Tensor | None = None


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


def load_packed_qwen3(packed_dir, config=QWEN3_0_6B, device="cuda"):
    tensors = load_file(f"{packed_dir}/qwen3_int_only.safetensors", device=device)
    blocks = []

    def optional(name):
        return tensors[name] if name in tensors else None

    def linear(name):
        return Int8LinearWeight(tensors[f"{name}.weight"], tensors[f"{name}.scale"])

    def cat_linear(prefix, names):
        if f"{prefix}.weight" in tensors:
            return linear(prefix)
        return Int8LinearWeight(torch.cat([tensors[f"{name}.weight"] for name in names], dim=0), torch.cat([tensors[f"{name}.scale"] for name in names], dim=0))

    for layer_idx in range(config.num_hidden_layers):
        p = f"layers.{layer_idx}"
        blocks.append(
            Qwen3BlockWeights(
                q_proj=linear(f"{p}.q_proj"),
                k_proj=linear(f"{p}.k_proj"),
                v_proj=linear(f"{p}.v_proj"),
                qkv_proj=cat_linear(f"{p}.qkv_proj", (f"{p}.q_proj", f"{p}.k_proj", f"{p}.v_proj")),
                o_proj=linear(f"{p}.o_proj"),
                gate_proj=linear(f"{p}.gate_proj"),
                up_proj=linear(f"{p}.up_proj"),
                gate_up_proj=cat_linear(f"{p}.gate_up_proj", (f"{p}.gate_proj", f"{p}.up_proj")),
                down_proj=linear(f"{p}.down_proj"),
                input_layernorm=tensors[f"{p}.input_layernorm"],
                post_attention_layernorm=tensors[f"{p}.post_attention_layernorm"],
                q_norm=tensors[f"{p}.q_norm"],
                k_norm=tensors[f"{p}.k_norm"],
                input_qkv_i8_scale=optional(f"{p}.input_qkv_i8.scale"),
                q_post_rope_i8_scale=optional(f"{p}.q_post_rope_i8.scale"),
                k_post_rope_i8_scale=optional(f"{p}.k_post_rope_i8.scale"),
                v_i8_scale=optional(f"{p}.v_i8.scale"),
                attn_i8_scale=optional(f"{p}.attn_i8.scale"),
                post_mlp_i8_scale=optional(f"{p}.post_mlp_i8.scale"),
                gated_mlp_i16_scale=optional(f"{p}.gated_mlp_i16.scale"),
            )
        )
    return tensors["model.embed_tokens.weight"], tensors["lm_head.weight"], tensors["model.norm.weight"], blocks


def load_embed_tokens(model_dir="/publicdata/huggingface.co/Qwen/Qwen3-0.6B", device="cuda"):
    with safe_open(f"{model_dir}/model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor("model.embed_tokens.weight").to(torch.float32).to(device)


def load_lm_head(model_dir="/publicdata/huggingface.co/Qwen/Qwen3-0.6B", device="cuda"):
    with safe_open(f"{model_dir}/model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor("lm_head.weight").to(torch.float32).to(device)


def rmsnorm_torch(x, weight):
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6) * weight


def rope_torch(x, cos, sin, heads, head_dim):
    y = x.reshape(x.shape[0], heads, head_dim)
    a, b = y[..., : head_dim // 2], y[..., head_dim // 2 :]
    out = torch.cat((a * cos[:, None, :] - b * sin[:, None, :], a * sin[:, None, :] + b * cos[:, None, :]), dim=-1)
    return out.reshape(x.shape[0], heads * head_dim)
