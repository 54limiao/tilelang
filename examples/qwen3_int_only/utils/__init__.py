from __future__ import annotations

from dataclasses import dataclass

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from examples.qwen3_int_only.kernels import Q15_16
from examples.qwen3_int_only.utils.quarot import (
    ROTATE_SEED,
    random_hadamard_rotation,
    rotate_head_input,
    rotate_head_output,
    rotate_input,
    rotate_norm_input,
    rotate_output,
)


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
    inv = rope_theta ** (-(2.0 * dim) / head_dim)
    theta = pos * inv
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
    o_proj: Int8LinearWeight
    gate_proj: Int8LinearWeight
    up_proj: Int8LinearWeight
    down_proj: Int8LinearWeight
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    q_pre_rope_i16_scale: torch.Tensor | None = None
    k_pre_rope_i16_scale: torch.Tensor | None = None
    q_post_rope_i8_scale: torch.Tensor | None = None
    k_post_rope_i8_scale: torch.Tensor | None = None
    v_i8_scale: torch.Tensor | None = None
    attn_i8_scale: torch.Tensor | None = None
    post_mlp_i8_scale: torch.Tensor | None = None
    gated_mlp_i16_scale: torch.Tensor | None = None
    q_proj_fp: torch.Tensor | None = None
    k_proj_fp: torch.Tensor | None = None
    v_proj_fp: torch.Tensor | None = None
    o_proj_fp: torch.Tensor | None = None
    gate_proj_fp: torch.Tensor | None = None
    up_proj_fp: torch.Tensor | None = None
    down_proj_fp: torch.Tensor | None = None

    @staticmethod
    def pack(
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        gate_proj,
        up_proj,
        down_proj,
        input_layernorm=None,
        post_attention_layernorm=None,
        q_norm=None,
        k_norm=None,
    ):
        hidden = q_proj.shape[1]
        head_dim = q_proj.shape[0] // 16 if q_norm is None else q_norm.numel()
        device = q_proj.device
        if input_layernorm is None:
            input_layernorm = torch.ones(hidden, device=device)
        if post_attention_layernorm is None:
            post_attention_layernorm = torch.ones(hidden, device=device)
        if q_norm is None:
            q_norm = torch.ones(head_dim, device=device)
        if k_norm is None:
            k_norm = torch.ones(head_dim, device=device)
        return Qwen3BlockWeights(
            Int8LinearWeight.pack(q_proj),
            Int8LinearWeight.pack(k_proj),
            Int8LinearWeight.pack(v_proj),
            Int8LinearWeight.pack(o_proj),
            Int8LinearWeight.pack(gate_proj),
            Int8LinearWeight.pack(up_proj),
            Int8LinearWeight.pack(down_proj),
            q15_16(input_layernorm),
            q15_16(post_attention_layernorm),
            q15_16(q_norm),
            q15_16(k_norm),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
        )


def load_packed_qwen3(packed_dir, config=QWEN3_0_6B, device="cuda"):
    tensors = load_file(f"{packed_dir}/qwen3_int_only.safetensors", device=device)
    blocks = []

    def optional(name):
        return tensors[name] if name in tensors else None

    for layer_idx in range(config.num_hidden_layers):
        p = f"layers.{layer_idx}"
        blocks.append(
            Qwen3BlockWeights(
                Int8LinearWeight(tensors[f"{p}.q_proj.weight"], tensors[f"{p}.q_proj.scale"]),
                Int8LinearWeight(tensors[f"{p}.k_proj.weight"], tensors[f"{p}.k_proj.scale"]),
                Int8LinearWeight(tensors[f"{p}.v_proj.weight"], tensors[f"{p}.v_proj.scale"]),
                Int8LinearWeight(tensors[f"{p}.o_proj.weight"], tensors[f"{p}.o_proj.scale"]),
                Int8LinearWeight(tensors[f"{p}.gate_proj.weight"], tensors[f"{p}.gate_proj.scale"]),
                Int8LinearWeight(tensors[f"{p}.up_proj.weight"], tensors[f"{p}.up_proj.scale"]),
                Int8LinearWeight(tensors[f"{p}.down_proj.weight"], tensors[f"{p}.down_proj.scale"]),
                tensors[f"{p}.input_layernorm"],
                tensors[f"{p}.post_attention_layernorm"],
                tensors[f"{p}.q_norm"],
                tensors[f"{p}.k_norm"],
                optional(f"{p}.q_pre_rope_i16.scale"),
                optional(f"{p}.k_pre_rope_i16.scale"),
                optional(f"{p}.q_post_rope_i8.scale"),
                optional(f"{p}.k_post_rope_i8.scale"),
                optional(f"{p}.v_i8.scale"),
                optional(f"{p}.attn_i8.scale"),
                optional(f"{p}.post_mlp_i8.scale"),
                optional(f"{p}.gated_mlp_i16.scale"),
            )
        )
    return tensors["model.embed_tokens.weight"], tensors["lm_head.weight"], tensors["model.norm.weight"], blocks


def load_all_qwen3_block_weights(model_dir="/publicdata/huggingface.co/Qwen/Qwen3-0.6B", config=QWEN3_0_6B, device="cuda"):
    blocks = []
    with safe_open(f"{model_dir}/model.safetensors", framework="pt", device="cpu") as f:
        def tensor(name):
            return f.get_tensor(name).to(torch.float32).to(device)

        for layer_idx in range(config.num_hidden_layers):
            p = f"model.layers.{layer_idx}"
            blocks.append(
                Qwen3BlockWeights.pack(
                    tensor(f"{p}.self_attn.q_proj.weight"),
                    tensor(f"{p}.self_attn.k_proj.weight"),
                    tensor(f"{p}.self_attn.v_proj.weight"),
                    tensor(f"{p}.self_attn.o_proj.weight"),
                    tensor(f"{p}.mlp.gate_proj.weight"),
                    tensor(f"{p}.mlp.up_proj.weight"),
                    tensor(f"{p}.mlp.down_proj.weight"),
                    tensor(f"{p}.input_layernorm.weight"),
                    tensor(f"{p}.post_attention_layernorm.weight"),
                    tensor(f"{p}.self_attn.q_norm.weight"),
                    tensor(f"{p}.self_attn.k_norm.weight"),
                )
            )
    return blocks


def load_embed_tokens(model_dir="/publicdata/huggingface.co/Qwen/Qwen3-0.6B", device="cuda"):
    with safe_open(f"{model_dir}/model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor("model.embed_tokens.weight").to(torch.float32).to(device)


def load_lm_head(model_dir="/publicdata/huggingface.co/Qwen/Qwen3-0.6B", device="cuda"):
    with safe_open(f"{model_dir}/model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor("lm_head.weight").to(torch.float32).to(device)


def load_final_norm(model_dir="/publicdata/huggingface.co/Qwen/Qwen3-0.6B", device="cuda"):
    with safe_open(f"{model_dir}/model.safetensors", framework="pt", device="cpu") as f:
        return q15_16(f.get_tensor("model.norm.weight").to(torch.float32).to(device))


def rmsnorm_torch(x, weight):
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6) * weight


def rope_torch(x, cos, sin, heads, head_dim):
    y = x.reshape(x.shape[0], heads, head_dim)
    a, b = y[..., : head_dim // 2], y[..., head_dim // 2 :]
    out = torch.cat((a * cos[:, None, :] - b * sin[:, None, :], a * sin[:, None, :] + b * cos[:, None, :]), dim=-1)
    return out.reshape(x.shape[0], heads * head_dim)


def causal_softmax(score):
    mask = torch.ones(score.shape[-2:], device=score.device, dtype=torch.bool).tril()
    return torch.softmax(score.masked_fill(~mask, torch.finfo(score.dtype).min), dim=-1)


def block_torch(x, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16, config=QWEN3_0_6B, r3=None):
    return block_torch_trace(x, weights, cos_q15_16, sin_q15_16, config, r3)["layer_out"]


def block_torch_trace(x, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16, config=QWEN3_0_6B, r3=None):
    cos, sin = cos_q15_16.float() / Q15_16, sin_q15_16.float() / Q15_16
    h = rmsnorm_torch(x, weights.input_layernorm.float() / Q15_16)
    q = h @ weights.q_proj_fp.T
    k = h @ weights.k_proj_fp.T
    v = h @ weights.v_proj_fp.T
    q = rmsnorm_torch(q.reshape(-1, config.num_attention_heads, config.head_dim), weights.q_norm.float() / Q15_16).reshape(-1, config.q_size)
    k = rmsnorm_torch(k.reshape(-1, config.num_key_value_heads, config.head_dim), weights.k_norm.float() / Q15_16).reshape(-1, config.kv_size)
    q = rope_torch(q, cos, sin, config.num_attention_heads, config.head_dim).reshape(-1, config.num_attention_heads, config.head_dim)
    k = rope_torch(k, cos, sin, config.num_key_value_heads, config.head_dim).reshape(-1, config.num_key_value_heads, config.head_dim)
    if r3 is not None:
        q = (q.to(torch.float64) @ r3.to(torch.float64)).to(torch.float32)
        k = (k.to(torch.float64) @ r3.to(torch.float64)).to(torch.float32)
    q = q.permute(1, 0, 2)
    group = config.num_attention_heads // config.num_key_value_heads
    k = k.repeat_interleave(group, dim=1).permute(1, 0, 2)
    v = v.reshape(-1, config.num_key_value_heads, config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2)
    attn = causal_softmax((q @ k.transpose(-1, -2)) / (config.head_dim**0.5)) @ v
    attn = attn.permute(1, 0, 2).reshape(x.shape[0], config.q_size)
    attn_out = attn @ weights.o_proj_fp.T
    h = x + attn_out
    post = rmsnorm_torch(h, weights.post_attention_layernorm.float() / Q15_16)
    gate = post @ weights.gate_proj_fp.T
    up = post @ weights.up_proj_fp.T
    gated = torch.nn.functional.silu(gate) * up
    mlp = gated @ weights.down_proj_fp.T
    return {
        "input_rms": rmsnorm_torch(x, weights.input_layernorm.float() / Q15_16),
        "q": q.permute(1, 0, 2).reshape(x.shape[0], config.q_size),
        "k": k.permute(1, 0, 2).reshape(x.shape[0], config.q_size),
        "v": v.permute(1, 0, 2).reshape(x.shape[0], config.q_size),
        "attn": attn,
        "attn_out": attn_out,
        "attn_residual": h,
        "post_rms": post,
        "gate": gate,
        "up": up,
        "gated": gated,
        "mlp": mlp,
        "layer_out": h + mlp,
    }
