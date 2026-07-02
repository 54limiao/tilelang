from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from examples.qwen3_int_only.kernels import (
    Q15_16,
    add_rmsnorm_dynamic_quant_q15_16_weighted_fast,
    add_rmsnorm_q15_16_weighted,
    add_q15_16,
    attention_i8v8_q15_16_gqa_cache_fused_static_current,
    attention_i8v8_q15_16_gqa_fused_static,
    compile_kernel,
    dynamic_quant_q15_16,
    exp_lut_neg,
    linear_dynamic_int8_pair_q15_16,
    linear_dynamic_int8_qkv_q15_16,
    linear_dynamic_int8_residual_q15_16,
    linear_dynamic_int8_q15_16,
    linear_dynamic_int16_residual_q15_16,
    linear_static_int8_pair_q15_16,
    linear_static_int16_residual_q15_16,
    rmsnorm_q15_16_grouped_weighted_rowwise,
    rmsnorm_dynamic_quant_q15_16_weighted_fast,
    rmsnorm_q15_16_weighted,
    rope_rotate_q15_16_heads,
    rope_rotate_static_quant_q15_16_attn,
    rope_rotate_static_quant_q15_16_attn_hadamard_approx,
    rope_rotate_static_quant_q15_16_attn_noscale,
    rope_q15_16_heads,
    rsqrt_lut,
    sigmoid_lut,
    static_quant_q15_16,
    static_quant_q15_16_per_head_attn,
    static_quant_q15_16_per_head_attn_noscale,
    silu_mul_dynamic_quant_q15_16,
    silu_mul_dynamic_quant_q15_16_fast,
    silu_mul_dynamic_quant_q15_16_i16_fast,
    silu_mul_static_quant_q15_16_i16_fast,
)
from examples.qwen3_int_only.quarot import ROTATE_SEED, random_hadamard_rotation


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
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
        )


def load_all_qwen3_block_weights(model_dir="/code/Qwen3-0.6B", config=QWEN3_0_6B, device="cuda"):
    path = f"{model_dir}/model.safetensors"
    blocks = []
    with safe_open(path, framework="pt", device="cpu") as f:
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


def load_embed_tokens(model_dir="/code/Qwen3-0.6B", device="cuda"):
    with safe_open(f"{model_dir}/model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor("model.embed_tokens.weight").to(torch.float32).to(device)


def load_lm_head(model_dir="/code/Qwen3-0.6B", device="cuda"):
    with safe_open(f"{model_dir}/model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor("lm_head.weight").to(torch.float32).to(device)


def load_final_norm(model_dir="/code/Qwen3-0.6B", device="cuda"):
    with safe_open(f"{model_dir}/model.safetensors", framework="pt", device="cpu") as f:
        return q15_16(f.get_tensor("model.norm.weight").to(torch.float32).to(device))


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
                optional(f"{p}.post_mlp_i8.scale"),
                optional(f"{p}.gated_mlp_i16.scale"),
            )
        )
    return tensors["model.embed_tokens.weight"], tensors["lm_head.weight"], tensors["model.norm.weight"], blocks


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
    trace = block_torch_trace(x, weights, cos_q15_16, sin_q15_16, config, r3)
    return trace["layer_out"]


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
        "input_rms": h.new_tensor(0) + rmsnorm_torch(x, weights.input_layernorm.float() / Q15_16),
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


def parse_layer_set(value):
    if not value or value == "none":
        return set()
    if value == "all":
        return set(range(QWEN3_0_6B.num_hidden_layers))
    layers = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(item))
    return layers


class Qwen3IntOnlyBlock:
    def __init__(self, seq_len, config=QWEN3_0_6B, cache_len=0, use_r3=True, fast_hadamard=True, mlp_i16=False, static_mlp=False):
        self.seq_len = seq_len
        self.cache_len = cache_len
        self.config = config
        self.use_r3 = use_r3
        self.fast_hadamard = fast_hadamard
        self.mlp_i16 = mlp_i16
        self.static_mlp = static_mlp
        h, hd, im = config.hidden_size, config.head_dim, config.intermediate_size
        qh, kvh = config.num_attention_heads, config.num_key_value_heads
        q_dim, kv_dim = config.q_size, config.kv_size
        self.rms_hidden_q15 = compile_kernel(rmsnorm_q15_16_weighted(seq_len, h), [3])
        self.rms_dq8_hidden_fast = compile_kernel(rmsnorm_dynamic_quant_q15_16_weighted_fast(seq_len, h), [3, 4])
        self.add_rms_hidden_q15 = compile_kernel(add_rmsnorm_q15_16_weighted(seq_len, h), [4, 5])
        self.add_rms_dq8_hidden_fast = compile_kernel(add_rmsnorm_dynamic_quant_q15_16_weighted_fast(seq_len, h), [4, 5, 6])
        self.rms_q_q15 = compile_kernel(rmsnorm_q15_16_grouped_weighted_rowwise(seq_len, qh, hd), [3])
        self.rms_k_q15 = compile_kernel(rmsnorm_q15_16_grouped_weighted_rowwise(seq_len, kvh, hd), [3])
        self.dq8_hidden = compile_kernel(dynamic_quant_q15_16(seq_len, h, "int8"), [1, 2])
        self.sq8_q_attn = compile_kernel(static_quant_q15_16_per_head_attn(seq_len, qh, hd, "int8"), [2, 3])
        self.sq8_kv_attn = compile_kernel(static_quant_q15_16_per_head_attn(seq_len, kvh, hd, "int8"), [2, 3])
        self.sq8_kv_attn_noscale = compile_kernel(static_quant_q15_16_per_head_attn_noscale(seq_len, kvh, hd, "int8"), [2])
        self.rope_sq8_q_attn = compile_kernel(rope_rotate_static_quant_q15_16_attn(seq_len, qh, hd), [5, 6]) if use_r3 else None
        self.rope_sq8_k_attn = compile_kernel(rope_rotate_static_quant_q15_16_attn(seq_len, kvh, hd), [5, 6]) if use_r3 else None
        self.rope_sq8_q_attn_noscale = compile_kernel(rope_rotate_static_quant_q15_16_attn_noscale(seq_len, qh, hd), [5]) if use_r3 else None
        self.rope_sq8_k_attn_noscale = compile_kernel(rope_rotate_static_quant_q15_16_attn_noscale(seq_len, kvh, hd), [5]) if use_r3 else None
        self.rope_sq8_q_attn_hadamard = compile_kernel(rope_rotate_static_quant_q15_16_attn_hadamard_approx(seq_len, qh, hd), [5]) if use_r3 and fast_hadamard else None
        self.rope_sq8_k_attn_hadamard = compile_kernel(rope_rotate_static_quant_q15_16_attn_hadamard_approx(seq_len, kvh, hd), [5]) if use_r3 and fast_hadamard else None
        self.dq8_q = compile_kernel(dynamic_quant_q15_16(seq_len, q_dim, "int8"), [1, 2])
        self.qkv_proj_i8 = compile_kernel(linear_dynamic_int8_qkv_q15_16(seq_len, h, q_dim, kv_dim, 64, 128, 64), [8, 9, 10])
        self.o_proj = compile_kernel(linear_dynamic_int8_q15_16(seq_len, q_dim, h, 64, 64, 64), [4])
        self.gate_up_proj_i8 = compile_kernel(linear_dynamic_int8_pair_q15_16(seq_len, h, im, 64, 128, 64), [6, 7])
        self.down_proj_i8 = compile_kernel(linear_dynamic_int8_q15_16(seq_len, im, h, 64, 64, 64), [4])
        self.down_residual_i8 = compile_kernel(linear_dynamic_int8_residual_q15_16(seq_len, im, h, 64, 64, 64), [5])
        self.down_residual_i16 = compile_kernel(linear_dynamic_int16_residual_q15_16(seq_len, im, h, 64, 64, 64), [5]) if mlp_i16 else None
        self.sq8_hidden = compile_kernel(static_quant_q15_16(seq_len, h, "int8"), [2, 3]) if static_mlp else None
        self.gate_up_proj_static = compile_kernel(linear_static_int8_pair_q15_16(seq_len, h, im, 64, 128, 64), [6, 7]) if static_mlp else None
        self.silu_mul_sq16_mid_fast = compile_kernel(silu_mul_static_quant_q15_16_i16_fast(seq_len, im), [4, 5]) if static_mlp else None
        self.down_residual_static = compile_kernel(linear_static_int16_residual_q15_16(seq_len, im, h, 64, 64, 64), [5]) if static_mlp else None
        self.rope_q = compile_kernel(rope_rotate_q15_16_heads(seq_len, qh, hd), [4]) if use_r3 else compile_kernel(rope_q15_16_heads(seq_len, qh, hd), [3])
        self.rope_k = compile_kernel(rope_rotate_q15_16_heads(seq_len, kvh, hd), [4]) if use_r3 else compile_kernel(rope_q15_16_heads(seq_len, kvh, hd), [3])
        self.silu_mul_dq8_mid = compile_kernel(silu_mul_dynamic_quant_q15_16(seq_len, im), [3, 4, 5])
        self.silu_mul_dq8_mid_fast = compile_kernel(silu_mul_dynamic_quant_q15_16_fast(seq_len, im), [3, 4])
        self.silu_mul_dq16_mid_fast = compile_kernel(silu_mul_dynamic_quant_q15_16_i16_fast(seq_len, im), [3, 4]) if mlp_i16 else None
        self.add_hidden = compile_kernel(add_q15_16(seq_len, h), [2])
        self.attn_i8v8_fused_static = compile_kernel(attention_i8v8_q15_16_gqa_fused_static(qh, kvh, seq_len, hd), [7])
        self.attn_i8v8_fused_cache_static = None
        if cache_len:
            self.attn_i8v8_fused_cache_static = compile_kernel(attention_i8v8_q15_16_gqa_cache_fused_static_current(qh, kvh, seq_len, cache_len, hd), [11])
        self.lut_rsqrt = torch.from_numpy(rsqrt_lut()).cuda()
        self.lut_sigmoid = torch.from_numpy(sigmoid_lut()).cuda()
        self.lut_exp = torch.from_numpy(exp_lut_neg()).cuda()

    def __call__(self, x_q15_16, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16, cache_k=None, cache_v=None, r3_q15=None, mlp_i16=None):
        return self.trace(x_q15_16, weights, cos_q15_16, sin_q15_16, cache_k, cache_v, r3_q15, collect=False, mlp_i16=mlp_i16)

    def trace(self, x_q15_16, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16, cache_k=None, cache_v=None, r3_q15=None, collect=True, mlp_i16=None):
        use_mlp_i16 = self.mlp_i16 if mlp_i16 is None else mlp_i16
        if collect:
            norm = self.rms_hidden_q15(x_q15_16, weights.input_layernorm, self.lut_rsqrt)
            x8, xs8 = self.dq8_hidden(norm)
        else:
            norm = None
            x8, xs8 = self.rms_dq8_hidden_fast(x_q15_16, weights.input_layernorm, self.lut_rsqrt)
        q, k, v = self.qkv_proj_i8(
            x8, xs8, weights.q_proj.weight, weights.q_proj.scale, weights.k_proj.weight, weights.k_proj.scale, weights.v_proj.weight, weights.v_proj.scale
        )
        v_heads = v.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim)
        q_heads = self.rms_q_q15(q, weights.q_norm, self.lut_rsqrt)
        k_heads = self.rms_k_q15(k, weights.k_norm, self.lut_rsqrt)
        pos_cos = cos_q15_16[self.cache_len : self.cache_len + self.seq_len]
        pos_sin = sin_q15_16[self.cache_len : self.cache_len + self.seq_len]
        if self.use_r3 and not collect:
            qr = None
            kr = None
            if self.fast_hadamard:
                q_attn = self.rope_sq8_q_attn_hadamard(q_heads, pos_cos, pos_sin, r3_q15, weights.q_post_rope_i8_scale)
                k_attn = self.rope_sq8_k_attn_hadamard(k_heads, pos_cos, pos_sin, r3_q15, weights.k_post_rope_i8_scale)
            else:
                q_attn = self.rope_sq8_q_attn_noscale(q_heads, pos_cos, pos_sin, r3_q15, weights.q_post_rope_i8_scale)
                k_attn = self.rope_sq8_k_attn_noscale(k_heads, pos_cos, pos_sin, r3_q15, weights.k_post_rope_i8_scale)
            v_attn = self.sq8_kv_attn_noscale(v_heads.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim), weights.v_i8_scale)
        else:
            if self.use_r3:
                qr = self.rope_q(q_heads, pos_cos, pos_sin, r3_q15)
                kr = self.rope_k(k_heads, pos_cos, pos_sin, r3_q15)
            else:
                qr = self.rope_q(q_heads, pos_cos, pos_sin)
                kr = self.rope_k(k_heads, pos_cos, pos_sin)
            q_attn, _qs_attn = self.sq8_q_attn(qr.reshape(self.seq_len, self.config.num_attention_heads, self.config.head_dim), weights.q_post_rope_i8_scale)
            k_attn, _ks_attn = self.sq8_kv_attn(kr.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim), weights.k_post_rope_i8_scale)
            v_attn = self.sq8_kv_attn_noscale(v_heads.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim), weights.v_i8_scale)
        if cache_k is None:
            prob_i16 = None
            attn = self.attn_i8v8_fused_static(q_attn, k_attn, v_attn, weights.q_post_rope_i8_scale, weights.k_post_rope_i8_scale, weights.v_i8_scale, self.lut_exp)
        else:
            prob_i16 = None
            attn = self.attn_i8v8_fused_cache_static(
                q_attn,
                cache_k[0],
                cache_v[0],
                k_attn,
                v_attn,
                weights.q_post_rope_i8_scale,
                cache_k[1],
                cache_v[1],
                weights.k_post_rope_i8_scale,
                weights.v_i8_scale,
                self.lut_exp,
            )
        attn8, attn_s8 = self.dq8_q(attn)
        attn_out = self.o_proj(attn8, attn_s8, weights.o_proj.weight, weights.o_proj.scale)
        if collect:
            h, post = self.add_rms_hidden_q15(x_q15_16, attn_out, weights.post_attention_layernorm, self.lut_rsqrt)
            h8, hs8 = self.dq8_hidden(post)
        else:
            post = None
            if self.static_mlp:
                h, post = self.add_rms_hidden_q15(x_q15_16, attn_out, weights.post_attention_layernorm, self.lut_rsqrt)
                h8, _hs8 = self.sq8_hidden(post, weights.post_mlp_i8_scale)
                hs8 = weights.post_mlp_i8_scale
            else:
                h, h8, hs8 = self.add_rms_dq8_hidden_fast(x_q15_16, attn_out, weights.post_attention_layernorm, self.lut_rsqrt)
        if self.static_mlp and not collect:
            gate, up = self.gate_up_proj_static(h8, hs8, weights.gate_proj.weight, weights.gate_proj.scale, weights.up_proj.weight, weights.up_proj.scale)
        else:
            gate, up = self.gate_up_proj_i8(h8, hs8, weights.gate_proj.weight, weights.gate_proj.scale, weights.up_proj.weight, weights.up_proj.scale)
        if collect:
            gated, gated8, gs8 = self.silu_mul_dq8_mid(gate, up, self.lut_sigmoid)
        elif self.static_mlp:
            gated = None
            gated8, _gs8 = self.silu_mul_sq16_mid_fast(gate, up, self.lut_sigmoid, weights.gated_mlp_i16_scale)
            gs8 = weights.gated_mlp_i16_scale
        elif use_mlp_i16:
            gated = None
            gated8, gs8 = self.silu_mul_dq16_mid_fast(gate, up, self.lut_sigmoid)
        else:
            gated = None
            gated8, gs8 = self.silu_mul_dq8_mid_fast(gate, up, self.lut_sigmoid)
        if collect:
            mlp = self.down_proj_i8(gated8, gs8, weights.down_proj.weight, weights.down_proj.scale)
            layer_out = self.add_hidden(h, mlp)
        elif self.static_mlp:
            mlp = None
            layer_out = self.down_residual_static(gated8, gs8, weights.down_proj.weight, weights.down_proj.scale, h)
        elif use_mlp_i16:
            mlp = None
            layer_out = self.down_residual_i16(gated8, gs8, weights.down_proj.weight, weights.down_proj.scale, h)
        else:
            mlp = None
            layer_out = self.down_residual_i8(gated8, gs8, weights.down_proj.weight, weights.down_proj.scale, h)
        if not collect:
            return layer_out
        q_trace = qr.reshape(self.seq_len, self.config.num_attention_heads, self.config.head_dim).reshape(self.seq_len, self.config.q_size)
        k_trace = kr.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(
            self.config.num_attention_heads // self.config.num_key_value_heads, dim=1
        ).reshape(self.seq_len, self.config.q_size)
        v_trace = v_heads.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(
            self.config.num_attention_heads // self.config.num_key_value_heads, dim=1
        ).reshape(self.seq_len, self.config.q_size)
        return {
            "input_rms": norm.float() / Q15_16,
            "q": q_trace.float() / Q15_16,
            "k": k_trace.float() / Q15_16,
            "v": v_trace.float() / Q15_16,
            "attn": attn.float() / Q15_16,
            "prob_i16": prob_i16,
            "q8": q_attn,
            "k8": k_attn,
            "v8": v_attn,
            "qs8": weights.q_post_rope_i8_scale[:, None].expand(self.config.num_attention_heads, self.seq_len),
            "ks8": weights.k_post_rope_i8_scale[:, None].expand(self.config.num_key_value_heads, self.seq_len),
            "vs8": weights.v_i8_scale[:, None].expand(self.config.num_key_value_heads, self.seq_len),
            "attn8": attn8,
            "attn_s8": attn_s8,
            "attn_out": attn_out.float() / Q15_16,
            "attn_residual": h.float() / Q15_16,
            "post_rms": post.float() / Q15_16,
            "post8": h8,
            "post_s8": hs8,
            "gate": gate.float() / Q15_16,
            "up": up.float() / Q15_16,
            "gated": gated.float() / Q15_16,
            "gated8": gated8,
            "gated_s8": gs8,
            "mlp": mlp.float() / Q15_16,
            "layer_out": layer_out.float() / Q15_16,
            "layer_out_q15": layer_out,
        }


class Qwen3IntOnlyModel:
    def __init__(self, seq_len, model_dir="/code/Qwen3-0.6B", packed_dir="/tmp/Qwen3-0.6B-static-calib-32x2048", config=QWEN3_0_6B, cache_len=0, use_r3=True, fast_hadamard=True, mlp_i16=False, mlp_i16_layers=None, static_mlp=False, rotate_seed=ROTATE_SEED):
        self.seq_len = seq_len
        self.cache_len = cache_len
        self.config = config
        self.mlp_i16 = mlp_i16
        self.static_mlp = static_mlp
        self.mlp_i16_layers = set() if mlp_i16_layers is None else set(mlp_i16_layers)
        self.r3_q15 = q15_16(random_hadamard_rotation(config.head_dim, rotate_seed + 2)) if use_r3 else None
        self.block = Qwen3IntOnlyBlock(seq_len, config, cache_len=cache_len, use_r3=use_r3, fast_hadamard=fast_hadamard, mlp_i16=mlp_i16 or bool(self.mlp_i16_layers), static_mlp=static_mlp)
        self.final_norm_kernel = compile_kernel(rmsnorm_q15_16_weighted(seq_len, config.hidden_size), [3])
        self.lut_rsqrt = torch.from_numpy(rsqrt_lut()).cuda()
        self.cos, self.sin, _ = rope_tables_q15_16(seq_len + cache_len, config.head_dim, config.rope_theta)
        self.embed, self.lm_head, self.final_norm, self.layers = load_packed_qwen3(packed_dir, config)

    def embed_input(self, input_ids):
        return q15_16(self.embed[input_ids])

    def hidden(self, input_ids, layers=None, verbose=False, cache_kv=None):
        x = self.embed_input(input_ids)
        n_layers = self.config.num_hidden_layers if layers is None else layers
        for layer_idx in range(n_layers):
            t0 = time.time()
            layer_cache = None if cache_kv is None else cache_kv[layer_idx]
            use_mlp_i16 = self.mlp_i16 or layer_idx in self.mlp_i16_layers
            if layer_cache is None:
                x = self.block(x, self.layers[layer_idx], self.cos, self.sin, r3_q15=self.r3_q15, mlp_i16=use_mlp_i16)
            else:
                x = self.block(x, self.layers[layer_idx], self.cos, self.sin, layer_cache[0], layer_cache[1], self.r3_q15, use_mlp_i16)
            if verbose:
                torch.cuda.synchronize()
                print(f"int-only layer {layer_idx} done in {time.time() - t0:.3f}s", flush=True)
        return self.final_norm_kernel(x, self.final_norm, self.lut_rsqrt)

    def logits(self, input_ids, layers=None, verbose=False, cache_kv=None):
        h = self.hidden(input_ids, layers=layers, verbose=verbose, cache_kv=cache_kv).float() / Q15_16
        return h @ self.lm_head.T


class Qwen3FloatModel:
    def __init__(self, seq_len, model_dir="/code/Qwen3-0.6B", config=QWEN3_0_6B):
        self.seq_len = seq_len
        self.config = config
        self.cos, self.sin, _ = rope_tables_q15_16(seq_len, config.head_dim, config.rope_theta)
        self.final_norm = load_final_norm(model_dir).float() / Q15_16
        self.embed = load_embed_tokens(model_dir)
        self.lm_head = load_lm_head(model_dir)
        self.layers = load_all_qwen3_block_weights(model_dir, config)

    def hidden(self, input_ids, layers=None, verbose=False):
        x = self.embed[input_ids]
        n_layers = self.config.num_hidden_layers if layers is None else layers
        for layer_idx in range(n_layers):
            t0 = time.time()
            x = block_torch(x, self.layers[layer_idx], self.cos, self.sin, self.config)
            if verbose:
                torch.cuda.synchronize()
                print(f"local-float layer {layer_idx} done in {time.time() - t0:.3f}s", flush=True)
        return rmsnorm_torch(x, self.final_norm)

    def logits(self, input_ids, layers=None, verbose=False):
        return self.hidden(input_ids, layers=layers, verbose=verbose) @ self.lm_head.T
