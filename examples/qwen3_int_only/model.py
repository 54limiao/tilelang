from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from safetensors import safe_open

from examples.qwen3_int_only.kernels import (
    Q15_16,
    add_q15_16,
    compile_kernel,
    dynamic_quant_q15_16,
    exp_lut,
    exp_lut_neg,
    flash_attention_i12_i12_per_scale,
    flash_attention_i12_q15_16_per_scale,
    flash_attention_i8_q15_16,
    flash_attention_i8_i8_per_scale,
    flash_attention_i8_q15_16_per_scale,
    linear_dynamic_int8_group_q15_16,
    linear_dynamic_int8_q15_16,
    linear_dynamic_int16_group_q15_16,
    linear_dynamic_int16_q15_16,
    linear_q15_16_int8_weight,
    mul_q15_16,
    packed_scale_matrix,
    packed_scale_tensor,
    recip_lut_i8,
    recip_lut_i12,
    recip_lut_i16,
    recip_lut_i16_norm,
    rmsnorm_i16_q15_16_weighted,
    rmsnorm_q15_16_weighted,
    rmsnorm_q15_16_weighted_safe,
    rope_q15_16,
    rsqrt_lut,
    silu_lut,
    silu_q15_16,
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
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
        )


def load_qwen3_block_weights(model_dir="/code/Qwen3-0.6B", layer_idx=0, device="cuda"):
    path = f"{model_dir}/model.safetensors"
    p = f"model.layers.{layer_idx}"
    with safe_open(path, framework="pt", device="cpu") as f:
        def tensor(name):
            return f.get_tensor(name).to(torch.float32).to(device)

        return Qwen3BlockWeights.pack(
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


def block_torch(x, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16, config=QWEN3_0_6B):
    cos, sin = cos_q15_16.float() / Q15_16, sin_q15_16.float() / Q15_16
    h = rmsnorm_torch(x, weights.input_layernorm.float() / Q15_16)
    q = h @ weights.q_proj_fp.T
    k = h @ weights.k_proj_fp.T
    v = h @ weights.v_proj_fp.T
    q = rmsnorm_torch(q.reshape(-1, config.num_attention_heads, config.head_dim), weights.q_norm.float() / Q15_16).reshape(-1, config.q_size)
    k = rmsnorm_torch(k.reshape(-1, config.num_key_value_heads, config.head_dim), weights.k_norm.float() / Q15_16).reshape(-1, config.kv_size)
    q = rope_torch(q, cos, sin, config.num_attention_heads, config.head_dim).reshape(-1, config.num_attention_heads, config.head_dim).permute(1, 0, 2)
    k = rope_torch(k, cos, sin, config.num_key_value_heads, config.head_dim).reshape(-1, config.num_key_value_heads, config.head_dim)
    group = config.num_attention_heads // config.num_key_value_heads
    k = k.repeat_interleave(group, dim=1).permute(1, 0, 2)
    v = v.reshape(-1, config.num_key_value_heads, config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2)
    attn = causal_softmax((q @ k.transpose(-1, -2)) / (config.head_dim**0.5)) @ v
    attn = attn.permute(1, 0, 2).reshape(x.shape[0], config.q_size)
    x = x + attn @ weights.o_proj_fp.T
    m = rmsnorm_torch(x, weights.post_attention_layernorm.float() / Q15_16)
    x = x + (torch.nn.functional.silu(m @ weights.gate_proj_fp.T) * (m @ weights.up_proj_fp.T)) @ weights.down_proj_fp.T
    return x


def attention_torch(x, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16, config=QWEN3_0_6B):
    cos, sin = cos_q15_16.float() / Q15_16, sin_q15_16.float() / Q15_16
    h = rmsnorm_torch(x, weights.input_layernorm.float() / Q15_16)
    q = h @ weights.q_proj_fp.T
    k = h @ weights.k_proj_fp.T
    v = h @ weights.v_proj_fp.T
    q = rmsnorm_torch(q.reshape(-1, config.num_attention_heads, config.head_dim), weights.q_norm.float() / Q15_16).reshape(-1, config.q_size)
    k = rmsnorm_torch(k.reshape(-1, config.num_key_value_heads, config.head_dim), weights.k_norm.float() / Q15_16).reshape(-1, config.kv_size)
    q = rope_torch(q, cos, sin, config.num_attention_heads, config.head_dim).reshape(-1, config.num_attention_heads, config.head_dim).permute(1, 0, 2)
    k = rope_torch(k, cos, sin, config.num_key_value_heads, config.head_dim).reshape(-1, config.num_key_value_heads, config.head_dim)
    group = config.num_attention_heads // config.num_key_value_heads
    k = k.repeat_interleave(group, dim=1).permute(1, 0, 2)
    v = v.reshape(-1, config.num_key_value_heads, config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2)
    attn = causal_softmax((q @ k.transpose(-1, -2)) / (config.head_dim**0.5)) @ v
    return attn.permute(1, 0, 2).reshape(x.shape[0], config.q_size) @ weights.o_proj_fp.T


def mlp_torch(x, weights: Qwen3BlockWeights):
    h = rmsnorm_torch(x, weights.post_attention_layernorm.float() / Q15_16)
    return (torch.nn.functional.silu(h @ weights.gate_proj_fp.T) * (h @ weights.up_proj_fp.T)) @ weights.down_proj_fp.T


class Qwen3IntOnlyBlock:
    def __init__(self, seq_len, config=QWEN3_0_6B):
        self.seq_len = seq_len
        self.config = config
        h, hd, im = config.hidden_size, config.head_dim, config.intermediate_size
        qh, kvh = config.num_attention_heads, config.num_key_value_heads
        q_dim, kv_dim = config.q_size, config.kv_size
        self.rms_hidden = compile_kernel(rmsnorm_q15_16_weighted(seq_len, h), [3])
        self.rms_hidden_safe = compile_kernel(rmsnorm_q15_16_weighted_safe(seq_len, h, 7), [3])
        self.rms_hidden_dyn = compile_kernel(rmsnorm_i16_q15_16_weighted(seq_len, h), [3])
        self.rms_q = compile_kernel(rmsnorm_q15_16_weighted_safe(seq_len * qh, hd, 7), [3])
        self.rms_k = compile_kernel(rmsnorm_q15_16_weighted_safe(seq_len * kvh, hd, 7), [3])
        self.rms_q_dyn = compile_kernel(rmsnorm_i16_q15_16_weighted(seq_len * qh, hd), [3])
        self.rms_k_dyn = compile_kernel(rmsnorm_i16_q15_16_weighted(seq_len * kvh, hd), [3])
        self.dq8_hidden = compile_kernel(dynamic_quant_q15_16(seq_len, h, "int8"), [2, 3])
        self.dq16_hidden = compile_kernel(dynamic_quant_q15_16(seq_len, h, "int16"), [2, 3])
        self.dq8_qhead = compile_kernel(dynamic_quant_q15_16(seq_len * qh, hd, "int8"), [2, 3])
        self.dq8_kvhead = compile_kernel(dynamic_quant_q15_16(seq_len * kvh, hd, "int8"), [2, 3])
        self.dq12_qhead = compile_kernel(dynamic_quant_q15_16(seq_len * qh, hd, "int16", 2047), [2, 3])
        self.dq12_kvhead = compile_kernel(dynamic_quant_q15_16(seq_len * kvh, hd, "int16", 2047), [2, 3])
        self.dq16_hidden_norm = compile_kernel(dynamic_quant_q15_16(seq_len, h, "int16", 4095), [2, 3])
        self.dq16_q_norm = compile_kernel(dynamic_quant_q15_16(seq_len, q_dim, "int16", 4095), [2, 3])
        self.dq16_kv_norm = compile_kernel(dynamic_quant_q15_16(seq_len, kv_dim, "int16", 4095), [2, 3])
        self.dq16_q = compile_kernel(dynamic_quant_q15_16(seq_len, q_dim, "int16"), [2, 3])
        self.dq16_kv = compile_kernel(dynamic_quant_q15_16(seq_len, kv_dim, "int16"), [2, 3])
        self.dq16_mid = compile_kernel(dynamic_quant_q15_16(seq_len, im, "int16"), [2, 3])
        self.q_proj = compile_kernel(linear_dynamic_int8_q15_16(seq_len, h, q_dim), [4])
        self.k_proj = compile_kernel(linear_dynamic_int8_q15_16(seq_len, h, kv_dim), [4])
        self.v_proj = compile_kernel(linear_dynamic_int8_q15_16(seq_len, h, kv_dim), [4])
        self.o_proj = compile_kernel(linear_dynamic_int16_q15_16(seq_len, q_dim, h), [4])
        self.o_proj_q15 = compile_kernel(linear_q15_16_int8_weight(seq_len, q_dim, h), [3])
        self.o_proj_i8 = compile_kernel(linear_dynamic_int8_q15_16(seq_len, q_dim, h), [4])
        self.o_proj_i8_group = compile_kernel(linear_dynamic_int8_group_q15_16(seq_len, q_dim, h, qh), [4])
        self.o_proj_i12_group = compile_kernel(linear_dynamic_int16_group_q15_16(seq_len, q_dim, h, qh), [4])
        self.gate_proj = compile_kernel(linear_dynamic_int16_q15_16(seq_len, h, im), [4])
        self.up_proj = compile_kernel(linear_dynamic_int16_q15_16(seq_len, h, im), [4])
        self.down_proj = compile_kernel(linear_dynamic_int16_q15_16(seq_len, im, h), [4])
        self.rope_q = compile_kernel(rope_q15_16(seq_len * qh, hd), [3])
        self.rope_k = compile_kernel(rope_q15_16(seq_len * kvh, hd), [3])
        self.silu_mid = compile_kernel(silu_q15_16(seq_len, im), [2])
        self.mul_mid = compile_kernel(mul_q15_16(seq_len, im), [2])
        self.add_hidden = compile_kernel(add_q15_16(seq_len, h), [2])
        self.attn = compile_kernel(flash_attention_i8_q15_16(qh, seq_len, hd, block_n=32), [6])
        self.attn_per_scale = compile_kernel(flash_attention_i8_q15_16_per_scale(qh, seq_len, hd, block_n=32), [6])
        self.attn_i8 = compile_kernel(flash_attention_i8_i8_per_scale(qh, seq_len, hd, block_n=32), [5])
        self.attn_i12 = compile_kernel(flash_attention_i12_i12_per_scale(qh, seq_len, hd, block_n=32), [5])
        self.attn_i12_q15 = compile_kernel(flash_attention_i12_q15_16_per_scale(qh, seq_len, hd, block_n=32), [6])
        self.lut_i8 = torch.from_numpy(recip_lut_i8()).cuda()
        self.lut_i12 = torch.from_numpy(recip_lut_i12()).cuda()
        self.lut_i16 = torch.from_numpy(recip_lut_i16()).cuda()
        self.lut_i16_norm = torch.from_numpy(recip_lut_i16_norm()).cuda()
        self.lut_rsqrt = torch.from_numpy(rsqrt_lut()).cuda()
        self.lut_silu = torch.from_numpy(silu_lut()).cuda()
        self.lut_exp = torch.from_numpy(exp_lut()).cuda()
        self.lut_exp_neg = torch.from_numpy(exp_lut_neg()).cuda()

    def __call__(self, x_q15_16, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16, safe_hidden=True, per_scale_attn=False, int8_attn_out=False, int12_attn_out=False, int12_attn_q15=False, float_mlp=False):
        rms_hidden = self.rms_hidden_dyn
        x16, _ = self.dq16_hidden_norm(x_q15_16, self.lut_i16_norm)
        norm = rms_hidden(x16, weights.input_layernorm, self.lut_rsqrt)
        x8, xs8 = self.dq8_hidden(norm, self.lut_i8)
        q = self.q_proj(x8, xs8, weights.q_proj.weight, weights.q_proj.scale)
        k = self.k_proj(x8, xs8, weights.k_proj.weight, weights.k_proj.scale)
        v = self.v_proj(x8, xs8, weights.v_proj.weight, weights.v_proj.scale)
        q_heads = q.reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim)
        k_heads = k.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim)
        v_heads = v.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim)
        qh16, _ = self.dq16_q_norm(q_heads.reshape(self.seq_len, self.config.q_size), self.lut_i16_norm)
        kh16, _ = self.dq16_kv_norm(k_heads.reshape(self.seq_len, self.config.kv_size), self.lut_i16_norm)
        q_heads = self.rms_q_dyn(qh16.reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim), weights.q_norm, self.lut_rsqrt)
        k_heads = self.rms_k_dyn(kh16.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim), weights.k_norm, self.lut_rsqrt)
        cos_q = cos_q15_16[:, None, :].expand(self.seq_len, self.config.num_attention_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim // 2).contiguous()
        sin_q = sin_q15_16[:, None, :].expand(self.seq_len, self.config.num_attention_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim // 2).contiguous()
        cos_k = cos_q15_16[:, None, :].expand(self.seq_len, self.config.num_key_value_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim // 2).contiguous()
        sin_k = sin_q15_16[:, None, :].expand(self.seq_len, self.config.num_key_value_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim // 2).contiguous()
        qr = self.rope_q(q_heads, cos_q, sin_q)
        kr = self.rope_k(k_heads, cos_k, sin_k)
        qr8, qrs8 = self.dq8_qhead(qr, self.lut_i8)
        kr8, krs8 = self.dq8_kvhead(kr, self.lut_i8)
        v8, vs8 = self.dq8_kvhead(v_heads, self.lut_i8)
        group = self.config.num_attention_heads // self.config.num_key_value_heads
        k_attn = kr8.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2).contiguous()
        v_attn = v8.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2).contiguous()
        q_attn = qr8.reshape(self.seq_len, self.config.num_attention_heads, self.config.head_dim).permute(1, 0, 2).contiguous().to(torch.int16)
        if per_scale_attn:
            q_scale = qrs8.reshape(self.seq_len, self.config.num_attention_heads).permute(1, 0).to(torch.float64) / Q15_16
            k_scale = krs8.reshape(self.seq_len, self.config.num_key_value_heads).repeat_interleave(group, dim=1).permute(1, 0).to(torch.float64) / Q15_16
            score_q = torch.from_numpy(packed_scale_matrix(q_scale[:, :, None] * k_scale[:, None, :] / (self.config.head_dim**0.5) * 64.0).reshape(-1)).cuda()
            value_q = vs8.reshape(self.seq_len, self.config.num_key_value_heads).repeat_interleave(group, dim=1).permute(1, 0).contiguous().to(torch.uint32)
            attn = self.attn_per_scale(q_attn, k_attn.to(torch.int16), v_attn.to(torch.int16), self.lut_exp, score_q, value_q)
        elif int8_attn_out:
            q_scale = qrs8.reshape(self.seq_len, self.config.num_attention_heads).permute(1, 0).to(torch.float64) / Q15_16
            k_scale = krs8.reshape(self.seq_len, self.config.num_key_value_heads).repeat_interleave(group, dim=1).permute(1, 0).to(torch.float64) / Q15_16
            score_q = torch.from_numpy(packed_scale_matrix(q_scale[:, :, None] * k_scale[:, None, :] / (self.config.head_dim**0.5) * 64.0).reshape(-1)).cuda()
            attn8 = self.attn_i8(q_attn, k_attn.to(torch.int16), v_attn.to(torch.int16), self.lut_exp, score_q)
            attn = attn8.permute(1, 0, 2).reshape(self.seq_len, self.config.q_size)
            attn_s = vs8.reshape(self.seq_len, self.config.num_key_value_heads).repeat_interleave(group, dim=1).contiguous()
            attn_out = self.o_proj_i8_group(attn, attn_s, weights.o_proj.weight, weights.o_proj.scale)
            h = self.add_hidden(x_q15_16, attn_out)
            h_norm16, _ = self.dq16_hidden_norm(h, self.lut_i16_norm)
            post = rms_hidden(h_norm16, weights.post_attention_layernorm, self.lut_rsqrt)
            h16, hs16 = self.dq16_hidden(post, self.lut_i16)
            gate = self.gate_proj(h16, hs16, weights.gate_proj.weight, weights.gate_proj.scale)
            up = self.up_proj(h16, hs16, weights.up_proj.weight, weights.up_proj.scale)
            gated = self.mul_mid(self.silu_mid(gate, self.lut_silu), up)
            gated16, gs16 = self.dq16_mid(gated, self.lut_i16)
            mlp = self.down_proj(gated16, gs16, weights.down_proj.weight, weights.down_proj.scale)
            return self.add_hidden(h, mlp)
        elif int12_attn_out:
            qr12, qrs12 = self.dq12_qhead(qr, self.lut_i12)
            kr12, krs12 = self.dq12_kvhead(kr, self.lut_i12)
            v12, vs12 = self.dq12_kvhead(v_heads, self.lut_i12)
            q_scale = qrs12.reshape(self.seq_len, self.config.num_attention_heads).permute(1, 0).to(torch.float64) / Q15_16
            k_scale = krs12.reshape(self.seq_len, self.config.num_key_value_heads).repeat_interleave(group, dim=1).permute(1, 0).to(torch.float64) / Q15_16
            score_q = torch.from_numpy(packed_scale_matrix(q_scale[:, :, None] * k_scale[:, None, :] / (self.config.head_dim**0.5) * 64.0).reshape(-1)).cuda()
            q12_attn = qr12.reshape(self.seq_len, self.config.num_attention_heads, self.config.head_dim).permute(1, 0, 2).contiguous()
            k12_attn = kr12.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2).contiguous()
            v12_attn = v12.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2).contiguous()
            attn12 = self.attn_i12(q12_attn, k12_attn, v12_attn, self.lut_exp_neg, score_q)
            attn = attn12.permute(1, 0, 2).reshape(self.seq_len, self.config.q_size)
            attn_s = vs12.reshape(self.seq_len, self.config.num_key_value_heads).repeat_interleave(group, dim=1).contiguous()
            attn_out = self.o_proj_i12_group(attn, attn_s, weights.o_proj.weight, weights.o_proj.scale)
            h = self.add_hidden(x_q15_16, attn_out)
            h_norm16, _ = self.dq16_hidden_norm(h, self.lut_i16_norm)
            post = rms_hidden(h_norm16, weights.post_attention_layernorm, self.lut_rsqrt)
            h16, hs16 = self.dq16_hidden(post, self.lut_i16)
            gate = self.gate_proj(h16, hs16, weights.gate_proj.weight, weights.gate_proj.scale)
            up = self.up_proj(h16, hs16, weights.up_proj.weight, weights.up_proj.scale)
            gated = self.mul_mid(self.silu_mid(gate, self.lut_silu), up)
            gated16, gs16 = self.dq16_mid(gated, self.lut_i16)
            mlp = self.down_proj(gated16, gs16, weights.down_proj.weight, weights.down_proj.scale)
            return self.add_hidden(h, mlp)
        elif int12_attn_q15:
            qr12, qrs12 = self.dq12_qhead(qr, self.lut_i12)
            kr12, krs12 = self.dq12_kvhead(kr, self.lut_i12)
            v12, vs12 = self.dq12_kvhead(v_heads, self.lut_i12)
            q_scale = qrs12.reshape(self.seq_len, self.config.num_attention_heads).permute(1, 0).to(torch.float64) / Q15_16
            k_scale = krs12.reshape(self.seq_len, self.config.num_key_value_heads).repeat_interleave(group, dim=1).permute(1, 0).to(torch.float64) / Q15_16
            score_q = torch.from_numpy(packed_scale_matrix(q_scale[:, :, None] * k_scale[:, None, :] / (self.config.head_dim**0.5) * 64.0).reshape(-1)).cuda()
            q12_attn = qr12.reshape(self.seq_len, self.config.num_attention_heads, self.config.head_dim).permute(1, 0, 2).contiguous()
            k12_attn = kr12.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2).contiguous()
            v12_attn = v12.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2).contiguous()
            value_q = vs12.reshape(self.seq_len, self.config.num_key_value_heads).repeat_interleave(group, dim=1).permute(1, 0).contiguous().to(torch.uint32)
            attn = self.attn_i12_q15(q12_attn, k12_attn, v12_attn, self.lut_exp_neg, score_q, value_q)
        else:
            score_scale = float((qrs8.to(torch.float64).mean() * krs8.to(torch.float64).mean()).item())
            score_scale = score_scale / (Q15_16 * Q15_16 * (self.config.head_dim**0.5)) * 64.0
            v_scale = int(vs8.to(torch.int64).float().mean().item())
            score_q = torch.from_numpy(packed_scale_tensor(score_scale)).cuda()
            value_q = torch.tensor([v_scale], device="cuda", dtype=torch.uint32)
            attn = self.attn(q_attn, k_attn.to(torch.int16), v_attn.to(torch.int16), self.lut_exp, score_q, value_q)
        attn = attn.permute(1, 0, 2).reshape(self.seq_len, self.config.q_size)
        attn16, attn_s16 = self.dq16_q(attn, self.lut_i16)
        attn_out = self.o_proj(attn16, attn_s16, weights.o_proj.weight, weights.o_proj.scale)
        h = self.add_hidden(x_q15_16, attn_out)
        if float_mlp:
            return q15_16(h.float() / Q15_16 + mlp_torch(h.float() / Q15_16, weights))
        h_norm16, _ = self.dq16_hidden_norm(h, self.lut_i16_norm)
        post = rms_hidden(h_norm16, weights.post_attention_layernorm, self.lut_rsqrt)
        h16, hs16 = self.dq16_hidden(post, self.lut_i16)
        gate = self.gate_proj(h16, hs16, weights.gate_proj.weight, weights.gate_proj.scale)
        up = self.up_proj(h16, hs16, weights.up_proj.weight, weights.up_proj.scale)
        gated = self.mul_mid(self.silu_mid(gate, self.lut_silu), up)
        gated16, gs16 = self.dq16_mid(gated, self.lut_i16)
        mlp = self.down_proj(gated16, gs16, weights.down_proj.weight, weights.down_proj.scale)
        return self.add_hidden(h, mlp)

    def proto_mlp_only(self, x_q15_16, weights: Qwen3BlockWeights):
        from examples.qwen3_int_only.proto import mlp_proto

        return mlp_proto(x_q15_16, weights)

    def proto_attention_q15_only(self, x_q15_16, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16):
        from examples.qwen3_int_only.proto import attention_i12_lut_proto, linear_dynamic_proto, linear_i16_kernel_proto, q15, rmsnorm_i16_proto

        cos, sin = cos_q15_16.float() / Q15_16, sin_q15_16.float() / Q15_16
        x = x_q15_16.float() / Q15_16
        norm = rmsnorm_i16_proto(q15(x), weights.input_layernorm).float() / Q15_16
        q, _, _ = linear_dynamic_proto(q15(norm), weights.q_proj.weight, weights.q_proj.scale, 127)
        k, _, _ = linear_dynamic_proto(q15(norm), weights.k_proj.weight, weights.k_proj.scale, 127)
        v, _, _ = linear_dynamic_proto(q15(norm), weights.v_proj.weight, weights.v_proj.scale, 127)
        qn = rmsnorm_i16_proto(q.reshape(-1, self.config.head_dim), weights.q_norm).reshape(self.seq_len, self.config.q_size).float() / Q15_16
        kn = rmsnorm_i16_proto(k.reshape(-1, self.config.head_dim), weights.k_norm).reshape(self.seq_len, self.config.kv_size).float() / Q15_16
        qr = rope_torch(qn, cos, sin, self.config.num_attention_heads, self.config.head_dim)
        kr = rope_torch(kn, cos, sin, self.config.num_key_value_heads, self.config.head_dim)
        group = self.config.num_attention_heads // self.config.num_key_value_heads
        kr = kr.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).reshape(self.seq_len, self.config.q_size)
        vv = (v.float() / Q15_16).reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).reshape(self.seq_len, self.config.q_size)
        attn = attention_i12_lut_proto(q15(qr), q15(kr), q15(vv), self.config)
        attn_out, _, _ = linear_i16_kernel_proto(attn, weights.o_proj.weight, weights.o_proj.scale)
        return self.add_hidden(x_q15_16, attn_out)

    def tile_attention_q15_only(self, x_q15_16, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16):
        x16, _ = self.dq16_hidden_norm(x_q15_16, self.lut_i16_norm)
        norm = self.rms_hidden_dyn(x16, weights.input_layernorm, self.lut_rsqrt)
        x8, xs8 = self.dq8_hidden(norm, self.lut_i8)
        q = self.q_proj(x8, xs8, weights.q_proj.weight, weights.q_proj.scale)
        k = self.k_proj(x8, xs8, weights.k_proj.weight, weights.k_proj.scale)
        v = self.v_proj(x8, xs8, weights.v_proj.weight, weights.v_proj.scale)
        q_heads = q.reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim)
        k_heads = k.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim)
        v_heads = v.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim)
        qh16, _ = self.dq16_q_norm(q_heads.reshape(self.seq_len, self.config.q_size), self.lut_i16_norm)
        kh16, _ = self.dq16_kv_norm(k_heads.reshape(self.seq_len, self.config.kv_size), self.lut_i16_norm)
        q_heads = self.rms_q_dyn(qh16.reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim), weights.q_norm, self.lut_rsqrt)
        k_heads = self.rms_k_dyn(kh16.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim), weights.k_norm, self.lut_rsqrt)
        cos_q = cos_q15_16[:, None, :].expand(self.seq_len, self.config.num_attention_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim // 2).contiguous()
        sin_q = sin_q15_16[:, None, :].expand(self.seq_len, self.config.num_attention_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim // 2).contiguous()
        cos_k = cos_q15_16[:, None, :].expand(self.seq_len, self.config.num_key_value_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim // 2).contiguous()
        sin_k = sin_q15_16[:, None, :].expand(self.seq_len, self.config.num_key_value_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim // 2).contiguous()
        qr = self.rope_q(q_heads, cos_q, sin_q)
        kr = self.rope_k(k_heads, cos_k, sin_k)
        qr12, qrs12 = self.dq12_qhead(qr, self.lut_i12)
        kr12, krs12 = self.dq12_kvhead(kr, self.lut_i12)
        v12, vs12 = self.dq12_kvhead(v_heads, self.lut_i12)
        group = self.config.num_attention_heads // self.config.num_key_value_heads
        q_scale = qrs12.reshape(self.seq_len, self.config.num_attention_heads).permute(1, 0).to(torch.float64) / Q15_16
        k_scale = krs12.reshape(self.seq_len, self.config.num_key_value_heads).repeat_interleave(group, dim=1).permute(1, 0).to(torch.float64) / Q15_16
        score_q = torch.from_numpy(packed_scale_matrix(q_scale[:, :, None] * k_scale[:, None, :] / (self.config.head_dim**0.5) * 64.0).reshape(-1)).cuda()
        q12_attn = qr12.reshape(self.seq_len, self.config.num_attention_heads, self.config.head_dim).permute(1, 0, 2).contiguous()
        k12_attn = kr12.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2).contiguous()
        v12_attn = v12.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2).contiguous()
        value_q = vs12.reshape(self.seq_len, self.config.num_key_value_heads).repeat_interleave(group, dim=1).permute(1, 0).contiguous().to(torch.uint32)
        attn = self.attn_i12_q15(q12_attn, k12_attn, v12_attn, self.lut_exp_neg, score_q, value_q)
        attn = attn.permute(1, 0, 2).reshape(self.seq_len, self.config.q_size)
        attn16, attn_s16 = self.dq16_q(attn, self.lut_i16)
        attn_out = self.o_proj(attn16, attn_s16, weights.o_proj.weight, weights.o_proj.scale)
        return self.add_hidden(x_q15_16, attn_out)

    def tile_pre_proto_attention_q15_only(self, x_q15_16, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16):
        from examples.qwen3_int_only.proto import attention_i12_lut_proto, linear_i16_kernel_proto

        x16, _ = self.dq16_hidden_norm(x_q15_16, self.lut_i16_norm)
        norm = self.rms_hidden_dyn(x16, weights.input_layernorm, self.lut_rsqrt)
        x8, xs8 = self.dq8_hidden(norm, self.lut_i8)
        q = self.q_proj(x8, xs8, weights.q_proj.weight, weights.q_proj.scale)
        k = self.k_proj(x8, xs8, weights.k_proj.weight, weights.k_proj.scale)
        v = self.v_proj(x8, xs8, weights.v_proj.weight, weights.v_proj.scale)
        q_heads = q.reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim)
        k_heads = k.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim)
        v_heads = v.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim)
        qh16, _ = self.dq16_q_norm(q_heads.reshape(self.seq_len, self.config.q_size), self.lut_i16_norm)
        kh16, _ = self.dq16_kv_norm(k_heads.reshape(self.seq_len, self.config.kv_size), self.lut_i16_norm)
        q_heads = self.rms_q_dyn(qh16.reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim), weights.q_norm, self.lut_rsqrt)
        k_heads = self.rms_k_dyn(kh16.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim), weights.k_norm, self.lut_rsqrt)
        cos_q = cos_q15_16[:, None, :].expand(self.seq_len, self.config.num_attention_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim // 2).contiguous()
        sin_q = sin_q15_16[:, None, :].expand(self.seq_len, self.config.num_attention_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim // 2).contiguous()
        cos_k = cos_q15_16[:, None, :].expand(self.seq_len, self.config.num_key_value_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim // 2).contiguous()
        sin_k = sin_q15_16[:, None, :].expand(self.seq_len, self.config.num_key_value_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim // 2).contiguous()
        qr = self.rope_q(q_heads, cos_q, sin_q)
        kr = self.rope_k(k_heads, cos_k, sin_k)
        group = self.config.num_attention_heads // self.config.num_key_value_heads
        kr = kr.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).reshape(self.seq_len, self.config.q_size)
        vv = (v.float() / Q15_16).reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).reshape(self.seq_len, self.config.q_size)
        attn = attention_i12_lut_proto(qr, kr, q15_16(vv), self.config)
        attn_out, _, _ = linear_i16_kernel_proto(attn, weights.o_proj.weight, weights.o_proj.scale)
        return self.add_hidden(x_q15_16, attn_out)

    def mlp_only(self, x_q15_16, weights: Qwen3BlockWeights, safe_hidden=True):
        x16, _ = self.dq16_hidden_norm(x_q15_16, self.lut_i16_norm)
        post = self.rms_hidden_dyn(x16, weights.post_attention_layernorm, self.lut_rsqrt)
        h16, hs16 = self.dq16_hidden(post, self.lut_i16)
        gate = self.gate_proj(h16, hs16, weights.gate_proj.weight, weights.gate_proj.scale)
        up = self.up_proj(h16, hs16, weights.up_proj.weight, weights.up_proj.scale)
        gated = self.mul_mid(self.silu_mid(gate, self.lut_silu), up)
        gated16, gs16 = self.dq16_mid(gated, self.lut_i16)
        mlp = self.down_proj(gated16, gs16, weights.down_proj.weight, weights.down_proj.scale)
        return self.add_hidden(x_q15_16, mlp)

    def attention_only(self, x_q15_16, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16, safe_hidden=True):
        x16, _ = self.dq16_hidden_norm(x_q15_16, self.lut_i16_norm)
        norm = self.rms_hidden_dyn(x16, weights.input_layernorm, self.lut_rsqrt)
        x8, xs8 = self.dq8_hidden(norm, self.lut_i8)
        q = self.q_proj(x8, xs8, weights.q_proj.weight, weights.q_proj.scale)
        k = self.k_proj(x8, xs8, weights.k_proj.weight, weights.k_proj.scale)
        v = self.v_proj(x8, xs8, weights.v_proj.weight, weights.v_proj.scale)
        qh16, _ = self.dq16_q_norm(q, self.lut_i16_norm)
        kh16, _ = self.dq16_kv_norm(k, self.lut_i16_norm)
        q_heads = self.rms_q_dyn(qh16.reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim), weights.q_norm, self.lut_rsqrt)
        k_heads = self.rms_k_dyn(kh16.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim), weights.k_norm, self.lut_rsqrt)
        v_heads = v.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim)
        cos_q = cos_q15_16[:, None, :].expand(self.seq_len, self.config.num_attention_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim // 2).contiguous()
        sin_q = sin_q15_16[:, None, :].expand(self.seq_len, self.config.num_attention_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim // 2).contiguous()
        cos_k = cos_q15_16[:, None, :].expand(self.seq_len, self.config.num_key_value_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim // 2).contiguous()
        sin_k = sin_q15_16[:, None, :].expand(self.seq_len, self.config.num_key_value_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim // 2).contiguous()
        qr = self.rope_q(q_heads, cos_q, sin_q)
        kr = self.rope_k(k_heads, cos_k, sin_k)
        qr8, qrs8 = self.dq8_qhead(qr, self.lut_i8)
        kr8, krs8 = self.dq8_kvhead(kr, self.lut_i8)
        v8, vs8 = self.dq8_kvhead(v_heads, self.lut_i8)
        group = self.config.num_attention_heads // self.config.num_key_value_heads
        k_attn = kr8.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2).contiguous()
        v_attn = v8.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2).contiguous()
        q_attn = qr8.reshape(self.seq_len, self.config.num_attention_heads, self.config.head_dim).permute(1, 0, 2).contiguous().to(torch.int16)
        score_scale = float((qrs8.to(torch.float64).mean() * krs8.to(torch.float64).mean()).item())
        score_scale = score_scale / (Q15_16 * Q15_16 * (self.config.head_dim**0.5)) * 64.0
        v_scale = int(vs8.to(torch.int64).float().mean().item())
        score_q = torch.from_numpy(packed_scale_tensor(score_scale)).cuda()
        value_q = torch.tensor([v_scale], device="cuda", dtype=torch.uint32)
        attn = self.attn(q_attn, k_attn.to(torch.int16), v_attn.to(torch.int16), self.lut_exp, score_q, value_q)
        attn = attn.permute(1, 0, 2).reshape(self.seq_len, self.config.q_size)
        attn16, attn_s16 = self.dq16_q(attn, self.lut_i16)
        attn_out = self.o_proj(attn16, attn_s16, weights.o_proj.weight, weights.o_proj.scale)
        return self.add_hidden(x_q15_16, attn_out)


class Qwen3IntOnlyModel:
    def __init__(self, seq_len, model_dir="/code/Qwen3-0.6B", config=QWEN3_0_6B, safe_from=2):
        self.seq_len = seq_len
        self.model_dir = model_dir
        self.config = config
        self.safe_from = safe_from
        self.block = Qwen3IntOnlyBlock(seq_len, config)
        self.final_norm_kernel = compile_kernel(rmsnorm_i16_q15_16_weighted(seq_len, config.hidden_size), [3])
        self.lut_i16 = torch.from_numpy(recip_lut_i16()).cuda()
        self.lut_i16_norm = torch.from_numpy(recip_lut_i16_norm()).cuda()
        self.lut_rsqrt = torch.from_numpy(rsqrt_lut()).cuda()
        self.cos, self.sin, _ = rope_tables_q15_16(seq_len, config.head_dim, config.rope_theta)
        self.final_norm = load_final_norm(model_dir)
        self.embed = load_embed_tokens(model_dir)
        self.lm_head = load_lm_head(model_dir)
        self.layers = load_all_qwen3_block_weights(model_dir, config)

    def embed_input(self, input_ids):
        return q15_16(self.embed[input_ids])

    def hidden(self, input_ids, layers=None, verbose=False, hybrid=None):
        x = self.embed_input(input_ids)
        n_layers = self.config.num_hidden_layers if layers is None else layers
        for layer_idx in range(n_layers):
            t0 = time.time()
            if hybrid == "float-block":
                x = q15_16(block_torch(x.float() / Q15_16, self.layers[layer_idx], self.cos, self.sin, self.config))
            elif hybrid == "float-attn":
                attn = attention_torch(x.float() / Q15_16, self.layers[layer_idx], self.cos, self.sin, self.config)
                x = self.block.mlp_only(q15_16(x.float() / Q15_16 + attn), self.layers[layer_idx], safe_hidden=layer_idx >= self.safe_from)
            elif hybrid == "float-mlp":
                h = self.block.attention_only(x, self.layers[layer_idx], self.cos, self.sin, safe_hidden=layer_idx >= self.safe_from)
                x = q15_16(h.float() / Q15_16 + mlp_torch(h.float() / Q15_16, self.layers[layer_idx]))
            elif hybrid == "per-scale-attn":
                x = self.block(x, self.layers[layer_idx], self.cos, self.sin, safe_hidden=layer_idx >= self.safe_from, per_scale_attn=True)
            elif hybrid == "int8-attn-out":
                x = self.block(x, self.layers[layer_idx], self.cos, self.sin, safe_hidden=layer_idx >= self.safe_from, int8_attn_out=True)
            elif hybrid == "int12-attn-out":
                x = self.block(x, self.layers[layer_idx], self.cos, self.sin, safe_hidden=layer_idx >= self.safe_from, int12_attn_out=True)
            elif hybrid == "int12-attn-q15":
                x = self.block(x, self.layers[layer_idx], self.cos, self.sin, safe_hidden=layer_idx >= self.safe_from, int12_attn_q15=True)
            elif hybrid == "int12-attn-q15-float-mlp":
                x = self.block(x, self.layers[layer_idx], self.cos, self.sin, safe_hidden=layer_idx >= self.safe_from, int12_attn_q15=True, float_mlp=True)
            elif hybrid == "int12-attn-q15-proto-mlp":
                h = self.block.tile_attention_q15_only(x, self.layers[layer_idx], self.cos, self.sin)
                x = self.block.proto_mlp_only(h, self.layers[layer_idx])
            elif hybrid == "proto-attn-q15-float-mlp":
                h = self.block.proto_attention_q15_only(x, self.layers[layer_idx], self.cos, self.sin)
                x = q15_16(h.float() / Q15_16 + mlp_torch(h.float() / Q15_16, self.layers[layer_idx]))
            elif hybrid == "tile-pre-proto-attn-q15-float-mlp":
                h = self.block.tile_pre_proto_attention_q15_only(x, self.layers[layer_idx], self.cos, self.sin)
                x = q15_16(h.float() / Q15_16 + mlp_torch(h.float() / Q15_16, self.layers[layer_idx]))
            elif hybrid == "proto-attn-q15":
                x = self.block.mlp_only(self.block.proto_attention_q15_only(x, self.layers[layer_idx], self.cos, self.sin), self.layers[layer_idx], safe_hidden=layer_idx >= self.safe_from)
            else:
                x = self.block(x, self.layers[layer_idx], self.cos, self.sin, safe_hidden=layer_idx >= self.safe_from)
            if verbose:
                torch.cuda.synchronize()
                print(f"int-only layer {layer_idx} done in {time.time() - t0:.3f}s", flush=True)
        x16, _ = self.block.dq16_hidden_norm(x, self.block.lut_i16_norm)
        return self.final_norm_kernel(x16, self.final_norm, self.lut_rsqrt)

    def logits(self, input_ids, layers=None, verbose=False, hybrid=None):
        h = self.hidden(input_ids, layers=layers, verbose=verbose, hybrid=hybrid).float() / Q15_16
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
