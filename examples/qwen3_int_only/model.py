from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from examples.qwen3_int_only.kernels import (
    Q15_16,
    add_rmsnorm_q15_16_weighted,
    add_q15_16,
    attention_i16v8_q15_16_gqa_cache,
    attention_i16v8_q15_16_gqa,
    attention_i8_q15_16_gqa_cache_softmax_i16,
    attention_i8_q15_16_gqa_softmax_i16,
    compile_kernel,
    dynamic_quant_q15_16,
    exp_lut_neg,
    attention_normalize_q15_16,
    flash_attention_i8_q15_16_gqa,
    flash_attention_i8_q15_16_gqa_cache,
    flash_attention_i8_q15_16_gqa_tiled,
    linear_dynamic_int8_pair_q15_16,
    linear_dynamic_int8_qkv_q15_16,
    linear_dynamic_int8_q15_16,
    rmsnorm_q15_16_grouped_weighted,
    rmsnorm_q15_16_weighted,
    rope_rotate_q15_16,
    rope_q15_16,
    rsqrt_lut,
    sigmoid_lut,
    silu_mul_dynamic_quant_q15_16,
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


class Qwen3IntOnlyBlock:
    def __init__(self, seq_len, config=QWEN3_0_6B, cache_len=0, use_r3=False, split_attn=False):
        self.seq_len = seq_len
        self.cache_len = cache_len
        self.config = config
        self.use_r3 = use_r3
        self.split_attn = split_attn
        h, hd, im = config.hidden_size, config.head_dim, config.intermediate_size
        qh, kvh = config.num_attention_heads, config.num_key_value_heads
        q_dim, kv_dim = config.q_size, config.kv_size
        self.rms_hidden_q15 = compile_kernel(rmsnorm_q15_16_weighted(seq_len, h), [3])
        self.add_rms_hidden_q15 = compile_kernel(add_rmsnorm_q15_16_weighted(seq_len, h), [4, 5])
        self.rms_q_q15 = compile_kernel(rmsnorm_q15_16_grouped_weighted(seq_len, qh, hd), [3])
        self.rms_k_q15 = compile_kernel(rmsnorm_q15_16_grouped_weighted(seq_len, kvh, hd), [3])
        self.dq8_hidden = compile_kernel(dynamic_quant_q15_16(seq_len, h, "int8"), [1, 2])
        self.dq8_q_head = compile_kernel(dynamic_quant_q15_16(seq_len * qh, hd, "int8"), [1, 2])
        self.dq8_kv_head = compile_kernel(dynamic_quant_q15_16(seq_len * kvh, hd, "int8"), [1, 2])
        self.dq8_q = compile_kernel(dynamic_quant_q15_16(seq_len, q_dim, "int8"), [1, 2])
        self.qkv_proj_i8 = compile_kernel(linear_dynamic_int8_qkv_q15_16(seq_len, h, q_dim, kv_dim, 32, 64, 128), [8, 9, 10])
        self.o_proj = compile_kernel(linear_dynamic_int8_q15_16(seq_len, q_dim, h, 32, 32, 128), [4])
        self.gate_up_proj_i8 = compile_kernel(linear_dynamic_int8_pair_q15_16(seq_len, h, im, 32, 64, 64), [6, 7])
        self.down_proj_i8 = compile_kernel(linear_dynamic_int8_q15_16(seq_len, im, h, 32, 32, 128), [4])
        self.rope_q = compile_kernel(rope_rotate_q15_16(seq_len * qh, hd), [4]) if use_r3 else compile_kernel(rope_q15_16(seq_len * qh, hd), [3])
        self.rope_k = compile_kernel(rope_rotate_q15_16(seq_len * kvh, hd), [4]) if use_r3 else compile_kernel(rope_q15_16(seq_len * kvh, hd), [3])
        self.silu_mul_dq8_mid = compile_kernel(silu_mul_dynamic_quant_q15_16(seq_len, im), [3, 4, 5])
        self.add_hidden = compile_kernel(add_q15_16(seq_len, h), [2])
        self.attn_i8_fixed = compile_kernel(flash_attention_i8_q15_16_gqa_tiled(qh, kvh, seq_len, hd), [7, 8])
        self.attn_norm = compile_kernel(attention_normalize_q15_16(qh, seq_len, hd), [2])
        self.attn_softmax_i16 = compile_kernel(attention_i8_q15_16_gqa_softmax_i16(qh, kvh, seq_len, hd), [5]) if split_attn else None
        self.attn_i16v8 = compile_kernel(attention_i16v8_q15_16_gqa(qh, kvh, seq_len, hd), [3]) if split_attn else None
        self.attn_i8_fixed_cache = None
        self.attn_cache_softmax_i16 = None
        self.attn_cache_i16v8 = None
        if cache_len:
            self.attn_i8_fixed_cache = compile_kernel(flash_attention_i8_q15_16_gqa_cache(qh, kvh, seq_len, cache_len, hd), [11])
            if split_attn:
                self.attn_cache_softmax_i16 = compile_kernel(attention_i8_q15_16_gqa_cache_softmax_i16(qh, kvh, seq_len, cache_len, hd), [7])
                self.attn_cache_i16v8 = compile_kernel(attention_i16v8_q15_16_gqa_cache(qh, kvh, seq_len, cache_len, hd), [5])
        self.lut_rsqrt = torch.from_numpy(rsqrt_lut()).cuda()
        self.lut_sigmoid = torch.from_numpy(sigmoid_lut()).cuda()
        self.lut_exp = torch.from_numpy(exp_lut_neg()).cuda()

    def __call__(self, x_q15_16, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16, cache_k=None, cache_v=None, r3_q15=None):
        norm = self.rms_hidden_q15(x_q15_16, weights.input_layernorm, self.lut_rsqrt)
        x8, xs8 = self.dq8_hidden(norm)
        q, k, v = self.qkv_proj_i8(
            x8, xs8, weights.q_proj.weight, weights.q_proj.scale, weights.k_proj.weight, weights.k_proj.scale, weights.v_proj.weight, weights.v_proj.scale
        )
        v_heads = v.reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim)
        q_heads = self.rms_q_q15(q, weights.q_norm, self.lut_rsqrt)
        k_heads = self.rms_k_q15(k, weights.k_norm, self.lut_rsqrt)
        pos_cos = cos_q15_16[self.cache_len : self.cache_len + self.seq_len]
        pos_sin = sin_q15_16[self.cache_len : self.cache_len + self.seq_len]
        cos_q = pos_cos[:, None, :].expand(self.seq_len, self.config.num_attention_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim // 2).contiguous()
        sin_q = pos_sin[:, None, :].expand(self.seq_len, self.config.num_attention_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_attention_heads, self.config.head_dim // 2).contiguous()
        cos_k = pos_cos[:, None, :].expand(self.seq_len, self.config.num_key_value_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim // 2).contiguous()
        sin_k = pos_sin[:, None, :].expand(self.seq_len, self.config.num_key_value_heads, self.config.head_dim // 2).reshape(self.seq_len * self.config.num_key_value_heads, self.config.head_dim // 2).contiguous()
        if self.use_r3:
            qr = self.rope_q(q_heads, cos_q, sin_q, r3_q15)
            kr = self.rope_k(k_heads, cos_k, sin_k, r3_q15)
        else:
            qr = self.rope_q(q_heads, cos_q, sin_q)
            kr = self.rope_k(k_heads, cos_k, sin_k)
        q8, qs8 = self.dq8_q_head(qr)
        k8, ks8 = self.dq8_kv_head(kr)
        v8, vs8 = self.dq8_kv_head(v_heads)
        q_attn = q8.reshape(self.seq_len, self.config.num_attention_heads, self.config.head_dim).permute(1, 0, 2).contiguous()
        k_attn = k8.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).permute(1, 0, 2).contiguous()
        v_attn = v8.reshape(self.seq_len, self.config.num_key_value_heads, self.config.head_dim).permute(1, 0, 2).contiguous()
        qs_attn = qs8.reshape(self.seq_len, self.config.num_attention_heads).permute(1, 0).contiguous()
        ks_attn = ks8.reshape(self.seq_len, self.config.num_key_value_heads).permute(1, 0).contiguous()
        vs_attn = vs8.reshape(self.seq_len, self.config.num_key_value_heads).permute(1, 0).contiguous()
        if cache_k is None:
            if self.split_attn:
                prob_i16 = self.attn_softmax_i16(q_attn, k_attn, qs_attn, ks_attn, self.lut_exp)
                attn = self.attn_i16v8(prob_i16, v_attn, vs_attn)
            else:
                attn_num, attn_den = self.attn_i8_fixed(q_attn, k_attn, v_attn, qs_attn, ks_attn, vs_attn, self.lut_exp)
                attn = self.attn_norm(attn_num, attn_den)
        else:
            if self.split_attn:
                prob_i16 = self.attn_cache_softmax_i16(q_attn, cache_k[0], k_attn, qs_attn, cache_k[1], ks_attn, self.lut_exp)
                attn = self.attn_cache_i16v8(prob_i16, cache_v[0], v_attn, cache_v[1], vs_attn)
            else:
                attn = self.attn_i8_fixed_cache(q_attn, cache_k[0], cache_v[0], k_attn, v_attn, qs_attn, cache_k[1], cache_v[1], ks_attn, vs_attn, self.lut_exp)
        if not self.split_attn:
            attn = attn.permute(1, 0, 2).reshape(self.seq_len, self.config.q_size)
        attn8, attn_s8 = self.dq8_q(attn)
        attn_out = self.o_proj(attn8, attn_s8, weights.o_proj.weight, weights.o_proj.scale)
        h, post = self.add_rms_hidden_q15(x_q15_16, attn_out, weights.post_attention_layernorm, self.lut_rsqrt)
        h8, hs8 = self.dq8_hidden(post)
        gate, up = self.gate_up_proj_i8(h8, hs8, weights.gate_proj.weight, weights.gate_proj.scale, weights.up_proj.weight, weights.up_proj.scale)
        gated, gated8, gs8 = self.silu_mul_dq8_mid(gate, up, self.lut_sigmoid)
        mlp = self.down_proj_i8(gated8, gs8, weights.down_proj.weight, weights.down_proj.scale)
        return self.add_hidden(h, mlp)


class Qwen3IntOnlyModel:
    def __init__(self, seq_len, model_dir="/code/Qwen3-0.6B", packed_dir=None, config=QWEN3_0_6B, cache_len=0, use_r3=False, split_attn=False, rotate_seed=ROTATE_SEED):
        self.seq_len = seq_len
        self.cache_len = cache_len
        self.config = config
        self.r3_q15 = q15_16(random_hadamard_rotation(config.head_dim, rotate_seed + 2)) if use_r3 else None
        self.block = Qwen3IntOnlyBlock(seq_len, config, cache_len=cache_len, use_r3=use_r3, split_attn=split_attn)
        self.final_norm_kernel = compile_kernel(rmsnorm_q15_16_weighted(seq_len, config.hidden_size), [3])
        self.lut_rsqrt = torch.from_numpy(rsqrt_lut()).cuda()
        self.cos, self.sin, _ = rope_tables_q15_16(seq_len + cache_len, config.head_dim, config.rope_theta)
        if packed_dir is None:
            self.final_norm = load_final_norm(model_dir)
            self.embed = load_embed_tokens(model_dir)
            self.lm_head = load_lm_head(model_dir)
            self.layers = load_all_qwen3_block_weights(model_dir, config)
        else:
            self.embed, self.lm_head, self.final_norm, self.layers = load_packed_qwen3(packed_dir, config)

    def embed_input(self, input_ids):
        return q15_16(self.embed[input_ids])

    def hidden(self, input_ids, layers=None, verbose=False, cache_kv=None):
        x = self.embed_input(input_ids)
        n_layers = self.config.num_hidden_layers if layers is None else layers
        for layer_idx in range(n_layers):
            t0 = time.time()
            layer_cache = None if cache_kv is None else cache_kv[layer_idx]
            if layer_cache is None:
                x = self.block(x, self.layers[layer_idx], self.cos, self.sin, r3_q15=self.r3_q15)
            else:
                x = self.block(x, self.layers[layer_idx], self.cos, self.sin, layer_cache[0], layer_cache[1], self.r3_q15)
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
