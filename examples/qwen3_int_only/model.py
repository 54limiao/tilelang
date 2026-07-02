from __future__ import annotations

import torch
import tilelang

from examples.qwen3_int_only.kernels import (
    rms_q15,
    rms_sq8,
    attention_i8,
    linear_i8,
    linear_i16,
    rope_sq8,
    silu_i16,
    quant_v_i8,
)
from examples.qwen3_int_only.utils import ROTATE_SEED, QWEN3_0_6B, Qwen3BlockWeights, load_packed_qwen3, q15_16, random_hadamard_rotation, rope_tables_q15_16
from examples.qwen3_int_only.utils.lut import exp_lut_neg, rsqrt_lut, sigmoid_lut

Q15_16 = 1 << 16


def compile_(func, out_idx):
    return tilelang.compile(func, out_idx=out_idx, target="cuda")


class Qwen3IntOnlyBlock:
    def __init__(self, seq_len, config=QWEN3_0_6B, cache_len=0):
        self.seq_len = seq_len
        self.cache_len = cache_len
        self.config = config
        h, hd, im = config.hidden_size, config.head_dim, config.intermediate_size
        qh, kvh = config.num_attention_heads, config.num_key_value_heads
        q_dim, kv_dim = config.q_size, config.kv_size
        self.zero_hidden = torch.zeros((seq_len, h), device="cuda", dtype=torch.int32)
        self.zero_q = torch.zeros((seq_len * qh, hd), device="cuda", dtype=torch.int32)
        self.zero_k = torch.zeros((seq_len * kvh, hd), device="cuda", dtype=torch.int32)
        self.empty_cache_k = torch.empty((kvh, cache_len, hd), device="cuda", dtype=torch.int8)
        self.empty_cache_v = torch.empty((kvh, cache_len, hd), device="cuda", dtype=torch.int8)
        self.empty_cache_s = torch.empty((kvh, cache_len), device="cuda", dtype=torch.uint32)
        self.rms_q15 = compile_(rms_q15(seq_len, h), [4, 5])
        self.rms_sq8 = compile_(rms_sq8(seq_len, h), [5, 6, 7])
        self.rms_q = compile_(rms_q15(seq_len * qh, hd), [4, 5])
        self.rms_k = compile_(rms_q15(seq_len * kvh, hd), [4, 5])
        self.quant_v = compile_(quant_v_i8(seq_len, kvh, hd, "int8"), [2])
        self.rope_q = compile_(rope_sq8(seq_len, qh, hd), [5])
        self.rope_k = compile_(rope_sq8(seq_len, kvh, hd), [5])
        self.qkv_proj = compile_(linear_i8(seq_len, h, q_dim + 2 * kv_dim, 64, 128, 64), [4])
        self.o_proj = compile_(linear_i8(seq_len, q_dim, h, 64, 64, 64), [4])
        self.gate_up_proj = compile_(linear_i8(seq_len, h, 2 * im, 64, 128, 64), [4])
        self.silu_mul = compile_(silu_i16(seq_len, im), [4, 5])
        self.down_proj = compile_(linear_i16(seq_len, im, h, 64, 64, 64), [4])
        self.attn = compile_(attention_i8(qh, kvh, seq_len, cache_len, hd), [12])
        self.lut_rsqrt = torch.from_numpy(rsqrt_lut()).cuda()
        self.lut_sigmoid = torch.from_numpy(sigmoid_lut()).cuda()
        self.lut_exp = torch.from_numpy(exp_lut_neg()).cuda()

    def input_rms_quant(self, x, weights: Qwen3BlockWeights):
        _res, x8, _scale = self.rms_sq8(x, self.zero_hidden, weights.input_layernorm, self.lut_rsqrt, weights.input_qkv_i8_scale)
        return x8

    def __call__(self, x, weights: Qwen3BlockWeights, cos, sin, cache_k=None, cache_v=None, r3_q15=None, x8=None):
        cfg = self.config
        cache_k = (self.empty_cache_k, self.empty_cache_s) if cache_k is None else cache_k
        cache_v = (self.empty_cache_v, self.empty_cache_s) if cache_v is None else cache_v
        qkv = self.qkv_proj(x8, weights.input_qkv_i8_scale, weights.qkv_proj.weight, weights.qkv_proj.scale)
        q = qkv[:, : cfg.q_size].contiguous()
        k = qkv[:, cfg.q_size : cfg.q_size + cfg.kv_size].contiguous()
        v = qkv[:, cfg.q_size + cfg.kv_size :].contiguous()
        _q, q_heads = self.rms_q(q.reshape(self.seq_len * cfg.num_attention_heads, cfg.head_dim).contiguous(), self.zero_q, weights.q_norm, self.lut_rsqrt)
        _k, k_heads = self.rms_k(k.reshape(self.seq_len * cfg.num_key_value_heads, cfg.head_dim).contiguous(), self.zero_k, weights.k_norm, self.lut_rsqrt)
        pos_cos = cos[self.cache_len : self.cache_len + self.seq_len]
        pos_sin = sin[self.cache_len : self.cache_len + self.seq_len]
        q_attn = self.rope_q(q_heads, pos_cos, pos_sin, r3_q15, weights.q_post_rope_i8_scale)
        k_attn = self.rope_k(k_heads, pos_cos, pos_sin, r3_q15, weights.k_post_rope_i8_scale)
        v_attn = self.quant_v(v.reshape(self.seq_len, cfg.num_key_value_heads, cfg.head_dim), weights.v_i8_scale)
        attn8 = self.attn(
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
            weights.attn_i8_scale,
        )
        attn_out = self.o_proj(attn8, weights.attn_i8_scale, weights.o_proj.weight, weights.o_proj.scale)
        residual, h8, _ = self.rms_sq8(x, attn_out, weights.post_attention_layernorm, self.lut_rsqrt, weights.post_mlp_i8_scale)
        gate_up = self.gate_up_proj(h8, weights.post_mlp_i8_scale, weights.gate_up_proj.weight, weights.gate_up_proj.scale)
        gate = gate_up[:, : cfg.intermediate_size].contiguous()
        up = gate_up[:, cfg.intermediate_size :].contiguous()
        gated, _ = self.silu_mul(gate, up, self.lut_sigmoid, weights.gated_mlp_i16_scale)
        mlp = self.down_proj(gated, weights.gated_mlp_i16_scale, weights.down_proj.weight, weights.down_proj.scale)
        return residual, mlp


class Qwen3IntOnlyModel:
    def __init__(self, seq_len, model_dir="/publicdata/huggingface.co/Qwen/Qwen3-0.6B", packed_dir="/tmp/Qwen3-0.6B-static-calib-32x2048", config=QWEN3_0_6B, cache_len=0, rotate_seed=ROTATE_SEED):
        self.seq_len = seq_len
        self.cache_len = cache_len
        self.config = config
        self.r3_q15 = q15_16(random_hadamard_rotation(config.head_dim, rotate_seed + 2))
        self.block = Qwen3IntOnlyBlock(seq_len, config, cache_len=cache_len)
        self.final_norm = compile_(rms_q15(seq_len, config.hidden_size), [4, 5])
        self.zero_hidden = torch.zeros((seq_len, config.hidden_size), device="cuda", dtype=torch.int32)
        self.lut_rsqrt = torch.from_numpy(rsqrt_lut()).cuda()
        self.cos, self.sin, _ = rope_tables_q15_16(seq_len + cache_len, config.head_dim, config.rope_theta)
        self.embed, self.lm_head, self.norm_weight, self.layers = load_packed_qwen3(packed_dir, config)

    def hidden(self, input_ids, layers=None, cache_kv=None):
        residual = q15_16(self.embed[input_ids])
        x8 = self.block.input_rms_quant(residual, self.layers[0])
        mlp = None
        n_layers = self.config.num_hidden_layers if layers is None else layers
        for layer_idx in range(n_layers):
            if layer_idx:
                residual, x8, _ = self.block.rms_sq8(residual, mlp, self.layers[layer_idx].input_layernorm, self.lut_rsqrt, self.layers[layer_idx].input_qkv_i8_scale)
            layer_cache = None if cache_kv is None else cache_kv[layer_idx]
            if layer_cache is None:
                residual, mlp = self.block(residual, self.layers[layer_idx], self.cos, self.sin, r3_q15=self.r3_q15, x8=x8)
            else:
                residual, mlp = self.block(residual, self.layers[layer_idx], self.cos, self.sin, layer_cache[0], layer_cache[1], self.r3_q15, x8=x8)
        if n_layers == 0:
            _hidden, norm = self.final_norm(residual, self.zero_hidden, self.norm_weight, self.lut_rsqrt)
            return norm
        _hidden, norm = self.final_norm(residual, mlp, self.norm_weight, self.lut_rsqrt)
        return norm

    def logits(self, input_ids, layers=None, cache_kv=None):
        h = self.hidden(input_ids, layers=layers, cache_kv=cache_kv).float() / Q15_16
        return h @ self.lm_head.T
