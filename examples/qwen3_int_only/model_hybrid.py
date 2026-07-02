from __future__ import annotations

import torch
import tilelang

from examples.qwen3_int_only.kernels_int_only import linear_i8, quant_v_i8
from examples.qwen3_int_only.kernels_hybrid import (
    attention_hybrid,
    qk_norm_rope_quant_hybrid,
    rms_hybrid,
    rms_quant_hybrid,
    silu_hadamard_quant_hybrid,
)
from examples.qwen3_int_only.utils import (
    ROTATE_SEED,
    Q15_16,
    QWEN3_0_6B,
    Qwen3Config,
    load_packed_qwen3,
    q15_16,
    random_hadamard_rotation,
    rope_tables,
)


class Qwen3HybridBlock:
    def __init__(self, seq_len, config=QWEN3_0_6B, cache_len=0, rotate_seed=ROTATE_SEED):
        self.seq_len = seq_len
        self.cache_len = cache_len
        self.config = config
        h, hd, im = config.hidden_size, config.head_dim, config.intermediate_size
        qh, kvh = config.num_attention_heads, config.num_key_value_heads
        q_dim, kv_dim = config.q_size, config.kv_size
        self.qkv_proj = tilelang.compile(linear_i8(seq_len, h, q_dim + 2 * kv_dim, 64, 128, 64), out_idx=[3], target="cuda")
        self.o_proj = tilelang.compile(linear_i8(seq_len, q_dim, h, 64, 64, 64), out_idx=[3], target="cuda")
        self.gate_up_proj = tilelang.compile(linear_i8(seq_len, h, 2 * im, 64, 128, 64), out_idx=[3], target="cuda")
        self.down_proj = tilelang.compile(linear_i8(seq_len, im, h, 64, 64, 64), out_idx=[3], target="cuda")
        self.quant_v = tilelang.compile(quant_v_i8(seq_len, kvh, hd), out_idx=[2], target="cuda")
        self.rms_quant_kernel = tilelang.compile(rms_quant_hybrid(seq_len, h), out_idx=[4, 5], target="cuda")
        self.rope_q = tilelang.compile(qk_norm_rope_quant_hybrid(seq_len, qh, hd), out_idx=[6], target="cuda")
        self.rope_k = tilelang.compile(qk_norm_rope_quant_hybrid(seq_len, kvh, hd), out_idx=[6], target="cuda")
        self.silu_kernel = tilelang.compile(silu_hadamard_quant_hybrid(seq_len, im), out_idx=[3], target="cuda")
        self.attn = tilelang.compile(attention_hybrid(qh, kvh, seq_len, cache_len, hd), out_idx=[8], target="cuda")
        self.empty_cache_k = torch.empty((kvh, cache_len, hd), device="cuda", dtype=torch.int8)
        self.empty_cache_v = torch.empty((kvh, cache_len, hd), device="cuda", dtype=torch.int8)
        self.r3 = q15_16(random_hadamard_rotation(config.head_dim, rotate_seed + 2, "cuda"))
        self.zero_hidden = torch.zeros((seq_len, h), device="cuda", dtype=torch.int32)

    def rms_quant(self, x_q15, residual_q15, weight_q15, out_scale):
        residual_q15 = self.zero_hidden if residual_q15 is None else residual_q15
        return self.rms_quant_kernel(x_q15, residual_q15, weight_q15, out_scale)

    def qk_norm_rope_quant(self, x_q15, weight_q15, cos, sin, heads, out_scale):
        kernel = self.rope_q if heads == self.config.num_attention_heads else self.rope_k
        return kernel(x_q15.reshape(self.seq_len * heads, self.config.head_dim).contiguous(), weight_q15, cos, sin, self.r3, out_scale)

    def attention_hybrid(self, q8, k8, v8, cache_k, cache_v, weights):
        return self.attn(q8, cache_k, cache_v, k8, v8, weights.attn_score_scale, weights.v_i8_scale, weights.attn_i8_scale)

    def silu_quant(self, gate_q15, up_q15, out_scale):
        return self.silu_kernel(gate_q15, up_q15, out_scale)

    def __call__(self, x, weights, cos, sin, cache_k=None, cache_v=None, x8=None):
        cfg = self.config
        cache_k = self.empty_cache_k if cache_k is None else cache_k
        cache_v = self.empty_cache_v if cache_v is None else cache_v
        qkv = self.qkv_proj(x8, weights.qkv_proj.weight, weights.qkv_out_qt)
        q = qkv[:, : cfg.q_size].contiguous()
        k = qkv[:, cfg.q_size : cfg.q_size + cfg.kv_size].contiguous()
        v = qkv[:, cfg.q_size + cfg.kv_size :].contiguous()
        pos_cos = cos[self.cache_len : self.cache_len + self.seq_len]
        pos_sin = sin[self.cache_len : self.cache_len + self.seq_len]
        q_attn = self.qk_norm_rope_quant(q, weights.q_norm, pos_cos, pos_sin, cfg.num_attention_heads, weights.q_post_rope_i8_scale)
        k_attn = self.qk_norm_rope_quant(k, weights.k_norm, pos_cos, pos_sin, cfg.num_key_value_heads, weights.k_post_rope_i8_scale)
        v_attn = self.quant_v(v.reshape(self.seq_len, cfg.num_key_value_heads, cfg.head_dim), weights.v_i8_qt)
        attn8 = self.attention_hybrid(q_attn, k_attn, v_attn, cache_k, cache_v, weights)
        attn_out = self.o_proj(attn8, weights.o_proj.weight, weights.o_out_qt)
        residual, h8 = self.rms_quant(x, attn_out, weights.post_attention_layernorm, weights.post_mlp_i8_scale)
        gate_up = self.gate_up_proj(h8, weights.gate_up_proj.weight, weights.gate_up_out_qt)
        gate = gate_up[:, : cfg.intermediate_size].contiguous()
        up = gate_up[:, cfg.intermediate_size :].contiguous()
        gated = self.silu_quant(gate, up, weights.gated_mlp_i8_scale)
        mlp = self.down_proj(gated, weights.down_proj.weight, weights.down_out_qt)
        return residual, mlp


class Qwen3HybridModel:
    def __init__(self, seq_len, model_dir="/publicdata/huggingface.co/Qwen/Qwen3-0.6B", packed_dir="/tmp/Qwen3-0.6B-static-calib-32x2048", config=None, cache_len=0, rotate_seed=ROTATE_SEED, layers=None):
        config = Qwen3Config.from_model_dir(model_dir) if config is None else config
        self.seq_len = seq_len
        self.cache_len = cache_len
        self.config = config
        self.block = Qwen3HybridBlock(seq_len, config, cache_len=cache_len, rotate_seed=rotate_seed)
        self.final_norm = tilelang.compile(rms_hybrid(seq_len, config.hidden_size), out_idx=[3, 4], target="cuda")
        self.zero_hidden = torch.zeros((seq_len, config.hidden_size), device="cuda", dtype=torch.int32)
        self.cos, self.sin, _ = rope_tables(seq_len + cache_len, config.head_dim, config.rope_theta)
        self.embed, self.lm_head, self.norm_weight, self.layers = load_packed_qwen3(packed_dir, config, layers=layers)

    def hidden(self, input_ids, layers=None, cache_kv=None):
        mlp = None
        n_layers = self.config.num_hidden_layers if layers is None else layers
        residual = q15_16(self.embed[input_ids])
        x8 = None
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                _, x8 = self.block.rms_quant(residual, None, self.layers[layer_idx].input_layernorm, self.layers[layer_idx].input_qkv_i8_scale)
            else:
                residual, x8 = self.block.rms_quant(residual, mlp, self.layers[layer_idx].input_layernorm, self.layers[layer_idx].input_qkv_i8_scale)
            layer_cache = None if cache_kv is None else cache_kv[layer_idx]
            if layer_cache is None:
                residual, mlp = self.block(residual, self.layers[layer_idx], self.cos, self.sin, x8=x8)
            else:
                residual, mlp = self.block(residual, self.layers[layer_idx], self.cos, self.sin, layer_cache[0], layer_cache[1], x8)
        _hidden, norm = self.final_norm(residual, self.zero_hidden if mlp is None else mlp, self.norm_weight)
        return norm

    def logits(self, input_ids, layers=None, cache_kv=None):
        h = self.hidden(input_ids, layers=layers, cache_kv=cache_kv).float() / Q15_16
        return h @ self.lm_head.T
