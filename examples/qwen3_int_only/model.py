from __future__ import annotations

import time

import torch

from examples.qwen3_int_only.kernels import (
    Q15_16,
    add_rmsnorm_q15_16_weighted,
    add_q15_16,
    attention_i8v8_q15_16_gqa_cache_fused_static_current,
    attention_i8v8_q15_16_gqa_fused_static,
    compile_kernel,
    dynamic_quant_q15_16,
    exp_lut_neg,
    linear_dynamic_int8_qkv_q15_16,
    linear_static_int8_q15_16,
    linear_static_int8_pair_q15_16,
    linear_static_int16_q15_16,
    rmsnorm_dynamic_quant_q15_16_weighted_fast,
    rmsnorm_q15_16_grouped_weighted_rowwise,
    rmsnorm_q15_16_weighted,
    rope_q15_16_heads,
    rope_rotate_q15_16_heads,
    rope_rotate_static_quant_q15_16_attn,
    rope_rotate_static_quant_q15_16_attn_hadamard_approx,
    rope_rotate_static_quant_q15_16_attn_noscale,
    rsqrt_lut,
    sigmoid_lut,
    silu_mul_static_quant_q15_16_i16_fast,
    static_quant_q15_16,
    static_quant_q15_16_per_head_attn,
    static_quant_q15_16_per_head_attn_noscale,
)
from examples.qwen3_int_only.utils import (
    ROTATE_SEED,
    QWEN3_0_6B,
    Qwen3BlockWeights,
    block_torch,
    load_all_qwen3_block_weights,
    load_embed_tokens,
    load_final_norm,
    load_lm_head,
    load_packed_qwen3,
    q15_16,
    random_hadamard_rotation,
    rmsnorm_torch,
    rope_tables_q15_16,
)

class Qwen3IntOnlyBlock:
    def __init__(self, seq_len, config=QWEN3_0_6B, cache_len=0, use_r3=True, fast_hadamard=True):
        self.seq_len = seq_len
        self.cache_len = cache_len
        self.config = config
        self.use_r3 = use_r3
        self.fast_hadamard = fast_hadamard
        h, hd, im = config.hidden_size, config.head_dim, config.intermediate_size
        qh, kvh = config.num_attention_heads, config.num_key_value_heads
        q_dim, kv_dim = config.q_size, config.kv_size
        self.rms_hidden_q15 = compile_kernel(rmsnorm_q15_16_weighted(seq_len, h), [3])
        self.rms_dq8_hidden_fast = compile_kernel(rmsnorm_dynamic_quant_q15_16_weighted_fast(seq_len, h), [3, 4])
        self.add_rms_hidden_q15 = compile_kernel(add_rmsnorm_q15_16_weighted(seq_len, h), [4, 5])
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
        self.o_proj = compile_kernel(linear_static_int8_q15_16(seq_len, q_dim, h, 64, 64, 64), [4])
        self.sq8_hidden = compile_kernel(static_quant_q15_16(seq_len, h, "int8"), [2, 3])
        self.gate_up_proj_static = compile_kernel(linear_static_int8_pair_q15_16(seq_len, h, im, 64, 128, 64), [6, 7])
        self.silu_mul_sq16_mid_fast = compile_kernel(silu_mul_static_quant_q15_16_i16_fast(seq_len, im), [4, 5])
        self.down_proj_static = compile_kernel(linear_static_int16_q15_16(seq_len, im, h, 64, 64, 64), [4])
        self.rope_q = compile_kernel(rope_rotate_q15_16_heads(seq_len, qh, hd), [4]) if use_r3 else compile_kernel(rope_q15_16_heads(seq_len, qh, hd), [3])
        self.rope_k = compile_kernel(rope_rotate_q15_16_heads(seq_len, kvh, hd), [4]) if use_r3 else compile_kernel(rope_q15_16_heads(seq_len, kvh, hd), [3])
        self.add_hidden = compile_kernel(add_q15_16(seq_len, h), [2])
        self.attn_i8v8_fused_static = compile_kernel(attention_i8v8_q15_16_gqa_fused_static(qh, kvh, seq_len, hd), [8])
        self.attn_i8v8_fused_cache_static = None
        if cache_len:
            self.attn_i8v8_fused_cache_static = compile_kernel(attention_i8v8_q15_16_gqa_cache_fused_static_current(qh, kvh, seq_len, cache_len, hd), [12])
        self.lut_rsqrt = torch.from_numpy(rsqrt_lut()).cuda()
        self.lut_sigmoid = torch.from_numpy(sigmoid_lut()).cuda()
        self.lut_exp = torch.from_numpy(exp_lut_neg()).cuda()

    def __call__(self, x_q15_16, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16, cache_k=None, cache_v=None, r3_q15=None):
        return self.trace(x_q15_16, weights, cos_q15_16, sin_q15_16, cache_k, cache_v, r3_q15, collect=False)

    def trace(self, x_q15_16, weights: Qwen3BlockWeights, cos_q15_16, sin_q15_16, cache_k=None, cache_v=None, r3_q15=None, collect=True):
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
            attn8 = self.attn_i8v8_fused_static(q_attn, k_attn, v_attn, weights.q_post_rope_i8_scale, weights.k_post_rope_i8_scale, weights.v_i8_scale, self.lut_exp, weights.attn_i8_scale)
        else:
            prob_i16 = None
            attn8 = self.attn_i8v8_fused_cache_static(
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
        attn_s8 = weights.attn_i8_scale
        attn = attn8.float() * attn_s8.float()[0] if collect else None
        attn_out = self.o_proj(attn8, attn_s8, weights.o_proj.weight, weights.o_proj.scale)
        h, post = self.add_rms_hidden_q15(x_q15_16, attn_out, weights.post_attention_layernorm, self.lut_rsqrt)
        h8, _hs8 = self.sq8_hidden(post, weights.post_mlp_i8_scale)
        hs8 = weights.post_mlp_i8_scale
        gate, up = self.gate_up_proj_static(h8, hs8, weights.gate_proj.weight, weights.gate_proj.scale, weights.up_proj.weight, weights.up_proj.scale)
        gated8, _gs8 = self.silu_mul_sq16_mid_fast(gate, up, self.lut_sigmoid, weights.gated_mlp_i16_scale)
        gs8 = weights.gated_mlp_i16_scale
        if collect:
            mlp_q15 = self.down_proj_static(gated8, gs8, weights.down_proj.weight, weights.down_proj.scale)
            gated = gated8.float() * gs8.float()[0] / Q15_16
            mlp = mlp_q15.float() / Q15_16
            layer_out = self.add_hidden(h, mlp_q15)
        else:
            gated = None
            mlp = None
            mlp_q15 = self.down_proj_static(gated8, gs8, weights.down_proj.weight, weights.down_proj.scale)
            layer_out = self.add_hidden(h, mlp_q15)
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
            "attn": attn / Q15_16,
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
            "gated": gated,
            "gated8": gated8,
            "gated_s8": gs8,
            "mlp": mlp,
            "layer_out": layer_out.float() / Q15_16,
            "layer_out_q15": layer_out,
        }


class Qwen3IntOnlyModel:
    def __init__(self, seq_len, model_dir="/publicdata/huggingface.co/Qwen/Qwen3-0.6B", packed_dir="/tmp/Qwen3-0.6B-static-calib-32x2048", config=QWEN3_0_6B, cache_len=0, use_r3=True, fast_hadamard=True, rotate_seed=ROTATE_SEED):
        self.seq_len = seq_len
        self.cache_len = cache_len
        self.config = config
        self.r3_q15 = q15_16(random_hadamard_rotation(config.head_dim, rotate_seed + 2)) if use_r3 else None
        self.block = Qwen3IntOnlyBlock(seq_len, config, cache_len=cache_len, use_r3=use_r3, fast_hadamard=fast_hadamard)
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
    def __init__(self, seq_len, model_dir="/publicdata/huggingface.co/Qwen/Qwen3-0.6B", config=QWEN3_0_6B):
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
