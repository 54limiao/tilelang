import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from examples.qwen3_int_only.model import (
    Q15_16,
    QWEN3_0_6B,
    Qwen3IntOnlyBlock,
    load_all_qwen3_block_weights,
    load_embed_tokens,
    load_packed_qwen3,
    parse_layer_set,
    q15_16,
    rope_tables_q15_16,
)
from examples.qwen3_int_only.kernels import (
    attention_i16v8_q15_16_gqa,
    attention_i16v8_q15_16_gqa_cache,
    attention_i8_q15_16_gqa_cache_softmax_i16,
    attention_i8_q15_16_gqa_softmax_i16,
    attention_i8v8_q15_16_gqa_cache_fused,
    attention_i8v8_q15_16_gqa_cache_fused_static_current,
    attention_i8v8_q15_16_gqa_fused,
    attention_i8v8_q15_16_gqa_fused_static,
    compile_kernel,
    exp_lut_neg,
)
from examples.qwen3_int_only.quarot import ROTATE_SEED, random_hadamard_rotation


TEXT_PATH = Path(__file__).resolve().parent / "data" / "declaration_of_independence.txt"


def packed_flags(packed_dir):
    if packed_dir is None:
        return False, False
    with safe_open(f"{packed_dir}/qwen3_int_only.safetensors", framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    return metadata.get("use_r1") == "1", metadata.get("use_r2") == "1"


def op_counts(seq_len, cfg, cache_len=0):
    h = cfg.hidden_size
    q_dim = cfg.q_size
    kv_dim = cfg.kv_size
    im = cfg.intermediate_size
    qh = cfg.num_attention_heads
    kvh = cfg.num_key_value_heads
    hd = cfg.head_dim
    kv_total = seq_len + cache_len
    qk_ops = 2 * qh * seq_len * kv_total * hd
    pv_gemm_ops = 2 * qh * seq_len * kv_total * hd
    pv_split_ops = 2 * pv_gemm_ops
    rope_q_ops = 2 * seq_len * qh * hd * hd
    rope_k_ops = 2 * seq_len * kvh * hd * hd
    return {
        "qkv_proj_i8": 2 * seq_len * h * (q_dim + 2 * kv_dim),
        "o_proj_i8": 2 * seq_len * q_dim * h,
        "gate_up_proj_i8": 2 * seq_len * h * (2 * im),
        "down_proj_i8": 2 * seq_len * im * h,
        "down_residual_i8": 2 * seq_len * im * h,
        "attention_softmax_i16": qk_ops,
        "attention_i16v8": pv_split_ops,
        "attention_i8v8_fused": qk_ops * 2 + pv_split_ops,
        "attention_i8v8_fused_static": qk_ops * 2 + pv_split_ops,
        "attention_cache_softmax_i16": qk_ops,
        "attention_cache_i16v8": pv_split_ops,
        "attention_cache_i8v8_fused": qk_ops * 2 + pv_split_ops,
        "attention_cache_i8v8_fused_static": qk_ops * 2 + pv_split_ops,
        "rope_q": rope_q_ops,
        "rope_k": rope_k_ops,
        "rope_sq8_q_attn": rope_q_ops,
        "rope_sq8_k_attn": rope_k_ops,
        "rope_sq8_q_attn_noscale": rope_q_ops,
        "rope_sq8_k_attn_noscale": rope_k_ops,
        "rope_sq8_q_attn_hadamard": rope_q_ops,
        "rope_sq8_k_attn_hadamard": rope_k_ops,
    }


class Profiler:
    def __init__(self, ops=None):
        self.rows = []
        self.ops = ops or {}

    def time(self, name, fn):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        self.rows.append((name, start.elapsed_time(end)))
        return out

    def report(self):
        return self.summary(print_rows=True)

    def summary(self, print_rows=False):
        totals = {}
        for name, ms in self.rows:
            totals[name] = totals.get(name, 0.0) + ms
        total = sum(totals.values())
        counts = {}
        for name, _ in self.rows:
            counts[name] = counts.get(name, 0) + 1
        items = []
        for name, ms in sorted(totals.items(), key=lambda x: x[1], reverse=True):
            row = {"name": name, "avg_ms": ms / counts[name], "total_ms": ms, "count": counts[name], "pct": ms / total * 100.0}
            if name in self.ops:
                row["gops"] = self.ops[name] * counts[name] / 1.0e9
                row["tops"] = row["gops"] / ms
            items.append(row)
            if print_rows:
                perf = f" {row['tops']:7.2f} TOPS" if "tops" in row else ""
                print(f"{name:22s} avg={row['avg_ms']:8.3f} ms total={ms:9.3f} ms {row['pct']:6.2f}%{perf}")
        if print_rows:
            print(f"{'total':22s} {total:9.3f} ms")
        return {"total_ms": total, "kernels": items}


def quant_i8_q15_16(x):
    xq = torch.clamp(torch.round(x * Q15_16), -(1 << 31), (1 << 31) - 1).to(torch.int32)
    scale = torch.div(xq.abs().amax(dim=-1) + 126, 127, rounding_mode="floor").clamp(min=1).to(torch.uint32)
    y = torch.div(xq.abs() + (scale.int()[..., None] >> 1), scale.int()[..., None], rounding_mode="floor")
    y = torch.where(xq < 0, -y, y).clamp(-128, 127).to(torch.int8)
    return y, scale


@torch.no_grad()
def build_cache_kv(model_dir, tokenizer, cache_prompt, layers, use_r2, use_r3):
    hf_model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    cache_ids = torch.tensor(tokenizer(cache_prompt, add_special_tokens=False).input_ids, device="cuda", dtype=torch.long)
    past = hf_model(cache_ids[None, :], use_cache=True).past_key_values
    if hasattr(past, "layers"):
        past = [(layer.keys, layer.values) for layer in past.layers]
    elif hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    r2 = random_hadamard_rotation(QWEN3_0_6B.head_dim, ROTATE_SEED + 1, "cuda") if use_r2 else None
    r3 = random_hadamard_rotation(QWEN3_0_6B.head_dim, ROTATE_SEED + 2, "cuda") if use_r3 else None
    cache_kv = []
    for k, v in past[:layers]:
        k = k[0].float().contiguous()
        v = v[0].float().contiguous()
        if r3 is not None:
            k = (k.to(torch.float64) @ r3.to(torch.float64)).to(torch.float32)
        if r2 is not None:
            v = (v.to(torch.float64) @ r2.to(torch.float64)).to(torch.float32)
        cache_kv.append((quant_i8_q15_16(k), quant_i8_q15_16(v)))
    return cache_kv, int(cache_ids.numel())


@torch.no_grad()
def run_block(block, x_q15_16, weights, cos_q15_16, sin_q15_16, r3_q15, prof, split_attn=False, fused_attn=False, cache_k=None, cache_v=None, mlp_i16=None):
    cfg = block.config
    seq_len = block.seq_len
    pos_cos = cos_q15_16[block.cache_len : block.cache_len + seq_len]
    pos_sin = sin_q15_16[block.cache_len : block.cache_len + seq_len]
    x8, xs8 = prof.time("rms_input_dq8_fast", lambda: block.rms_dq8_hidden_fast(x_q15_16, weights.input_layernorm, block.lut_rsqrt))
    q, k, v = prof.time(
        "qkv_proj_i8",
        lambda: block.qkv_proj_i8(
            x8, xs8, weights.q_proj.weight, weights.q_proj.scale, weights.k_proj.weight, weights.k_proj.scale, weights.v_proj.weight, weights.v_proj.scale
        ),
    )
    v_heads = v.reshape(seq_len * cfg.num_key_value_heads, cfg.head_dim)
    q_heads = prof.time("rms_q_q15", lambda: block.rms_q_q15(q, weights.q_norm, block.lut_rsqrt))
    k_heads = prof.time("rms_k_q15", lambda: block.rms_k_q15(k, weights.k_norm, block.lut_rsqrt))
    if weights.q_post_rope_i8_scale is not None and weights.k_post_rope_i8_scale is not None and weights.v_i8_scale is not None:
        static_attn_scale = True
        static_fused_fast = fused_attn
        if block.use_r3 and static_fused_fast:
            if block.fast_hadamard:
                q_attn = prof.time("rope_sq8_q_attn_hadamard", lambda: block.rope_sq8_q_attn_hadamard(q_heads, pos_cos, pos_sin, r3_q15, weights.q_post_rope_i8_scale))
                k_attn = prof.time("rope_sq8_k_attn_hadamard", lambda: block.rope_sq8_k_attn_hadamard(k_heads, pos_cos, pos_sin, r3_q15, weights.k_post_rope_i8_scale))
            else:
                q_attn = prof.time("rope_sq8_q_attn_noscale", lambda: block.rope_sq8_q_attn_noscale(q_heads, pos_cos, pos_sin, r3_q15, weights.q_post_rope_i8_scale))
                k_attn = prof.time("rope_sq8_k_attn_noscale", lambda: block.rope_sq8_k_attn_noscale(k_heads, pos_cos, pos_sin, r3_q15, weights.k_post_rope_i8_scale))
            v_attn = prof.time("sq8_v_attn_noscale", lambda: block.sq8_kv_attn_noscale(v_heads.reshape(seq_len, cfg.num_key_value_heads, cfg.head_dim), weights.v_i8_scale))
            qs_attn = None
            ks_attn = None
            vs_attn = None
        elif block.use_r3:
            q_attn, qs_attn = prof.time("rope_sq8_q_attn", lambda: block.rope_sq8_q_attn(q_heads, pos_cos, pos_sin, r3_q15, weights.q_post_rope_i8_scale))
            k_attn, ks_attn = prof.time("rope_sq8_k_attn", lambda: block.rope_sq8_k_attn(k_heads, pos_cos, pos_sin, r3_q15, weights.k_post_rope_i8_scale))
        else:
            qr = prof.time("rope_q", lambda: block.rope_q(q_heads, pos_cos, pos_sin))
            kr = prof.time("rope_k", lambda: block.rope_k(k_heads, pos_cos, pos_sin))
            q_attn, qs_attn = prof.time("sq8_q_attn", lambda: block.sq8_q_attn(qr.reshape(seq_len, cfg.num_attention_heads, cfg.head_dim), weights.q_post_rope_i8_scale))
            k_attn, ks_attn = prof.time("sq8_kv_attn", lambda: block.sq8_kv_attn(kr.reshape(seq_len, cfg.num_key_value_heads, cfg.head_dim), weights.k_post_rope_i8_scale))
        if not static_fused_fast:
            v_attn, vs_attn = prof.time("sq8_v_attn", lambda: block.sq8_kv_attn(v_heads.reshape(seq_len, cfg.num_key_value_heads, cfg.head_dim), weights.v_i8_scale))
    else:
        static_attn_scale = False
        if block.use_r3:
            qr = prof.time("rope_q", lambda: block.rope_q(q_heads, pos_cos, pos_sin, r3_q15))
            kr = prof.time("rope_k", lambda: block.rope_k(k_heads, pos_cos, pos_sin, r3_q15))
        else:
            qr = prof.time("rope_q", lambda: block.rope_q(q_heads, pos_cos, pos_sin))
            kr = prof.time("rope_k", lambda: block.rope_k(k_heads, pos_cos, pos_sin))
        q8, qs8 = prof.time("dq8_q_head", lambda: block.dq8_q_head(qr))
        k8, ks8 = prof.time("dq8_kv_head", lambda: block.dq8_kv_head(kr))
        v8, vs8 = prof.time("dq8_v_head", lambda: block.dq8_kv_head(v_heads))

        def attn_layout():
            return (
                q8.reshape(seq_len, cfg.num_attention_heads, cfg.head_dim).permute(1, 0, 2).contiguous(),
                k8.reshape(seq_len, cfg.num_key_value_heads, cfg.head_dim).permute(1, 0, 2).contiguous(),
                v8.reshape(seq_len, cfg.num_key_value_heads, cfg.head_dim).permute(1, 0, 2).contiguous(),
                qs8.reshape(seq_len, cfg.num_attention_heads).permute(1, 0).contiguous(),
                ks8.reshape(seq_len, cfg.num_key_value_heads).permute(1, 0).contiguous(),
                vs8.reshape(seq_len, cfg.num_key_value_heads).permute(1, 0).contiguous(),
            )

        q_attn, k_attn, v_attn, qs_attn, ks_attn, vs_attn = prof.time("attn_layout", attn_layout)
    if cache_k is not None and fused_attn:
        if static_attn_scale:
            attn = prof.time(
                "attention_cache_i8v8_fused_static",
                lambda: block.attn_i8v8_fused_cache_static(
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
                    block.lut_exp,
                ),
            )
        else:
            attn = prof.time(
                "attention_cache_i8v8_fused",
                lambda: block.attn_i8v8_fused_cache(q_attn, cache_k[0], cache_v[0], k_attn, v_attn, qs_attn, cache_k[1], cache_v[1], ks_attn, vs_attn, block.lut_exp),
            )
    elif cache_k is not None and split_attn:
        prob_i16 = prof.time("attention_cache_softmax_i16", lambda: block.attn_cache_softmax_i16(q_attn, cache_k[0], k_attn, qs_attn, cache_k[1], ks_attn, block.lut_exp))
        attn = prof.time("attention_cache_i16v8", lambda: block.attn_cache_i16v8(prob_i16, cache_v[0], v_attn, cache_v[1], vs_attn))
    elif fused_attn:
        if static_attn_scale:
            if getattr(block, "attn_i8v8_fused_static", None) is None:
                block.attn_i8v8_fused_static = compile_kernel(attention_i8v8_q15_16_gqa_fused_static(cfg.num_attention_heads, cfg.num_key_value_heads, seq_len, cfg.head_dim), [7])
            attn = prof.time(
                "attention_i8v8_fused_static",
                lambda: block.attn_i8v8_fused_static(q_attn, k_attn, v_attn, weights.q_post_rope_i8_scale, weights.k_post_rope_i8_scale, weights.v_i8_scale, block.lut_exp),
            )
        else:
            if getattr(block, "attn_i8v8_fused", None) is None:
                block.attn_i8v8_fused = compile_kernel(attention_i8v8_q15_16_gqa_fused(cfg.num_attention_heads, cfg.num_key_value_heads, seq_len, cfg.head_dim), [7])
            attn = prof.time("attention_i8v8_fused", lambda: block.attn_i8v8_fused(q_attn, k_attn, v_attn, qs_attn, ks_attn, vs_attn, block.lut_exp))
    elif split_attn:
        if getattr(block, "attn_softmax_i16", None) is None:
            block.attn_softmax_i16 = compile_kernel(attention_i8_q15_16_gqa_softmax_i16(cfg.num_attention_heads, cfg.num_key_value_heads, seq_len, cfg.head_dim), [5])
            block.attn_i16v8 = compile_kernel(attention_i16v8_q15_16_gqa(cfg.num_attention_heads, cfg.num_key_value_heads, seq_len, cfg.head_dim), [3])
        prob_i16 = prof.time("attention_softmax_i16", lambda: block.attn_softmax_i16(q_attn, k_attn, qs_attn, ks_attn, block.lut_exp))
        attn = prof.time("attention_i16v8", lambda: block.attn_i16v8(prob_i16, v_attn, vs_attn))
    else:
        attn_num, attn_den = prof.time("attention_i8_fixed", lambda: block.attn_i8_fixed(q_attn, k_attn, v_attn, qs_attn, ks_attn, vs_attn, block.lut_exp))
        attn = prof.time("attention_norm", lambda: block.attn_norm(attn_num, attn_den))
        attn = attn.permute(1, 0, 2).reshape(seq_len, cfg.q_size)
    attn8, attn_s8 = prof.time("dq8_attn", lambda: block.dq8_q(attn))
    attn_out = prof.time("o_proj_i8", lambda: block.o_proj(attn8, attn_s8, weights.o_proj.weight, weights.o_proj.scale))
    h, h8, hs8 = prof.time("residual_attn_rms_dq8_fast", lambda: block.add_rms_dq8_hidden_fast(x_q15_16, attn_out, weights.post_attention_layernorm, block.lut_rsqrt))
    gate, up = prof.time(
        "gate_up_proj_i8",
        lambda: block.gate_up_proj_i8(h8, hs8, weights.gate_proj.weight, weights.gate_proj.scale, weights.up_proj.weight, weights.up_proj.scale),
    )
    use_mlp_i16 = block.mlp_i16 if mlp_i16 is None else mlp_i16
    if use_mlp_i16:
        gated, gs = prof.time("silu_mul_dq16_mid_fast", lambda: block.silu_mul_dq16_mid_fast(gate, up, block.lut_sigmoid))
        return prof.time("down_residual_i16", lambda: block.down_residual_i16(gated, gs, weights.down_proj.weight, weights.down_proj.scale, h))
    gated, gs = prof.time("silu_mul_dq8_mid_fast", lambda: block.silu_mul_dq8_mid_fast(gate, up, block.lut_sigmoid))
    return prof.time("down_residual_i8", lambda: block.down_residual_i8(gated, gs, weights.down_proj.weight, weights.down_proj.scale, h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/code/Qwen3-0.6B")
    parser.add_argument("--packed-dir")
    parser.add_argument("--max-tokens", type=int, default=129)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--use-r2", action="store_true")
    parser.add_argument("--use-r3", action="store_true")
    parser.add_argument("--split-attn", action="store_true")
    parser.add_argument("--fused-attn", action="store_true")
    parser.add_argument("--fast-hadamard", action="store_true")
    parser.add_argument("--mlp-i16", action="store_true")
    parser.add_argument("--mlp-i16-layers", default="")
    parser.add_argument("--cache-len", type=int, default=0)
    parser.add_argument("--cache-block", action="store_true")
    parser.add_argument("--static-cache", action="store_true")
    parser.add_argument("--cache-prompt", default="你是一个有用而无害的聊天助手。")
    parser.add_argument("--jsonl-out", default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    ids = tokenizer(TEXT_PATH.read_text(encoding="utf-8"), add_special_tokens=False).input_ids[: args.max_tokens]
    seq_len = len(ids) - 1
    seq_len -= seq_len % 32
    ids = torch.tensor(ids[:seq_len], device="cuda", dtype=torch.long)
    cfg = QWEN3_0_6B
    if args.cache_len and not args.cache_block:
        torch.manual_seed(0)
        qh, kvh, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = torch.randint(-127, 128, (qh, seq_len, hd), device="cuda", dtype=torch.int8)
        k = torch.randint(-127, 128, (kvh, seq_len, hd), device="cuda", dtype=torch.int8)
        v = torch.randint(-127, 128, (kvh, seq_len, hd), device="cuda", dtype=torch.int8)
        ck = torch.randint(-127, 128, (kvh, args.cache_len, hd), device="cuda", dtype=torch.int8)
        cv = torch.randint(-127, 128, (kvh, args.cache_len, hd), device="cuda", dtype=torch.int8)
        qs = torch.randint(320, 1600, (qh, seq_len), device="cuda", dtype=torch.uint32)
        ks = torch.randint(320, 1600, (kvh, seq_len), device="cuda", dtype=torch.uint32)
        vs = torch.randint(320, 2000, (kvh, seq_len), device="cuda", dtype=torch.uint32)
        cks = torch.randint(320, 1600, (kvh, args.cache_len), device="cuda", dtype=torch.uint32)
        cvs = torch.randint(320, 2000, (kvh, args.cache_len), device="cuda", dtype=torch.uint32)
        qs_head = torch.randint(320, 1600, (qh,), device="cuda", dtype=torch.uint32)
        ks_head = torch.randint(320, 1600, (kvh,), device="cuda", dtype=torch.uint32)
        vs_head = torch.randint(320, 2000, (kvh,), device="cuda", dtype=torch.uint32)
        lut = torch.from_numpy(exp_lut_neg()).cuda()
        softmax = compile_kernel(attention_i8_q15_16_gqa_cache_softmax_i16(qh, kvh, seq_len, args.cache_len, hd), [7])
        pv = compile_kernel(attention_i16v8_q15_16_gqa_cache(qh, kvh, seq_len, args.cache_len, hd), [5])
        fused = compile_kernel(attention_i8v8_q15_16_gqa_cache_fused(qh, kvh, seq_len, args.cache_len, hd), [11])
        fused_static = compile_kernel(attention_i8v8_q15_16_gqa_cache_fused_static_current(qh, kvh, seq_len, args.cache_len, hd), [11])

        def run_cache(prof):
            if args.static_cache:
                return prof.time("attention_cache_i8v8_fused_static", lambda: fused_static(q, ck, cv, k, v, qs_head, cks, cvs, ks_head, vs_head, lut))
            if args.fused_attn:
                return prof.time("attention_cache_i8v8_fused", lambda: fused(q, ck, cv, k, v, qs, cks, cvs, ks, vs, lut))
            p = prof.time("attention_cache_softmax_i16", lambda: softmax(q, ck, k, qs, cks, ks, lut))
            return prof.time("attention_cache_i16v8", lambda: pv(p, cv, v, cvs, vs))

        ops = op_counts(seq_len, cfg, args.cache_len)
        for _ in range(args.warmup):
            run_cache(Profiler(ops))
        prof = Profiler(ops)
        for _ in range(args.repeat):
            run_cache(prof)
        summary = prof.report()
        if args.jsonl_out:
            row = {
                "max_tokens": args.max_tokens,
                "seq_len": seq_len,
                "cache_len": args.cache_len,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "fused_attn": args.fused_attn,
                "static_cache": args.static_cache,
                **summary,
            }
            with open(args.jsonl_out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        return

    cache_kv, cache_len = None, 0
    if args.cache_block:
        _packed_r1, packed_r2 = packed_flags(args.packed_dir)
        cache_kv, cache_len = build_cache_kv(args.model_dir, tokenizer, args.cache_prompt, args.layers, args.use_r2 or packed_r2, args.use_r3)
    mlp_i16_layers = parse_layer_set(args.mlp_i16_layers)
    block = Qwen3IntOnlyBlock(seq_len, cfg, cache_len=cache_len, use_r3=args.use_r3, split_attn=args.split_attn, fused_attn=args.fused_attn, fast_hadamard=args.fast_hadamard, mlp_i16=args.mlp_i16 or bool(mlp_i16_layers))
    r3_q15 = q15_16(random_hadamard_rotation(cfg.head_dim, ROTATE_SEED + 2)) if args.use_r3 else None
    if args.packed_dir:
        embed, _, _, weights = load_packed_qwen3(args.packed_dir, cfg)
    else:
        weights = load_all_qwen3_block_weights(args.model_dir, cfg)
        embed = load_embed_tokens(args.model_dir)
    cos, sin, _ = rope_tables_q15_16(seq_len + cache_len, cfg.head_dim, cfg.rope_theta)
    def run_layers(prof):
        x = q15_16(embed[ids])
        for layer_idx in range(args.layers):
            layer_cache = None if cache_kv is None else cache_kv[layer_idx]
            cache_k = None if layer_cache is None else layer_cache[0]
            cache_v = None if layer_cache is None else layer_cache[1]
            x = run_block(block, x, weights[layer_idx], cos, sin, r3_q15, prof, args.split_attn, args.fused_attn, cache_k, cache_v, args.mlp_i16 or layer_idx in mlp_i16_layers)
        return x

    ops = op_counts(seq_len, cfg, cache_len)
    for _ in range(args.warmup):
        run_layers(Profiler(ops))
    prof = Profiler(ops)
    for _ in range(args.repeat):
        run_layers(prof)
    summary = prof.report()
    if args.jsonl_out:
        row = {
            "max_tokens": args.max_tokens,
            "seq_len": seq_len,
            "layers": args.layers,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "use_r2": args.use_r2,
            "use_r3": args.use_r3,
            "split_attn": args.split_attn,
            "fused_attn": args.fused_attn,
            "fast_hadamard": args.fast_hadamard,
            "mlp_i16": args.mlp_i16,
            "mlp_i16_layers": args.mlp_i16_layers,
            "cache_block": args.cache_block,
            "cache_len": cache_len,
            **summary,
        }
        with open(args.jsonl_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
