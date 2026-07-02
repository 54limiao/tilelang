import argparse
import json

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.qwen3_int_only.model import Q15_16, QWEN3_0_6B, Qwen3IntOnlyBlock
from examples.qwen3_int_only.utils.ppl import iter_texts, quant_i8_q15_16
from examples.qwen3_int_only.utils import ROTATE_SEED, load_packed_qwen3, q15_16, random_hadamard_rotation, rope_tables_q15_16


DEFAULT_MODEL_DIR = "/publicdata/huggingface.co/Qwen/Qwen3-0.6B"


def packed_flags(packed_dir):
    with safe_open(f"{packed_dir}/qwen3_int_only.safetensors", framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    return metadata.get("use_r1") == "1", metadata.get("use_r2") == "1"


def op_counts(seq_len, cfg, cache_len=0):
    qk_ops = 2 * cfg.num_attention_heads * seq_len * (seq_len + cache_len) * cfg.head_dim
    pv_ops = 2 * cfg.num_attention_heads * seq_len * (seq_len + cache_len) * cfg.head_dim
    return {
        "qkv_proj_i8": 2 * seq_len * cfg.hidden_size * (cfg.q_size + 2 * cfg.kv_size),
        "o_proj_i8_static": 2 * seq_len * cfg.q_size * cfg.hidden_size,
        "gate_up_proj_static": 2 * seq_len * cfg.hidden_size * (2 * cfg.intermediate_size),
        "down_residual_static": 2 * seq_len * cfg.intermediate_size * cfg.hidden_size,
        "attention_i8v8_fused_static": qk_ops + pv_ops,
        "attention_cache_i8v8_fused_static": qk_ops + pv_ops,
    }


class Profiler:
    def __init__(self, ops):
        self.rows = []
        self.ops = ops

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

    def summary(self, print_rows=False):
        totals = {}
        counts = {}
        for name, ms in self.rows:
            totals[name] = totals.get(name, 0.0) + ms
            counts[name] = counts.get(name, 0) + 1
        total = sum(totals.values())
        items = []
        for name, ms in sorted(totals.items(), key=lambda x: x[1], reverse=True):
            row = {"name": name, "avg_ms": ms / counts[name], "total_ms": ms, "count": counts[name], "pct": ms / total * 100.0}
            if name in self.ops:
                row["gops"] = self.ops[name] * counts[name] / 1.0e9
                row["tops"] = row["gops"] / ms
            items.append(row)
            if print_rows:
                perf = f" {row['tops']:7.2f} TOPS" if "tops" in row else ""
                print(f"{name:28s} avg={row['avg_ms']:8.3f} ms total={ms:9.3f} ms {row['pct']:6.2f}%{perf}")
        if print_rows:
            print(f"{'total':28s} {total:9.3f} ms")
        return {"total_ms": total, "kernels": items}


@torch.no_grad()
def build_cache_kv(model_dir, tokenizer, cache_prompt, layers, use_r2=True, use_r3=True):
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
def run_block(block, x, weights, cos, sin, r3_q15, prof, cache_k=None, cache_v=None):
    cfg = block.config
    pos_cos = cos[block.cache_len : block.cache_len + block.seq_len]
    pos_sin = sin[block.cache_len : block.cache_len + block.seq_len]
    x8, xs8 = prof.time("rms_input_dq8_fast", lambda: block.rms_dq8_hidden_fast(x, weights.input_layernorm, block.lut_rsqrt))
    q, k, v = prof.time("qkv_proj_i8", lambda: block.qkv_proj_i8(x8, xs8, weights.q_proj.weight, weights.q_proj.scale, weights.k_proj.weight, weights.k_proj.scale, weights.v_proj.weight, weights.v_proj.scale))
    v_heads = v.reshape(block.seq_len * cfg.num_key_value_heads, cfg.head_dim)
    q_heads = prof.time("rms_q_q15", lambda: block.rms_q_q15(q, weights.q_norm, block.lut_rsqrt))
    k_heads = prof.time("rms_k_q15", lambda: block.rms_k_q15(k, weights.k_norm, block.lut_rsqrt))
    q_attn = prof.time("rope_sq8_q_attn_hadamard", lambda: block.rope_sq8_q_attn_hadamard(q_heads, pos_cos, pos_sin, r3_q15, weights.q_post_rope_i8_scale))
    k_attn = prof.time("rope_sq8_k_attn_hadamard", lambda: block.rope_sq8_k_attn_hadamard(k_heads, pos_cos, pos_sin, r3_q15, weights.k_post_rope_i8_scale))
    v_attn = prof.time("sq8_v_attn_noscale", lambda: block.sq8_kv_attn_noscale(v_heads.reshape(block.seq_len, cfg.num_key_value_heads, cfg.head_dim), weights.v_i8_scale))
    if cache_k is None:
        attn8 = prof.time("attention_i8v8_fused_static", lambda: block.attn_i8v8_fused_static(q_attn, k_attn, v_attn, weights.q_post_rope_i8_scale, weights.k_post_rope_i8_scale, weights.v_i8_scale, block.lut_exp, weights.attn_i8_scale))
    else:
        attn8 = prof.time("attention_cache_i8v8_fused_static", lambda: block.attn_i8v8_fused_cache_static(q_attn, cache_k[0], cache_v[0], k_attn, v_attn, weights.q_post_rope_i8_scale, cache_k[1], cache_v[1], weights.k_post_rope_i8_scale, weights.v_i8_scale, block.lut_exp, weights.attn_i8_scale))
    attn_out = prof.time("o_proj_i8_static", lambda: block.o_proj(attn8, weights.attn_i8_scale, weights.o_proj.weight, weights.o_proj.scale))
    h, post = prof.time("residual_attn_rms_q15", lambda: block.add_rms_hidden_q15(x, attn_out, weights.post_attention_layernorm, block.lut_rsqrt))
    h8, _hs8 = prof.time("sq8_hidden_static", lambda: block.sq8_hidden(post, weights.post_mlp_i8_scale))
    hs8 = weights.post_mlp_i8_scale
    gate, up = prof.time("gate_up_proj_static", lambda: block.gate_up_proj_static(h8, hs8, weights.gate_proj.weight, weights.gate_proj.scale, weights.up_proj.weight, weights.up_proj.scale))
    gated, _gs = prof.time("silu_mul_sq16_mid_fast", lambda: block.silu_mul_sq16_mid_fast(gate, up, block.lut_sigmoid, weights.gated_mlp_i16_scale))
    gs = weights.gated_mlp_i16_scale
    return prof.time("down_residual_static", lambda: block.down_residual_static(gated, gs, weights.down_proj.weight, weights.down_proj.scale, h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--packed-dir", default="/tmp/Qwen3-0.6B-static-calib-32x2048")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cache-prompt", default="你是一个有用而无害的聊天助手。")
    parser.add_argument("--jsonl-out", default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    ids = []
    for text in iter_texts("fineweb", "text"):
        ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
        if len(ids) >= args.max_tokens:
            break
    ids = ids[: args.max_tokens]
    seq_len = len(ids)
    seq_len -= seq_len % 32
    ids = torch.tensor(ids[:seq_len], device="cuda", dtype=torch.long)
    _packed_r1, packed_r2 = packed_flags(args.packed_dir)
    cache_kv, cache_len = build_cache_kv(args.model_dir, tokenizer, args.cache_prompt, args.layers, packed_r2, True)
    embed, _, _, weights = load_packed_qwen3(args.packed_dir, QWEN3_0_6B)
    block = Qwen3IntOnlyBlock(seq_len, QWEN3_0_6B, cache_len=cache_len, use_r3=True, fast_hadamard=True)
    cos, sin, _ = rope_tables_q15_16(seq_len + cache_len, QWEN3_0_6B.head_dim, QWEN3_0_6B.rope_theta)
    r3_q15 = q15_16(random_hadamard_rotation(QWEN3_0_6B.head_dim, ROTATE_SEED + 2))

    def run_layers(prof):
        x = q15_16(embed[ids])
        for layer_idx in range(args.layers):
            cache_k, cache_v = cache_kv[layer_idx]
            x = run_block(block, x, weights[layer_idx], cos, sin, r3_q15, prof, cache_k, cache_v)
        return x

    ops = op_counts(seq_len, QWEN3_0_6B, cache_len)
    for _ in range(args.warmup):
        run_layers(Profiler(ops))
    prof = Profiler(ops)
    for _ in range(args.repeat):
        run_layers(prof)
    summary = prof.summary(print_rows=True)
    counted_ops_top = sum(row.get("gops", 0.0) for row in summary["kernels"]) / 1000.0
    print(
        f"profile seq_len={seq_len} tokens layers={args.layers} repeats={args.repeat} "
        f"total_ms={summary['total_ms']:.3f} per_pass_ms={summary['total_ms'] / args.repeat:.3f} "
        f"counted_ops_per_pass={counted_ops_top / args.repeat:.3f} TOP counted_tops={counted_ops_top / (summary['total_ms'] / 1000.0):.2f}"
    )
    if args.jsonl_out:
        row = {
            "max_tokens": args.max_tokens,
            "seq_len": seq_len,
            "layers": args.layers,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "static_mlp": True,
            "cache_len": cache_len,
            "fused_static": True,
            "fast_hadamard": True,
            "counted_ops_top": counted_ops_top,
            "counted_ops_per_pass_top": counted_ops_top / args.repeat,
            "counted_tops": counted_ops_top / (summary["total_ms"] / 1000.0),
            **summary,
        }
        with open(args.jsonl_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
