import argparse
import json

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.qwen3_int_only.model import Q15_16, Qwen3IntOnlyBlock
from examples.qwen3_int_only.utils.ppl import iter_texts, quant_i8_static_q15_16
from examples.qwen3_int_only.utils import ROTATE_SEED, Qwen3Config, load_packed_qwen3, q15_16, random_hadamard_rotation, rope_tables_q15_16


DEFAULT_MODEL_DIR = "/publicdata/huggingface.co/Qwen/Qwen3-0.6B"


def packed_flags(packed_dir):
    with safe_open(f"{packed_dir}/qwen3_int_only.safetensors", framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    return metadata.get("use_r1") == "1", metadata.get("use_r2") == "1"


def op_counts(seq_len, cfg, cache_len=0):
    qk_ops = 2 * cfg.num_attention_heads * seq_len * (seq_len + cache_len) * cfg.head_dim
    pv_ops = 2 * cfg.num_attention_heads * seq_len * (seq_len + cache_len) * cfg.head_dim
    return {
        "linear_i8_qkv": 2 * seq_len * cfg.hidden_size * (cfg.q_size + 2 * cfg.kv_size),
        "linear_i8_o": 2 * seq_len * cfg.q_size * cfg.hidden_size,
        "linear_i8_gate_up": 2 * seq_len * cfg.hidden_size * (2 * cfg.intermediate_size),
        "linear_i8_down": 2 * seq_len * cfg.intermediate_size * cfg.hidden_size,
        "attention_i8": qk_ops + pv_ops,
    }


def tc_op_counts(seq_len, cfg, cache_len=0):
    # math_ops is the model matmul work. tc_ops is the int8 tensorcore work we
    # actually issue: attention recomputes QK and splits P16@V8 into two int8 GEMMs.
    ops = op_counts(seq_len, cfg, cache_len)
    qk_ops = 2 * cfg.num_attention_heads * seq_len * (seq_len + cache_len) * cfg.head_dim
    pv_ops = qk_ops
    ops["attention_i8"] = 2 * qk_ops + 2 * pv_ops
    return ops


def default_int8_peak_tops():
    name = torch.cuda.get_device_name().lower()
    if "a100" in name:
        return 624.0
    return 0.0


class Profiler:
    def __init__(self, peak_tops=0.0):
        self.rows = []
        self.peak_tops = peak_tops

    def time(self, name, fn, math_ops=0, tc_ops=None):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        self.rows.append((name, start.elapsed_time(end), math_ops, math_ops if tc_ops is None else tc_ops))
        return out

    def summary(self, print_rows=False):
        totals = {}
        counts = {}
        math_ops = {}
        tc_ops = {}
        for name, ms, math, tc in self.rows:
            totals[name] = totals.get(name, 0.0) + ms
            counts[name] = counts.get(name, 0) + 1
            math_ops[name] = math_ops.get(name, 0) + math
            tc_ops[name] = tc_ops.get(name, 0) + tc
        total = sum(totals.values())
        items = []
        for name, ms in sorted(totals.items(), key=lambda x: x[1], reverse=True):
            row = {"name": name, "total_ms": ms, "pct": ms / total * 100.0}
            if math_ops[name]:
                row["math_gops"] = math_ops[name] / 1.0e9
                row["math_tops"] = row["math_gops"] / ms
                if self.peak_tops:
                    row["math_util_pct"] = row["math_tops"] / self.peak_tops * 100.0
                row["tc_gops"] = tc_ops[name] / 1.0e9
                row["tc_tops"] = row["tc_gops"] / ms
                if self.peak_tops:
                    row["tc_util_pct"] = row["tc_tops"] / self.peak_tops * 100.0
            items.append(row)
            if print_rows:
                perf = ""
                if "math_tops" in row:
                    math_util = f" util={row['math_util_pct']:5.2f}%" if "math_util_pct" in row else ""
                    tc_util = f" util={row['tc_util_pct']:5.2f}%" if "tc_util_pct" in row else ""
                    perf = f" math={row['math_tops']:7.2f} TOPS{math_util} tc={row['tc_tops']:7.2f} TOPS{tc_util}"
                print(f"{name:28s} total={ms:9.3f} ms {row['pct']:6.2f}%{perf}")
        if print_rows:
            print(f"{'total':28s} {total:9.3f} ms")
        return {"total_ms": total, "kernels": items}


@torch.no_grad()
def build_cache_kv(model_dir, tokenizer, cache_prompt, layer_weights, config, use_r2=True):
    hf_model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    cache_ids = torch.tensor(tokenizer(cache_prompt, add_special_tokens=False).input_ids, device="cuda", dtype=torch.long)
    past = hf_model(cache_ids[None, :], use_cache=True).past_key_values
    if hasattr(past, "layers"):
        past = [(layer.keys, layer.values) for layer in past.layers]
    elif hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    r2 = random_hadamard_rotation(config.head_dim, ROTATE_SEED + 1, "cuda") if use_r2 else None
    r3 = random_hadamard_rotation(config.head_dim, ROTATE_SEED + 2, "cuda")
    cache_kv = []
    for weights, (k, v) in zip(layer_weights, past[: len(layer_weights)]):
        k = k[0].float().contiguous()
        v = v[0].float().contiguous()
        k = (k.to(torch.float64) @ r3.to(torch.float64)).to(torch.float32)
        if r2 is not None:
            v = (v.to(torch.float64) @ r2.to(torch.float64)).to(torch.float32)
        kq = quant_i8_static_q15_16(k, weights.k_post_rope_i8_scale[:, None])
        vq = quant_i8_static_q15_16(v, weights.v_i8_scale[:, None])
        cache_kv.append((kq, vq))
    return cache_kv, int(cache_ids.numel())


@torch.no_grad()
def run_block(block, x, x8, xs8, weights, cos, sin, r3_q15, prof, cache_k=None, cache_v=None):
    cfg = block.config
    ops = op_counts(block.seq_len, cfg, block.cache_len)
    tc_ops = tc_op_counts(block.seq_len, cfg, block.cache_len)
    pos_cos = cos[block.cache_len : block.cache_len + block.seq_len]
    pos_sin = sin[block.cache_len : block.cache_len + block.seq_len]
    qkv = prof.time("linear_i8", lambda: block.qkv_proj(x8, weights.qkv_proj.weight, weights.qkv_out_qt), ops["linear_i8_qkv"], tc_ops["linear_i8_qkv"])
    q = qkv[:, : cfg.q_size].contiguous()
    k = qkv[:, cfg.q_size : cfg.q_size + cfg.kv_size].contiguous()
    v = qkv[:, cfg.q_size + cfg.kv_size :].contiguous()
    v_heads = v.reshape(block.seq_len * cfg.num_key_value_heads, cfg.head_dim)
    _q, q_heads = prof.time("rms_q15", lambda: block.rms_q(q.reshape(block.seq_len * cfg.num_attention_heads, cfg.head_dim).contiguous(), block.zero_q, weights.q_norm, block.lut_rsqrt))
    _k, k_heads = prof.time("rms_q15", lambda: block.rms_k(k.reshape(block.seq_len * cfg.num_key_value_heads, cfg.head_dim).contiguous(), block.zero_k, weights.k_norm, block.lut_rsqrt))
    q_attn = prof.time("rope_sq8", lambda: block.rope_q(q_heads, pos_cos, pos_sin, r3_q15, weights.q_post_rope_i8_qt))
    k_attn = prof.time("rope_sq8", lambda: block.rope_k(k_heads, pos_cos, pos_sin, r3_q15, weights.k_post_rope_i8_qt))
    v_attn = prof.time("quant_v_i8", lambda: block.quant_v(v_heads.reshape(block.seq_len, cfg.num_key_value_heads, cfg.head_dim), weights.v_i8_qt))
    cache_k = block.empty_cache_k if cache_k is None else cache_k
    cache_v = block.empty_cache_v if cache_v is None else cache_v
    attn8 = prof.time("attention_i8", lambda: block.attn(q_attn, cache_k, cache_v, k_attn, v_attn, weights.q_post_rope_i8_scale, weights.k_post_rope_i8_scale, block.lut_exp, weights.attn_out_qt), ops["attention_i8"], tc_ops["attention_i8"])
    attn_out = prof.time("linear_i8", lambda: block.o_proj(attn8, weights.o_proj.weight, weights.o_out_qt), ops["linear_i8_o"], tc_ops["linear_i8_o"])
    h, h8, _hs8 = prof.time("rms_sq8", lambda: block.rms_sq8(x, attn_out, weights.post_attention_layernorm, block.lut_rsqrt, weights.post_mlp_i8_qt))
    gate_up = prof.time("linear_i8", lambda: block.gate_up_proj(h8, weights.gate_up_proj.weight, weights.gate_up_out_qt), ops["linear_i8_gate_up"], tc_ops["linear_i8_gate_up"])
    gate = gate_up[:, : cfg.intermediate_size].contiguous()
    up = gate_up[:, cfg.intermediate_size :].contiguous()
    gated, _gs = prof.time("silu_hadamard_i8", lambda: block.silu_hadamard(gate, up, block.lut_sigmoid, weights.gated_mlp_i8_qt))
    mlp = prof.time("linear_i8", lambda: block.down_proj(gated, weights.down_proj.weight, weights.down_out_qt), ops["linear_i8_down"], tc_ops["linear_i8_down"])
    return h, mlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--packed-dir", default="/tmp/Qwen3-0.6B-static-calib-32x2048")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--peak-tops", type=float, default=0.0)
    parser.add_argument("--cache-prompt", default="你是一个有用而无害的聊天助手。")
    parser.add_argument("--jsonl-out", default="")
    args = parser.parse_args()

    config = Qwen3Config.from_model_dir(args.model_dir)
    if args.layers == 0:
        args.layers = config.num_hidden_layers
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
    embed, _, _, weights = load_packed_qwen3(args.packed_dir, config)
    cache_kv, cache_len = build_cache_kv(args.model_dir, tokenizer, args.cache_prompt, weights[: args.layers], config, packed_r2)
    block = Qwen3IntOnlyBlock(seq_len, config, cache_len=cache_len)
    cos, sin, _ = rope_tables_q15_16(seq_len + cache_len, config.head_dim, config.rope_theta)
    r3_q15 = q15_16(random_hadamard_rotation(config.head_dim, ROTATE_SEED + 2))

    def run_layers(prof):
        residual = q15_16(embed[ids])
        _res, x8, xs8 = prof.time(
            "rms_sq8",
            lambda: block.rms_sq8(residual, block.zero_hidden, weights[0].input_layernorm, block.lut_rsqrt, weights[0].input_qkv_i8_qt),
        )
        mlp = None
        for layer_idx in range(args.layers):
            if layer_idx:
                residual, x8, xs8 = prof.time(
                    "rms_sq8",
                    lambda: block.rms_sq8(residual, mlp, weights[layer_idx].input_layernorm, block.lut_rsqrt, weights[layer_idx].input_qkv_i8_qt),
                )
            cache_k, cache_v = cache_kv[layer_idx]
            residual, mlp = run_block(block, residual, x8, xs8, weights[layer_idx], cos, sin, r3_q15, prof, cache_k, cache_v)
        return residual, mlp

    peak_tops = args.peak_tops or default_int8_peak_tops()
    for _ in range(args.warmup):
        run_layers(Profiler(peak_tops))
    prof = Profiler(peak_tops)
    for _ in range(args.repeat):
        run_layers(prof)
    summary = prof.summary(print_rows=True)
    math_ops_top = sum(row.get("math_gops", 0.0) for row in summary["kernels"]) / 1000.0
    tc_ops_top = sum(row.get("tc_gops", row.get("math_gops", 0.0)) for row in summary["kernels"]) / 1000.0
    math_tops = math_ops_top / (summary["total_ms"] / 1000.0)
    tc_tops = tc_ops_top / (summary["total_ms"] / 1000.0)
    math_util = math_tops / peak_tops * 100.0 if peak_tops else 0.0
    tc_util = tc_tops / peak_tops * 100.0 if peak_tops else 0.0
    print(
        f"profile seq_len={seq_len} tokens layers={args.layers} repeats={args.repeat} "
        f"total_ms={summary['total_ms']:.3f} per_pass_ms={summary['total_ms'] / args.repeat:.3f} "
        f"math_ops_per_pass={math_ops_top / args.repeat:.3f} TOP math_tops={math_tops:.2f} "
        f"tc_ops_per_pass={tc_ops_top / args.repeat:.3f} TOP tc_tops={tc_tops:.2f} "
        f"peak_tops={peak_tops:.2f} math_util={math_util:.2f}% tc_util={tc_util:.2f}%"
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
            "math_ops_top": math_ops_top,
            "math_ops_per_pass_top": math_ops_top / args.repeat,
            "tc_ops_top": tc_ops_top,
            "tc_ops_per_pass_top": tc_ops_top / args.repeat,
            "math_tops": math_tops,
            "tc_tops": tc_tops,
            "peak_tops": peak_tops,
            "math_util_pct": math_util,
            "tc_util_pct": tc_util,
            **summary,
        }
        with open(args.jsonl_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
