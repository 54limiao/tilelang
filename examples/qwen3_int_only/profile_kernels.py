import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from examples.qwen3_int_only.model import (
    QWEN3_0_6B,
    Qwen3IntOnlyBlock,
    load_all_qwen3_block_weights,
    load_embed_tokens,
    q15_16,
    rope_tables_q15_16,
)
from examples.qwen3_int_only.quarot import ROTATE_SEED, random_hadamard_rotation


TEXT_PATH = Path(__file__).resolve().parent / "data" / "declaration_of_independence.txt"


class Profiler:
    def __init__(self):
        self.rows = []

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
        totals = {}
        for name, ms in self.rows:
            totals[name] = totals.get(name, 0.0) + ms
        total = sum(totals.values())
        counts = {}
        for name, _ in self.rows:
            counts[name] = counts.get(name, 0) + 1
        for name, ms in sorted(totals.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:22s} avg={ms / counts[name]:8.3f} ms total={ms:9.3f} ms {ms / total * 100.0:6.2f}%")
        print(f"{'total':22s} {total:9.3f} ms")


@torch.no_grad()
def run_block(block, x_q15_16, weights, cos_q15_16, sin_q15_16, r3_q15, prof):
    cfg = block.config
    seq_len = block.seq_len
    norm = prof.time("rms_input_q15", lambda: block.rms_hidden_q15(x_q15_16, weights.input_layernorm, block.lut_rsqrt))
    x8, xs8 = prof.time("dq8_hidden", lambda: block.dq8_hidden(norm))
    q = prof.time("q_proj_i8", lambda: block.q_proj(x8, xs8, weights.q_proj.weight, weights.q_proj.scale))
    k = prof.time("k_proj_i8", lambda: block.k_proj(x8, xs8, weights.k_proj.weight, weights.k_proj.scale))
    v = prof.time("v_proj_i8", lambda: block.v_proj(x8, xs8, weights.v_proj.weight, weights.v_proj.scale))
    q_heads = q.reshape(seq_len * cfg.num_attention_heads, cfg.head_dim)
    k_heads = k.reshape(seq_len * cfg.num_key_value_heads, cfg.head_dim)
    v_heads = v.reshape(seq_len * cfg.num_key_value_heads, cfg.head_dim)
    qh16, _ = prof.time("dq16_q_norm", lambda: block.dq16_q_norm(q_heads.reshape(seq_len, cfg.q_size)))
    kh16, _ = prof.time("dq16_kv_norm", lambda: block.dq16_kv_norm(k_heads.reshape(seq_len, cfg.kv_size)))
    q_heads = prof.time("rms_q", lambda: block.rms_q_dyn(qh16.reshape(seq_len * cfg.num_attention_heads, cfg.head_dim), weights.q_norm, block.lut_rsqrt))
    k_heads = prof.time("rms_k", lambda: block.rms_k_dyn(kh16.reshape(seq_len * cfg.num_key_value_heads, cfg.head_dim), weights.k_norm, block.lut_rsqrt))
    cos_q = cos_q15_16[:, None, :].expand(seq_len, cfg.num_attention_heads, cfg.head_dim // 2).reshape(seq_len * cfg.num_attention_heads, cfg.head_dim // 2).contiguous()
    sin_q = sin_q15_16[:, None, :].expand(seq_len, cfg.num_attention_heads, cfg.head_dim // 2).reshape(seq_len * cfg.num_attention_heads, cfg.head_dim // 2).contiguous()
    cos_k = cos_q15_16[:, None, :].expand(seq_len, cfg.num_key_value_heads, cfg.head_dim // 2).reshape(seq_len * cfg.num_key_value_heads, cfg.head_dim // 2).contiguous()
    sin_k = sin_q15_16[:, None, :].expand(seq_len, cfg.num_key_value_heads, cfg.head_dim // 2).reshape(seq_len * cfg.num_key_value_heads, cfg.head_dim // 2).contiguous()
    if block.use_r3:
        qr = prof.time("rope_q", lambda: block.rope_q(q_heads, cos_q, sin_q, r3_q15))
        kr = prof.time("rope_k", lambda: block.rope_k(k_heads, cos_k, sin_k, r3_q15))
    else:
        qr = prof.time("rope_q", lambda: block.rope_q(q_heads, cos_q, sin_q))
        kr = prof.time("rope_k", lambda: block.rope_k(k_heads, cos_k, sin_k))
    q8, qs8 = prof.time("dq8_q_head", lambda: block.dq8_q_head(qr))
    k8, ks8 = prof.time("dq8_kv_head", lambda: block.dq8_kv_head(kr))
    v8, vs8 = prof.time("dq8_v_head", lambda: block.dq8_kv_head(v_heads))
    q_attn = q8.reshape(seq_len, cfg.num_attention_heads, cfg.head_dim).permute(1, 0, 2).contiguous()
    k_attn = k8.reshape(seq_len, cfg.num_key_value_heads, cfg.head_dim).permute(1, 0, 2).contiguous()
    v_attn = v8.reshape(seq_len, cfg.num_key_value_heads, cfg.head_dim).permute(1, 0, 2).contiguous()
    qs_attn = qs8.reshape(seq_len, cfg.num_attention_heads).permute(1, 0).contiguous()
    ks_attn = ks8.reshape(seq_len, cfg.num_key_value_heads).permute(1, 0).contiguous()
    vs_attn = vs8.reshape(seq_len, cfg.num_key_value_heads).permute(1, 0).contiguous()
    attn = prof.time("attention_i8_fixed", lambda: block.attn_i8_fixed(q_attn, k_attn, v_attn, qs_attn, ks_attn, vs_attn, block.lut_exp))
    attn = attn.permute(1, 0, 2).reshape(seq_len, cfg.q_size)
    attn8, attn_s8 = prof.time("dq8_attn", lambda: block.dq8_q(attn))
    attn_out = prof.time("o_proj_i8", lambda: block.o_proj(attn8, attn_s8, weights.o_proj.weight, weights.o_proj.scale))
    h, post = prof.time("residual_attn_rms_q15", lambda: block.add_rms_hidden_q15(x_q15_16, attn_out, weights.post_attention_layernorm, block.lut_rsqrt))
    h8, hs8 = prof.time("dq8_hidden_mlp", lambda: block.dq8_hidden(post))
    gate, up = prof.time(
        "gate_up_proj_i8",
        lambda: block.gate_up_proj_i8(h8, hs8, weights.gate_proj.weight, weights.gate_proj.scale, weights.up_proj.weight, weights.up_proj.scale),
    )
    gated, gated8, gs8 = prof.time("silu_mul_dq8_mid", lambda: block.silu_mul_dq8_mid(gate, up, block.lut_sigmoid))
    mlp = prof.time("down_proj_i8", lambda: block.down_proj_i8(gated8, gs8, weights.down_proj.weight, weights.down_proj.scale))
    return prof.time("residual_mlp", lambda: block.add_hidden(h, mlp))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/code/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=129)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--use-r3", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    ids = tokenizer(TEXT_PATH.read_text(encoding="utf-8"), add_special_tokens=False).input_ids[: args.max_tokens]
    seq_len = len(ids) - 1
    seq_len -= seq_len % 32
    ids = torch.tensor(ids[:seq_len], device="cuda", dtype=torch.long)
    cfg = QWEN3_0_6B
    block = Qwen3IntOnlyBlock(seq_len, cfg, use_r3=args.use_r3)
    r3_q15 = q15_16(random_hadamard_rotation(cfg.head_dim, ROTATE_SEED + 2)) if args.use_r3 else None
    weights = load_all_qwen3_block_weights(args.model_dir, cfg)
    embed = load_embed_tokens(args.model_dir)
    cos, sin, _ = rope_tables_q15_16(seq_len, cfg.head_dim, cfg.rope_theta)
    def run_layers(prof):
        x = q15_16(embed[ids])
        for layer_idx in range(args.layers):
            x = run_block(block, x, weights[layer_idx], cos, sin, r3_q15, prof)
        return x

    for _ in range(args.warmup):
        run_layers(Profiler())
    prof = Profiler()
    for _ in range(args.repeat):
        run_layers(prof)
    prof.report()


if __name__ == "__main__":
    main()
