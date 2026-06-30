import argparse
import math

import torch
from transformers import AutoTokenizer

from examples.qwen3_int_only.model import (
    Q15_16,
    Qwen3FloatModel,
    causal_softmax,
    q15_16,
    rmsnorm_torch,
    rope_torch,
)
from examples.qwen3_int_only.ppl import archives_declaration_text, archives_founding_text, tokens_from_texts
from examples.qwen3_int_only.proto import attention_i12_lut_proto, linear_dynamic_proto, linear_i16_kernel_proto, q15, rmsnorm_i16_proto


STAGES = ["float", "q15-io", "rms", "qkv", "qknorm-rope", "attn-q15", "attn-i12-out", "o-proj", "post-rms", "gate-up", "silu-mul", "down"]


def metric(name, ref, got):
    a, b = ref.float().reshape(-1), got.float().reshape(-1)
    rmse = torch.sqrt(torch.mean((a - b) * (a - b)))
    rel = rmse / torch.sqrt(torch.mean(a * a)).clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0)
    print(f"{name:14s} cos={float(cos): .6f} rel={float(rel): .6f} rmse={float(rmse): .6f}")


def float_attention(q, k, v, cfg):
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    qh = q.reshape(-1, cfg.num_attention_heads, cfg.head_dim).permute(1, 0, 2)
    kh = k.reshape(-1, cfg.num_key_value_heads, cfg.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2)
    vh = v.reshape(-1, cfg.num_key_value_heads, cfg.head_dim).repeat_interleave(group, dim=1).permute(1, 0, 2)
    return (causal_softmax((qh @ kh.transpose(-1, -2)) / math.sqrt(cfg.head_dim)) @ vh).permute(1, 0, 2).reshape(-1, cfg.q_size)

def block_stage(x, w, cos, sin, cfg, stage):
    float_norm = rmsnorm_torch(x, w.input_layernorm.float() / Q15_16)
    norm = float_norm if stage in ["float", "q15-io"] else rmsnorm_i16_proto(q15(x), w.input_layernorm).float() / Q15_16

    if stage in ["float", "q15-io", "rms"]:
        q = norm @ w.q_proj_fp.T
        k = norm @ w.k_proj_fp.T
        v = norm @ w.v_proj_fp.T
    else:
        q_i, _, _ = linear_dynamic_proto(q15(norm), w.q_proj.weight, w.q_proj.scale, 127)
        k_i, _, _ = linear_dynamic_proto(q15(norm), w.k_proj.weight, w.k_proj.scale, 127)
        v_i, _, _ = linear_dynamic_proto(q15(norm), w.v_proj.weight, w.v_proj.scale, 127)
        q, k, v = q_i.float() / Q15_16, k_i.float() / Q15_16, v_i.float() / Q15_16

    if stage in ["float", "q15-io", "rms", "qkv"]:
        qn = rmsnorm_torch(q.reshape(-1, cfg.num_attention_heads, cfg.head_dim), w.q_norm.float() / Q15_16).reshape(-1, cfg.q_size)
        kn = rmsnorm_torch(k.reshape(-1, cfg.num_key_value_heads, cfg.head_dim), w.k_norm.float() / Q15_16).reshape(-1, cfg.kv_size)
    else:
        qn = rmsnorm_i16_proto(q15(q).reshape(-1, cfg.head_dim), w.q_norm).reshape(-1, cfg.q_size).float() / Q15_16
        kn = rmsnorm_i16_proto(q15(k).reshape(-1, cfg.head_dim), w.k_norm).reshape(-1, cfg.kv_size).float() / Q15_16
    qr = rope_torch(qn, cos, sin, cfg.num_attention_heads, cfg.head_dim)
    kr = rope_torch(kn, cos, sin, cfg.num_key_value_heads, cfg.head_dim)

    if stage in ["attn-q15", "o-proj", "post-rms", "gate-up", "silu-mul", "down", "attn-i12-out"]:
        group = cfg.num_attention_heads // cfg.num_key_value_heads
        kr_g = kr.reshape(-1, cfg.num_key_value_heads, cfg.head_dim).repeat_interleave(group, dim=1).reshape(-1, cfg.q_size)
        v_g = v.reshape(-1, cfg.num_key_value_heads, cfg.head_dim).repeat_interleave(group, dim=1).reshape(-1, cfg.q_size)
        if stage == "attn-i12-out":
            attn_i, attn_s = attention_i12_lut_proto(q15(qr), q15(kr_g), q15(v_g), cfg, out_int=True)
            scale = attn_s.repeat_interleave(cfg.head_dim, dim=1).float() / Q15_16
            attn_out = (attn_i.float() * scale) @ w.o_proj_fp.T
        elif stage == "attn-q15":
            attn_q15 = attention_i12_lut_proto(q15(qr), q15(kr_g), q15(v_g), cfg)
            attn = attn_q15.float() / Q15_16
            attn_out = attn @ w.o_proj_fp.T
        else:
            attn_q15 = attention_i12_lut_proto(q15(qr), q15(kr_g), q15(v_g), cfg)
            attn_out_i, _, _ = linear_i16_kernel_proto(attn_q15, w.o_proj.weight, w.o_proj.scale)
            attn_out = attn_out_i.float() / Q15_16
    else:
        attn_out = float_attention(qr, kr, v, cfg) @ w.o_proj_fp.T

    h = x + attn_out
    if stage == "down":
        post = rmsnorm_i16_proto(q15(h), w.post_attention_layernorm)
        gate, _, _ = linear_i16_kernel_proto(post, w.gate_proj.weight, w.gate_proj.scale)
        up, _, _ = linear_i16_kernel_proto(post, w.up_proj.weight, w.up_proj.scale)
        gated = q15(torch.nn.functional.silu(gate.float() / Q15_16) * (up.float() / Q15_16))
        down, _, _ = linear_i16_kernel_proto(gated, w.down_proj.weight, w.down_proj.scale)
        return h + down.float() / Q15_16

    if stage == "silu-mul":
        post = rmsnorm_i16_proto(q15(h), w.post_attention_layernorm)
        gate, _, _ = linear_i16_kernel_proto(post, w.gate_proj.weight, w.gate_proj.scale)
        up, _, _ = linear_i16_kernel_proto(post, w.up_proj.weight, w.up_proj.scale)
        gated = q15(torch.nn.functional.silu(gate.float() / Q15_16) * (up.float() / Q15_16)).float() / Q15_16
        return h + gated @ w.down_proj_fp.T

    if stage == "gate-up":
        post = rmsnorm_i16_proto(q15(h), w.post_attention_layernorm).float() / Q15_16
        gate_i, _, _ = linear_i16_kernel_proto(q15(post), w.gate_proj.weight, w.gate_proj.scale)
        up_i, _, _ = linear_i16_kernel_proto(q15(post), w.up_proj.weight, w.up_proj.scale)
        gate, up = gate_i.float() / Q15_16, up_i.float() / Q15_16
        return h + (torch.nn.functional.silu(gate) * up) @ w.down_proj_fp.T

    if stage == "post-rms":
        m = rmsnorm_i16_proto(q15(h), w.post_attention_layernorm).float() / Q15_16
        return h + (torch.nn.functional.silu(m @ w.gate_proj_fp.T) * (m @ w.up_proj_fp.T)) @ w.down_proj_fp.T

    m = rmsnorm_torch(h, w.post_attention_layernorm.float() / Q15_16)
    return h + (torch.nn.functional.silu(m @ w.gate_proj_fp.T) * (m @ w.up_proj_fp.T)) @ w.down_proj_fp.T


def run_stage(model, ids, labels, stage):
    x = model.embed[ids]
    if stage == "q15-io":
        x = q15_16(x).float() / Q15_16
    cos, sin = model.cos.float() / Q15_16, model.sin.float() / Q15_16
    for w in model.layers:
        x = block_stage(x, w, cos, sin, model.config, stage)
        if stage == "q15-io":
            x = q15_16(x).float() / Q15_16
    h = rmsnorm_torch(x, model.final_norm)
    logits = h @ model.lm_head.T
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return logits, loss


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/code/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=129)
    parser.add_argument("--corpus", choices=["declaration", "founding"], default="declaration")
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    text = archives_declaration_text() if args.corpus == "declaration" else archives_founding_text()
    input_ids = tokens_from_texts(tokenizer, [text], args.max_tokens, "cuda").squeeze(0)
    ids, labels = input_ids[:-1], input_ids[1:]
    model = Qwen3FloatModel(ids.numel(), model_dir=args.model_dir)
    ref_logits, ref_loss = run_stage(model, ids, labels, "float")
    stages = [args.stage] if args.stage else STAGES
    for stage in stages:
        logits, loss = (ref_logits, ref_loss) if stage == "float" else run_stage(model, ids, labels, stage)
        ppl = math.exp(float(loss))
        print(f"{stage:14s} loss={float(loss):.6f} ppl={ppl:.6f}")
        metric(stage, ref_logits, logits)


if __name__ == "__main__":
    main()
