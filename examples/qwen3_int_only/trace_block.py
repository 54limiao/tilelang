import argparse

import torch
from transformers import AutoTokenizer

from examples.qwen3_int_only.model import (
    Q15_16,
    Qwen3BlockWeights,
    Qwen3IntOnlyModel,
    block_torch,
    block_torch_trace,
    load_packed_qwen3,
    q15_16,
    rmsnorm_torch,
)
from examples.qwen3_int_only.ppl import FINEWEB_PATH, load_ids
from examples.qwen3_int_only.quarot import ROTATE_SEED, random_hadamard_rotation


def metrics(a, b):
    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    diff = x - y
    dot = torch.dot(x, y)
    x2 = torch.dot(x, x)
    y2 = torch.dot(y, y)
    se = torch.dot(diff, diff)
    return float(dot / torch.sqrt((x2 * y2).clamp_min(1e-30))), float(se / x.numel()), float(se / y2.clamp_min(1e-30))


def attention_refs(trace):
    q = trace["q8"].float()
    group = q.shape[0] // trace["k8"].shape[0]
    k = trace["k8"].repeat_interleave(group, dim=0).float()
    v = trace["v8"].repeat_interleave(group, dim=0).float()
    qs = trace["qs8"].float()[:, :, None]
    ks = trace["ks8"].repeat_interleave(group, dim=0).float()[:, None, :]
    vs = trace["vs8"].repeat_interleave(group, dim=0).float()[:, :, None]
    score = torch.matmul(q, k.transpose(-1, -2)) * (qs * ks) / (Q15_16 * Q15_16 * (q.shape[-1] ** 0.5))
    mask = torch.ones(score.shape[-2:], device=score.device, dtype=torch.bool).tril()
    prob = torch.softmax(score.masked_fill(~mask, torch.finfo(score.dtype).min), dim=-1)
    pv = torch.matmul(prob, v * vs / Q15_16).permute(1, 0, 2).reshape(trace["attn"].shape)
    return prob, pv


def dequant_qkv(trace):
    group = trace["q8"].shape[0] // trace["k8"].shape[0]
    q = (trace["q8"].float() * trace["qs8"].float()[:, :, None] / Q15_16).permute(1, 0, 2).reshape(trace["q"].shape)
    k = (trace["k8"].float() * trace["ks8"].float()[:, :, None] / Q15_16).repeat_interleave(group, dim=0).permute(1, 0, 2).reshape(trace["k"].shape)
    v = (trace["v8"].float() * trace["vs8"].float()[:, :, None] / Q15_16).repeat_interleave(group, dim=0).permute(1, 0, 2).reshape(trace["v"].shape)
    return q, k, v


def dequant_rows(q, scale):
    return q.float() * scale.float()[:, None] / Q15_16


def linear_ref(x, weights, name):
    packed = getattr(weights, name)
    w = packed.weight.float() * (packed.scale.float() / Q15_16)[:, None]
    return x @ w.T


def attach_dequant_fp(weights: Qwen3BlockWeights):
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        packed = getattr(weights, name)
        setattr(weights, f"{name}_fp", packed.weight.float() * (packed.scale.float() / Q15_16)[:, None])
    return weights


def print_metric(name, value, ref):
    cos, mse, rel = metrics(value, ref)
    print(f"{name:14s} cos={cos:.8f} mse={mse:.8e} rel_mse={rel:.8e}")


def print_rel_to(name, value, ref, denom):
    diff = value.float().reshape(-1) - ref.float().reshape(-1)
    se = torch.dot(diff, diff)
    den = torch.dot(denom.float().reshape(-1), denom.float().reshape(-1)).clamp_min(1e-30)
    energy = torch.dot(ref.float().reshape(-1), ref.float().reshape(-1)) / den
    print(f"{name:14s} rel_to_out={float(se / den):.8e} ref_energy={float(energy):.8e}")


def parse_layers(value, fallback):
    if not value:
        return [fallback]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def print_trace(layer_idx, seq_len, x_int, xf, itrace, ftrace, layer):
    print(f"layer={layer_idx} tokens={seq_len}")
    print_metric("hidden_in", x_int, xf)
    for name in ("input_rms", "q", "k", "v", "attn", "attn_out", "attn_residual", "post_rms", "gate", "up", "gated", "mlp", "layer_out"):
        print_metric(name, itrace[name], ftrace[name])
    print_rel_to("hidden_in_out", x_int, xf, ftrace["layer_out"])
    for name in ("attn_out", "mlp", "layer_out"):
        print_rel_to(name + "_out", itrace[name], ftrace[name], ftrace["layer_out"])
    input_rms_ref = rmsnorm_torch(x_int, layer.input_layernorm.float() / Q15_16)
    post_rms_ref = rmsnorm_torch(itrace["attn_residual"], layer.post_attention_layernorm.float() / Q15_16)
    print_metric("input_rms_kern", itrace["input_rms"], input_rms_ref)
    print_metric("input_rms_int", input_rms_ref, ftrace["input_rms"])
    print_metric("attn_resid_add", itrace["attn_residual"], x_int + itrace["attn_out"])
    print_metric("post_rms_kern", itrace["post_rms"], post_rms_ref)
    print_metric("post_rms_int", post_rms_ref, ftrace["post_rms"])
    print_metric("layer_out_add", itrace["layer_out"], itrace["attn_residual"] + itrace["mlp"])
    qdq = dequant_qkv(itrace)
    for name, value in zip(("q_qdq", "k_qdq", "v_qdq"), qdq):
        base = name[:1]
        print_metric(name, value, ftrace[base])
        print_metric(name + "_loss", value, itrace[base])
    for name, q_name, s_name, base in (
        ("attn_qdq", "attn8", "attn_s8", "attn"),
        ("post_qdq", "post8", "post_s8", "post_rms"),
        ("gated_qdq", "gated8", "gated_s8", "gated"),
    ):
        value = dequant_rows(itrace[q_name], itrace[s_name])
        print_metric(name, value, ftrace[base])
        print_metric(name + "_loss", value, itrace[base])
    post_qdq = dequant_rows(itrace["post8"], itrace["post_s8"])
    gate_qdq_ref = linear_ref(post_qdq, layer, "gate_proj")
    up_qdq_ref = linear_ref(post_qdq, layer, "up_proj")
    for name, value, ref in (("gate_from_post_qdq", itrace["gate"], gate_qdq_ref), ("up_from_post_qdq", itrace["up"], up_qdq_ref)):
        print_metric(name, value, ref)
    silu_ref = torch.nn.functional.silu(itrace["gate"]) * itrace["up"]
    print_metric("silu_mul", itrace["gated"], silu_ref)
    if itrace["prob_i16"] is not None:
        prob_ref, pv_ref = attention_refs(itrace)
        print_metric("softmax_i16", itrace["prob_i16"].float() / 16383.0, prob_ref)
        print_metric("pv_i16v8", itrace["attn"], pv_ref)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/code/Qwen3-0.6B")
    parser.add_argument("--packed-dir", default="/tmp/Qwen3-0.6B-int-only-static")
    parser.add_argument("--eval-parquet", default="fineweb")
    parser.add_argument("--eval-column", default="text")
    parser.add_argument("--eval-text", default="examples/qwen3_int_only/data/declaration_of_independence.txt")
    parser.add_argument("--max-tokens", type=int, default=257)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--layers", default="")
    parser.add_argument("--use-r3", action="store_true")
    parser.add_argument("--split-attn", action="store_true")
    args = parser.parse_args()
    if args.eval_parquet == "fineweb":
        args.eval_parquet = FINEWEB_PATH

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    ids = load_ids(tokenizer, args, args.max_tokens, "cuda")
    seq_len = ids.numel() - 1
    ids = ids[: seq_len + 1]
    imodel = Qwen3IntOnlyModel(seq_len, model_dir=args.model_dir, packed_dir=args.packed_dir, use_r3=args.use_r3, split_attn=args.split_attn)
    embed, _lm_head, _final_norm, fweights = load_packed_qwen3(args.packed_dir)
    for weights in fweights:
        attach_dequant_fp(weights)
    r3 = random_hadamard_rotation(imodel.config.head_dim, ROTATE_SEED + 2, "cuda") if args.use_r3 else None
    targets = sorted(set(parse_layers(args.layers, args.layer)))
    max_layer = targets[-1]
    xf = embed[ids[:-1]]
    xi = q15_16(imodel.embed[ids[:-1]])
    for layer_idx in range(max_layer + 1):
        if layer_idx in targets:
            ftrace = block_torch_trace(xf, fweights[layer_idx], imodel.cos, imodel.sin, imodel.config, r3)
            itrace = imodel.block.trace(xi, imodel.layers[layer_idx], imodel.cos, imodel.sin, r3_q15=imodel.r3_q15)
            print_trace(layer_idx, seq_len, xi.float() / Q15_16, xf, itrace, ftrace, imodel.layers[layer_idx])
            xf = ftrace["layer_out"]
            xi = itrace["layer_out_q15"]
        else:
            xf = block_torch(xf, fweights[layer_idx], imodel.cos, imodel.sin, imodel.config, r3)
            xi = imodel.block(xi, imodel.layers[layer_idx], imodel.cos, imodel.sin, r3_q15=imodel.r3_q15)


if __name__ == "__main__":
    main()
