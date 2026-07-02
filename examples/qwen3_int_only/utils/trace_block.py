import argparse
import json

import torch
from transformers import AutoTokenizer

from examples.qwen3_int_only.model import Qwen3IntOnlyModel
from examples.qwen3_int_only.utils import (
    ROTATE_SEED,
    Q15_16,
    Qwen3BlockWeights,
    block_torch,
    block_torch_trace,
    load_packed_qwen3,
    q15_16,
    random_hadamard_rotation,
    rmsnorm_torch,
)
from examples.qwen3_int_only.utils.ppl import load_ids


def metrics(a, b):
    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    diff = x - y
    dot = torch.dot(x, y)
    x2 = torch.dot(x, x)
    y2 = torch.dot(y, y)
    se = torch.dot(diff, diff)
    ae = torch.sum(torch.abs(diff))
    return {
        "cos": float(dot / torch.sqrt((x2 * y2).clamp_min(1e-30))),
        "mse": float(se / x.numel()),
        "mae": float(ae / x.numel()),
        "max_abs": float(torch.max(torch.abs(diff))),
        "rel_mse": float(se / y2.clamp_min(1e-30)),
    }


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


def dequant_static(q, scale):
    return q.float() * scale.float()[0] / Q15_16


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
    row = metrics(value, ref)
    print(
        f"{name:14s} cos={row['cos']:.8f} mse={row['mse']:.8e} "
        f"mae={row['mae']:.8e} max_abs={row['max_abs']:.8e} rel_mse={row['rel_mse']:.8e}"
    )
    return row


def print_rel_to(name, value, ref, denom):
    diff = value.float().reshape(-1) - ref.float().reshape(-1)
    se = torch.dot(diff, diff)
    den = torch.dot(denom.float().reshape(-1), denom.float().reshape(-1)).clamp_min(1e-30)
    energy = torch.dot(ref.float().reshape(-1), ref.float().reshape(-1)) / den
    row = {"rel_to_out": float(se / den), "ref_energy": float(energy)}
    print(f"{name:14s} rel_to_out={row['rel_to_out']:.8e} ref_energy={row['ref_energy']:.8e}")
    return row


def parse_layers(value, fallback):
    if not value:
        return [fallback]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_row(rows, layer_idx, seq_len, name, kind, values):
    row = {"layer": layer_idx, "tokens": seq_len, "name": name, "kind": kind}
    row.update(values)
    rows.append(row)


def print_trace(layer_idx, seq_len, x_int, xf, itrace, ftrace, layer):
    rows = []
    print(f"layer={layer_idx} tokens={seq_len}")
    add_row(rows, layer_idx, seq_len, "hidden_in", "metric", print_metric("hidden_in", x_int, xf))
    for name in ("input_rms", "q", "k", "v", "attn", "attn_out", "attn_residual", "post_rms", "gate", "up", "gated", "mlp", "layer_out"):
        add_row(rows, layer_idx, seq_len, name, "metric", print_metric(name, itrace[name], ftrace[name]))
    add_row(rows, layer_idx, seq_len, "hidden_in_out", "rel_to_out", print_rel_to("hidden_in_out", x_int, xf, ftrace["layer_out"]))
    for name in ("attn_out", "mlp", "layer_out"):
        add_row(rows, layer_idx, seq_len, name + "_out", "rel_to_out", print_rel_to(name + "_out", itrace[name], ftrace[name], ftrace["layer_out"]))
    input_rms_ref = rmsnorm_torch(x_int, layer.input_layernorm.float() / Q15_16)
    post_rms_ref = rmsnorm_torch(itrace["attn_residual"], layer.post_attention_layernorm.float() / Q15_16)
    add_row(rows, layer_idx, seq_len, "input_rms_kern", "metric", print_metric("input_rms_kern", itrace["input_rms"], input_rms_ref))
    add_row(rows, layer_idx, seq_len, "input_rms_int", "metric", print_metric("input_rms_int", input_rms_ref, ftrace["input_rms"]))
    add_row(rows, layer_idx, seq_len, "attn_resid_add", "metric", print_metric("attn_resid_add", itrace["attn_residual"], x_int + itrace["attn_out"]))
    add_row(rows, layer_idx, seq_len, "post_rms_kern", "metric", print_metric("post_rms_kern", itrace["post_rms"], post_rms_ref))
    add_row(rows, layer_idx, seq_len, "post_rms_int", "metric", print_metric("post_rms_int", post_rms_ref, ftrace["post_rms"]))
    add_row(rows, layer_idx, seq_len, "layer_out_add", "metric", print_metric("layer_out_add", itrace["layer_out"], itrace["attn_residual"] + itrace["mlp"]))
    qdq = dequant_qkv(itrace)
    for name, value in zip(("q_qdq", "k_qdq", "v_qdq"), qdq):
        base = name[:1]
        add_row(rows, layer_idx, seq_len, name, "metric", print_metric(name, value, ftrace[base]))
        add_row(rows, layer_idx, seq_len, name + "_loss", "metric", print_metric(name + "_loss", value, itrace[base]))
    attn_qdq = dequant_static(itrace["attn8"], itrace["attn_s8"])
    add_row(rows, layer_idx, seq_len, "attn_qdq", "metric", print_metric("attn_qdq", attn_qdq, ftrace["attn"]))
    add_row(rows, layer_idx, seq_len, "attn_qdq_loss", "metric", print_metric("attn_qdq_loss", attn_qdq, itrace["attn"]))
    for name, q_name, s_name, base in (
        ("post_qdq", "post8", "post_s8", "post_rms"),
        ("gated_qdq", "gated8", "gated_s8", "gated"),
    ):
        value = dequant_rows(itrace[q_name], itrace[s_name])
        add_row(rows, layer_idx, seq_len, name, "metric", print_metric(name, value, ftrace[base]))
        add_row(rows, layer_idx, seq_len, name + "_loss", "metric", print_metric(name + "_loss", value, itrace[base]))
    post_qdq = dequant_rows(itrace["post8"], itrace["post_s8"])
    gated_qdq = dequant_rows(itrace["gated8"], itrace["gated_s8"])
    o_qdq_ref = linear_ref(attn_qdq, layer, "o_proj")
    gate_qdq_ref = linear_ref(post_qdq, layer, "gate_proj")
    up_qdq_ref = linear_ref(post_qdq, layer, "up_proj")
    down_qdq_ref = linear_ref(gated_qdq, layer, "down_proj")
    for name, value, ref in (
        ("o_from_attn_qdq", itrace["attn_out"], o_qdq_ref),
        ("gate_from_post_qdq", itrace["gate"], gate_qdq_ref),
        ("up_from_post_qdq", itrace["up"], up_qdq_ref),
        ("down_from_gated_qdq", itrace["mlp"], down_qdq_ref),
    ):
        add_row(rows, layer_idx, seq_len, name, "metric", print_metric(name, value, ref))
    silu_ref = torch.nn.functional.silu(itrace["gate"]) * itrace["up"]
    add_row(rows, layer_idx, seq_len, "silu_mul", "metric", print_metric("silu_mul", itrace["gated"], silu_ref))
    if itrace["prob_i16"] is not None:
        prob_ref, pv_ref = attention_refs(itrace)
        add_row(rows, layer_idx, seq_len, "softmax_i16", "metric", print_metric("softmax_i16", itrace["prob_i16"].float() / 16383.0, prob_ref))
        add_row(rows, layer_idx, seq_len, "pv_i16v8", "metric", print_metric("pv_i16v8", itrace["attn"], pv_ref))
    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/publicdata/huggingface.co/Qwen/Qwen3-0.6B")
    parser.add_argument("--packed-dir", default="/tmp/Qwen3-0.6B-int-only-static")
    parser.add_argument("--eval-dataset", default="fineweb")
    parser.add_argument("--eval-parquet", default="")
    parser.add_argument("--eval-column", default="text")
    parser.add_argument("--eval-text", default="")
    parser.add_argument("--max-tokens", type=int, default=257)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--layers", default="")
    parser.add_argument("--jsonl-out", default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    ids = load_ids(tokenizer, args, args.max_tokens, "cuda")
    seq_len = ids.numel() - 1
    ids = ids[: seq_len + 1]
    imodel = Qwen3IntOnlyModel(seq_len, model_dir=args.model_dir, packed_dir=args.packed_dir, use_r3=True, fast_hadamard=False)
    embed, _lm_head, _final_norm, fweights = load_packed_qwen3(args.packed_dir)
    for weights in fweights:
        attach_dequant_fp(weights)
    r3 = random_hadamard_rotation(imodel.config.head_dim, ROTATE_SEED + 2, "cuda")
    targets = sorted(set(parse_layers(args.layers, args.layer)))
    max_layer = targets[-1]
    xf = embed[ids[:-1]]
    xi = q15_16(imodel.embed[ids[:-1]])
    json_rows = []
    for layer_idx in range(max_layer + 1):
        if layer_idx in targets:
            ftrace = block_torch_trace(xf, fweights[layer_idx], imodel.cos, imodel.sin, imodel.config, r3)
            itrace = imodel.block.trace(xi, imodel.layers[layer_idx], imodel.cos, imodel.sin, r3_q15=imodel.r3_q15)
            json_rows.extend(print_trace(layer_idx, seq_len, xi.float() / Q15_16, xf, itrace, ftrace, imodel.layers[layer_idx]))
            xf = ftrace["layer_out"]
            xi = itrace["layer_out_q15"]
        else:
            xf = block_torch(xf, fweights[layer_idx], imodel.cos, imodel.sin, imodel.config, r3)
            xi = imodel.block(xi, imodel.layers[layer_idx], imodel.cos, imodel.sin, r3_q15=imodel.r3_q15)
    if args.jsonl_out:
        with open(args.jsonl_out, "a", encoding="utf-8") as f:
            for row in json_rows:
                row.update(
                    {
                        "model_dir": args.model_dir,
                        "packed_dir": args.packed_dir,
                        "eval_dataset": args.eval_dataset,
                        "eval_parquet": args.eval_parquet,
                        "eval_text": args.eval_text,
                        "use_r3": True,
                        "fused_static": True,
                    }
                )
                f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
