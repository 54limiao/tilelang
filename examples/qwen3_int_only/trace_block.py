import argparse

import torch
from transformers import AutoTokenizer

from examples.qwen3_int_only.model import Q15_16, Qwen3BlockWeights, Qwen3IntOnlyModel, block_torch, block_torch_trace, load_packed_qwen3, q15_16
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


def attach_dequant_fp(weights: Qwen3BlockWeights):
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        packed = getattr(weights, name)
        setattr(weights, f"{name}_fp", packed.weight.float() * (packed.scale.float() / Q15_16)[:, None])
    return weights


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
    xf = embed[ids[:-1]]
    xi = q15_16(imodel.embed[ids[:-1]])
    for layer_idx in range(args.layer):
        xf = block_torch(xf, fweights[layer_idx], imodel.cos, imodel.sin, imodel.config, r3)
        xi = imodel.block(xi, imodel.layers[layer_idx], imodel.cos, imodel.sin, r3_q15=imodel.r3_q15)
    ftrace = block_torch_trace(xf, fweights[args.layer], imodel.cos, imodel.sin, imodel.config, r3)
    itrace = imodel.block.trace(xi, imodel.layers[args.layer], imodel.cos, imodel.sin, r3_q15=imodel.r3_q15)
    print(f"layer={args.layer} tokens={seq_len}")
    for name in ("input_rms", "q", "k", "v", "attn", "attn_out", "attn_residual", "post_rms", "gate", "up", "gated", "mlp", "layer_out"):
        cos, mse, rel = metrics(itrace[name], ftrace[name])
        print(f"{name:14s} cos={cos:.8f} mse={mse:.8e} rel_mse={rel:.8e}")
    qdq = dequant_qkv(itrace)
    for name, value in zip(("q_qdq", "k_qdq", "v_qdq"), qdq):
        base = name[:1]
        cos, mse, rel = metrics(value, ftrace[base])
        print(f"{name:14s} cos={cos:.8f} mse={mse:.8e} rel_mse={rel:.8e}")
        cos, mse, rel = metrics(value, itrace[base])
        print(f"{name + '_loss':14s} cos={cos:.8f} mse={mse:.8e} rel_mse={rel:.8e}")
    if itrace["prob_i16"] is not None:
        prob_ref, pv_ref = attention_refs(itrace)
        cos, mse, rel = metrics(itrace["prob_i16"].float() / 16383.0, prob_ref)
        print(f"{'softmax_i16':14s} cos={cos:.8f} mse={mse:.8e} rel_mse={rel:.8e}")
        cos, mse, rel = metrics(itrace["attn"], pv_ref)
        print(f"{'pv_i16v8':14s} cos={cos:.8f} mse={mse:.8e} rel_mse={rel:.8e}")


if __name__ == "__main__":
    main()
