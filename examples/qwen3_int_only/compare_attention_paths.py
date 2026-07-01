import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from examples.qwen3_int_only.model import (
    QWEN3_0_6B,
    Qwen3IntOnlyBlock,
    load_all_qwen3_block_weights,
    load_embed_tokens,
    load_final_norm,
    load_lm_head,
    load_packed_qwen3,
    q15_16,
    rmsnorm_torch,
    rope_tables_q15_16,
)
from examples.qwen3_int_only.ppl import ppl_from_logits


TEXT_PATH = Path(__file__).resolve().parent / "data" / "declaration_of_independence.txt"


def metric(name, a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
    rel_mse = (torch.mean((af - bf) ** 2) / (torch.mean(af * af) + 1e-12)).item()
    print(f"{name:10s} cos={cos:.8f} rel_mse={rel_mse:.6e} amax_ref={a.abs().max().item()} amax_split={b.abs().max().item()}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/code/Qwen3-0.6B")
    parser.add_argument("--packed-dir", default="/tmp/Qwen3-0.6B-int-only-r12")
    parser.add_argument("--max-tokens", type=int, default=513)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    ids = torch.tensor(tok(TEXT_PATH.read_text(encoding="utf-8"), add_special_tokens=False).input_ids[: args.max_tokens], device="cuda")
    seq_len = ids.numel() - 1
    seq_len -= seq_len % 32
    ids = ids[: seq_len + 1]

    cfg = QWEN3_0_6B
    if args.packed_dir:
        embed, lm_head, final_norm, weights = load_packed_qwen3(args.packed_dir, cfg)
    else:
        embed = load_embed_tokens(args.model_dir)
        lm_head = load_lm_head(args.model_dir)
        final_norm = load_final_norm(args.model_dir)
        weights = load_all_qwen3_block_weights(args.model_dir, cfg)
    cos, sin, _ = rope_tables_q15_16(seq_len, cfg.head_dim, cfg.rope_theta)

    ref_block = Qwen3IntOnlyBlock(seq_len, cfg, split_attn=False)
    split_block = Qwen3IntOnlyBlock(seq_len, cfg, split_attn=True)
    x_ref = q15_16(embed[ids[:-1]])
    x_split = x_ref.clone()

    for layer_idx in range(args.layers):
        x_ref = ref_block(x_ref, weights[layer_idx], cos, sin)
        x_split = split_block(x_split, weights[layer_idx], cos, sin)
        metric(f"layer{layer_idx}", x_ref, x_split)

    norm_ref = rmsnorm_torch(x_ref.float() / 65536.0, final_norm.float() / 65536.0)
    norm_split = rmsnorm_torch(x_split.float() / 65536.0, final_norm.float() / 65536.0)
    logits_ref = norm_ref @ lm_head.T
    logits_split = norm_split @ lm_head.T
    labels = ids[1:]
    ppl_ref, loss_ref, _ = ppl_from_logits(logits_ref, labels)
    ppl_split, loss_split, _ = ppl_from_logits(logits_split, labels)
    print(f"default loss={loss_ref:.6f} ppl={ppl_ref:.6f}")
    print(f"split   loss={loss_split:.6f} ppl={ppl_split:.6f}")


if __name__ == "__main__":
    main()
