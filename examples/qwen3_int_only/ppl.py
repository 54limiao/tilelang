import argparse
import math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.qwen3_int_only.model import (
    Q15_16,
    QWEN3_0_6B,
    Qwen3FloatModel,
    Qwen3IntOnlyModel,
)
from examples.qwen3_int_only.quarot import ROTATE_SEED, random_hadamard_rotation


TEXT_PATH = Path(__file__).resolve().parent / "data" / "declaration_of_independence.txt"


def packed_flags(packed_dir):
    if packed_dir is None:
        return False, False
    with safe_open(f"{packed_dir}/qwen3_int_only.safetensors", framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    return metadata.get("use_r1") == "1", metadata.get("use_r2") == "1"


def input_tokens(tokenizer, max_tokens, device):
    ids = tokenizer(TEXT_PATH.read_text(encoding="utf-8"), add_special_tokens=False).input_ids[:max_tokens]
    return torch.tensor(ids, device=device, dtype=torch.long)


def quant_i8_q15_16(x):
    xq = torch.clamp(torch.round(x * Q15_16), -(1 << 31), (1 << 31) - 1).to(torch.int32)
    scale = torch.div(xq.abs().amax(dim=-1), 127, rounding_mode="floor").clamp(min=1).to(torch.uint32)
    y = torch.div(xq, scale.int()[..., None], rounding_mode="floor").clamp(-128, 127).to(torch.int8)
    return y, scale


def ppl_from_logits(logits, labels):
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return math.exp(float(loss)), float(loss), int(labels.numel())


@torch.no_grad()
def hf_ppl(model_dir, max_tokens, cache_prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    ids = input_tokens(tokenizer, max_tokens, model.device).unsqueeze(0)
    if cache_prompt:
        prefix = torch.tensor(tokenizer(cache_prompt, add_special_tokens=False).input_ids, device=model.device, dtype=torch.long).unsqueeze(0)
        full = torch.cat((prefix, ids), dim=1)
        logits = model(full[:, :-1]).logits.float()[:, prefix.size(1) :]
    else:
        logits = model(ids[:, :-1]).logits.float()
    return ppl_from_logits(logits.reshape(-1, model.config.vocab_size), ids[:, 1:].reshape(-1))


@torch.no_grad()
def local_float_ppl(model_dir, max_tokens, layers, verbose):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    ids = input_tokens(tokenizer, max_tokens, "cuda")
    model = Qwen3FloatModel(ids.numel() - 1, model_dir=model_dir)
    return ppl_from_logits(model.logits(ids[:-1], layers=layers, verbose=verbose).float(), ids[1:])


@torch.no_grad()
def int_only_ppl(model_dir, packed_dir, max_tokens, layers, verbose, cache_prompt, use_r2, use_r3, split_attn):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    ids = input_tokens(tokenizer, max_tokens, "cuda")
    cache_kv = None
    cache_len = 0
    if cache_prompt:
        cache_ids = torch.tensor(tokenizer(cache_prompt, add_special_tokens=False).input_ids, device="cuda", dtype=torch.long)
        cache_len = int(cache_ids.numel())
        hf_model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
        past = hf_model(cache_ids[None, :], use_cache=True).past_key_values
        if hasattr(past, "layers"):
            past = [(layer.keys, layer.values) for layer in past.layers]
        elif hasattr(past, "to_legacy_cache"):
            past = past.to_legacy_cache()
        n_layers = QWEN3_0_6B.num_hidden_layers if layers is None else layers
        r2 = random_hadamard_rotation(QWEN3_0_6B.head_dim, ROTATE_SEED + 1, "cuda") if use_r2 else None
        r3 = random_hadamard_rotation(QWEN3_0_6B.head_dim, ROTATE_SEED + 2, "cuda") if use_r3 else None
        cache_kv = []
        for k, v in past[:n_layers]:
            k = k[0].float().contiguous()
            v = v[0].float().contiguous()
            if r3 is not None:
                k = (k.to(torch.float64) @ r3.to(torch.float64)).to(torch.float32)
            if r2 is not None:
                v = (v.to(torch.float64) @ r2.to(torch.float64)).to(torch.float32)
            cache_kv.append((quant_i8_q15_16(k), quant_i8_q15_16(v)))
        del hf_model
    model = Qwen3IntOnlyModel(ids.numel() - 1, model_dir=model_dir, packed_dir=packed_dir, cache_len=cache_len, use_r3=use_r3, split_attn=split_attn)
    return ppl_from_logits(model.logits(ids[:-1], layers=layers, verbose=verbose, cache_kv=cache_kv).float(), ids[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/code/Qwen3-0.6B")
    parser.add_argument("--packed-dir")
    parser.add_argument("--backend", choices=["hf", "local-float", "int-only"], default="local-float")
    parser.add_argument("--max-tokens", type=int, default=2049)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cache-prompt")
    parser.add_argument("--use-r1", action="store_true")
    parser.add_argument("--use-r2", action="store_true")
    parser.add_argument("--use-r3", action="store_true")
    parser.add_argument("--split-attn", action="store_true")
    args = parser.parse_args()

    if args.backend == "hf":
        ppl, loss, ntokens = hf_ppl(args.model_dir, args.max_tokens, args.cache_prompt)
    elif args.backend == "local-float":
        ppl, loss, ntokens = local_float_ppl(args.model_dir, args.max_tokens, args.layers, args.verbose)
    else:
        _packed_r1, packed_r2 = packed_flags(args.packed_dir)
        ppl, loss, ntokens = int_only_ppl(
            args.model_dir,
            args.packed_dir,
            args.max_tokens,
            args.layers,
            args.verbose,
            args.cache_prompt,
            args.use_r2 or packed_r2,
            args.use_r3,
            args.split_attn,
        )
    print(f"backend={args.backend} tokens={ntokens} loss={loss:.6f} ppl={ppl:.6f}")


if __name__ == "__main__":
    main()
