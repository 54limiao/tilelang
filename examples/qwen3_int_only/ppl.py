import argparse
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.qwen3_int_only.model import Qwen3FloatModel, Qwen3IntOnlyModel


TEXT_PATH = Path(__file__).resolve().parent / "data" / "declaration_of_independence.txt"


def input_tokens(tokenizer, max_tokens, device):
    ids = tokenizer(TEXT_PATH.read_text(encoding="utf-8"), add_special_tokens=False).input_ids[:max_tokens]
    return torch.tensor(ids, device=device, dtype=torch.long)


def ppl_from_logits(logits, labels):
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return math.exp(float(loss)), float(loss), int(labels.numel())


@torch.no_grad()
def hf_ppl(model_dir, max_tokens):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    ids = input_tokens(tokenizer, max_tokens, model.device).unsqueeze(0)
    return ppl_from_logits(model(ids[:, :-1]).logits.float().reshape(-1, model.config.vocab_size), ids[:, 1:].reshape(-1))


@torch.no_grad()
def local_float_ppl(model_dir, max_tokens, layers, verbose):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    ids = input_tokens(tokenizer, max_tokens, "cuda")
    model = Qwen3FloatModel(ids.numel() - 1, model_dir=model_dir)
    return ppl_from_logits(model.logits(ids[:-1], layers=layers, verbose=verbose).float(), ids[1:])


@torch.no_grad()
def int_only_ppl(model_dir, max_tokens, layers, verbose):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    ids = input_tokens(tokenizer, max_tokens, "cuda")
    model = Qwen3IntOnlyModel(ids.numel() - 1, model_dir=model_dir)
    return ppl_from_logits(model.logits(ids[:-1], layers=layers, verbose=verbose).float(), ids[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/code/Qwen3-0.6B")
    parser.add_argument("--backend", choices=["hf", "local-float", "int-only"], default="local-float")
    parser.add_argument("--max-tokens", type=int, default=2049)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.backend == "hf":
        ppl, loss, ntokens = hf_ppl(args.model_dir, args.max_tokens)
    elif args.backend == "local-float":
        ppl, loss, ntokens = local_float_ppl(args.model_dir, args.max_tokens, args.layers, args.verbose)
    else:
        ppl, loss, ntokens = int_only_ppl(args.model_dir, args.max_tokens, args.layers, args.verbose)
    print(f"backend={args.backend} tokens={ntokens} loss={loss:.6f} ppl={ppl:.6f}")


if __name__ == "__main__":
    main()
