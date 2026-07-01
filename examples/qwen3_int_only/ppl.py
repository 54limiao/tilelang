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
FINEWEB_PATH = "/publicdata/huggingface.co/datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet"


def packed_flags(packed_dir):
    if packed_dir is None:
        return False, False
    with safe_open(f"{packed_dir}/qwen3_int_only.safetensors", framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    return metadata.get("use_r1") == "1", metadata.get("use_r2") == "1"


def load_ids(tokenizer, args, total_tokens, device):
    if args.eval_parquet:
        import pyarrow.parquet as pq

        ids = []
        parquet = pq.ParquetFile(args.eval_parquet)
        for batch in parquet.iter_batches(batch_size=256, columns=[args.eval_column]):
            for item in batch.column(args.eval_column).to_pylist():
                if item:
                    ids.extend(tokenizer(str(item), add_special_tokens=False).input_ids)
                    if len(ids) >= total_tokens:
                        return torch.tensor(ids[:total_tokens], device=device, dtype=torch.long)
    text = Path(args.eval_text).read_text(encoding="utf-8")
    ids = tokenizer(text, add_special_tokens=False).input_ids[:total_tokens]
    return torch.tensor(ids, device=device, dtype=torch.long)


def eval_windows(tokenizer, args, device):
    windows = args.batch_size * args.num_batches
    ids = load_ids(tokenizer, args, args.max_tokens * windows, device)
    if ids.numel() < args.max_tokens:
        return ids[None, :]
    windows = min(windows, ids.numel() // args.max_tokens)
    return ids[: args.max_tokens * windows].reshape(windows, args.max_tokens)


def quant_i8_q15_16(x):
    xq = torch.clamp(torch.round(x * Q15_16), -(1 << 31), (1 << 31) - 1).to(torch.int32)
    scale = torch.div(xq.abs().amax(dim=-1) + 126, 127, rounding_mode="floor").clamp(min=1).to(torch.uint32)
    y = torch.div(xq, scale.int()[..., None], rounding_mode="floor").clamp(-128, 127).to(torch.int8)
    return y, scale


def add_metrics(acc, logits, labels, golden=None):
    logits = logits.float()
    labels = labels.reshape(-1)
    flat = logits.reshape(-1, logits.shape[-1])
    loss_sum = torch.nn.functional.cross_entropy(flat, labels, reduction="sum")
    acc["loss_sum"] += float(loss_sum)
    acc["tokens"] += int(labels.numel())
    if golden is not None:
        ref = golden.float().reshape(-1)
        got = logits.reshape(-1)
        diff = got - ref
        acc["dot"] += float(torch.dot(got, ref))
        acc["got2"] += float(torch.dot(got, got))
        acc["ref2"] += float(torch.dot(ref, ref))
        acc["se"] += float(torch.dot(diff, diff))
        acc["logits"] += int(got.numel())


def finish_metrics(acc):
    loss = acc["loss_sum"] / acc["tokens"]
    out = {"tokens": acc["tokens"], "loss": loss, "ppl": math.exp(loss)}
    if acc["logits"]:
        out["cos"] = acc["dot"] / math.sqrt(max(acc["got2"] * acc["ref2"], 1e-30))
        out["mse"] = acc["se"] / acc["logits"]
        out["rel_mse"] = acc["se"] / max(acc["ref2"], 1e-30)
    return out


def print_metrics(backend, metrics, compare_backend):
    msg = f"backend={backend} tokens={metrics['tokens']} loss={metrics['loss']:.6f} ppl={metrics['ppl']:.6f}"
    if "cos" in metrics:
        msg += f" compare={compare_backend} cos={metrics['cos']:.8f} mse={metrics['mse']:.8e} rel_mse={metrics['rel_mse']:.8e}"
    print(msg)


def parse_layer_sweep(value, default_layers):
    if not value:
        return [default_layers]
    if value == "all":
        return list(range(1, default_layers + 1))
    layers = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        layers.append(default_layers if item == "full" else int(item))
    return layers


@torch.no_grad()
def hf_logits(model, windows, cache_prompt, tokenizer):
    if cache_prompt:
        prefix = torch.tensor(tokenizer(cache_prompt, add_special_tokens=False).input_ids, device=windows.device, dtype=torch.long)
        prefix = prefix[None, :].expand(windows.shape[0], prefix.numel())
        full = torch.cat((prefix, windows), dim=1)
        return model(full[:, :-1]).logits.float()[:, prefix.shape[1] :]
    return model(windows[:, :-1]).logits.float()


@torch.no_grad()
def local_float_logits(model, window, layers, verbose):
    return model.logits(window[:-1], layers=layers, verbose=verbose).float()


@torch.no_grad()
def build_cache_kv(hf_model, tokenizer, cache_prompt, layers, use_r2, use_r3):
    if not cache_prompt:
        return None, 0
    cache_ids = torch.tensor(tokenizer(cache_prompt, add_special_tokens=False).input_ids, device="cuda", dtype=torch.long)
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
    return cache_kv, int(cache_ids.numel())


@torch.no_grad()
def prepare_eval(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    windows = eval_windows(tokenizer, args, "cuda")
    seq_len = windows.shape[1] - 1
    layer_sweep = parse_layer_sweep(args.layer_sweep, QWEN3_0_6B.num_hidden_layers if args.layers is None else args.layers)
    need_hf = args.backend == "hf" or args.compare_backend == "hf" or (args.backend == "int-only" and args.cache_prompt)
    hf_model = None
    if need_hf:
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16
        ).to("cuda")
    local_model = None
    if args.backend == "local-float" or args.compare_backend == "local-float":
        local_model = Qwen3FloatModel(seq_len, model_dir=args.model_dir)

    cache_kv, cache_len = None, 0
    if args.backend == "int-only":
        _packed_r1, packed_r2 = packed_flags(args.packed_dir)
        cache_kv, cache_len = build_cache_kv(
            hf_model, tokenizer, args.cache_prompt, max(layer_sweep), args.use_r2 or packed_r2, args.use_r3
        )
        int_model = Qwen3IntOnlyModel(
            seq_len,
            model_dir=args.model_dir,
            packed_dir=args.packed_dir,
            cache_len=cache_len,
            use_r3=args.use_r3,
            split_attn=args.split_attn,
        )
    else:
        int_model = None
    return tokenizer, windows, hf_model, local_model, int_model, cache_kv, layer_sweep


@torch.no_grad()
def run_eval(args, tokenizer, windows, hf_model, local_model, int_model, cache_kv, layers):
    acc = {"loss_sum": 0.0, "tokens": 0, "dot": 0.0, "got2": 0.0, "ref2": 0.0, "se": 0.0, "logits": 0}
    for start in range(0, windows.shape[0], args.batch_size):
        batch = windows[start : start + args.batch_size]
        golden = None
        if args.compare_backend == "hf":
            golden = hf_logits(hf_model, batch, args.cache_prompt, tokenizer)
        elif args.compare_backend == "local-float":
            golden = torch.stack([local_float_logits(local_model, row, layers, False) for row in batch])

        if args.backend == "hf":
            logits = hf_logits(hf_model, batch, args.cache_prompt, tokenizer)
            add_metrics(acc, logits, batch[:, 1:], golden)
        elif args.backend == "local-float":
            for idx, row in enumerate(batch):
                logits = local_float_logits(local_model, row, layers, args.verbose)
                ref = None if golden is None else golden[idx]
                add_metrics(acc, logits, row[1:], ref)
        else:
            for idx, row in enumerate(batch):
                logits = int_model.logits(row[:-1], layers=layers, verbose=args.verbose, cache_kv=cache_kv).float()
                ref = None if golden is None else golden[idx]
                add_metrics(acc, logits, row[1:], ref)
    return finish_metrics(acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/code/Qwen3-0.6B")
    parser.add_argument("--packed-dir")
    parser.add_argument("--backend", choices=["hf", "local-float", "int-only"], default="local-float")
    parser.add_argument("--compare-backend", choices=["none", "hf", "local-float"], default="none")
    parser.add_argument("--max-tokens", type=int, default=2049)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--eval-text", default=str(TEXT_PATH))
    parser.add_argument("--eval-parquet", default="")
    parser.add_argument("--eval-column", default="text")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--layer-sweep", default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cache-prompt")
    parser.add_argument("--use-r1", action="store_true")
    parser.add_argument("--use-r2", action="store_true")
    parser.add_argument("--use-r3", action="store_true")
    parser.add_argument("--split-attn", action="store_true")
    args = parser.parse_args()
    if args.eval_parquet == "fineweb":
        args.eval_parquet = FINEWEB_PATH
    tokenizer, windows, hf_model, local_model, int_model, cache_kv, layer_sweep = prepare_eval(args)
    for layers in layer_sweep:
        metrics = run_eval(args, tokenizer, windows, hf_model, local_model, int_model, cache_kv, layers)
        if args.layer_sweep:
            print(f"layers={layers} ", end="")
        print_metrics(args.backend, metrics, args.compare_backend)


if __name__ == "__main__":
    main()
