import argparse
import gzip
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.qwen3_int_only.model import (
    Q15_16,
    QWEN3_0_6B,
    Qwen3IntOnlyModel,
)
from examples.qwen3_int_only.utils import ROTATE_SEED, random_hadamard_rotation


DEFAULT_MODEL_DIR = "/publicdata/huggingface.co/Qwen/Qwen3-0.6B"
FINEWEB_PATH = "/publicdata/huggingface.co/datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet"
C4_PATH = "/publicdata/huggingface.co/datasets/allenai/c4/en/c4-train.00000-of-01024.json.gz"
DATASETS = {
    "fineweb": ("parquet", FINEWEB_PATH),
    "c4": ("jsonl.gz", C4_PATH),
}


def packed_flags(packed_dir):
    if packed_dir is None:
        return False, False
    with safe_open(f"{packed_dir}/qwen3_int_only.safetensors", framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    return metadata.get("use_r1") == "1", metadata.get("use_r2") == "1"


def resolve_dataset(name):
    return DATASETS.get(name, ("", name))


def iter_texts(source, column):
    kind, path = resolve_dataset(source)
    if kind == "parquet" or path.endswith(".parquet"):
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=[column]):
            for item in batch.column(column).to_pylist():
                if item:
                    yield str(item)
    elif kind == "jsonl.gz" or path.endswith(".json.gz") or path.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line).get(column)
                if item:
                    yield str(item)
    else:
        yield Path(path).read_text(encoding="utf-8")


def load_ids(tokenizer, args, total_tokens, device):
    ids = []
    source = args.eval_parquet or args.eval_dataset or args.eval_text
    column = args.eval_column
    for text in iter_texts(source, column):
        ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
        if len(ids) >= total_tokens:
            return torch.tensor(ids[:total_tokens], device=device, dtype=torch.long)
    return torch.tensor(ids[:total_tokens], device=device, dtype=torch.long)


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
    y = torch.div(xq.abs() + (scale.int()[..., None] >> 1), scale.int()[..., None], rounding_mode="floor")
    y = torch.where(xq < 0, -y, y).clamp(-128, 127).to(torch.int8)
    return y, scale


def new_acc():
    return {
        "loss_sum": 0.0,
        "tokens": 0,
        "dot": 0.0,
        "got2": 0.0,
        "ref2": 0.0,
        "se": 0.0,
        "ae": 0.0,
        "max_abs": 0.0,
        "logits": 0,
    }


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
        acc["ae"] += float(torch.sum(torch.abs(diff)))
        acc["max_abs"] = max(acc["max_abs"], float(torch.max(torch.abs(diff))))
        acc["logits"] += int(got.numel())


def finish_metrics(acc):
    loss = acc["loss_sum"] / acc["tokens"]
    out = {"tokens": acc["tokens"], "loss": loss, "ppl": math.exp(loss)}
    if acc["logits"]:
        out["cos"] = acc["dot"] / math.sqrt(max(acc["got2"] * acc["ref2"], 1e-30))
        out["mse"] = acc["se"] / acc["logits"]
        out["mae"] = acc["ae"] / acc["logits"]
        out["max_abs"] = acc["max_abs"]
        out["rel_mse"] = acc["se"] / max(acc["ref2"], 1e-30)
    return out


def print_metrics(backend, metrics, compare_backend):
    msg = f"backend={backend} tokens={metrics['tokens']} loss={metrics['loss']:.6f} ppl={metrics['ppl']:.6f}"
    if "cos" in metrics:
        msg += (
            f" compare={compare_backend} cos={metrics['cos']:.8f} mse={metrics['mse']:.8e}"
            f" mae={metrics['mae']:.8e} max_abs={metrics['max_abs']:.8e} rel_mse={metrics['rel_mse']:.8e}"
        )
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

    cache_kv, cache_len = None, 0
    if args.backend == "int-only":
        _packed_r1, packed_r2 = packed_flags(args.packed_dir)
        cache_kv, cache_len = build_cache_kv(
            hf_model, tokenizer, args.cache_prompt, max(layer_sweep), args.use_r2 or packed_r2, True
        )
        int_model = Qwen3IntOnlyModel(
            seq_len,
            model_dir=args.model_dir,
            packed_dir=args.packed_dir,
            cache_len=cache_len,
            use_r3=True,
            fast_hadamard=True,
        )
    else:
        int_model = None
    return tokenizer, windows, hf_model, int_model, cache_kv, layer_sweep


@torch.no_grad()
def run_eval(args, tokenizer, windows, hf_model, int_model, cache_kv, layers):
    acc = new_acc()
    window_rows = []
    for start in range(0, windows.shape[0], args.batch_size):
        batch = windows[start : start + args.batch_size]
        golden = None
        if args.compare_backend == "hf":
            golden = hf_logits(hf_model, batch, args.cache_prompt, tokenizer)

        if args.backend == "hf":
            logits = hf_logits(hf_model, batch, args.cache_prompt, tokenizer)
            add_metrics(acc, logits, batch[:, 1:], golden)
            if args.jsonl_windows:
                for idx in range(batch.shape[0]):
                    wacc = new_acc()
                    ref = None if golden is None else golden[idx]
                    add_metrics(wacc, logits[idx], batch[idx, 1:], ref)
                    window_rows.append((start + idx, finish_metrics(wacc)))
        else:
            for idx, row in enumerate(batch):
                logits = int_model.logits(row[:-1], layers=layers, verbose=args.verbose, cache_kv=cache_kv).float()
                ref = None if golden is None else golden[idx]
                add_metrics(acc, logits, row[1:], ref)
                if args.jsonl_windows:
                    wacc = new_acc()
                    add_metrics(wacc, logits, row[1:], ref)
                    window_rows.append((start + idx, finish_metrics(wacc)))
    return finish_metrics(acc), window_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--packed-dir", default="/tmp/Qwen3-0.6B-static-calib-32x2048")
    parser.add_argument("--backend", choices=["hf", "int-only"], default="int-only")
    parser.add_argument("--compare-backend", choices=["none", "hf"], default="hf")
    parser.add_argument("--max-tokens", type=int, default=2049)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--eval-text", default="")
    parser.add_argument("--eval-dataset", default="fineweb")
    parser.add_argument("--eval-parquet", default="")
    parser.add_argument("--eval-column", default="text")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--layer-sweep", default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cache-prompt", default="你是一个有用而无害的聊天助手。")
    parser.add_argument("--use-r1", action="store_true")
    parser.add_argument("--use-r2", action="store_true")
    parser.add_argument("--jsonl-out", default="")
    parser.add_argument("--jsonl-windows", action="store_true")
    args = parser.parse_args()
    tokenizer, windows, hf_model, int_model, cache_kv, layer_sweep = prepare_eval(args)
    for layers in layer_sweep:
        metrics, window_rows = run_eval(args, tokenizer, windows, hf_model, int_model, cache_kv, layers)
        if args.layer_sweep:
            print(f"layers={layers} ", end="")
        print_metrics(args.backend, metrics, args.compare_backend)
        if args.jsonl_out:
            common = {
                "backend": args.backend,
                "compare_backend": args.compare_backend,
                "layers": layers,
                "max_tokens": args.max_tokens,
                "batch_size": args.batch_size,
                "num_batches": args.num_batches,
                "eval_dataset": args.eval_dataset,
                "eval_parquet": args.eval_parquet,
                "eval_text": args.eval_text,
                "eval_column": args.eval_column,
                "use_r1": args.use_r1,
                "use_r2": args.use_r2,
                "use_r3": True,
                "fused_static": True,
                "fast_hadamard": True,
                "static_mlp": True,
            }
            row = {
                "kind": "summary",
                **common,
                **metrics,
            }
            with open(args.jsonl_out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                for window, values in window_rows:
                    f.write(json.dumps({"kind": "window", "window": window, **common, **values}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
