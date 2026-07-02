import argparse
import gzip
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.qwen3_int_only.model import Q15_16, QWEN3_0_6B, Qwen3IntOnlyModel
from examples.qwen3_int_only.utils import ROTATE_SEED, load_packed_qwen3, random_hadamard_rotation


DEFAULT_MODEL_DIR = "/publicdata/huggingface.co/Qwen/Qwen3-0.6B"
FINEWEB_PATH = "/publicdata/huggingface.co/datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet"
C4_PATH = "/publicdata/huggingface.co/datasets/allenai/c4/en/c4-train.00000-of-01024.json.gz"
DATASETS = {
    "fineweb": ("parquet", FINEWEB_PATH),
    "c4": ("jsonl.gz", C4_PATH),
}


def packed_use_r2(packed_dir):
    with safe_open(f"{packed_dir}/qwen3_int_only.safetensors", framework="pt", device="cpu") as f:
        return (f.metadata() or {}).get("use_r2") == "1"


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
    for text in iter_texts(source, args.eval_column):
        ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
        if len(ids) >= total_tokens:
            break
    return torch.tensor(ids[:total_tokens], device=device, dtype=torch.long)


def quant_i8_static_q15_16(x, scale):
    xq = torch.clamp(torch.round(x * Q15_16), -(1 << 31), (1 << 31) - 1).to(torch.int32)
    y = torch.div(xq.abs() + (scale.int()[..., None] >> 1), scale.int()[..., None], rounding_mode="floor")
    return torch.where(xq < 0, -y, y).clamp(-128, 127).to(torch.int8)


def new_acc():
    return {"loss_sum": 0.0, "tokens": 0, "dot": 0.0, "got2": 0.0, "ref2": 0.0, "se": 0.0, "ae": 0.0, "max_abs": 0.0, "logits": 0}


def add_metrics(acc, logits, labels, golden=None):
    logits = logits.float()
    labels = labels.reshape(-1)
    loss_sum = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels, reduction="sum")
    acc["loss_sum"] += float(loss_sum)
    acc["tokens"] += int(labels.numel())
    if golden is not None:
        got = logits.reshape(-1)
        ref = golden.float().reshape(-1)
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
        out.update(
            cos=acc["dot"] / math.sqrt(max(acc["got2"] * acc["ref2"], 1e-30)),
            mse=acc["se"] / acc["logits"],
            mae=acc["ae"] / acc["logits"],
            max_abs=acc["max_abs"],
            rel_mse=acc["se"] / max(acc["ref2"], 1e-30),
        )
    return out


def print_metrics(backend, metrics, compare_backend="hf"):
    msg = f"backend={backend} tokens={metrics['tokens']} loss={metrics['loss']:.6f} ppl={metrics['ppl']:.6f}"
    if "cos" in metrics:
        msg += f" compare={compare_backend} cos={metrics['cos']:.8f} mse={metrics['mse']:.8e} mae={metrics['mae']:.8e} max_abs={metrics['max_abs']:.8e} rel_mse={metrics['rel_mse']:.8e}"
    print(msg)


@torch.no_grad()
def hf_logits(model, windows, cache_prompt, tokenizer):
    prefix = torch.tensor(tokenizer(cache_prompt, add_special_tokens=False).input_ids, device=windows.device, dtype=torch.long)
    full = torch.cat((prefix[None, :].expand(windows.shape[0], prefix.numel()), windows), dim=1)
    return model(full[:, :-1]).logits.float()[:, prefix.numel() :]


@torch.no_grad()
def build_cache_kv(hf_model, tokenizer, cache_prompt, layer_weights, use_r2):
    cache_ids = torch.tensor(tokenizer(cache_prompt, add_special_tokens=False).input_ids, device="cuda", dtype=torch.long)
    past = hf_model(cache_ids[None, :], use_cache=True).past_key_values
    if hasattr(past, "layers"):
        past = [(layer.keys, layer.values) for layer in past.layers]
    elif hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    r2 = random_hadamard_rotation(QWEN3_0_6B.head_dim, ROTATE_SEED + 1, "cuda") if use_r2 else None
    r3 = random_hadamard_rotation(QWEN3_0_6B.head_dim, ROTATE_SEED + 2, "cuda")
    cache_kv = []
    for weights, (k, v) in zip(layer_weights, past[: len(layer_weights)]):
        k = (k[0].float().contiguous().to(torch.float64) @ r3.to(torch.float64)).to(torch.float32)
        v = v[0].float().contiguous()
        if r2 is not None:
            v = (v.to(torch.float64) @ r2.to(torch.float64)).to(torch.float32)
        kq = quant_i8_static_q15_16(k, weights.k_post_rope_i8_scale[:, None])
        vq = quant_i8_static_q15_16(v, weights.v_i8_scale[:, None])
        cache_kv.append((kq, vq))
    return cache_kv, int(cache_ids.numel())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--packed-dir", default="/tmp/Qwen3-0.6B-static-calib-32x2048")
    parser.add_argument("--backend", choices=["hf", "int-only"], default="int-only")
    parser.add_argument("--compare-backend", choices=["none", "hf"], default="hf")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--eval-text", default="")
    parser.add_argument("--eval-dataset", default="fineweb")
    parser.add_argument("--eval-parquet", default="")
    parser.add_argument("--eval-column", default="text")
    parser.add_argument("--layers", type=int, default=QWEN3_0_6B.num_hidden_layers)
    parser.add_argument("--cache-prompt", default="你是一个有用而无害的聊天助手。")
    parser.add_argument("--use-r1", action="store_true")
    parser.add_argument("--use-r2", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    window_tokens = args.max_tokens + 1
    ids = load_ids(tokenizer, args, window_tokens * args.batch_size * args.num_batches, "cuda")
    windows = ids[: (ids.numel() // window_tokens) * window_tokens].reshape(-1, window_tokens)
    windows = windows[: args.batch_size * args.num_batches]
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")

    if args.backend == "hf":
        acc = new_acc()
        logits = hf_logits(hf_model, windows, args.cache_prompt, tokenizer)
        add_metrics(acc, logits, windows[:, 1:])
        print_metrics("hf", finish_metrics(acc), "none")
        return

    _, _, _, packed_layers = load_packed_qwen3(args.packed_dir, QWEN3_0_6B)
    cache_kv, cache_len = build_cache_kv(hf_model, tokenizer, args.cache_prompt, packed_layers[: args.layers], args.use_r2 or packed_use_r2(args.packed_dir))
    int_model = Qwen3IntOnlyModel(windows.shape[1] - 1, model_dir=args.model_dir, packed_dir=args.packed_dir, cache_len=cache_len)
    golden = hf_logits(hf_model, windows, args.cache_prompt, tokenizer) if args.compare_backend == "hf" else None
    acc = new_acc()
    for idx, row in enumerate(windows):
        logits = int_model.logits(row[:-1], layers=args.layers, cache_kv=cache_kv).float()
        add_metrics(acc, logits, row[1:], None if golden is None else golden[idx])
    print_metrics("int-only", finish_metrics(acc), args.compare_backend)


if __name__ == "__main__":
    main()
