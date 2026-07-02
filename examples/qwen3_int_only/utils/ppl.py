import argparse
import gzip
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.qwen3_int_only.model import Q15_16, Qwen3IntOnlyModel
from examples.qwen3_int_only.model_hybrid import Qwen3HybridModel
from examples.qwen3_int_only.utils import ROTATE_SEED, Qwen3Config, random_hadamard_rotation


DEFAULT_MODEL_DIR = "/publicdata/huggingface.co/Qwen/Qwen3-0.6B"
FINEWEB_PATH = "/publicdata/huggingface.co/datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet"
C4_PATH = "/publicdata/huggingface.co/datasets/allenai/c4/en/c4-train.00000-of-01024.json.gz"
DATASETS = {
    "fineweb": ("parquet", FINEWEB_PATH),
    "c4": ("jsonl.gz", C4_PATH),
}


def packed_flag(packed_dir, name):
    with safe_open(f"{packed_dir}/qwen3_int_only.safetensors", framework="pt", device="cpu") as f:
        return (f.metadata() or {}).get(name) == "1"


def load_cache_scales(packed_dir, layers, device="cuda"):
    with safe_open(f"{packed_dir}/qwen3_int_only.safetensors", framework="pt", device="cpu") as f:
        return [
            (
                f.get_tensor(f"layers.{idx}.k_post_rope_i8.scale").to(device),
                f.get_tensor(f"layers.{idx}.v_i8.scale").to(device),
            )
            for idx in range(layers)
        ]


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
    if scale.dtype.is_floating_point:
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(-1)
        return torch.round(x / scale).clamp(-128, 127).to(torch.int8)
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
        got = logits.detach().cpu().reshape(-1)
        ref = golden.detach().float().cpu().reshape(-1)
        for start in range(0, got.numel(), 16 * 1024 * 1024):
            end = min(start + 16 * 1024 * 1024, got.numel())
            got_chunk = got[start:end]
            ref_chunk = ref[start:end]
            diff = got_chunk - ref_chunk
            acc["dot"] += float(torch.dot(got_chunk, ref_chunk))
            acc["got2"] += float(torch.dot(got_chunk, got_chunk))
            acc["ref2"] += float(torch.dot(ref_chunk, ref_chunk))
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


def load_hf_model(model_dir):
    print("loading HF model", file=sys.stderr, flush=True)
    kwargs = dict(local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16)
    if importlib.util.find_spec("accelerate") is not None:
        kwargs["device_map"] = {"": "cuda"}
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs).to("cuda")
    print("loaded HF model", file=sys.stderr, flush=True)
    return model


def hf_metrics(model, windows, cache_prompt, tokenizer):
    acc = new_acc()
    logits = hf_logits(model, windows, cache_prompt, tokenizer)
    add_metrics(acc, logits, windows[:, 1:])
    return finish_metrics(acc), logits.cpu()


@torch.no_grad()
def hf_logits(model, windows, cache_prompt, tokenizer):
    prefix = torch.tensor(tokenizer(cache_prompt, add_special_tokens=False).input_ids, device=windows.device, dtype=torch.long)
    full = torch.cat((prefix[None, :].expand(windows.shape[0], prefix.numel()), windows), dim=1)
    return model(full[:, :-1]).logits.float()[:, prefix.numel() :]


@torch.no_grad()
def build_cache_kv(hf_model, tokenizer, cache_prompt, cache_scales, use_r2, config):
    cache_ids = torch.tensor(tokenizer(cache_prompt, add_special_tokens=False).input_ids, device="cuda", dtype=torch.long)
    if cache_ids.numel() == 0:
        return None, 0
    past = hf_model(cache_ids[None, :], use_cache=True).past_key_values
    if hasattr(past, "layers"):
        past = [(layer.keys, layer.values) for layer in past.layers]
    elif hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    r2 = random_hadamard_rotation(config.head_dim, ROTATE_SEED + 1, "cuda") if use_r2 else None
    r3 = random_hadamard_rotation(config.head_dim, ROTATE_SEED + 2, "cuda")
    cache_kv = []
    for (k_scale, v_scale), (k, v) in zip(cache_scales, past[: len(cache_scales)]):
        k = (k[0].float().contiguous().to(torch.float64) @ r3.to(torch.float64)).to(torch.float32)
        v = v[0].float().contiguous()
        if r2 is not None:
            v = (v.to(torch.float64) @ r2.to(torch.float64)).to(torch.float32)
        kq = quant_i8_static_q15_16(k, k_scale[:, None])
        vq = quant_i8_static_q15_16(v, v_scale[:, None])
        cache_kv.append((kq, vq))
    return cache_kv, int(cache_ids.numel())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--packed-dir", default="/tmp/Qwen3-0.6B-static-calib-32x2048")
    parser.add_argument("--backend", choices=["hf", "int-only", "hybrid"], default="int-only")
    parser.add_argument("--compare-backend", choices=["none", "hf"], default="hf")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--eval-text", default="")
    parser.add_argument("--eval-dataset", default="fineweb")
    parser.add_argument("--eval-parquet", default="")
    parser.add_argument("--eval-column", default="text")
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--cache-prompt", default="你是一个有用而无害的聊天助手。")
    parser.add_argument("--use-r1", action="store_true")
    parser.add_argument("--use-r2", action="store_true")
    args = parser.parse_args()

    config = Qwen3Config.from_model_dir(args.model_dir)
    if args.layers == 0:
        args.layers = config.num_hidden_layers
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    window_tokens = args.max_tokens + 1
    ids = load_ids(tokenizer, args, window_tokens * args.batch_size * args.num_batches, "cuda")
    windows = ids[: (ids.numel() // window_tokens) * window_tokens].reshape(-1, window_tokens)
    windows = windows[: args.batch_size * args.num_batches]
    hf_model = load_hf_model(args.model_dir)

    if args.backend == "hf":
        metrics, _ = hf_metrics(hf_model, windows, args.cache_prompt, tokenizer)
        print_metrics("hf", metrics, "none")
        return

    print("running HF golden", file=sys.stderr, flush=True)
    if args.compare_backend == "hf":
        hf_stats, golden = hf_metrics(hf_model, windows, args.cache_prompt, tokenizer)
        print_metrics("hf", hf_stats, "none")
    else:
        golden = None
    use_r2 = args.use_r2 or packed_flag(args.packed_dir, "use_r2")
    use_r3 = packed_flag(args.packed_dir, "use_r3")
    cache_scales = load_cache_scales(args.packed_dir, args.layers)
    print("building cache kv", file=sys.stderr, flush=True)
    cache_kv, cache_len = build_cache_kv(hf_model, tokenizer, args.cache_prompt, cache_scales, use_r2, config)
    del hf_model, cache_scales
    torch.cuda.empty_cache()
    print(f"loading {args.backend} model", file=sys.stderr, flush=True)
    if args.backend == "hybrid":
        int_model = Qwen3HybridModel(windows.shape[1] - 1, model_dir=args.model_dir, packed_dir=args.packed_dir, config=config, cache_len=cache_len, layers=args.layers)
    else:
        int_model = Qwen3IntOnlyModel(windows.shape[1] - 1, model_dir=args.model_dir, packed_dir=args.packed_dir, config=config, cache_len=cache_len, layers=args.layers, use_r3=use_r3)
    print(f"running {args.backend} logits", file=sys.stderr, flush=True)
    acc = new_acc()
    for idx, row in enumerate(windows):
        logits = int_model.logits(row[:-1], layers=args.layers, cache_kv=cache_kv).float()
        add_metrics(acc, logits, row[1:], None if golden is None else golden[idx])
    print_metrics(args.backend, finish_metrics(acc), args.compare_backend)


if __name__ == "__main__":
    main()
