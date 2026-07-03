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

from examples.qwen3_int_only.model_int_only import Q15_16, Qwen3IntOnlyModel
from examples.qwen3_int_only.model_hybrid import Qwen3HybridModel
from examples.qwen3_int_only.utils import Qwen3Config, fast_hadamard, load_packed_qwen3, rmsnorm_torch, rope_tables, rope_torch


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


def pack_metadata(packed_dir):
    with safe_open(f"{packed_dir}/qwen3_int_only.safetensors", framework="pt", device="cpu") as f:
        return f.metadata() or {}


def load_cache_scales(packed_dir, layers, device="cuda"):
    with safe_open(f"{packed_dir}/qwen3_int_only.safetensors", framework="pt", device="cpu") as f:
        return [
            (
                f.get_tensor(f"layers.{idx}.k_post_rope_i8.scale").to(device),
                f.get_tensor(f"layers.{idx}.v_i8.scale").to(device),
                f.get_tensor(f"layers.{idx}.r2").to(device),
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


def qdq_i8(x, scale):
    while scale.ndim < x.ndim:
        scale = scale.unsqueeze(-1)
    return torch.round(x / scale).clamp(-128, 127) * scale


def linear_qdq(x, packed):
    w = packed.weight.float() * packed.scale.float()[:, None]
    return x.float() @ w.T


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


class Qwen3FakeQuantModel:
    def __init__(self, seq_len, model_dir, packed_dir, config, cache_len=0, layers=None):
        self.seq_len = seq_len
        self.cache_len = cache_len
        self.config = config
        self.cos, self.sin, _ = rope_tables(seq_len + cache_len, config.head_dim, config.rope_theta)
        packed = load_packed_qwen3(packed_dir, config, layers=layers)
        self.embed, self.lm_head, self.norm_weight, self.layers = packed[0], packed[4], packed[5], packed[6]

    def qk_norm_rope_qdq(self, x, weight, cos, sin, heads, scale):
        cfg = self.config
        x = rmsnorm_torch(x.reshape(self.seq_len, heads, cfg.head_dim), weight)
        x = rope_torch(x.reshape(self.seq_len, heads * cfg.head_dim), cos, sin, heads, cfg.head_dim)
        x = fast_hadamard(x.reshape(self.seq_len, heads, cfg.head_dim))
        return qdq_i8(x.permute(1, 0, 2).contiguous(), scale[:, None])

    def attention(self, q, k, v, cache_k, cache_v, weights):
        cfg = self.config
        group = cfg.num_attention_heads // cfg.num_key_value_heads
        if cache_k is not None:
            ks = weights.k_post_rope_i8_scale[:, None, None]
            vs = weights.v_i8_scale[:, None, None]
            k = torch.cat((cache_k.float() * ks, k), dim=1)
            v = torch.cat((cache_v.float() * vs, v), dim=1)
        k = k.repeat_interleave(group, dim=0)
        v = v.repeat_interleave(group, dim=0)
        score = torch.matmul(q, k.transpose(-1, -2)) / (cfg.head_dim**0.5)
        cur = torch.arange(self.seq_len, device=score.device)[:, None] + self.cache_len
        pos = torch.arange(score.shape[-1], device=score.device)[None, :]
        score = score.masked_fill(pos > cur, torch.finfo(score.dtype).min)
        out = torch.matmul(torch.softmax(score, dim=-1), v)
        out = out.permute(1, 0, 2).reshape(self.seq_len, cfg.q_size)
        return qdq_i8(out, weights.attn_i8_scale)

    def hidden(self, input_ids, layers=None, cache_kv=None):
        cfg = self.config
        n_layers = cfg.num_hidden_layers if layers is None else layers
        residual = self.embed[input_ids].float()
        for layer_idx in range(n_layers):
            weights = self.layers[layer_idx]
            h = rmsnorm_torch(residual, weights.input_layernorm)
            h = qdq_i8(h, weights.input_qkv_i8_scale)
            qkv = linear_qdq(h, weights.qkv_proj)
            q = qkv[:, : cfg.q_size].contiguous()
            k = qkv[:, cfg.q_size : cfg.q_size + cfg.kv_size].contiguous()
            v = qkv[:, cfg.q_size + cfg.kv_size :].reshape(self.seq_len, cfg.num_key_value_heads, cfg.head_dim)
            pos_cos = self.cos[self.cache_len : self.cache_len + self.seq_len]
            pos_sin = self.sin[self.cache_len : self.cache_len + self.seq_len]
            q = self.qk_norm_rope_qdq(q, weights.q_norm, pos_cos, pos_sin, cfg.num_attention_heads, weights.q_post_rope_i8_scale)
            k = self.qk_norm_rope_qdq(k, weights.k_norm, pos_cos, pos_sin, cfg.num_key_value_heads, weights.k_post_rope_i8_scale)
            v = qdq_i8(v.permute(1, 0, 2).contiguous(), weights.v_i8_scale[:, None])
            layer_cache = None if cache_kv is None else cache_kv[layer_idx]
            cache_k = None if layer_cache is None else layer_cache[0]
            cache_v = None if layer_cache is None else layer_cache[1]
            attn = self.attention(q, k, v, cache_k, cache_v, weights)
            residual = residual + linear_qdq(attn, weights.o_proj)
            h = rmsnorm_torch(residual, weights.post_attention_layernorm)
            h = qdq_i8(h, weights.post_mlp_i8_scale)
            gate_up = linear_qdq(h, weights.gate_up_proj)
            gate = gate_up[:, : cfg.intermediate_size]
            up = gate_up[:, cfg.intermediate_size :]
            gated = fast_hadamard(torch.nn.functional.silu(gate) * up, cfg.head_dim)
            gated = qdq_i8(gated, weights.gated_mlp_i8_scale)
            residual = residual + linear_qdq(gated, weights.down_proj)
        return rmsnorm_torch(residual, self.norm_weight)

    def logits(self, input_ids, layers=None, cache_kv=None):
        return self.hidden(input_ids, layers=layers, cache_kv=cache_kv) @ self.lm_head.float().T


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
    cache_kv = []
    for (k_scale, v_scale, r2), (k, v) in zip(cache_scales, past[: len(cache_scales)]):
        k = fast_hadamard(k[0].float().contiguous())
        v = v[0].float().contiguous()
        if use_r2 and r2.numel():
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
    parser.add_argument("--backend", choices=["hf", "fake-quant", "int-only", "hybrid"], default="int-only")
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
    metadata = pack_metadata(args.packed_dir)
    use_r2 = args.use_r2 or metadata.get("use_r2") == "1"
    cache_scales = load_cache_scales(args.packed_dir, args.layers)
    print("building cache kv", file=sys.stderr, flush=True)
    cache_kv, cache_len = build_cache_kv(hf_model, tokenizer, args.cache_prompt, cache_scales, use_r2, config)
    del hf_model, cache_scales
    torch.cuda.empty_cache()
    print(f"loading {args.backend} model", file=sys.stderr, flush=True)
    if args.backend == "fake-quant":
        int_model = Qwen3FakeQuantModel(windows.shape[1] - 1, args.model_dir, args.packed_dir, config, cache_len=cache_len, layers=args.layers)
    elif args.backend == "hybrid":
        int_model = Qwen3HybridModel(windows.shape[1] - 1, model_dir=args.model_dir, packed_dir=args.packed_dir, config=config, cache_len=cache_len, layers=args.layers)
    else:
        int_model = Qwen3IntOnlyModel(windows.shape[1] - 1, model_dir=args.model_dir, packed_dir=args.packed_dir, config=config, cache_len=cache_len, layers=args.layers)
    print(f"running {args.backend} logits", file=sys.stderr, flush=True)
    acc = new_acc()
    for idx, row in enumerate(windows):
        logits = int_model.logits(row[:-1], layers=args.layers, cache_kv=cache_kv).float()
        add_metrics(acc, logits, row[1:], None if golden is None else golden[idx])
    print_metrics(args.backend, finish_metrics(acc), args.compare_backend)


if __name__ == "__main__":
    main()
