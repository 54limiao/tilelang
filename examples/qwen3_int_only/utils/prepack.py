import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers import AutoTokenizer

from examples.qwen3_int_only.utils.ppl import iter_texts
from examples.qwen3_int_only.utils import (
    ROTATE_SEED,
    Q15_16,
    Qwen3Config,
    SafeTensorReader,
    per_channel_i8_weight,
    q15_16,
    hadamard_rotation,
    fast_hadamard,
    random_hadamard_rotation,
    rmsnorm_torch,
    rope_tables,
    rope_torch,
    rotate_block_input,
    rotate_head_input,
    rotate_head_output,
    rotate_input,
    rotate_norm_input,
    rotate_output,
)


DEFAULT_MODEL_DIR = "/publicdata/huggingface.co/Qwen/Qwen3-0.6B"
FLEX_BLOCK = 128
FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)
FLEX_MASKS = {}


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def fused_causal_attention(q, k, v):
    seq_len = q.shape[-2]
    if seq_len < FLEX_BLOCK or seq_len % FLEX_BLOCK:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    key = (q.device, q.shape[0], q.shape[1], seq_len)
    if key not in FLEX_MASKS:
        FLEX_MASKS[key] = create_block_mask(causal_mask, q.shape[0], q.shape[1], seq_len, seq_len, device=q.device, BLOCK_SIZE=FLEX_BLOCK)
    return FLEX_ATTENTION(q, k, v, block_mask=FLEX_MASKS[key], enable_gqa=True)


def update_head_amax(acc, x):
    cur = x.abs().amax(dim=(0, 2)).to(torch.int64)
    return cur if acc is None else torch.maximum(acc, cur)


def update_tensor_amax(acc, x):
    cur = x.abs().amax().reshape(1).to(torch.int64)
    return cur if acc is None else torch.maximum(acc, cur)


def update_stats_amax(acc, stats):
    out = {} if acc is None else acc
    for name, value in stats.items():
        if name in ("input_qkv_i8", "attn_i8", "post_mlp_i8", "gated_mlp_i8"):
            out[name] = update_tensor_amax(out.get(name), value)
        else:
            out[name] = update_head_amax(out.get(name), value)
    return out


def scale_from_amax(amax, qmax):
    return (amax.float() / float(qmax * Q15_16)).clamp(min=1.0 / Q15_16).to(torch.float32)


def scales_from_amax(layer):
    return {
        "q_pre_rope_i16": scale_from_amax(layer["q_pre_rope_i16"], 32767),
        "k_pre_rope_i16": scale_from_amax(layer["k_pre_rope_i16"], 32767),
        "input_qkv_i8": scale_from_amax(layer["input_qkv_i8"], 127),
        "q_post_rope_i8": scale_from_amax(layer["q_post_rope_i8"], 127),
        "k_post_rope_i8": scale_from_amax(layer["k_post_rope_i8"], 127),
        "v_i8": scale_from_amax(layer["v_i8"], 127),
        "attn_i8": scale_from_amax(layer["attn_i8"], 127),
        "post_mlp_i8": scale_from_amax(layer["post_mlp_i8"], 127),
        "gated_mlp_i8": scale_from_amax(layer["gated_mlp_i8"], 127),
    }


def load_calib_ids(model_dir, calib_text, calib_dataset, calib_parquet, calib_column, tokens, device, cache_prompt=""):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    ids = []
    prefix = tokenizer(cache_prompt, add_special_tokens=False).input_ids if cache_prompt else []
    source = calib_parquet or calib_dataset or calib_text
    for text in iter_texts(source, calib_column):
        ids.extend(prefix)
        ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
        if len(ids) >= tokens:
            return torch.tensor(ids[:tokens], device=device, dtype=torch.long)
    return torch.tensor(ids, device=device, dtype=torch.long)


def resolve_calib_shape(args):
    if args.calib_seq_len and args.calib_batches:
        return args.calib_seq_len * args.calib_batches, args.calib_seq_len
    if args.calib_tokens:
        return args.calib_tokens, args.calib_tokens
    return 0, args.calib_seq_len


def write_timestamp(path, args, calib_tokens, calib_seq_len, metadata=None):
    def value(name, default):
        return (metadata or {}).get(name, str(default))

    path.write_text(
        "\n".join(
            (
                f"packed_at_utc={datetime.now(timezone.utc).isoformat()}",
                f"model_dir={args.model_dir}",
                f"calib_dataset={value('calib_dataset', args.calib_dataset)}",
                f"calib_parquet={value('calib_parquet', args.calib_parquet or '')}",
                f"calib_tokens={value('calib_tokens', calib_tokens)}",
                f"calib_seq_len={value('calib_seq_len', calib_seq_len)}",
                f"calib_batches={value('calib_batches', args.calib_batches)}",
                f"calib_prefix_tokens={value('calib_prefix_tokens', args.calib_prefix_tokens)}",
                f"cache_prompt={value('cache_prompt', args.cache_prompt)}",
                f"packed_layers={value('packed_layers', args.max_layers or 'all')}",
                f"use_r1={value('use_r1', int(args.use_r1))}",
                f"use_r2={value('use_r2', int(args.use_r2))}",
                f"r2_impl={value('r2_impl', 'per_layer')}",
                f"r3_impl={value('r3_impl', 'fwht')}",
                f"weight_scale_dtype={value('weight_scale_dtype', 'fp32')}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def current_pack_metadata(path, args, config):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        metadata = f.metadata() or {}
    if metadata.get("model_dir") != args.model_dir:
        return None
    if metadata.get("hidden_size") != str(config.hidden_size) or metadata.get("num_hidden_layers") != str(config.num_hidden_layers):
        return None
    if metadata.get("weight_scale_dtype") != "fp32":
        return None
    if metadata.get("r2_impl") != "per_layer":
        return None
    if metadata.get("r3_impl") != "fwht":
        return None
    expected_layers = str(args.max_layers or config.num_hidden_layers)
    if metadata.get("packed_layers") != expected_layers:
        return None
    ok = all(
        key in keys
        for key in (
            "layers.0.qkv_proj.weight",
            "layers.0.qkv_proj.scale",
            "layers.0.gate_up_proj.weight",
            "layers.0.gate_up_proj.scale",
            "layers.0.q_post_rope_i8.scale",
            "layers.0.input_qkv_i8.scale",
            "layers.0.k_post_rope_i8.scale",
            "layers.0.v_i8.scale",
            "layers.0.attn_i8.scale",
            "layers.0.post_mlp_i8.scale",
            "layers.0.gated_mlp_i8.scale",
            "quarot.r1",
            "quarot.r4",
            "layers.0.r2",
        )
    )
    return metadata if ok else None


@torch.no_grad()
def run_calib_segment(x, w, norms, cos, sin, config, prefix_tokens):
    input_norm, post_norm, q_norm, k_norm = norms
    h = rmsnorm_torch(x, input_norm)
    q = h @ w["q_proj"].T
    k = h @ w["k_proj"].T
    v = h @ w["v_proj"].T
    batch, seq_len = x.shape[:2]
    cos_b = cos[None, :, :].expand(batch, seq_len, config.head_dim // 2).reshape(batch * seq_len, config.head_dim // 2)
    sin_b = sin[None, :, :].expand(batch, seq_len, config.head_dim // 2).reshape(batch * seq_len, config.head_dim // 2)
    q = rmsnorm_torch(q.reshape(batch, seq_len, config.num_attention_heads, config.head_dim), q_norm)
    k = rmsnorm_torch(k.reshape(batch, seq_len, config.num_key_value_heads, config.head_dim), k_norm)
    q_rope = rope_torch(q.reshape(batch * seq_len, config.q_size), cos_b, sin_b, config.num_attention_heads, config.head_dim).reshape(batch, seq_len, config.num_attention_heads, config.head_dim)
    k_rope = rope_torch(k.reshape(batch * seq_len, config.kv_size), cos_b, sin_b, config.num_key_value_heads, config.head_dim).reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)
    q_rope = fast_hadamard(q_rope)
    k_rope = fast_hadamard(k_rope)
    v = v.reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)
    stats = {
        "input_qkv_i8": q15_16(h[:, prefix_tokens:]),
        "q_pre_rope_i16": q15_16(q).reshape(-1, config.num_attention_heads, config.head_dim),
        "k_pre_rope_i16": q15_16(k).reshape(-1, config.num_key_value_heads, config.head_dim),
        "q_post_rope_i8": q15_16(q_rope).reshape(-1, config.num_attention_heads, config.head_dim),
        "k_post_rope_i8": q15_16(k_rope).reshape(-1, config.num_key_value_heads, config.head_dim),
        "v_i8": q15_16(v).reshape(-1, config.num_key_value_heads, config.head_dim),
    }
    q_attn = q_rope.permute(0, 2, 1, 3)
    k_attn = k_rope.permute(0, 2, 1, 3)
    v_attn = v.permute(0, 2, 1, 3)
    attn = fused_causal_attention(q_attn, k_attn, v_attn)
    attn = attn.permute(0, 2, 1, 3).reshape(batch, seq_len, config.q_size)
    stats["attn_i8"] = q15_16(attn[:, prefix_tokens:])
    x = x + attn @ w["o_proj"].T
    m = rmsnorm_torch(x, post_norm)
    gate = m @ w["gate_proj"].T
    up = m @ w["up_proj"].T
    gated = torch.nn.functional.silu(gate) * up
    gated_h = fast_hadamard(gated, config.head_dim)
    stats["post_mlp_i8"] = q15_16(m[:, prefix_tokens:])
    stats["gated_mlp_i8"] = q15_16(gated_h[:, prefix_tokens:])
    return x + gated_h @ w["down_proj"].T, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", default="/tmp/Qwen3-0.6B-static-calib")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-r1", action="store_true")
    parser.add_argument("--use-r2", action="store_true")
    parser.add_argument("--rotate-seed", type=int, default=ROTATE_SEED)
    parser.add_argument("--calib-text", default="")
    parser.add_argument("--calib-dataset", default="fineweb")
    parser.add_argument("--calib-parquet")
    parser.add_argument("--calib-column", default="text")
    parser.add_argument("--calib-tokens", type=int, default=0)
    parser.add_argument("--calib-seq-len", type=int, default=0)
    parser.add_argument("--calib-batches", type=int, default=0)
    parser.add_argument("--calib-micro-batch", type=int, default=1)
    parser.add_argument("--calib-prefix-tokens", type=int, default=0)
    parser.add_argument("--cache-prompt", default="")
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "qwen3_int_only.safetensors"
    timestamp = out_dir / "timestamp"
    calib_tokens, calib_seq_len = resolve_calib_shape(args)
    config = Qwen3Config.from_model_dir(args.model_dir)
    metadata = current_pack_metadata(path, args, config) if path.exists() and not args.force else None
    pack_layers = args.max_layers or config.num_hidden_layers
    if metadata is not None:
        if not timestamp.exists():
            write_timestamp(timestamp, args, calib_tokens, calib_seq_len, metadata)
        print(f"skip existing pack: {path}")
        print(timestamp.read_text(encoding="utf-8").strip())
        return

    tensors = {}
    r1 = random_hadamard_rotation(config.hidden_size, args.rotate_seed, args.device) if args.use_r1 else None
    down_hadamard = hadamard_rotation(config.head_dim, args.device)
    if args.use_r1:
        tensors["quarot.r1"] = r1.cpu().contiguous()
    else:
        tensors["quarot.r1"] = torch.empty((0,), dtype=torch.float32)
    tensors["quarot.r4"] = down_hadamard.cpu().contiguous()
    with SafeTensorReader(args.model_dir) as reader:
        def tensor(name, device=args.device, dtype=torch.float32):
            out = reader.get_tensor(name, device)
            return out if dtype is None else out.to(dtype)

        embed = tensor("model.embed_tokens.weight", "cpu" if not args.use_r1 else args.device, None)
        lm_head = tensor("lm_head.weight", "cpu" if not args.use_r1 else args.device, None)
        final_norm = tensor("model.norm.weight")
        if args.use_r1:
            embed = embed.to(torch.float32)
            lm_head = lm_head.to(torch.float32)
            embed = rotate_input(embed, r1)
            lm_head = rotate_norm_input(lm_head, final_norm, r1)
            final_norm = torch.ones_like(final_norm)
        tensors["model.embed_tokens.weight"] = embed.cpu().contiguous()
        tensors["lm_head.weight"] = lm_head.cpu().contiguous()
        tensors["model.norm.weight"] = q15_16(final_norm).cpu()
        calib_x = None
        cos = sin = None
        if calib_tokens:
            ids = load_calib_ids(args.model_dir, args.calib_text, args.calib_dataset, args.calib_parquet, args.calib_column, calib_tokens, args.device, args.cache_prompt)
            calib_x = embed.to(args.device, torch.float32)[ids.reshape(-1, calib_seq_len)]
            cos, sin, _ = rope_tables(calib_seq_len, config.head_dim, config.rope_theta, args.device)
        for layer_idx in range(pack_layers):
            r2 = random_hadamard_rotation(config.head_dim, args.rotate_seed + 1 + layer_idx, args.device) if args.use_r2 else None
            src = f"model.layers.{layer_idx}"
            dst = f"layers.{layer_idx}"
            tensors[f"{dst}.r2"] = r2.cpu().contiguous() if args.use_r2 else torch.empty((0,), dtype=torch.float32)
            input_norm = tensor(f"{src}.input_layernorm.weight")
            post_norm = tensor(f"{src}.post_attention_layernorm.weight")
            layer_float = {}
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                owner = "self_attn" if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                weight = tensor(f"{src}.{owner}.{name}.weight")
                if args.use_r1 and name in ("q_proj", "k_proj", "v_proj"):
                    weight = rotate_norm_input(weight, input_norm, r1)
                if args.use_r1 and name in ("gate_proj", "up_proj"):
                    weight = rotate_norm_input(weight, post_norm, r1)
                if args.use_r2 and name == "v_proj":
                    weight = rotate_head_output(weight, config.head_dim, r2)
                if args.use_r2 and name == "o_proj":
                    weight = rotate_head_input(weight, config.head_dim, r2)
                if args.use_r1 and name in ("o_proj", "down_proj"):
                    weight = rotate_output(weight, r1)
                if name == "down_proj":
                    weight = rotate_block_input(weight, config.head_dim, down_hadamard)
                layer_float[name] = weight
                w, s = per_channel_i8_weight(weight)
                tensors[f"{dst}.{name}.weight"] = w.cpu().contiguous()
                tensors[f"{dst}.{name}.scale"] = s.cpu().contiguous()
            for packed_name, parts in (
                ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
                ("gate_up_proj", ("gate_proj", "up_proj")),
            ):
                w, s = per_channel_i8_weight(torch.cat([layer_float[name] for name in parts], dim=0))
                tensors[f"{dst}.{packed_name}.weight"] = w.cpu().contiguous()
                tensors[f"{dst}.{packed_name}.scale"] = s.cpu().contiguous()
            if args.use_r1:
                input_norm = torch.ones_like(input_norm)
                post_norm = torch.ones_like(post_norm)
            tensors[f"{dst}.input_layernorm"] = q15_16(input_norm).cpu()
            tensors[f"{dst}.post_attention_layernorm"] = q15_16(post_norm).cpu()
            q_norm = tensor(f"{src}.self_attn.q_norm.weight")
            k_norm = tensor(f"{src}.self_attn.k_norm.weight")
            tensors[f"{dst}.q_norm"] = q15_16(q_norm).cpu()
            tensors[f"{dst}.k_norm"] = q15_16(k_norm).cpu()
            layer_float["down_hadamard"] = down_hadamard
            if calib_x is not None:
                next_x = torch.empty_like(calib_x)
                stats_amax = None
                micro = max(args.calib_micro_batch, 1)
                for start in range(0, calib_x.shape[0], micro):
                    end = min(start + micro, calib_x.shape[0])
                    next_x[start:end], stats = run_calib_segment(calib_x[start:end], layer_float, (input_norm, post_norm, q_norm, k_norm), cos, sin, config, args.calib_prefix_tokens)
                    stats_amax = update_stats_amax(stats_amax, stats)
                calib_x = next_x
                for name, scale in scales_from_amax(stats_amax).items():
                    tensors[f"{dst}.{name}.scale"] = scale.cpu().contiguous()

    save_file(
        tensors,
        str(path),
        metadata={
            "use_r1": str(int(args.use_r1)),
            "use_r2": str(int(args.use_r2)),
            "r2_impl": "per_layer",
            "r3_impl": "fwht",
            "model_dir": args.model_dir,
            "hidden_size": str(config.hidden_size),
            "intermediate_size": str(config.intermediate_size),
            "num_hidden_layers": str(config.num_hidden_layers),
            "num_attention_heads": str(config.num_attention_heads),
            "num_key_value_heads": str(config.num_key_value_heads),
            "head_dim": str(config.head_dim),
            "packed_layers": str(pack_layers),
            "calib_tokens": str(calib_tokens),
            "calib_dataset": args.calib_dataset,
            "calib_parquet": str(args.calib_parquet or ""),
            "calib_seq_len": str(calib_seq_len),
            "calib_batches": str(args.calib_batches),
            "calib_prefix_tokens": str(args.calib_prefix_tokens),
            "cache_prompt": args.cache_prompt,
            "weight_scale_dtype": "fp32",
        },
    )
    write_timestamp(timestamp, args, calib_tokens, calib_seq_len)
    print(path)
    print(timestamp)


if __name__ == "__main__":
    main()
