import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from examples.qwen3_int_only.utils.ppl import iter_texts
from examples.qwen3_int_only.utils import (
    ROTATE_SEED,
    QWEN3_0_6B,
    per_channel_i8_weight,
    q15_16,
    random_hadamard_rotation,
    rmsnorm_torch,
    rope_tables_q15_16,
    rope_torch,
    rotate_head_input,
    rotate_head_output,
    rotate_input,
    rotate_norm_input,
    rotate_output,
)


DEFAULT_MODEL_DIR = "/publicdata/huggingface.co/Qwen/Qwen3-0.6B"


def update_head_amax(acc, x):
    cur = x.abs().amax(dim=(0, 2)).to(torch.int64)
    return cur if acc is None else torch.maximum(acc, cur)


def update_tensor_amax(acc, x):
    cur = x.abs().amax().reshape(1).to(torch.int64)
    return cur if acc is None else torch.maximum(acc, cur)


def scale_from_amax(amax, qmax):
    return torch.div(amax + qmax - 1, qmax, rounding_mode="floor").clamp(min=1).to(torch.uint32)


def load_calib_ids(model_dir, calib_text, calib_dataset, calib_parquet, calib_column, tokens, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    ids = []
    source = calib_parquet or calib_dataset or calib_text
    for text in iter_texts(source, calib_column):
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
                f"use_r1={value('use_r1', int(args.use_r1))}",
                f"use_r2={value('use_r2', int(args.use_r2))}",
                f"use_r3={value('use_r3', int(args.use_r3))}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def current_pack_metadata(path):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        metadata = f.metadata() or {}
    ok = all(
        key in keys
        for key in (
            "layers.0.q_post_rope_i8.scale",
            "layers.0.k_post_rope_i8.scale",
            "layers.0.v_i8.scale",
            "layers.0.attn_i8.scale",
            "layers.0.post_mlp_i8.scale",
            "layers.0.gated_mlp_i16.scale",
        )
    )
    return metadata if ok else None


@torch.no_grad()
def run_calib_segment(x, w, norms, cos, sin, config, r3, prefix_tokens):
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
    if r3 is not None:
        q_rope = (q_rope.to(torch.float64) @ r3.to(torch.float64)).to(torch.float32)
        k_rope = (k_rope.to(torch.float64) @ r3.to(torch.float64)).to(torch.float32)
    v = v.reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)
    stats = {
        "q_pre_rope_i16": q15_16(q[:, prefix_tokens:]).reshape(-1, config.num_attention_heads, config.head_dim),
        "k_pre_rope_i16": q15_16(k[:, prefix_tokens:]).reshape(-1, config.num_key_value_heads, config.head_dim),
        "q_post_rope_i8": q15_16(q_rope[:, prefix_tokens:]).reshape(-1, config.num_attention_heads, config.head_dim),
        "k_post_rope_i8": q15_16(k_rope[:, prefix_tokens:]).reshape(-1, config.num_key_value_heads, config.head_dim),
        "v_i8": q15_16(v[:, prefix_tokens:]).reshape(-1, config.num_key_value_heads, config.head_dim),
    }
    group = config.num_attention_heads // config.num_key_value_heads
    q_attn = q_rope.permute(0, 2, 1, 3)
    k_attn = k_rope.repeat_interleave(group, dim=2).permute(0, 2, 1, 3)
    v_attn = v.repeat_interleave(group, dim=2).permute(0, 2, 1, 3)
    score = q_attn @ k_attn.transpose(-1, -2) / (config.head_dim**0.5)
    mask = torch.ones(score.shape[-2:], device=score.device, dtype=torch.bool).tril()
    attn = torch.softmax(score.masked_fill(~mask, torch.finfo(score.dtype).min), dim=-1) @ v_attn
    attn = attn.permute(0, 2, 1, 3).reshape(batch, seq_len, config.q_size)
    stats["attn_i8"] = q15_16(attn[:, prefix_tokens:])
    x = x + attn @ w["o_proj"].T
    m = rmsnorm_torch(x, post_norm)
    gate = m @ w["gate_proj"].T
    up = m @ w["up_proj"].T
    gated = torch.nn.functional.silu(gate) * up
    stats["post_mlp_i8"] = q15_16(m[:, prefix_tokens:])
    stats["gated_mlp_i16"] = q15_16(gated[:, prefix_tokens:])
    return x + gated @ w["down_proj"].T, stats


@torch.no_grad()
def calibrate_attention_scales(embed, layer_weights, norm_weights, ids, config, r3=None, prefix_tokens=0, seq_len=2048):
    cos_q15, sin_q15, _ = rope_tables_q15_16(seq_len, config.head_dim, config.rope_theta, ids.device)
    cos, sin = cos_q15.float() / 65536.0, sin_q15.float() / 65536.0
    x = embed[ids.reshape(-1, seq_len)]
    scales = [None for _ in layer_weights]
    for layer_idx, (w, norms) in enumerate(zip(layer_weights, norm_weights)):
        x, stats = run_calib_segment(x, w, norms, cos, sin, config, r3, prefix_tokens)
        scales[layer_idx] = {name: update_head_amax(None, value) for name, value in stats.items() if name not in ("post_mlp_i8", "gated_mlp_i16")}
        scales[layer_idx]["attn_i8"] = update_tensor_amax(None, stats["attn_i8"])
        scales[layer_idx]["post_mlp_i8"] = update_tensor_amax(None, stats["post_mlp_i8"])
        scales[layer_idx]["gated_mlp_i16"] = update_tensor_amax(None, stats["gated_mlp_i16"])
    return [
        {
            "q_pre_rope_i16": scale_from_amax(layer["q_pre_rope_i16"], 32767),
            "k_pre_rope_i16": scale_from_amax(layer["k_pre_rope_i16"], 32767),
            "q_post_rope_i8": scale_from_amax(layer["q_post_rope_i8"], 127),
            "k_post_rope_i8": scale_from_amax(layer["k_post_rope_i8"], 127),
            "v_i8": scale_from_amax(layer["v_i8"], 127),
            "attn_i8": scale_from_amax(layer["attn_i8"], 127),
            "post_mlp_i8": scale_from_amax(layer["post_mlp_i8"], 127),
            "gated_mlp_i16": scale_from_amax(layer["gated_mlp_i16"], 32767),
        }
        for layer in scales
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", default="/tmp/Qwen3-0.6B-static-calib")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-r1", action="store_true")
    parser.add_argument("--use-r2", action="store_true")
    parser.add_argument("--use-r3", action="store_true")
    parser.add_argument("--rotate-seed", type=int, default=ROTATE_SEED)
    parser.add_argument("--calib-text", default="")
    parser.add_argument("--calib-dataset", default="fineweb")
    parser.add_argument("--calib-parquet")
    parser.add_argument("--calib-column", default="text")
    parser.add_argument("--calib-tokens", type=int, default=0)
    parser.add_argument("--calib-seq-len", type=int, default=0)
    parser.add_argument("--calib-batches", type=int, default=0)
    parser.add_argument("--calib-prefix-tokens", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "qwen3_int_only.safetensors"
    timestamp = out_dir / "timestamp"
    calib_tokens, calib_seq_len = resolve_calib_shape(args)
    metadata = current_pack_metadata(path) if path.exists() and not args.force else None
    if metadata is not None:
        if not timestamp.exists():
            write_timestamp(timestamp, args, calib_tokens, calib_seq_len, metadata)
        print(f"skip existing pack: {path}")
        print(timestamp.read_text(encoding="utf-8").strip())
        return

    tensors = {}
    calib_weights = []
    calib_norms = []
    r1 = random_hadamard_rotation(QWEN3_0_6B.hidden_size, args.rotate_seed, args.device) if args.use_r1 else None
    r2 = random_hadamard_rotation(QWEN3_0_6B.head_dim, args.rotate_seed + 1, args.device) if args.use_r2 else None
    r3 = random_hadamard_rotation(QWEN3_0_6B.head_dim, args.rotate_seed + 2, args.device) if args.use_r3 else None
    with safe_open(f"{args.model_dir}/model.safetensors", framework="pt", device="cpu") as f:
        def tensor(name, device=args.device):
            return f.get_tensor(name).to(torch.float32).to(device)

        embed = tensor("model.embed_tokens.weight")
        lm_head = tensor("lm_head.weight")
        final_norm = tensor("model.norm.weight")
        if args.use_r1:
            embed = rotate_input(embed, r1)
            lm_head = rotate_norm_input(lm_head, final_norm, r1)
            final_norm = torch.ones_like(final_norm)
        tensors["model.embed_tokens.weight"] = embed.cpu().contiguous()
        tensors["lm_head.weight"] = lm_head.cpu().contiguous()
        tensors["model.norm.weight"] = q15_16(final_norm).cpu()
        for layer_idx in range(QWEN3_0_6B.num_hidden_layers):
            src = f"model.layers.{layer_idx}"
            dst = f"layers.{layer_idx}"
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
                    weight = rotate_head_output(weight, QWEN3_0_6B.head_dim, r2)
                if args.use_r2 and name == "o_proj":
                    weight = rotate_head_input(weight, QWEN3_0_6B.head_dim, r2)
                if args.use_r1 and name in ("o_proj", "down_proj"):
                    weight = rotate_output(weight, r1)
                layer_float[name] = weight
                w, s = per_channel_i8_weight(weight)
                tensors[f"{dst}.{name}.weight"] = w.cpu().contiguous()
                tensors[f"{dst}.{name}.scale"] = s.cpu().contiguous()
            if args.use_r1:
                input_norm = torch.ones_like(input_norm)
                post_norm = torch.ones_like(post_norm)
            tensors[f"{dst}.input_layernorm"] = q15_16(input_norm).cpu()
            tensors[f"{dst}.post_attention_layernorm"] = q15_16(post_norm).cpu()
            q_norm = tensor(f"{src}.self_attn.q_norm.weight")
            k_norm = tensor(f"{src}.self_attn.k_norm.weight")
            tensors[f"{dst}.q_norm"] = q15_16(q_norm).cpu()
            tensors[f"{dst}.k_norm"] = q15_16(k_norm).cpu()
            calib_weights.append(layer_float)
            calib_norms.append((input_norm, post_norm, q_norm, k_norm))

    if calib_tokens:
        ids = load_calib_ids(args.model_dir, args.calib_text, args.calib_dataset, args.calib_parquet, args.calib_column, calib_tokens, args.device)
        for layer_idx, scales in enumerate(calibrate_attention_scales(embed, calib_weights, calib_norms, ids, QWEN3_0_6B, r3, args.calib_prefix_tokens, calib_seq_len)):
            dst = f"layers.{layer_idx}"
            for name, scale in scales.items():
                tensors[f"{dst}.{name}.scale"] = scale.cpu().contiguous()

    save_file(
        tensors,
        str(path),
        metadata={
            "use_r1": str(int(args.use_r1)),
            "use_r2": str(int(args.use_r2)),
            "use_r3": str(int(args.use_r3)),
            "calib_tokens": str(calib_tokens),
            "calib_dataset": args.calib_dataset,
            "calib_parquet": str(args.calib_parquet or ""),
            "calib_seq_len": str(calib_seq_len),
            "calib_batches": str(args.calib_batches),
            "calib_prefix_tokens": str(args.calib_prefix_tokens),
        },
    )
    write_timestamp(timestamp, args, calib_tokens, calib_seq_len)
    print(path)
    print(timestamp)


if __name__ == "__main__":
    main()
