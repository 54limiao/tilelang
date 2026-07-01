import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from examples.qwen3_int_only.model import QWEN3_0_6B, per_channel_i8_weight, q15_16
from examples.qwen3_int_only.quarot import (
    ROTATE_SEED,
    random_hadamard_rotation,
    rotate_head_input,
    rotate_head_output,
    rotate_input,
    rotate_norm_input,
    rotate_output,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/code/Qwen3-0.6B")
    parser.add_argument("--out-dir", default="/code/Qwen3-0.6B-int-only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-r1", action="store_true")
    parser.add_argument("--use-r2", action="store_true")
    parser.add_argument("--rotate-seed", type=int, default=ROTATE_SEED)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    r1 = random_hadamard_rotation(QWEN3_0_6B.hidden_size, args.rotate_seed, args.device) if args.use_r1 else None
    r2 = random_hadamard_rotation(QWEN3_0_6B.head_dim, args.rotate_seed + 1, args.device) if args.use_r2 else None
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
                w, s = per_channel_i8_weight(weight)
                tensors[f"{dst}.{name}.weight"] = w.cpu().contiguous()
                tensors[f"{dst}.{name}.scale"] = s.cpu().contiguous()
            if args.use_r1:
                input_norm = torch.ones_like(input_norm)
                post_norm = torch.ones_like(post_norm)
            tensors[f"{dst}.input_layernorm"] = q15_16(input_norm).cpu()
            tensors[f"{dst}.post_attention_layernorm"] = q15_16(post_norm).cpu()
            tensors[f"{dst}.q_norm"] = q15_16(tensor(f"{src}.self_attn.q_norm.weight")).cpu()
            tensors[f"{dst}.k_norm"] = q15_16(tensor(f"{src}.self_attn.k_norm.weight")).cpu()

    path = out_dir / "qwen3_int_only.safetensors"
    save_file(tensors, str(path), metadata={"use_r1": str(int(args.use_r1)), "use_r2": str(int(args.use_r2))})
    print(path)


if __name__ == "__main__":
    main()
