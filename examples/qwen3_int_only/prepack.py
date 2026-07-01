import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from examples.qwen3_int_only.model import QWEN3_0_6B, per_channel_i8_weight, q15_16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/code/Qwen3-0.6B")
    parser.add_argument("--out-dir", default="/code/Qwen3-0.6B-int-only")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    with safe_open(f"{args.model_dir}/model.safetensors", framework="pt", device="cpu") as f:
        def tensor(name, device=args.device):
            return f.get_tensor(name).to(torch.float32).to(device)

        tensors["model.embed_tokens.weight"] = tensor("model.embed_tokens.weight", "cpu")
        tensors["lm_head.weight"] = tensor("lm_head.weight", "cpu")
        tensors["model.norm.weight"] = q15_16(tensor("model.norm.weight")).cpu()
        for layer_idx in range(QWEN3_0_6B.num_hidden_layers):
            src = f"model.layers.{layer_idx}"
            dst = f"layers.{layer_idx}"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                owner = "self_attn" if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                w, s = per_channel_i8_weight(tensor(f"{src}.{owner}.{name}.weight"))
                tensors[f"{dst}.{name}.weight"] = w.cpu()
                tensors[f"{dst}.{name}.scale"] = s.cpu()
            tensors[f"{dst}.input_layernorm"] = q15_16(tensor(f"{src}.input_layernorm.weight")).cpu()
            tensors[f"{dst}.post_attention_layernorm"] = q15_16(tensor(f"{src}.post_attention_layernorm.weight")).cpu()
            tensors[f"{dst}.q_norm"] = q15_16(tensor(f"{src}.self_attn.q_norm.weight")).cpu()
            tensors[f"{dst}.k_norm"] = q15_16(tensor(f"{src}.self_attn.k_norm.weight")).cpu()

    path = out_dir / "qwen3_int_only.safetensors"
    save_file(tensors, str(path))
    print(path)


if __name__ == "__main__":
    main()
