import torch

from examples.qwen3_int_only.model import QWEN3_0_6B, Qwen3IntOnlyBlock, load_qwen3_block_weights, q15_16, rope_tables_q15_16


def main(model_dir="/code/Qwen3-0.6B"):
    seq_len = 32
    cfg = QWEN3_0_6B
    block = Qwen3IntOnlyBlock(seq_len, cfg)
    weights = load_qwen3_block_weights(model_dir, layer_idx=0)
    token = torch.arange(seq_len, device="cuda").float()[:, None]
    dim = torch.arange(cfg.hidden_size, device="cuda").float()[None, :]
    x = 0.08 * torch.sin(token * 0.17 + dim * 0.013)
    cos, sin, _ = rope_tables_q15_16(seq_len, cfg.head_dim, cfg.rope_theta)
    y = block(q15_16(x), weights, cos, sin)
    print("qwen3-0.6b layer0 int-only block passed", y.shape, y.dtype)


if __name__ == "__main__":
    main()
