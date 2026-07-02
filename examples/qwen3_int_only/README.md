# Qwen3 0.6B Static Int-Only Path

Run the standard 2048-token quality check:

```bash
examples/qwen3_int_only/test_static_path.sh
```

Default inputs:

```text
model:   /publicdata/huggingface.co/Qwen/Qwen3-0.6B
data:    /publicdata/huggingface.co/datasets
pack:    /tmp/Qwen3-0.6B-static-calib-32x2048
prefix:  你是一个有用而无害的聊天助手。
```

Current 2048-token FineWeb quality result against HF bf16:

| backend | tokens | loss | ppl | cos | mse | rel_mse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HF bf16 | 2048 | 3.801437 | 44.765483 | - | - | - |
| int-only | 2048 | 3.820530 | 45.628371 | 0.99206903 | 1.83985954e-01 | 1.61278185e-02 |

Kernel profile for the int-only LLM block path: 2048 tokens, 28 layers, prefix KV cache enabled, 3 measured repeats.

| kernel | avg ms | total ms | pct | TOPS |
| --- | ---: | ---: | ---: | ---: |
| attention_cache_i8v8_fused_static | 1.079 | 90.626 | 49.43 | 32.00 |
| down_proj_static | 0.200 | 16.841 | 9.18 | 64.27 |
| qkv_proj_i8 | 0.152 | 12.774 | 6.97 | 112.97 |
| gate_up_proj_static | 0.147 | 12.345 | 6.73 | 175.35 |
| rope_sq8_q_attn_hadamard | 0.096 | 8.072 | 4.40 | - |
| rms_q_q15 | 0.072 | 6.080 | 3.32 | - |
| o_proj_i8_static | 0.070 | 5.892 | 3.21 | 122.46 |
| silu_mul_sq16_mid_fast | 0.067 | 5.610 | 3.06 | - |
| rope_sq8_k_attn_hadamard | 0.060 | 5.010 | 2.73 | - |
| rms_k_q15 | 0.047 | 3.988 | 2.18 | - |
| residual_attn_rms_q15 | 0.043 | 3.599 | 1.96 | - |
| rms_input_dq8_fast | 0.043 | 3.584 | 1.95 | - |
| sq8_v_attn_noscale | 0.041 | 3.479 | 1.90 | - |
| sq8_hidden_static | 0.034 | 2.856 | 1.56 | - |
| add_mlp_residual | 0.031 | 2.603 | 1.42 | - |
| total | 2.183 | 183.359 | 100.00 | 45.33 |

The first run writes `qwen3_int_only.safetensors` and `timestamp`; later runs skip packing when the pack matches the current static schema. Use `FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh` to rebuild.

Main quantization path:

- weights: per-channel static int8 for all linear weights
- activations: calibrated static scales from FineWeb, 32 batches x 2048 tokens, ignoring the first 512 prefix tokens
- residual stream: Q15.16 int32
- attention: q/k/v int8, r2/r3 QuaRot head rotations, prefix KV cache, fused static int8 attention path with static pertensor int8 output into O projection
- MLP: static int8 input, static int16 gated activation, int-only TileLang kernels
- comparison: int-only logits are compared with HF bf16 logits using PPL, cosine, MSE, MAE, max_abs, and rel_mse

Useful knobs:

```bash
EVAL_TOKENS=2048 NUM_BATCHES=1 examples/qwen3_int_only/test_static_path.sh
FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh
PACKED_DIR=/tmp/custom-pack examples/qwen3_int_only/test_static_path.sh
/root/venv/bin/python examples/qwen3_int_only/utils/profile_kernels.py --max-tokens 2048 --layers 28 --warmup 1 --repeat 3
```

The showcase implementation is in `model.py` and `kernels.py`; packing, QuaRot, PPL, profiling, and trace helpers live under `utils/`.
