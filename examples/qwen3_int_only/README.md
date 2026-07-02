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
| int-only | 2048 | 3.814808 | 45.368040 | 0.99172074 | 1.93770385e-01 | 1.69855011e-02 |

Kernel profile for the int-only LLM block path: 2048 tokens, 28 layers, prefix KV cache enabled, 3 measured repeats.

| kernel | avg ms | total ms | pct | TOPS |
| --- | ---: | ---: | ---: | ---: |
| attention_cache_i8v8_fused_static | 0.992 | 83.319 | 50.22 | 34.81 |
| down_proj_static | 0.186 | 15.625 | 9.42 | 69.27 |
| qkv_proj_i8 | 0.141 | 11.805 | 7.11 | 122.25 |
| gate_up_proj_static | 0.135 | 11.308 | 6.82 | 191.44 |
| rope_sq8_q_attn_hadamard | 0.088 | 7.430 | 4.48 | - |
| rms_q_q15 | 0.067 | 5.595 | 3.37 | - |
| silu_mul_sq16_mid_fast | 0.063 | 5.290 | 3.19 | - |
| o_proj_i8_static | 0.063 | 5.288 | 3.19 | 136.45 |
| rope_sq8_k_attn_hadamard | 0.054 | 4.553 | 2.74 | - |
| residual_attn_rms_sq8 | 0.043 | 3.610 | 2.18 | - |
| rms_k_q15 | 0.043 | 3.593 | 2.17 | - |
| rms_input_sq8_fast | 0.037 | 3.150 | 1.90 | - |
| sq8_v_attn_noscale | 0.037 | 3.068 | 1.85 | - |
| add_mlp_residual | 0.027 | 2.288 | 1.38 | - |
| total | 1.975 | 165.921 | 100.00 | 50.10 |

The first run writes `qwen3_int_only.safetensors` and `timestamp`; later runs skip packing when the pack matches the current static schema. Use `FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh` to rebuild.

Main quantization path:

- weights: per-channel static int8 for all linear weights
- activations: calibrated static scales from FineWeb, 32 batches x 2048 tokens, ignoring the first 512 prefix tokens
- residual stream: Q15.16 int32
- attention: input RMSNorm directly emits static int8 for QKV, Q/K RMSNorm stays Q15.16, q/k/v are static int8 with r2/r3 QuaRot head rotations, prefix KV cache, fused static int8 attention output feeds O projection
- MLP: post-attention residual RMSNorm emits both Q15.16 residual and static int8 linear input, SiLU-gated activation is static int16, all compute kernels are int-only TileLang kernels
- comparison: int-only logits are compared with HF bf16 logits using PPL, cosine, MSE, MAE, max_abs, and rel_mse

Useful knobs:

```bash
EVAL_TOKENS=2048 NUM_BATCHES=1 examples/qwen3_int_only/test_static_path.sh
FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh
PACKED_DIR=/tmp/custom-pack examples/qwen3_int_only/test_static_path.sh
/root/venv/bin/python examples/qwen3_int_only/utils/profile_kernels.py --max-tokens 2048 --layers 28 --warmup 1 --repeat 3
```

The showcase implementation is in `model.py` and `kernels.py`; packing, QuaRot, PPL, profiling, and trace helpers live under `utils/`.
