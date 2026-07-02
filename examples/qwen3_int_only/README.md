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

| backend | tokens | loss | ppl | compare | cos | mse | rel_mse |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| HF bf16 | 2048 | 3.801437 | 44.765483 | - | - | - | - |
| int-only | 2048 | 3.820530 | 45.628371 | HF bf16 logits | 0.99206903 | 1.83985954e-01 | 1.61278185e-02 |

Kernel profile for the int-only LLM block path:

| item | value |
| --- | ---: |
| sequence length | 2048 tokens |
| layers | 28 |
| prefix KV cache | enabled |
| measured repeats | 3 |
| time per 28-layer pass | 57.305 ms |

| kernel | avg ms | total ms | pct | TOPS |
| --- | ---: | ---: | ---: | ---: |
| attention_cache_i8v8_fused_static | 1.021 | 85.775 | 49.89 | 67.63 |
| down_residual_static | 0.196 | 16.465 | 9.58 | 131.47 |
| qkv_proj_i8 | 0.146 | 12.265 | 7.13 | 117.66 |
| gate_up_proj_static | 0.139 | 11.702 | 6.81 | 184.99 |
| rope_sq8_q_attn_hadamard | 0.092 | 7.770 | 4.52 | 11.61 |
| rms_q_q15 | 0.070 | 5.852 | 3.40 | - |
| o_proj_i8_static | 0.066 | 5.571 | 3.24 | 129.51 |
| silu_mul_sq16_mid_fast | 0.065 | 5.420 | 3.15 | - |
| rope_sq8_k_attn_hadamard | 0.057 | 4.751 | 2.76 | 9.49 |
| rms_k_q15 | 0.045 | 3.795 | 2.21 | - |
| rms_input_dq8_fast | 0.041 | 3.434 | 2.00 | - |
| sq8_v_attn_noscale | 0.039 | 3.246 | 1.89 | - |
| residual_attn_rms_q15 | 0.038 | 3.196 | 1.86 | - |
| sq8_hidden_static | 0.032 | 2.674 | 1.56 | - |

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
