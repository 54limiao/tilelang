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
| attention_cache_i8v8_fused_static | 0.925 | 77.702 | 51.12 | 37.33 |
| down_proj_static | 0.173 | 14.564 | 9.58 | 74.32 |
| qkv_proj_i8 | 0.128 | 10.771 | 7.09 | 133.98 |
| gate_up_proj_static | 0.124 | 10.408 | 6.85 | 207.99 |
| rope_sq8_q_attn_hadamard | 0.082 | 6.908 | 4.54 | - |
| residual_rms_sq8 | 0.038 | 6.278 | 4.13 | - |
| rms_q_q15 | 0.062 | 5.209 | 3.43 | - |
| silu_mul_sq16_mid_fast | 0.060 | 5.007 | 3.29 | - |
| o_proj_i8_static | 0.057 | 4.809 | 3.16 | 150.04 |
| rope_sq8_k_attn_hadamard | 0.049 | 4.124 | 2.71 | - |
| rms_k_q15 | 0.040 | 3.394 | 2.23 | - |
| sq8_v_attn_noscale | 0.033 | 2.736 | 1.80 | - |
| rms_input_sq8_fast | 0.031 | 0.093 | 0.06 | - |
| total | 1.810 | 152.004 | 100.00 | 54.68 |

The first run writes `qwen3_int_only.safetensors` and `timestamp`; later runs skip packing when the pack matches the current static schema. Use `FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh` to rebuild.

Main quantization path:

- weights: per-channel static int8 for all linear weights
- activations: calibrated static scales from FineWeb, 32 batches x 2048 tokens, ignoring the first 512 prefix tokens
- residual stream: Q15.16 int32
- attention: the first input RMSNorm directly emits static int8 for QKV; later input RMSNorms use the same residual RMS kernel as post-attention RMSNorm
- residual flow: follows the mini-sglang `RMSNormFused` layout; layer 0 starts with no residual, then residual RMS kernels update the Q15.16 residual stream and emit static int8 linear inputs
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
