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
| int-only | 2048 | 3.810040 | 45.152234 | 0.99131307 | 2.01618288e-01 | 1.76734316e-02 |

Kernel profile for the int-only LLM block path: 2048 tokens, 28 layers, prefix KV cache enabled, 3 measured repeats.

| kernel | avg ms | total ms | pct | TOPS |
| --- | ---: | ---: | ---: | ---: |
| attention_cache_i8v8_fused_static | 0.978 | 82.154 | 51.07 | 35.30 |
| down_proj_static | 0.185 | 15.545 | 9.66 | 69.62 |
| gate_up_proj_static | 0.127 | 10.635 | 6.61 | 203.55 |
| qkv_proj_i8 | 0.092 | 7.698 | 4.79 | 187.45 |
| residual_rms_sq8 | 0.045 | 7.459 | 4.64 | - |
| rope_sq8_q_attn_hadamard | 0.088 | 7.400 | 4.60 | - |
| rms_q_q15 | 0.079 | 6.607 | 4.11 | - |
| silu_mul_sq16_mid_fast | 0.066 | 5.534 | 3.44 | - |
| o_proj_i8_static | 0.064 | 5.389 | 3.35 | 133.89 |
| rope_sq8_k_attn_hadamard | 0.055 | 4.621 | 2.87 | - |
| rms_k_q15 | 0.053 | 4.479 | 2.78 | - |
| sq8_v_attn_noscale | 0.038 | 3.203 | 1.99 | - |
| rms_input_sq8 | 0.052 | 0.155 | 0.10 | - |
| total | 1.915 | 160.879 | 100.00 | 51.67 |

The first run writes `qwen3_int_only.safetensors` and `timestamp`; later runs skip packing when the pack matches the current static schema. Use `FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh` to rebuild.

Main quantization path:

- weights: per-channel static int8 for all linear weights; QKV and gate/up are packed offline
- activations: calibrated static scales from FineWeb, 32 batches x 2048 tokens, ignoring the first 512 prefix tokens
- residual stream: Q15.16 int32
- attention: one cache-aware int8 attention kernel handles prefix KV and current-token causal mask; Q/K RMSNorm reuse the same Q15.16 RMS kernel on head-shaped tensors
- residual flow: follows the mini-sglang `RMSNormFused` layout; layer 0 starts with no residual, then residual RMS kernels update the Q15.16 residual stream and emit static int8 linear inputs
- MLP: post-attention residual RMSNorm emits both Q15.16 residual and static int8 linear input; gate/up use one packed int8 GEMM; SiLU-gated activation is static int16
- comparison: int-only logits are compared with HF bf16 logits using PPL, cosine, MSE, MAE, max_abs, and rel_mse

Useful knobs:

```bash
EVAL_TOKENS=2048 NUM_BATCHES=1 examples/qwen3_int_only/test_static_path.sh
FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh
PACKED_DIR=/tmp/custom-pack examples/qwen3_int_only/test_static_path.sh
/root/venv/bin/python examples/qwen3_int_only/utils/profile_kernels.py --max-tokens 2048 --layers 28 --warmup 1 --repeat 3
```

The showcase implementation is in `model.py` and `kernels.py`; packing, QuaRot, PPL, and profiling live under `utils/`.
