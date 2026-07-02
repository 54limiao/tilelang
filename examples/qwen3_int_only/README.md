# Qwen3 0.6B Static Int-Only Path

Run the standard 2048-token quality check:

```bash
examples/qwen3_int_only/test_static_path.sh
```

Default inputs:

```text
model:   /publicdata/huggingface.co/Qwen/Qwen3-0.6B
data:    /publicdata/huggingface.co/datasets
```

Current 2048-token FineWeb quality result against HF bf16:

| backend | tokens | loss | ppl | cos | mse | rel_mse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HF bf16 | 2048 | 3.801437 | 44.765483 | - | - | - |
| int-only | 2048 | 3.813066 | 45.289083 | 0.99066291 | 2.19819066e-01 | 1.92688732e-02 |

Kernel profile for the int-only LLM block path: 2048 tokens, 28 layers, prefix KV cache enabled, 3 measured repeats.

| kernel | avg ms | total ms | math TOPS | math util | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attention_i8 | 0.978 | 82.154 | 35.30 | 5.66% | 88.25 | 14.14% | 51.07% |
| linear_i16_down | 0.185 | 15.545 | 69.62 | 11.16% | 139.24 | 22.31% | 9.66% |
| linear_i8_gate_up | 0.127 | 10.635 | 203.55 | 32.62% | 203.55 | 32.62% | 6.61% |
| linear_i8_qkv | 0.092 | 7.698 | 187.45 | 30.04% | 187.45 | 30.04% | 4.79% |
| rms_sq8_post_attn | 0.045 | 7.459 | - | - | - | - | 4.64% |
| rope_sq8_q | 0.088 | 7.400 | - | - | - | - | 4.60% |
| rms_q15_q | 0.079 | 6.607 | - | - | - | - | 4.11% |
| silu_i16 | 0.066 | 5.534 | - | - | - | - | 3.44% |
| linear_i8_o | 0.064 | 5.389 | 133.89 | 21.46% | 133.89 | 21.46% | 3.35% |
| rope_sq8_k | 0.055 | 4.621 | - | - | - | - | 2.87% |
| rms_q15_k | 0.053 | 4.479 | - | - | - | - | 2.78% |
| quant_v_i8 | 0.038 | 3.203 | - | - | - | - | 1.99% |
| rms_sq8_input | 0.052 | 0.155 | - | - | - | - | 0.10% |
| total | 1.915 | 160.879 | 51.67 | 8.28% | 85.43 | 13.69% | 100.00% |

Utilization uses the A100 int8 tensorcore peak, 624 TOPS. `math TOPS` is the model matmul work, counted as `2MNK / time`. `tc TOPS` is the int8 tensorcore-equivalent work issued by the current kernels. The `linear_i16_down` kernel implements `int16@int8` as two `int8@int8` GEMMs over the high 8 bits and middle 7 bits, so its `tc TOPS` is counted as 2x the math work. The current `attention_i8` kernel recomputes QK and computes P16@V8 as two int8 GEMMs over the high 8 bits and middle 7 bits of P, so its `tc TOPS` is counted as `2 * QK + 2 * PV`.

The first run writes `qwen3_int_only.safetensors` and `timestamp`; later runs skip packing when the pack matches the current static schema. Use `FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh` to rebuild.

Main quantization path:

- weights: per-channel static int8 for all linear weights; QKV and gate/up are packed offline
- activations: calibrated static scales from FineWeb, 32 batches x 2048 tokens, with the cache prompt prepended; QKV scales use the full sequence, while non-QKV activation scales ignore the first 512 prefix tokens
- residual stream: Q15.16 int32
- attention: one cache-aware int8 attention kernel handles prefix KV and current-token causal mask; Q/K/V cache and current tokens use static per-head scales; P16@V8 is computed with two int8 GEMMs
- residual flow: layer 0 starts with no residual, then residual RMS kernels update the Q15.16 residual stream and emit static int8 linear inputs
- MLP: post-attention residual RMSNorm emits both Q15.16 residual and static int8 linear input; gate/up use one packed int8 GEMM; SiLU-gated activation is static int16
- comparison: int-only logits are compared with HF bf16 logits using PPL, cosine, MSE, MAE, max_abs, and rel_mse

The showcase implementation is in `model.py` and `kernels.py`; packing, QuaRot, PPL, and profiling live under `utils/`.
