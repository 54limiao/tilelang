# Qwen3 0.6B Static Int-Only Path

Run the standard 2048-token quality check:

```bash
examples/qwen3_int_only/test_static_path.sh
```

Default inputs:

```text
model:   /publicdata/huggingface.co/Qwen/Qwen3-0.6B
data:    /publicdata/huggingface.co/datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet
```

Current 2048-token FineWeb quality result against HF bf16:

| backend | tokens | loss | ppl | cos | mse | rel_mse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HF bf16 | 2048 | 3.801437 | 44.765483 | - | - | - |
| int-only | 2048 | 3.813066 | 45.289083 | 0.99066291 | 2.19819066e-01 | 1.92688732e-02 |

Kernel profile for the int-only LLM block path: 2048 tokens, 28 layers, prefix KV cache enabled, 16 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention_i8 | 347.574 | 44.50 | 89.01 | 14.26% | 46.18% |
| linear_i8 | 121.594 | 189.89 | 189.89 | 30.43% | 16.16% |
| linear_i16 | 79.248 | 72.84 | 145.68 | 23.35% | 10.53% |
| rope_sq8 | 61.556 | - | - | - | 8.18% |
| rms_q15 | 57.412 | - | - | - | 7.63% |
| rms_sq8 | 39.517 | - | - | - | 5.25% |
| silu_i16 | 29.464 | - | - | - | 3.92% |
| quant_v_i8 | 16.205 | - | - | - | 2.15% |
| total | 752.570 | 58.91 | 87.13 | 13.96% | 100.00% |

Utilization uses the A100 int8 tensorcore peak, 624 TOPS. `math TOPS` is the model matmul work, counted as `2MNK / time`. `tc TOPS` is the int8 tensorcore-equivalent work issued by the current kernels. The `linear_i16_down` kernel implements `int16@int8` as two `int8@int8` GEMMs over the high 8 bits and middle 7 bits, so its `tc TOPS` is counted as 2x the math work. The current `attention_i8` kernel computes QK once for online softmax statistics and recomputes QK for PV, then computes P16@V8 as two int8 GEMMs over the high 8 bits and middle 7 bits of P, so its `tc TOPS` counts two QK GEMMs plus two PV GEMMs.

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

Kernel numpy prototypes live under `utils/proto/` and can be checked with `python -m examples.qwen3_int_only.utils.proto.run_all`.
