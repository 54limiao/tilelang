# Qwen3 Static Quantized Path

Run the standard 2048-token quality check:

```bash
# export MODEL_DIR=/publicdata/huggingface.co/Qwen/Qwen3-14B
# export BACKEND=hybrid
examples/qwen3_int_only/test_static_path.sh
```

The script defaults to Qwen3-0.6B and `BACKEND=int-only`; use `BACKEND=hybrid` for the hybrid path. Export `MODEL_DIR=/publicdata/huggingface.co/Qwen/Qwen3-14B` to run another Qwen3 model; the default pack path follows the model name under `/tmp`, and `PACKED_DIR=...` can override it. For large models, copy the HF model directory to `/tmp` or `/code` first and export that local path to avoid slow publicdata reads.

Default inputs:

```text
model:   /publicdata/huggingface.co/Qwen/Qwen3-0.6B
data:    /publicdata/huggingface.co/datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet
```

## Qwen3-0.6B

2048-token FineWeb quality result against HF bf16:

| backend | tokens | loss | ppl | cos | mse |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF bf16 | 2048 | 3.801437 | 44.765483 | - | - |
| fake-quant | 2048 | 3.819883 | 45.598856 | 0.99490349 | 1.16091984e-01 |
| hybrid | 2048 | 3.816845 | 45.460562 | 0.99481226 | 1.18138879e-01 |
| int-only | 2048 | 3.801799 | 44.781688 | 0.99203995 | 1.83226401e-01 |

Kernel profile for the int-only LLM block path: 2048 tokens, 28 layers, prefix KV cache enabled, 16 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention_i8 | 336.126 | 46.02 | 92.04 | 14.75% | 46.67% |
| linear_i8 | 185.461 | 155.62 | 155.62 | 24.94% | 25.75% |
| rms_q15 | 57.459 | - | - | - | 7.98% |
| rope_sq8 | 55.054 | - | - | - | 7.64% |
| rms_sq8 | 36.517 | - | - | - | 5.07% |
| silu_hadamard_i8 | 34.078 | - | - | - | 4.73% |
| quant_v_i8 | 15.461 | - | - | - | 2.15% |
| total | 720.156 | 61.56 | 83.04 | 13.31% | 100.00% |

Kernel profile for the hybrid LLM block path: 2048 tokens, 28 layers, prefix KV cache enabled, 16 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention_hybrid | 312.233 | 49.54 | 74.31 | 11.91% | 40.62% |
| linear_i8 | 187.536 | 153.90 | 153.90 | 24.66% | 24.40% |
| qk_norm_rope_quant_hybrid | 181.160 | - | - | - | 23.57% |
| silu_hadamard_quant_hybrid | 41.486 | - | - | - | 5.40% |
| rms_quant_hybrid | 31.018 | - | - | - | 4.04% |
| quant_v_i8 | 15.196 | - | - | - | 1.98% |
| total | 768.628 | 57.67 | 67.74 | 10.86% | 100.00% |

## Qwen3-14B

2048-token FineWeb quality result against HF bf16:

| backend | tokens | loss | ppl | cos | mse |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF bf16 | 2048 | 3.031365 | 20.725512 | - | - |
| fake-quant | 2048 | 3.048030 | 21.073791 | 0.98760375 | 4.06879236e-01 |
| hybrid | 2048 | 3.065444 | 21.443991 | 0.98531953 | 4.83506536e-01 |
| int-only | 2048 | 3.093172 | 22.046891 | 0.96337582 | 1.24137745e+00 |

Torch fake-quant uses the same packed int8 weights and static activation scales, but computes matmul, attention, and SiLU in torch float after QDQ. This gives the current pure-i8 quantization ceiling before kernel fixed-point error. The int-only kernel path should first close the gap from the current baseline to this fake-quant ceiling.

Kernel profile for the same static int-only path: 2048 tokens, 40 layers, prefix KV cache enabled, 16 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear_i8 | 5684.166 | 152.33 | 152.33 | 24.41% | 77.07% |
| attention_i8 | 1072.657 | 51.50 | 103.00 | 16.51% | 14.54% |
| silu_hadamard_i8 | 205.398 | - | - | - | 2.79% |
| rms_sq8 | 134.289 | - | - | - | 1.82% |
| rope_sq8 | 129.244 | - | - | - | 1.75% |
| rms_q15 | 126.678 | - | - | - | 1.72% |
| quant_v_i8 | 22.601 | - | - | - | 0.31% |
| total | 7375.033 | 124.90 | 132.39 | 21.22% | 100.00% |

Kernel profile for the hybrid LLM block path: 2048 tokens, 40 layers, prefix KV cache enabled, 16 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear_i8 | 5702.455 | 151.84 | 151.84 | 24.33% | 75.69% |
| attention_hybrid | 953.176 | 57.96 | 86.94 | 13.93% | 12.65% |
| qk_norm_rope_quant_hybrid | 453.162 | - | - | - | 6.02% |
| silu_hadamard_quant_hybrid | 269.475 | - | - | - | 3.58% |
| rms_quant_hybrid | 132.985 | - | - | - | 1.77% |
| quant_v_i8 | 22.506 | - | - | - | 0.30% |
| total | 7533.759 | 122.26 | 125.93 | 20.18% | 100.00% |

On 0.6B, attention is a large share because the MLP/linear matrices are small. On 14B, the same 2048-token attention work is much less dominant relative to the hidden/intermediate-size linear work, so `linear_i8` becomes the main cost.

Utilization uses the A100 int8 tensorcore peak, 624 TOPS. `math TOPS` is the model matmul work, counted as `2MNK / time`. `tc TOPS` is the int8 tensorcore-equivalent work issued by the current kernels. The int-only attention kernel computes QK once for online softmax statistics and recomputes QK for PV, then computes P16@V8 as two int8 GEMMs over the high 8 bits and middle 7 bits of P, so int-only attention `tc TOPS` counts two QK GEMMs plus two PV GEMMs. The hybrid attention kernel uses a single-pass online softmax, so its `tc TOPS` counts one QK GEMM plus two PV GEMMs.

The first run writes `qwen3_int_only.safetensors` and `timestamp`; later runs skip packing when the pack matches the current static schema. Use `FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh` to rebuild.

Main quantization path:

- weights: per-channel static int8 for all linear weights; QKV and gate/up are packed offline
- activations: calibrated static scales from FineWeb, 32 batches x 2048 tokens, with the cache prompt prepended; QKV scales use the full sequence, while non-QKV activation scales ignore the first 512 prefix tokens; calibration attention uses PyTorch fused flex attention with SDPA fallback
- linear output quantization: activation-scale x weight-scale factors are precomputed as packed XP5 quant parameters, so linear kernels finish with one `T.fix.quant`
- residual stream: Q15.16 int32
- attention: cache-aware kernels handle prefix KV and current-token causal mask; Q/K/V cache and current tokens use static per-head scales; P16@V8 is computed with two int8 GEMMs
- RoPE tables: int-only kernels use q15.16 cos/sin tables; hybrid kernels use fp32 cos/sin tables
- residual flow: layer 0 starts with no residual, then residual RMS kernels update the Q15.16 residual stream and emit static int8 linear inputs
- MLP: post-attention residual RMSNorm emits both Q15.16 residual and static int8 linear input; gate/up use one packed int8 GEMM; SiLU-gated activation is block-Hadamard rotated and statically quantized to int8, and down_proj uses the pre-rotated int8 weight
- comparison: int-only logits are compared with HF bf16 logits using PPL, cosine, MSE, MAE, max_abs, and rel_mse
- hybrid path: GEMM inputs/outputs remain integer, while RMSNorm, RoPE, SiLU, and online softmax are TileLang fp32 kernels; QK and P16@V8 still use int8 tensorcore GEMMs

The int-only showcase implementation is in `model.py` and `kernels_int_only.py`; the hybrid implementation is in `model_hybrid.py` and `kernels_hybrid.py`. Packing, QuaRot, PPL, and profiling live under `utils/`.

Kernel numpy prototypes live under `utils/proto/` and can be checked with `python -m examples.qwen3_int_only.utils.proto.run_all`.
