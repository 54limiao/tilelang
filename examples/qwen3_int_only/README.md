# Qwen3 Static Quantized Path

Run the standard 2048-token quality check:

```bash
# export MODEL_DIR=/publicdata/huggingface.co/Qwen/Qwen3-14B
examples/qwen3_int_only/test_static_path.sh
```

The script defaults to Qwen3-0.6B and `BACKEND=all`, which reports HF bf16, fake-quant, hybrid, and int-only quality. Export `MODEL_DIR=/publicdata/huggingface.co/Qwen/Qwen3-14B` to run another Qwen3 model; the default pack path follows the model name under `/tmp`, and `PACKED_DIR=...` can override it. For large models, copy the HF model directory to `/tmp` or `/code` first and export that local path to avoid slow publicdata reads.

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
| fake-quant | 2048 | 3.818876 | 45.552979 | 0.99498089 | 1.18436021e-01 |
| hybrid | 2048 | 3.813385 | 45.303522 | 0.99503829 | 1.16245255e-01 |
| int-only | 2048 | 3.810946 | 45.193192 | 0.99113492 | 2.01657777e-01 |

Kernel profile for the int-only LLM block path: 2048 tokens, 28 layers, prefix KV cache enabled, 16 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention_i8 | 272.246 | 56.82 | 85.23 | 13.66% | 36.20% |
| qk_norm_rope_i8 | 204.872 | - | - | - | 27.24% |
| linear_i8 | 187.152 | 154.22 | 154.22 | 24.71% | 24.88% |
| rms_sq8 | 37.184 | - | - | - | 4.94% |
| silu_hadamard_i8 | 35.081 | - | - | - | 4.66% |
| quant_v_i8 | 15.577 | - | - | - | 2.07% |
| total | 752.113 | 58.94 | 69.22 | 11.09% | 100.00% |

Kernel profile for the hybrid LLM block path: 2048 tokens, 28 layers, prefix KV cache enabled, 16 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention_hybrid | 303.655 | 50.94 | 76.41 | 12.25% | 42.22% |
| linear_i8 | 186.663 | 154.62 | 154.62 | 24.78% | 25.95% |
| qk_norm_rope_quant_hybrid | 138.976 | - | - | - | 19.32% |
| silu_hadamard_quant_hybrid | 41.861 | - | - | - | 5.82% |
| rms_quant_hybrid | 32.467 | - | - | - | 4.51% |
| quant_v_i8 | 15.653 | - | - | - | 2.18% |
| total | 719.276 | 61.63 | 72.38 | 11.60% | 100.00% |

## Qwen3-14B

2048-token FineWeb quality result against HF bf16:

| backend | tokens | loss | ppl | cos | mse |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF bf16 | 2048 | 3.031365 | 20.725512 | - | - |
| fake-quant | 2048 | 3.056284 | 21.248450 | 0.98738441 | 4.13797397e-01 |
| hybrid | 2048 | 3.062984 | 21.391299 | 0.98538696 | 4.82325177e-01 |
| int-only | 2048 | 3.091172 | 22.002855 | 0.96762299 | 1.09244404e+00 |

Kernel profile for the int-only LLM block path: 2048 tokens, 40 layers, prefix KV cache enabled, 16 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear_i8 | 5682.797 | 152.37 | 152.37 | 24.42% | 76.14% |
| attention_i8 | 876.265 | 63.04 | 94.57 | 15.16% | 11.74% |
| qk_norm_rope_i8 | 540.440 | - | - | - | 7.24% |
| silu_hadamard_i8 | 205.857 | - | - | - | 2.76% |
| rms_sq8 | 136.035 | - | - | - | 1.82% |
| quant_v_i8 | 22.401 | - | - | - | 0.30% |
| total | 7463.795 | 123.41 | 127.11 | 20.37% | 100.00% |

Kernel profile for the hybrid LLM block path: 2048 tokens, 40 layers, prefix KV cache enabled, 16 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear_i8 | 5679.004 | 152.47 | 152.47 | 24.43% | 76.46% |
| attention_hybrid | 952.606 | 57.99 | 86.99 | 13.94% | 12.83% |
| qk_norm_rope_quant_hybrid | 362.882 | - | - | - | 4.89% |
| silu_hadamard_quant_hybrid | 275.569 | - | - | - | 3.71% |
| rms_quant_hybrid | 133.326 | - | - | - | 1.80% |
| quant_v_i8 | 23.568 | - | - | - | 0.32% |
| total | 7426.955 | 124.02 | 127.74 | 20.47% | 100.00% |

On 0.6B, attention is a large share because the MLP/linear matrices are small. On 14B, the same 2048-token attention work is much less dominant relative to the hidden/intermediate-size linear work, so `linear_i8` becomes the main cost.

Utilization uses the A100 int8 tensorcore peak, 624 TOPS. `math TOPS` is the model matmul work, counted as `2MNK / time`. `tc TOPS` is the int8 tensorcore-equivalent work issued by the current kernels. Attention computes QK once with online softmax, then computes PV as `P int16 x V int8`; the current tensorcore lowering counts this PV as two int8-equivalent GEMMs.

The first run writes `qwen3_int_only.safetensors` and `timestamp`; later script runs reuse the pack when `timestamp` exists and print that timestamp. Use `FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh` to rebuild.

## Quantization Scheme

This example uses one static calibration pass to pack the model and then runs both `int-only` and `hybrid` inference from the same packed weights. Linear inputs are statically quantized to `int8`; all linear weights are per-channel static `int8`. All GEMMs in both backends are integer GEMMs. Hybrid only changes the non-GEMM scalar work: RMSNorm, RoPE, SiLU, Hadamard scaling, and online softmax are TileLang fp32 kernels, while QK, PV, QKV/O/Gate-Up/Down projections remain integer GEMMs.

The int-only backend stores the residual stream as Q15.16 `int32`. The hybrid backend stores the residual stream as `fp32`; each RMSNorm consumes `fp32 residual + Q15.16 linear output`, then emits `int8` activations for the next integer GEMM.

Calibration is data driven. The default script uses FineWeb from `/publicdata/huggingface.co/datasets`, 32 batches of 2048 tokens, and the same Chinese cache prompt used at evaluation time. QKV activation scales use the full calibrated sequence because prefix outliers matter for KV cache quality; non-QKV activation scales ignore the first 512 prefix tokens to avoid over-scaling MLP residual outliers that do not represent normal generated tokens.

QuaRot is applied during packing. R1 smooths residual-channel activation ranges through a single model-wide weight rotation, R2 rotates the V/O path with one saved matrix per layer, R3 is a fixed exact fast Hadamard on the Q/K head dimension, and R4 is the exact Hadamard rotation used before down_proj. The pack saves `quarot.r1`, `quarot.r4`, and `layers.N.r2`; cache builders read those matrices from the pack instead of reconstructing them from random seeds. R2 and R3 affect KV-cache semantics, so the cache builder applies the saved R2 matrix and the fixed R3 transform when it quantizes HF-generated prefix KV tensors.

Linear kernels consume `int8` activations and per-channel `int8` weights. Activation-scale x weight-scale factors are precomputed into packed XP5 quant parameters, so each linear kernel ends with one `T.fix.quant` from the `int32` accumulator back to Q15.16. The packed QKV and gate/up matrices are concatenated offline to avoid runtime packing kernels.

Attention is the main difference between the two backends. `int-only` keeps online softmax in fixed-point with `T.fix.lut_10bit`; `hybrid` keeps Q/K/V and all GEMMs integer but uses TileLang fp32 for online softmax and normalization work. Both backends compute QK once and compute PV as `P int16 x V int8`.

The hybrid backend is intended for hardware where tensorcore/NPU handles the GEMM-heavy work and a DSP can cheaply handle scalar fp32/fp16-style operations. Hybrid kernels do not use `T.fix`; int-only kernels keep the fixed-point path for the pure integer target.

Quality is reported against HF bf16 logits with PPL, cosine, MSE, MAE, max_abs, and rel_mse. `fake-quant` uses the same packed int8 weights and static activation scales but performs QDQ math in torch float, giving the current quantization ceiling before kernel arithmetic error.

## Runtime Flow

Int-only keeps the whole block in fixed-point integer form. The block order is RMSNorm, attention, RMSNorm, MLP. Every matrix multiply is an integer GEMM, including QK and PV in attention.

```mermaid
flowchart TD
  A[Q15.16 residual int32] --> B[Residual RMSNorm\nT.fix rsqrt LUT]
  B --> C[input_qkv int8 quant]:::quant
  C --> D[QKV int8 x int8 GEMM]
  D --> E[Q/K RMSNorm + RoPE + R3\nT.fix, int8 output]:::quant
  D --> F[V int8 quant]:::quant
  F --> G[V cache/current int8]
  E --> H[QK int8 x int8 GEMM]
  H --> I[Online softmax -> P int16\nT.fix.lut_10bit]
  G --> K[PV integer GEMM\nP int16 x V int8]
  I --> K
  K --> L[attention int8 quant]:::quant
  L --> M[O int8 x int8 GEMM]
  M --> N[Residual RMSNorm + MLP int8 quant\nT.fix rsqrt LUT]:::quant
  N --> O[Gate/Up int8 x int8 GEMM]
  O --> P[SiLU LUT + R4 Hadamard\nint8 quant]:::quant
  P --> Q[Down int8 x int8 GEMM]
  Q --> R[next Q15.16 residual int32]
  classDef quant fill:#fff3cd,stroke:#d39e00,stroke-width:2px,color:#24292f;
```

Hybrid uses the same integer GEMM dataflow but moves scalar-heavy work to TileLang fp32 kernels. The block order is RMSNorm, attention, RMSNorm, MLP. QK and PV are still integer GEMMs; fp32 is used only around them for residual accumulation, normalization, RoPE, SiLU, and online softmax state.

```mermaid
flowchart TD
  A[fp32 residual] --> B[RMSNorm + input int8 quant\nfp32 residual + Q15.16 linear]:::quant
  B --> C[QKV int8 x int8 GEMM]
  C --> D[Q/K RMSNorm + RoPE + R3\nTileLang fp32, int8 output]:::quant
  C --> E[V int8 quant]:::quant
  E --> F[V cache/current int8]
  D --> G[QK int8 x int8 GEMM]
  G --> H[Single-pass online softmax -> P int16\nTileLang fp32 state]
  F --> J[PV integer GEMM\nP int16 x V int8]
  H --> J
  J --> K[attention int8 quant\nTileLang fp32 scale]:::quant
  K --> L[O int8 x int8 GEMM]
  L --> M[RMSNorm + MLP int8 quant\nfp32 residual + Q15.16 O output]:::quant
  M --> N[Gate/Up int8 x int8 GEMM]
  N --> O[SiLU + R4 Hadamard + int8 quant\nTileLang fp32]:::quant
  O --> P[Down int8 x int8 GEMM]
  P --> Q[next fp32 residual path]
  classDef quant fill:#fff3cd,stroke:#d39e00,stroke-width:2px,color:#24292f;
```

The int-only showcase implementation is in `model_int_only.py` and `kernels_int_only.py`; the hybrid implementation is in `model_hybrid.py` and `kernels_hybrid.py`. Packing, QuaRot, PPL, and profiling live under `utils/`.

Kernel numpy prototypes live under `utils/proto/` and can be checked with `python -m examples.qwen3_int_only.utils.proto.run_all`.
