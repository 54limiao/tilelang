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

## Quantization Scheme

This example uses one static calibration pass to pack the model and then runs both `hybrid` and `int-only` inference from the same packed weights. Linear inputs are statically quantized to `int8`; all linear weights are per-channel static `int8`. All GEMMs in both backends are integer GEMMs. Hybrid only changes the non-GEMM scalar work: RMSNorm, RoPE, SiLU, Hadamard scaling, and online softmax are TileLang fp32 kernels, while QK, PV, QKV/O/Gate-Up/Down projections remain integer GEMMs.

The hybrid backend stores the residual stream as `fp32`; each RMSNorm consumes `fp32 residual + Q15.16 linear output`, then emits `int8` activations for the next integer GEMM. The int-only backend stores the residual stream as Q15.16 `int32` and keeps the whole block in fixed-point integer form.

Calibration is data driven. The default script uses FineWeb from `/publicdata/huggingface.co/datasets`, 32 batches of 2048 tokens, and the same Chinese cache prompt used at evaluation time. QKV activation scales use the full calibrated sequence because prefix outliers matter for KV cache quality; non-QKV activation scales ignore the first 512 prefix tokens to avoid over-scaling MLP residual outliers that do not represent normal generated tokens.

QuaRot is applied during packing. R1 smooths residual-channel activation ranges through a single model-wide weight rotation, R2 rotates the V/O path with one saved matrix per layer, R3 is a fixed exact fast Hadamard on the Q/K head dimension, and R4 is the exact Hadamard rotation used before down_proj. The pack saves `quarot.r1`, `quarot.r4`, and `layers.N.r2`; cache builders read those matrices from the pack instead of reconstructing them from random seeds. R2 and R3 affect KV-cache semantics, so the cache builder applies the saved R2 matrix and the fixed R3 transform when it quantizes HF-generated prefix KV tensors.

Linear kernels consume `int8` activations and per-channel `int8` weights. Activation-scale x weight-scale factors are precomputed into packed XP5 quant parameters, so each linear kernel ends with one `T.fix.quant` from the `int32` accumulator back to Q15.16. The packed QKV and gate/up matrices are concatenated offline to avoid runtime packing kernels.

Attention is the main difference between the two backends. `hybrid` keeps Q/K/V and all GEMMs integer but uses TileLang fp32 for online softmax and normalization work. `int-only` keeps online softmax in fixed-point with `T.fix.lut_10bit`. Both backends compute QK once and compute PV as `P int16 x V int8`.

The hybrid backend is intended for hardware where tensorcore/NPU handles the GEMM-heavy work and a DSP can cheaply handle scalar fp32/fp16-style operations. Hybrid kernels do not use `T.fix`; int-only kernels keep the fixed-point path for the pure integer target.

Quality is reported against HF bf16 logits with PPL, cosine, MSE. `fake-quant` uses the same packed int8 weights and static activation scales but performs QDQ math in torch float, giving the current quantization ceiling before kernel arithmetic error.

## Runtime Flow

Hybrid uses the same integer GEMM dataflow but moves scalar-heavy work to TileLang fp32 kernels. The block order is RMSNorm, attention, RMSNorm, MLP. QK and PV are still integer GEMMs; fp32 is used only around them for residual accumulation, normalization, RoPE, SiLU, and online softmax state.

```mermaid
flowchart TD
  A["fp32 residual"] --> B["RMSNorm and input int8 quant<br/>fp32 residual plus Q15.16 linear"]
  B --> C["QKV int8 x int8 GEMM"]
  C --> D["QK RMSNorm, RoPE, R3<br/>TileLang fp32 int8 output"]
  C --> E["V int8 quant"]
  E --> F["V cache and current int8"]
  D --> G["QK int8 x int8 GEMM"]
  G --> H["Single-pass online softmax to P int16<br/>TileLang fp32 state"]
  F --> J["PV integer GEMM<br/>P int16 x V int8"]
  H --> J
  J --> K["attention int8 quant<br/>TileLang fp32 scale"]
  K --> L["O int8 x int8 GEMM"]
  L --> M["RMSNorm and MLP int8 quant<br/>fp32 residual plus Q15.16 O output"]
  M --> N["Gate Up int8 x int8 GEMM"]
  N --> O["SiLU, R4 Hadamard, int8 quant<br/>TileLang fp32"]
  O --> P["Down int8 x int8 GEMM"]
  P --> Q["next fp32 residual path"]
  class B,D,E,K,M,O quant;
  classDef quant fill:#fff3cd,stroke:#d39e00,stroke-width:2px,color:#24292f;
```

Int-only keeps the whole block in fixed-point integer form. The block order is RMSNorm, attention, RMSNorm, MLP. Every matrix multiply is an integer GEMM, including QK and PV in attention.

```mermaid
flowchart TD
  A["Q15.16 residual int32"] --> B["Residual RMSNorm<br/>T.fix rsqrt LUT"]
  B --> C["input_qkv int8 quant"]
  C --> D["QKV int8 x int8 GEMM"]
  D --> E["QK RMSNorm, RoPE, R3<br/>T.fix int8 output"]
  D --> F["V int8 quant"]
  F --> G["V cache and current int8"]
  E --> H["QK int8 x int8 GEMM"]
  H --> I["Online softmax to P int16<br/>T.fix.lut_10bit"]
  G --> K["PV integer GEMM<br/>P int16 x V int8"]
  I --> K
  K --> L["attention int8 quant"]
  L --> M["O int8 x int8 GEMM"]
  M --> N["Residual RMSNorm and MLP int8 quant<br/>T.fix rsqrt LUT"]
  N --> O["Gate Up int8 x int8 GEMM"]
  O --> P["SiLU LUT and R4 Hadamard<br/>int8 quant"]
  P --> Q["Down int8 x int8 GEMM"]
  Q --> R["next Q15.16 residual int32"]
  class C,E,F,L,N,P quant;
  classDef quant fill:#fff3cd,stroke:#d39e00,stroke-width:2px,color:#24292f;
```

The hybrid implementation is in `model_hybrid.py` and `kernels_hybrid.py`; the int-only showcase implementation is in `model_int_only.py` and `kernels_int_only.py`. Packing, QuaRot, PPL, and profiling live under `utils/`.

Kernel numpy prototypes live under `utils/proto/` and can be checked with `python -m examples.qwen3_int_only.utils.proto.run_all`.

## Qwen3-0.6B

2048-token FineWeb quality result against HF bf16:

| backend | tokens | loss | ppl | cos | mse |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF bf16 | 2048 | 3.801437 | 44.765483 | - | - |
| fake-quant | 2048 | 3.818876 | 45.552979 | 0.99498089 | 1.18436021e-01 |
| hybrid | 2048 | 3.813385 | 45.303522 | 0.99503829 | 1.16245255e-01 |
| int-only | 2048 | 3.805970 | 44.968847 | 0.99117421 | 2.01085618e-01 |

Kernel profile for the hybrid LLM block path: 2048 tokens, 28 layers, prefix KV cache enabled, 16 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention_hybrid | 306.169 | 50.52 | 75.78 | 12.14% | 45.13% |
| linear_i8 | 149.406 | 193.18 | 193.18 | 30.96% | 22.02% |
| qk_norm_rope_quant_hybrid | 139.517 | - | - | - | 20.56% |
| silu_hadamard_quant_hybrid | 36.374 | - | - | - | 5.36% |
| rms_quant_hybrid | 31.554 | - | - | - | 4.65% |
| quant_v_i8 | 15.460 | - | - | - | 2.28% |
| total | 678.480 | 65.33 | 76.75 | 12.30% | 100.00% |

Kernel profile for the int-only LLM block path: 2048 tokens, 28 layers, prefix KV cache enabled, 16 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention_i8 | 248.985 | 62.13 | 93.19 | 14.93% | 36.41% |
| qk_norm_rope_i8 | 202.130 | - | - | - | 29.56% |
| linear_i8 | 148.222 | 194.72 | 194.72 | 31.21% | 21.68% |
| rms_sq8 | 37.520 | - | - | - | 5.49% |
| silu_hadamard_i8 | 31.480 | - | - | - | 4.60% |
| quant_v_i8 | 15.464 | - | - | - | 2.26% |
| total | 683.801 | 64.83 | 76.14 | 12.20% | 100.00% |

## Qwen3-14B

2048-token FineWeb quality result against HF bf16:

| backend | tokens | loss | ppl | cos | mse |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF bf16 | 2048 | 3.031365 | 20.725512 | - | - |
| fake-quant | 2048 | 3.056284 | 21.248450 | 0.98738441 | 4.13797397e-01 |
| hybrid | 2048 | 3.062984 | 21.391299 | 0.98538696 | 4.82325177e-01 |
| int-only | 2048 | 3.087627 | 21.924997 | 0.96796332 | 1.08189079e+00 |

Kernel profile for the hybrid LLM block path: 2048 tokens, 40 layers, prefix KV cache enabled, 8 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear_i8 | 1040.170 | 416.21 | 416.21 | 66.70% | 55.86% |
| attention_hybrid | 475.812 | 58.05 | 87.08 | 13.95% | 25.55% |
| qk_norm_rope_quant_hybrid | 182.468 | - | - | - | 9.80% |
| silu_hadamard_quant_hybrid | 84.561 | - | - | - | 4.54% |
| rms_quant_hybrid | 68.081 | - | - | - | 3.66% |
| quant_v_i8 | 10.944 | - | - | - | 0.59% |
| total | 1862.037 | 247.34 | 254.76 | 40.83% | 100.00% |

Kernel profile for the int-only LLM block path: 2048 tokens, 40 layers, prefix KV cache enabled, 8 measured repeats.

| kernel | total ms | math TOPS | tc TOPS | tc util | pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear_i8 | 1039.107 | 416.64 | 416.64 | 66.77% | 56.05% |
| attention_i8 | 384.670 | 71.81 | 107.71 | 17.26% | 20.75% |
| qk_norm_rope_i8 | 272.695 | - | - | - | 14.71% |
| silu_hadamard_i8 | 76.718 | - | - | - | 4.14% |
| rms_sq8 | 69.580 | - | - | - | 3.75% |
| quant_v_i8 | 11.035 | - | - | - | 0.60% |
| total | 1853.806 | 248.44 | 255.89 | 41.01% | 100.00% |

On 0.6B, attention is a large share because the MLP/linear matrices are small. On 14B, the same 2048-token attention work is much less dominant relative to the hidden/intermediate-size linear work, so `linear_i8` becomes the main cost.

Utilization uses the A100 int8 tensorcore peak, 624 TOPS. `math TOPS` is the model matmul work, counted as `2MNK / time`. `tc TOPS` is the int8 tensorcore-equivalent work issued by the current kernels. Attention computes QK once with online softmax, then computes PV as `P int16 x V int8`; the current tensorcore lowering counts this PV as two int8-equivalent GEMMs.

The first run writes `qwen3_int_only.safetensors` and `timestamp`; later script runs reuse the pack when `timestamp` exists and print that timestamp. Use `FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh` to rebuild.
