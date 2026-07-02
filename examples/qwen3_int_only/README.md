# Qwen3 0.6B Integer-Only Prototype

This example runs a Qwen3-0.6B inference path using Q15.16 activations, per-token dynamic quantization, int8 per-channel weights, int8 attention, and integer TileLang kernels for the default path.

The committed TileLang path is W8A8: dynamic quant, fused q/k/v int8 GEMM, fused Q15 RMSNorm, RoPE/R3, SiLU+mul+dynamic-quant, fixed-point attention with GQA/cache, fused attention residual+RMSNorm, and paired gate/up projection.

The expected local model path is:

```bash
/code/Qwen3-0.6B
```

Use a packed directory with QuaRot metadata, for example:

```bash
/tmp/Qwen3-0.6B-int-only-r12
```

For reproducible static-quantization experiments, use the shared model and dataset roots:

```bash
/publicdata/huggingface.co/Qwen/Qwen3-0.6B/
/publicdata/huggingface.co/datasets/
```

Static attention activation scales are packed with the weights. The current packed scale slots are per-head `q_pre_rope_i16`, `k_pre_rope_i16`, `q_post_rope_i8`, `k_post_rope_i8`, and `v_i8`; runtime kernels should read these scales instead of calibrating.
Quantization scales use no-clip ceil amax: `(amax + qmax - 1) // qmax`, clamped to at least 1. Runtime quantization uses integer round-to-nearest on magnitude and restores sign. Re-run `prepack.py` after changing the scale rule because static attention scales are serialized into the pack.

```bash
/root/venv/bin/python examples/qwen3_int_only/prepack.py \
  --model-dir /publicdata/huggingface.co/Qwen/Qwen3-0.6B \
  --out-dir /tmp/Qwen3-0.6B-int-only-static \
  --use-r1 \
  --use-r2 \
  --use-r3 \
  --calib-dataset fineweb \
  --calib-column text \
  --calib-seq-len 2048 \
  --calib-batches 32 \
  --calib-prefix-tokens 512
```

Run a quick perplexity smoke on the bundled Declaration of Independence text:

```bash
/root/venv/bin/python examples/qwen3_int_only/ppl.py \
  --backend int-only \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-int-only-r12 \
  --max-tokens 2049 \
  --cache-prompt "你是一个有用而无害的聊天助手。" \
  --use-r1 \
  --use-r2 \
  --use-r3
```

Run the real-text PPL/cos/MSE/MAE/max-abs baseline on fixed FineWeb windows. HF runs as a torch batch; the TileLang int-only path currently keeps the single-sequence kernel ABI and streams the same windows while accumulating identical metrics. Add `--jsonl-out /tmp/qwen_ppl_metrics.jsonl --jsonl-windows` to append one summary row and one row per 2048-token window for regression tracking. `--eval-dataset fineweb` uses `/publicdata/huggingface.co/datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet`; `--eval-dataset c4` uses `/publicdata/huggingface.co/datasets/allenai/c4/en/c4-train.00000-of-01024.json.gz`.

```bash
/root/venv/bin/python examples/qwen3_int_only/ppl.py \
  --backend hf \
  --model-dir /code/Qwen3-0.6B \
  --eval-dataset fineweb \
  --max-tokens 2049 \
  --batch-size 2 \
  --num-batches 1 \
  --cache-prompt "你是一个有用而无害的聊天助手。"
```

```bash
/root/venv/bin/python examples/qwen3_int_only/ppl.py \
  --backend int-only \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-int-only-static \
  --eval-dataset fineweb \
  --max-tokens 2049 \
  --batch-size 2 \
  --num-batches 1 \
  --cache-prompt "你是一个有用而无害的聊天助手。" \
  --use-r1 \
  --use-r2 \
  --use-r3 \
  --split-attn \
  --compare-backend hf \
  --jsonl-out /tmp/qwen_ppl_metrics.jsonl \
  --jsonl-windows
```

Run a cumulative layer sweep against the local float path to locate where fixed-point error starts accumulating:

```bash
/root/venv/bin/python examples/qwen3_int_only/ppl.py \
  --backend int-only \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-int-only-static \
  --eval-dataset fineweb \
  --max-tokens 257 \
  --batch-size 1 \
  --num-batches 1 \
  --use-r1 \
  --use-r2 \
  --use-r3 \
  --split-attn \
  --compare-backend local-float \
  --layer-sweep 1,2,4,8 \
  --jsonl-out /tmp/qwen_layer_sweep.jsonl
```

Trace a single block against the dequantized packed-weight float path. This compares the same packed weights and rotations, so the numbers isolate TileLang fixed-point op error instead of QuaRot basis changes:

```bash
/root/venv/bin/python examples/qwen3_int_only/trace_block.py \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-int-only-static \
  --eval-dataset fineweb \
  --max-tokens 65 \
  --layer 0 \
  --use-r3 \
  --split-attn \
  --jsonl-out /tmp/qwen_trace_metrics.jsonl
```

Trace multiple layers in one model run with `--layers`:

```bash
/root/venv/bin/python examples/qwen3_int_only/trace_block.py \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-int-only-static \
  --eval-dataset fineweb \
  --max-tokens 65 \
  --layers 4,7 \
  --use-r3 \
  --split-attn \
  --jsonl-out /tmp/qwen_trace_metrics.jsonl
```

Run the focused tests:

```bash
/root/venv/bin/python -m pytest -q examples/qwen3_int_only/test_example_qwen3_int_only.py -q
```

Run the single-layer kernel profile baseline. Use this no-cache single-layer profile for same-machine relative kernel changes:

```bash
/root/venv/bin/python examples/qwen3_int_only/profile_kernels.py \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-int-only-static \
  --max-tokens 2049 \
  --layers 1 \
  --warmup 1 \
  --repeat 5 \
  --use-r3 \
  --jsonl-out /tmp/qwen_profile_metrics.jsonl
```

Run the split attention profile that materializes `softmax_i16` and uses the `int16 x int8` PV GEMM path:

```bash
/root/venv/bin/python examples/qwen3_int_only/profile_kernels.py \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-int-only-static \
  --max-tokens 2049 \
  --layers 1 \
  --warmup 1 \
  --repeat 5 \
  --use-r3 \
  --split-attn \
  --jsonl-out /tmp/qwen_profile_metrics.jsonl
```

Run the fused no-cache attention profile. This keeps the same two-stage fixed-point probability and `int16 x int8` PV semantics as split attention, but computes PV inside the attention kernel instead of writing and reloading the full `P[q_heads, seqlen, seqlen]` tensor:

```bash
/root/venv/bin/python examples/qwen3_int_only/profile_kernels.py \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-int-only-static \
  --max-tokens 2049 \
  --layers 1 \
  --warmup 1 \
  --repeat 5 \
  --use-r3 \
  --fused-attn \
  --jsonl-out /tmp/qwen_profile_metrics.jsonl
```

Add `--fast-hadamard` to profile the optional approximate warp-shuffle R3 path. It is off by default because it uses a symmetric `+/-22` coefficient instead of the exact dense `+22/-23` fixed-point R3 coefficients.

Run the prefix/cache attention micro-profile. This isolates the cache attention body without the rest of the block:

```bash
/root/venv/bin/python examples/qwen3_int_only/profile_kernels.py \
  --model-dir /code/Qwen3-0.6B \
  --max-tokens 257 \
  --cache-len 32 \
  --warmup 1 \
  --repeat 10
```

```bash
/root/venv/bin/python examples/qwen3_int_only/profile_kernels.py \
  --model-dir /code/Qwen3-0.6B \
  --max-tokens 257 \
  --cache-len 32 \
  --warmup 1 \
  --repeat 10 \
  --fused-attn
```

Use `--static-cache` to isolate the static per-head current-K/V cache fused kernel used by the packed static path:

```bash
/root/venv/bin/python examples/qwen3_int_only/profile_kernels.py \
  --model-dir /code/Qwen3-0.6B \
  --max-tokens 257 \
  --cache-len 32 \
  --warmup 1 \
  --repeat 10 \
  --static-cache \
  --jsonl-out /tmp/qwen_profile_metrics.jsonl
```

Run the prefix/cache full-block profile. This builds a HF cache for the cache prompt, quantizes cache K/V with the active rotations, and profiles the whole TileLang block rather than only the attention body:

```bash
/root/venv/bin/python examples/qwen3_int_only/profile_kernels.py \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-int-only-static \
  --max-tokens 257 \
  --layers 1 \
  --warmup 2 \
  --repeat 10 \
  --use-r2 \
  --use-r3 \
  --fused-attn \
  --cache-block \
  --jsonl-out /tmp/qwen_profile_metrics.jsonl
```

`--split-attn` and `--fused-attn` are wired for both no-cache and prefix/cache PPL paths. Fused attention keeps the same two-stage fixed-point probability semantics as split attention but computes PV inside the attention kernel instead of writing and reloading `P`.

Recent 2048-token-class Declaration PPL record:

```text
backend=hf tokens=1902 loss=3.106583 ppl=22.344565
backend=int-only --use-r1 --use-r2 --use-r3 tokens=1902 loss=3.372760 ppl=29.158889
backend=int-only --use-r1 --use-r2 --use-r3 --split-attn tokens=1902 loss=3.299825 ppl=27.107887 compare=hf cos=0.97377499 mse=1.27046896e+00 rel_mse=5.34204678e-02
```

Current FineWeb 2x2048-token baseline with the Chinese cache prompt and static 32x2048 calibration pack:

```text
backend=hf tokens=4096 loss=3.451550 ppl=31.549241
backend=int-only --use-r1 --use-r2 --use-r3 --split-attn tokens=4096 loss=3.628089 ppl=37.640823 compare=hf cos=0.93488973 mse=1.55581174e+00 rel_mse=1.25986741e-01
```

Prefix/cache is the required quality measurement path because it removes the early-token outlier regime. On a single 2048-token FineWeb window with the Chinese cache prompt, the current float QuaRot/cache upper bound is close to HF (`rotated_cache cos=0.999855`, `mse=1.0534e-02` on the 256-token diagnostic), so the target `cos > 0.99` is not blocked by the rotation/cache semantics. The fused PV path must split the non-negative `prob * v_scale / 16383` coefficient into three base-128 int8 digits; the old two-byte split overflowed when late-layer V scales exceeded the signed high byte range.

```text
hf prefix tokens=2048 loss=3.801437 ppl=44.765483
dynamic int-only prefix tokens=2048 loss=4.037960 ppl=56.710538 compare=hf cos=0.95941293 mse=9.18609851e-01 rel_mse=8.05233916e-02
static int-only prefix fused+fast-hadamard tokens=2048 loss=3.835020 ppl=46.294334 compare=hf cos=0.98933302 mse=2.42206294e-01 rel_mse=2.12312901e-02
static int-only prefix fused+fast-hadamard --mlp-i16-layers 20-27 tokens=2048 loss=3.827422 ppl=45.943948 compare=hf cos=0.99145196 mse=1.97211390e-01 rel_mse=1.72871323e-02
static int-only prefix fused+fast-hadamard --mlp-i16-layers 16-27 tokens=2048 loss=3.826508 ppl=45.901949 compare=hf cos=0.99214346 mse=1.80577228e-01 rel_mse=1.58290168e-02
static int-only prefix fused+fast-hadamard --mlp-i16-layers 12-27 tokens=2048 loss=3.824328 ppl=45.802009 compare=hf cos=0.99238650 mse=1.74848317e-01 rel_mse=1.53268327e-02
static int-only prefix fused+fast-hadamard --mlp-i16 tokens=2048 loss=3.817359 ppl=45.483925 compare=hf cos=0.99371496 mse=1.44382801e-01 rel_mse=1.26562902e-02
static int-only prefix fused+fast-hadamard tokens=256 loss=4.485878 ppl=88.754843 compare=hf cos=0.98688868 mse=2.12787863e-01 rel_mse=2.61091964e-02
static int-only prefix fused+fast-hadamard --mlp-i16-layers 20-27 tokens=256 loss=4.481172 ppl=88.338152 compare=hf cos=0.99050064 mse=1.54476767e-01 rel_mse=1.89543905e-02
static int-only prefix fused+fast-hadamard --mlp-i16-layers 16-27 tokens=256 loss=4.490720 ppl=89.185661 compare=hf cos=0.99118570 mse=1.43219328e-01 rel_mse=1.75730962e-02
static int-only prefix fused+fast-hadamard --mlp-i16-layers 12-27 tokens=256 loss=4.479905 ppl=88.226302 compare=hf cos=0.99160920 mse=1.36349685e-01 rel_mse=1.67301869e-02
static int-only prefix fused+fast-hadamard --mlp-i16 tokens=256 loss=4.451707 ppl=85.773265 compare=hf cos=0.99361168 mse=1.03831085e-01 rel_mse=1.27401354e-02
static int-only prefix split tokens=256 loss=4.509197 ppl=90.848816 compare=hf cos=0.98297748 mse=2.75460453e-01 rel_mse=3.37991602e-02
```

The remaining W8A8 gap is mainly hidden-state error amplified by the final LM head. On the 256-token prefix diagnostic, the packed-float reference is still close to HF (`cos=0.99909288`), while the int hidden state before final RMSNorm is `cos=0.99377662` and the resulting logits are `cos=0.98688871`. Replacing the MLP branch with the float reference in a local diagnostic lifts logits to `cos=0.99771351`; replacing attention lifts them to `cos=0.99233556`. The `--mlp-i16` quality path keeps attention/projections W8A8 but quantizes the SiLU-gated MLP activation to int16 and uses int16xint8 down projection; it crosses the `cos > 0.99` target on the 2048-token prefix run. `--mlp-i16-layers` enables this path only for selected zero-based layers. The current speed/quality recommendation is `--mlp-i16-layers 20-27`, which crosses `cos > 0.99`; `12-27` gives better error metrics with more int16 down-projection cost.

Current FineWeb 256-token cumulative layer sweep against local float:

```text
layers=1 backend=int-only tokens=256 loss=14.298169 ppl=1620376.578481 compare=local-float cos=0.99740373 mse=1.80444243e-01 rel_mse=5.19308812e-03
layers=2 backend=int-only tokens=256 loss=12.875274 ppl=390535.324770 compare=local-float cos=0.99801277 mse=1.95146106e-01 rel_mse=4.00037221e-03
layers=4 backend=int-only tokens=256 loss=12.392824 ppl=241065.551394 compare=local-float cos=0.99683737 mse=3.35850729e-01 rel_mse=6.44836483e-03
layers=8 backend=int-only tokens=256 loss=11.571200 ppl=106000.635758 compare=local-float cos=0.99299909 mse=1.00256420e+00 rel_mse=1.48564244e-02
```

Use the metrics at different levels. Real-text PPL is the end-to-end language-modeling result. Logit `cos`, `mse`, `mae`, `max_abs`, and `rel_mse` are the quantitative regression metrics for comparing int-only runs against HF or local float. The cumulative layer sweep localizes where hidden-state drift starts, the block trace localizes whether the drift is from q/k/v quantization, softmax, PV, MLP, residual add, or RMSNorm, and the profile JSONL records the full kernel time split for same-machine optimization tracking. `trace_block.py --jsonl-out` appends one row per layer/op metric, so quality regressions can be filtered by `layer`, `name`, and `kind` instead of copying console text. `profile_kernels.py` also reports estimated `gops` and `tops` for GEMM-like and dense-RoPE kernels; this is a same-machine utilization signal, not a hardware counter.

Current 64-token block trace highlights:

```text
layer=0 input_rms rel_mse=1.06552034e-06 q rel_mse=7.11495290e-04 attn rel_mse=2.02661175e-02 softmax_i16 rel_mse=8.41666572e-03 pv_i16v8 rel_mse=2.45846040e-03 mlp rel_mse=3.71044017e-02 layer_out rel_mse=1.50453513e-02
layer=1 input_rms rel_mse=1.55158173e-02 q rel_mse=1.29401488e-02 attn rel_mse=4.41492461e-02 softmax_i16 rel_mse=1.66352175e-03 pv_i16v8 rel_mse=1.23040669e-03 mlp rel_mse=4.23145220e-02 layer_out rel_mse=1.83776282e-02
layer=2 input_rms rel_mse=1.77999847e-02 q rel_mse=1.67836715e-02 attn rel_mse=4.16141599e-02 mlp rel_mse=4.86866618e-03 layer_out rel_mse=4.85706003e-03
```

The split attention sub-trace compares `softmax_i16` and `pv_i16v8` against torch references using the already-quantized q/k/v inputs. On these samples their own rel_mse is small, so the larger attention rel_mse is dominated by quantized inputs and layer-to-layer accumulation rather than the PV GEMM math.

Current q/k/v QDQ loss is also small. On layer 1, `q_qdq_loss rel_mse=5.37294778e-04`, `k_qdq_loss rel_mse=1.07717264e-04`, and `v_qdq_loss rel_mse=8.31166573e-04`, while the pre-quant q/k/v tensors are already at `2-4%` rel_mse. That points to accumulated hidden-state error before q/k/v quantization rather than bad static q/k/v scales.

MLP-side QDQ improved after switching runtime quantization to integer round-to-nearest. On layer 2, `gated_qdq_loss rel_mse` dropped from `3.66984569e-02` to `3.33172247e-05`, and `layer_out rel_mse` dropped from `1.36468485e-02` to `4.85706003e-03`. Projection lowering remains small: `down_from_gated_qdq rel_mse=1.79821334e-04` on layer 2.

The deeper MLP trace shows the SiLU kernel itself is accurate against torch using the same fixed-point gate/up inputs: `silu_mul rel_mse=4.55698144e-04` on layer 0 and `2.64857546e-04` on layer 1. Gate/up projection from post-RMS QDQ is also small (`gate_from_post_qdq rel_mse=2.11212205e-06`, `up_from_post_qdq rel_mse=2.06542627e-05` on layer 1). The large gated-vs-float error is therefore inherited from earlier hidden-state/residual/RMSNorm drift, not the SiLU or gate/up kernels.

Residual/RMSNorm trace now splits kernel-local error from input drift. `attn_resid_add` and `layer_out_add` are exact on layers 0 and 1. Dynamic quantization now uses no-clip ceil amax scales; this removes the previous RMSNorm-local max clipping error. Layer 0 has `post_rms_kern rel_mse=5.62097284e-07` while `post_rms_int rel_mse=1.22313248e-02`; layer 1 has `post_rms_kern rel_mse=6.15960857e-07` while `post_rms_int rel_mse=2.08080132e-02`. Layer 1 input RMSNorm has the same split: `input_rms_kern rel_mse=6.40880103e-07` and `input_rms_int rel_mse=1.60594936e-02`. So residual add and RMSNorm kernels are no longer quality sources; the remaining larger error is already present in the int hidden state entering RMSNorm.

Repacking the static attention calibration with the same no-clip ceil scale rule makes the serialized q/k/v scales consistent, but it is not the main quality lever on the current FineWeb 256-token sweep. With `/tmp/Qwen3-0.6B-static-ceil-calib-32x2048`, the layer sweep was `layers=1 rel_mse=6.17165126e-03`, `layers=2 rel_mse=5.93324587e-03`, `layers=4 rel_mse=1.59930011e-02`, and `layers=8 rel_mse=3.27977759e-02`. The main RMSNorm improvement comes from runtime dynamic no-clip scale, while deeper-layer drift still needs attention/MLP hidden-state work.

Deeper block traces should be read with the `*_out` normalized metrics, not only branch-local rel_mse. On layer 4, `attn_out rel_mse=3.04542363e-01` and `mlp rel_mse=2.01169401e-01`, but their contribution relative to final hidden energy is only `attn_out_out rel_to_out=1.13196293e-05` and `mlp_out rel_to_out=2.01086641e-05`; `hidden_in_out rel_to_out=1.36730000e-02` already matches the final `layer_out rel_mse=1.36817088e-02`. Layer 7 is similar: `hidden_in_out rel_to_out=1.37537895e-02`, while `attn_out_out rel_to_out=2.94227393e-05` and `mlp_out rel_to_out=9.25156637e-05`. This means the large branch-local rel_mse is mostly a small-energy branch effect; the final hidden drift is inherited from earlier layers.

Early-layer attribution now shows MLP gated quantization was a real quality source and projection GEMM lowering is still small. After round-to-nearest, layer 1 has `hidden_in_out rel_to_out=7.86277559e-03`, `attn_out_out rel_to_out=2.98309419e-03`, and `mlp_out rel_to_out=1.18715856e-02`; layer 2 has `mlp_out rel_to_out=4.86669596e-03` with `gated_qdq_loss rel_mse=3.33172247e-05`. The projection checks stay small: layer 1 has `o_from_attn_qdq rel_mse=3.48534813e-04` and `down_from_gated_qdq rel_mse=3.15667829e-04`, while layer 2 has `down_from_gated_qdq rel_mse=1.79821334e-04`.

Current same-machine single-layer 2048-token-class profile baseline. Compare `avg` and `TOPS` across rows; `total` is the sum over that run's `repeat`. Default and split rows are the earlier `--warmup 1 --repeat 5` records, fused no-cache uses `--warmup 2 --repeat 10`, and fused fast-hadamard uses `--warmup 1 --repeat 5`:

```text
default attention:
attention_i8_fixed     avg=   1.745 ms total=    8.725 ms  48.13%
rope_sq8_q_attn        avg=   0.557 ms total=    2.784 ms  15.36%
rope_sq8_k_attn        avg=   0.288 ms total=    1.438 ms   7.93%
gate_up_proj_i8        avg=   0.169 ms total=    0.844 ms   4.65%
qkv_proj_i8            avg=   0.160 ms total=    0.801 ms   4.42%
down_proj_i8           avg=   0.116 ms total=    0.581 ms   3.20%
o_proj_i8              avg=   0.088 ms total=    0.440 ms   2.43%
rms_q_q15              avg=   0.087 ms total=    0.435 ms   2.40%
silu_mul_dq8_mid       avg=   0.072 ms total=    0.358 ms   1.97%
rms_k_q15              avg=   0.049 ms total=    0.246 ms   1.36%
attention_norm         avg=   0.049 ms total=    0.244 ms   1.34%
sq8_v_attn             avg=   0.044 ms total=    0.222 ms   1.22%
residual_attn_rms_q15  avg=   0.037 ms total=    0.187 ms   1.03%
rms_input_q15          avg=   0.037 ms total=    0.186 ms   1.03%
dq8_attn               avg=   0.036 ms total=    0.181 ms   1.00%
dq8_hidden             avg=   0.032 ms total=    0.162 ms   0.90%
dq8_hidden_mlp         avg=   0.031 ms total=    0.153 ms   0.85%
residual_mlp           avg=   0.028 ms total=    0.141 ms   0.78%
total                     18.127 ms

split attention:
rope_sq8_q_attn        avg=   0.558 ms total=    2.790 ms  20.67%
attention_softmax_i16  avg=   0.509 ms total=    2.543 ms  18.85%
attention_i16v8        avg=   0.354 ms total=    1.770 ms  13.12%
rope_sq8_k_attn        avg=   0.288 ms total=    1.442 ms  10.69%
gate_up_proj_i8        avg=   0.169 ms total=    0.844 ms   6.25%
qkv_proj_i8            avg=   0.162 ms total=    0.810 ms   6.00%
down_proj_i8           avg=   0.115 ms total=    0.573 ms   4.24%
rms_q_q15              avg=   0.088 ms total=    0.442 ms   3.28%
o_proj_i8              avg=   0.086 ms total=    0.432 ms   3.20%
silu_mul_dq8_mid       avg=   0.070 ms total=    0.352 ms   2.61%
rms_k_q15              avg=   0.050 ms total=    0.248 ms   1.84%
sq8_v_attn             avg=   0.046 ms total=    0.232 ms   1.72%
rms_input_q15          avg=   0.039 ms total=    0.194 ms   1.44%
residual_attn_rms_q15  avg=   0.037 ms total=    0.183 ms   1.35%
dq8_attn               avg=   0.034 ms total=    0.171 ms   1.27%
dq8_hidden             avg=   0.033 ms total=    0.167 ms   1.23%
dq8_hidden_mlp         avg=   0.031 ms total=    0.155 ms   1.15%
residual_mlp           avg=   0.029 ms total=    0.147 ms   1.09%
total                     13.495 ms

fused no-cache attention:
attention_i8v8_fused_static avg=   0.611 ms total=    6.107 ms  25.83%   71.73 TOPS
rope_sq8_q_attn_noscale avg=   0.553 ms total=    5.533 ms  23.40%    1.79 TOPS
rope_sq8_k_attn_noscale avg=   0.280 ms total=    2.800 ms  11.84%    1.77 TOPS
qkv_proj_i8            avg=   0.160 ms total=    1.600 ms   6.77%   99.00 TOPS
gate_up_proj_i8        avg=   0.158 ms total=    1.583 ms   6.70%  150.06 TOPS
down_proj_i8           avg=   0.098 ms total=    0.977 ms   4.13%  121.53 TOPS
rms_q_q15              avg=   0.087 ms total=    0.871 ms   3.68%
silu_mul_dq8_mid       avg=   0.070 ms total=    0.702 ms   2.97%
o_proj_i8              avg=   0.065 ms total=    0.654 ms   2.76%  121.15 TOPS
rms_k_q15              avg=   0.049 ms total=    0.487 ms   2.06%
sq8_v_attn_noscale     avg=   0.038 ms total=    0.378 ms   1.60%
residual_attn_rms_q15  avg=   0.037 ms total=    0.366 ms   1.55%
rms_input_q15          avg=   0.036 ms total=    0.359 ms   1.52%
dq8_attn               avg=   0.032 ms total=    0.324 ms   1.37%
dq8_hidden             avg=   0.032 ms total=    0.321 ms   1.36%
dq8_hidden_mlp         avg=   0.031 ms total=    0.306 ms   1.30%
residual_mlp           avg=   0.028 ms total=    0.275 ms   1.16%
total                     23.642 ms

fused no-cache attention with `--fast-hadamard`:
attention_i8v8_fused_static avg=   0.613 ms total=    3.063 ms  38.36%   95.35 TOPS
qkv_proj_i8            avg=   0.157 ms total=    0.787 ms   9.85%  100.67 TOPS
gate_up_proj_i8        avg=   0.137 ms total=    0.683 ms   8.55%  173.93 TOPS
rope_sq8_q_attn_hadamard avg=   0.094 ms total=    0.468 ms   5.86%   10.57 TOPS
down_residual_i8       avg=   0.091 ms total=    0.453 ms   5.67%  131.23 TOPS
rms_q_q15              avg=   0.090 ms total=    0.448 ms   5.61%
o_proj_i8              avg=   0.064 ms total=    0.322 ms   4.03%  122.99 TOPS
silu_mul_dq8_mid_fast  avg=   0.063 ms total=    0.314 ms   3.94%
rms_input_dq8_fast     avg=   0.058 ms total=    0.290 ms   3.64%
rope_sq8_k_attn_hadamard avg=   0.057 ms total=    0.286 ms   3.58%    8.66 TOPS
rms_k_q15              avg=   0.051 ms total=    0.257 ms   3.22%
residual_attn_rms_dq8_fast avg=   0.047 ms total=    0.233 ms   2.92%
sq8_v_attn_noscale     avg=   0.040 ms total=    0.200 ms   2.51%
dq8_attn               avg=   0.036 ms total=    0.180 ms   2.26%
total                      7.984 ms

The `--mlp-i16` quality path adds a third MLP precision mode. On the same single-layer 2048-token-class fused fast-Hadamard profile with `--warmup 1 --repeat 3`, the default W8A8 path measured `total=5.166 ms`; `--mlp-i16` measured `total=5.755 ms`. The added cost is concentrated in `down_residual_i16 avg=0.289 ms` versus `down_residual_i8 avg=0.087 ms`, while attention remains about `0.757 ms`. On the full 28-layer prefix/cache profile with `--warmup 1 --repeat 2`, `--mlp-i16-layers 20-27` measured `total=105.981 ms` and `--mlp-i16-layers 12-27` measured `total=108.670 ms`; attention stayed the largest item at about `47-50%`.

fast-hadamard short quality check:
backend=int-only --use-r1 --use-r2 --use-r3 --fused-attn --fast-hadamard tokens=256 loss=13.333147 ppl=617322.618637 compare=local-float cos=0.99898713 mse=8.64596025e-02 rel_mse=2.04769744e-03

prefix/cache attention micro-profile (`--max-tokens 257 --cache-len 32 --warmup 1 --repeat 10`):
split cache:
attention_cache_softmax_i16 avg=   0.068 ms total=    0.676 ms  61.35%
attention_cache_i16v8  avg=   0.043 ms total=    0.426 ms  38.65%
total                      1.101 ms

fused cache:
attention_cache_i8v8_fused avg=   0.053 ms total=    0.529 ms 100.00%
total                      0.529 ms

static fused cache:
attention_cache_i8v8_fused_static avg=   0.061 ms total=    0.608 ms 100.00%   19.86 TOPS
total                      0.608 ms

prefix/cache full block (`--max-tokens 257 --warmup 2 --repeat 10 --use-r2 --use-r3 --fused-attn --cache-block`):
rope_sq8_q_attn_noscale avg=   0.092 ms total=    0.915 ms  13.25%
attention_cache_i8v8_fused_static avg=   0.062 ms total=    0.617 ms   8.94%
qkv_proj_i8            avg=   0.058 ms total=    0.584 ms   8.46%
rope_sq8_k_attn_noscale avg=   0.055 ms total=    0.551 ms   7.98%
gate_up_proj_i8        avg=   0.046 ms total=    0.463 ms   6.70%
silu_mul_dq8_mid       avg=   0.036 ms total=    0.362 ms   5.24%
rms_input_q15          avg=   0.035 ms total=    0.351 ms   5.09%
sq8_v_attn_noscale     avg=   0.034 ms total=    0.340 ms   4.92%
down_proj_i8           avg=   0.034 ms total=    0.337 ms   4.88%
residual_attn_rms_q15  avg=   0.031 ms total=    0.314 ms   4.55%
dq8_hidden             avg=   0.031 ms total=    0.313 ms   4.53%
dq8_attn               avg=   0.031 ms total=    0.310 ms   4.48%
o_proj_i8              avg=   0.031 ms total=    0.306 ms   4.42%
rms_q_q15              avg=   0.030 ms total=    0.304 ms   4.40%
dq8_hidden_mlp         avg=   0.030 ms total=    0.300 ms   4.34%
rms_k_q15              avg=   0.028 ms total=    0.278 ms   4.02%
residual_mlp           avg=   0.026 ms total=    0.262 ms   3.80%
total                      6.907 ms

prefix/cache full block with `--fast-hadamard` (`--warmup 1 --repeat 5`):
attention_cache_i8v8_fused_static avg=   0.063 ms total=    0.317 ms  10.16%   17.58 TOPS
qkv_proj_i8            avg=   0.059 ms total=    0.297 ms   9.51%   36.15 TOPS
gate_up_proj_i8        avg=   0.047 ms total=    0.237 ms   7.60%   67.89 TOPS
rms_input_q15          avg=   0.037 ms total=    0.187 ms   5.98%
sq8_v_attn_noscale     avg=   0.036 ms total=    0.179 ms   5.74%
silu_mul_dq8_mid       avg=   0.036 ms total=    0.179 ms   5.72%
down_proj_i8           avg=   0.035 ms total=    0.173 ms   5.54%   46.56 TOPS
rope_sq8_q_attn_hadamard avg=   0.033 ms total=    0.167 ms   5.36%    4.01 TOPS
residual_attn_rms_q15  avg=   0.032 ms total=    0.162 ms   5.17%
dq8_hidden             avg=   0.032 ms total=    0.161 ms   5.14%
rms_q_q15              avg=   0.032 ms total=    0.158 ms   5.05%
o_proj_i8              avg=   0.031 ms total=    0.157 ms   5.03%   34.21 TOPS
dq8_hidden_mlp         avg=   0.031 ms total=    0.156 ms   4.99%
dq8_attn               avg=   0.031 ms total=    0.155 ms   4.95%
rms_k_q15              avg=   0.030 ms total=    0.148 ms   4.74%
residual_mlp           avg=   0.029 ms total=    0.146 ms   4.68%
rope_sq8_k_attn_hadamard avg=   0.029 ms total=    0.145 ms   4.63%    2.32 TOPS
total                      3.122 ms
```

The single-layer profile intentionally includes every block component, not only attention kernels. The static q/k/v quantizers emit attention-native head-major tensors, and the static fused hot paths now keep q/k/v scales as per-head vectors instead of materializing expanded `(heads, seq)` scale matrices. q/k fuse `RoPE + R3 + static quant` into `rope_sq8_*_attn`, removing the previous q15_16 intermediate tensors from the packed static path. The fused no-cache attention path removes the global `P` write/read and reduces the attention body from about `0.819 ms` (`attention_softmax_i16 + attention_i16v8`) to about `0.613 ms` with static per-head scale ABI and one precomputed score scale per q/k head pair. The fused cache attention path similarly reduces the isolated prefix/cache attention body from about `0.111 ms` per repeat to about `0.053 ms`; the full cache-block profile shows the decode/prefix bottlenecks are now spread across `rope_sq8_q_attn_noscale`, `attention_cache_i8v8_fused_static`, `qkv_proj_i8`, `rope_sq8_k_attn_noscale`, and `sq8_v_attn_noscale`, not only the attention body. Its row tile is `block_m=16`, which was bitwise identical to the previous cache fused output and faster on the cache micro-profile. The projection tiles now use real-tensor sweep winners that were bitwise identical to the previous outputs: qkv `64x128x64`, o `64x64x64`, gate/up `64x128x64`, and down `64x64x64`. This mainly improves gate/up and down in the current no-cache profile. The MLP hot path now uses `silu_mul_dq8_mid_fast`, which keeps the same fixed-point SiLU and dynamic quantization output but skips writing the debug-only int32 gated tensor; the full-output kernel is still used when tracing with `collect=True`. The final MLP projection now uses `down_residual_i8` in inference/profile, fusing down projection with the final residual add while trace keeps separate `mlp` and `layer_out` tensors. Input RMSNorm now uses `rms_input_dq8_fast` in inference/profile, fusing RMSNorm and dynamic int8 quantization while trace keeps the full input-RMS tensor. The post-attention residual path similarly uses `residual_attn_rms_dq8_fast`, fusing residual add, post-RMSNorm, and dynamic int8 quantization while trace keeps the full post-RMS tensor. q/k RMSNorm now uses a rowwise grouped kernel that computes the row amax once and serializes groups within that row; it is bitwise identical to the previous grouped kernel and reduces the latest fused fast-Hadamard profile from about `7.817 ms` to `7.662 ms` (`rms_q_q15` about `0.088 -> 0.074 ms`). The exact dense-R3 no-cache profile is dominated by `attention_i8v8_fused_static + rope_sq8_q_attn_noscale + rope_sq8_k_attn_noscale`. The optional `--fast-hadamard` path replaces dense R3 with a warp-shuffle Hadamard pass; it is approximate because it uses `+/-22` instead of the exact dense `+22/-23` coefficients. On the local 256-token, 16-head q test it gives `cos=0.99961817`, `rel_mse=1.25598e-03`, and reduces the q-side R3/RoPE/static-quant kernel from about `0.084 ms` to about `0.021 ms`. In the latest full 2048-token-class profile it reduces q/k RoPE-R3 from `0.553/0.280 ms` to `0.094/0.057 ms`; the same profile reports `attention_i8v8_fused_static` at `0.613 ms` and `95.35 TOPS` after counting the split PV path as two int8 GEMMs, with total block time at `7.984 ms`. In the prefix/cache full-block profile, `--fast-hadamard` reduces q/k RoPE-R3 from `0.092/0.055 ms` to `0.033/0.029 ms`, while `attention_cache_i8v8_fused_static` stays about `0.063 ms`. A Q/K RMSNorm launch-fusion trial was bitwise identical to separate `rms_q_q15 + rms_k_q15`, but measured `0.137 ms`, the same as the separated stable profile (`0.087 + 0.050 ms`), so it was removed. A QKV+V-attention-quant fusion trial emitted `V_ATT` directly from qkv projection; it was bitwise identical to `qkv_proj_i8 + sq8_v_attn_noscale`, but real layer-0 1888-token timing regressed from about `0.141 ms` to `0.202 ms`, so it was removed. A GQA2 fused-static trial combined the two query heads that share one kv head so K/V shared-memory loads are reused; it is bitwise identical to the current fused-static kernel, but on the 1888-token synthetic micro-benchmark it stabilized around `0.713 ms` versus current fused-static around `0.598 ms`, so it is not the default path. A trial that fused q/k RMSNorm with RoPE+R3+static quant was bitwise identical to the current separated kernels, but regressed the cache-block profile (`rms_rope_sq8_q_attn` about `0.288 ms`, `rms_rope_sq8_k_attn` about `0.159 ms`, total about `2.955 ms`), so it was removed. A trial that replaced dense R3 rotation with a shared-memory signed-Hadamard butterfly was exact but regressed badly (`rope_q` about `5.434 ms`, `rope_k` about `2.736 ms`, total about `50.565 ms`), so that implementation was not kept. A second fast-Hadamard prototype tried to exploit the structured R3 matrix directly, but current TileLang lowering rejects the butterfly's dynamic paired fragment indices, so that code was removed before commit. Two smaller R3 experiments were also rejected: storing `R >> 8` as int8 was exact but slowed the fused profile to about `13.130 ms`, and merging q/k RoPE-static-quant into one conditional kernel slowed it to about `16.763 ms`. A single-pass online fused attention experiment avoided the second QK GEMM and ran the attention body at about `0.541 ms` versus `0.621 ms` on a synthetic 1888-token micro-benchmark, but rescaling already-quantized PV accumulators across K blocks only reached about `cos=0.972` versus the two-pass fused path on the 128-token test. That path was removed; the current fused kernel keeps final-denominator probability semantics. The latest real-input fused-static tile sweep confirms `block_m=32, block_n=64` is the current no-cache default: `block_m=16/32/64, block_n=64` are bitwise identical, but on the layer-0 1888-token attention tensors `block_m=32` is fastest at about `0.592-0.595 ms`, while `block_m=16` is about `0.621 ms` and `block_m=64` is about `0.692 ms`. `block_n=64` remains the semantic anchor; `block_n=32` or `128` changes fixed-point softmax block rounding (`cos` around `0.99994`, max_abs up to `6953`), so those variants are not defaults.

Current default-vs-split attention check on the 256-token Declaration window with the static calibration pack shows they are close after the q/k fused path: layer 0 `cos=0.99930203 rel_mse=1.396672e-03`, layer 1 `cos=0.99869299 rel_mse=2.615922e-03`; default loss was `12.107508` and split loss was `12.134119` on this short diagnostic. The split path remains the better current speed path because it uses block GEMMs for both score and PV, while the default online path reduces IO but is still per-token and slower in the profile above.
