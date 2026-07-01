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

```bash
/root/venv/bin/python examples/qwen3_int_only/prepack.py \
  --model-dir /publicdata/huggingface.co/Qwen/Qwen3-0.6B \
  --out-dir /tmp/Qwen3-0.6B-int-only-static \
  --use-r1 \
  --use-r2 \
  --use-r3 \
  --calib-parquet /publicdata/huggingface.co/datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet \
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

Run the real-text PPL/cos/MSE baseline on FineWeb windows. HF runs as a torch batch; the TileLang int-only path currently keeps the single-sequence kernel ABI and streams the same windows while accumulating the same metrics.

```bash
/root/venv/bin/python examples/qwen3_int_only/ppl.py \
  --backend hf \
  --model-dir /code/Qwen3-0.6B \
  --eval-parquet fineweb \
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
  --eval-parquet fineweb \
  --max-tokens 2049 \
  --batch-size 2 \
  --num-batches 1 \
  --cache-prompt "你是一个有用而无害的聊天助手。" \
  --use-r1 \
  --use-r2 \
  --use-r3 \
  --split-attn \
  --compare-backend hf
```

Run a cumulative layer sweep against the local float path to locate where fixed-point error starts accumulating:

```bash
/root/venv/bin/python examples/qwen3_int_only/ppl.py \
  --backend int-only \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-int-only-static \
  --eval-parquet fineweb \
  --max-tokens 257 \
  --batch-size 1 \
  --num-batches 1 \
  --use-r1 \
  --use-r2 \
  --use-r3 \
  --split-attn \
  --compare-backend local-float \
  --layer-sweep 1,2,4,8
```

Trace a single block against the dequantized packed-weight float path. This compares the same packed weights and rotations, so the numbers isolate TileLang fixed-point op error instead of QuaRot basis changes:

```bash
/root/venv/bin/python examples/qwen3_int_only/trace_block.py \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-int-only-static \
  --eval-parquet fineweb \
  --max-tokens 65 \
  --layer 0 \
  --use-r3 \
  --split-attn
```

Run the focused tests:

```bash
/root/venv/bin/python -m pytest -q examples/qwen3_int_only/test_example_qwen3_int_only.py -q
```

Run the single-layer kernel profile baseline. Use this no-cache single-layer profile for same-machine relative kernel changes:

```bash
/root/venv/bin/python examples/qwen3_int_only/profile_kernels.py \
  --model-dir /code/Qwen3-0.6B \
  --max-tokens 2049 \
  --layers 1 \
  --warmup 1 \
  --repeat 5 \
  --use-r3
```

Run the split attention profile that materializes `softmax_i16` and uses the `int16 x int8` PV GEMM path:

```bash
/root/venv/bin/python examples/qwen3_int_only/profile_kernels.py \
  --model-dir /code/Qwen3-0.6B \
  --max-tokens 2049 \
  --layers 1 \
  --warmup 1 \
  --repeat 5 \
  --use-r3 \
  --split-attn
```

`--split-attn` is wired for both no-cache and prefix/cache PPL paths.

Recent 2048-token-class Declaration PPL record:

```text
backend=hf tokens=1902 loss=3.106583 ppl=22.344565
backend=int-only --use-r1 --use-r2 --use-r3 tokens=1902 loss=3.372760 ppl=29.158889
backend=int-only --use-r1 --use-r2 --use-r3 --split-attn tokens=1902 loss=3.365579 ppl=28.950245
```

Current FineWeb 2x2048-token baseline with the Chinese cache prompt and static 32x2048 calibration pack:

```text
backend=hf tokens=4096 loss=3.451550 ppl=31.549241
backend=int-only --use-r1 --use-r2 --use-r3 --split-attn tokens=4096 loss=3.628089 ppl=37.640823 compare=hf cos=0.93488973 mse=1.55581174e+00 rel_mse=1.25986741e-01
```

Current FineWeb 256-token cumulative layer sweep against local float:

```text
layers=1 backend=int-only tokens=256 loss=14.241476 ppl=1531067.953902 compare=local-float cos=0.99556057 mse=3.10909846e-01 rel_mse=8.94781789e-03
layers=2 backend=int-only tokens=256 loss=12.809982 ppl=365851.319804 compare=local-float cos=0.99526169 mse=4.73308714e-01 rel_mse=9.70253042e-03
layers=4 backend=int-only tokens=256 loss=12.286460 ppl=216741.336274 compare=local-float cos=0.99096497 mse=9.77206377e-01 rel_mse=1.87624521e-02
layers=8 backend=int-only tokens=256 loss=11.688217 ppl=119159.375157 compare=local-float cos=0.98516666 mse=2.15970396e+00 rel_mse=3.20034156e-02
```

Current 64-token block trace highlights:

```text
layer=0 input_rms rel_mse=1.06552034e-06 q rel_mse=7.91176164e-04 attn rel_mse=2.00207401e-02 mlp rel_mse=4.60865684e-02 layer_out rel_mse=1.74910743e-02
layer=1 input_rms rel_mse=2.37806290e-02 q rel_mse=2.27384176e-02 attn rel_mse=9.51652229e-02 mlp rel_mse=9.14253369e-02 layer_out rel_mse=3.40638570e-02
layer=2 input_rms rel_mse=3.90793644e-02 q rel_mse=4.33117785e-02 attn rel_mse=9.49960873e-02 mlp rel_mse=1.28384978e-02 layer_out rel_mse=1.28336456e-02
```

Current same-machine single-layer 2048-token-class profile baseline (`--warmup 1 --repeat 5`):

```text
default attention:
attention_i8_fixed     avg=   1.748 ms total=    8.740 ms  46.26%
rope_q                 avg=   0.575 ms total=    2.873 ms  15.21%
rope_k                 avg=   0.298 ms total=    1.489 ms   7.88%
gate_up_proj_i8        avg=   0.171 ms total=    0.855 ms   4.53%
qkv_proj_i8            avg=   0.162 ms total=    0.810 ms   4.29%
down_proj_i8           avg=   0.118 ms total=    0.592 ms   3.13%
rms_q_q15              avg=   0.090 ms total=    0.451 ms   2.39%
o_proj_i8              avg=   0.088 ms total=    0.440 ms   2.33%
silu_mul_dq8_mid       avg=   0.072 ms total=    0.361 ms   1.91%
dq8_q_head             avg=   0.057 ms total=    0.287 ms   1.52%
attention_norm         avg=   0.053 ms total=    0.263 ms   1.39%
rms_k_q15              avg=   0.050 ms total=    0.249 ms   1.32%
rms_input_q15          avg=   0.041 ms total=    0.203 ms   1.07%
residual_attn_rms_q15  avg=   0.040 ms total=    0.200 ms   1.06%
dq8_attn               avg=   0.040 ms total=    0.198 ms   1.05%
dq8_v_head             avg=   0.039 ms total=    0.197 ms   1.04%
dq8_kv_head            avg=   0.039 ms total=    0.195 ms   1.03%
dq8_hidden             avg=   0.035 ms total=    0.174 ms   0.92%
dq8_hidden_mlp         avg=   0.034 ms total=    0.170 ms   0.90%
residual_mlp           avg=   0.029 ms total=    0.146 ms   0.77%
total                     18.893 ms

split attention:
rope_q                 avg=   0.578 ms total=    2.888 ms  20.08%
attention_softmax_i16  avg=   0.513 ms total=    2.564 ms  17.82%
attention_i16v8        avg=   0.356 ms total=    1.782 ms  12.39%
rope_k                 avg=   0.306 ms total=    1.530 ms  10.64%
gate_up_proj_i8        avg=   0.172 ms total=    0.862 ms   5.99%
qkv_proj_i8            avg=   0.163 ms total=    0.816 ms   5.67%
down_proj_i8           avg=   0.117 ms total=    0.585 ms   4.07%
rms_q_q15              avg=   0.091 ms total=    0.455 ms   3.17%
o_proj_i8              avg=   0.088 ms total=    0.440 ms   3.06%
dq8_q_head             avg=   0.074 ms total=    0.368 ms   2.56%
silu_mul_dq8_mid       avg=   0.072 ms total=    0.359 ms   2.50%
rms_k_q15              avg=   0.053 ms total=    0.266 ms   1.85%
dq8_kv_head            avg=   0.043 ms total=    0.213 ms   1.48%
rms_input_q15          avg=   0.040 ms total=    0.202 ms   1.41%
residual_attn_rms_q15  avg=   0.040 ms total=    0.202 ms   1.40%
dq8_v_head             avg=   0.037 ms total=    0.187 ms   1.30%
dq8_attn               avg=   0.036 ms total=    0.179 ms   1.24%
dq8_hidden             avg=   0.034 ms total=    0.168 ms   1.17%
dq8_hidden_mlp         avg=   0.033 ms total=    0.163 ms   1.13%
residual_mlp           avg=   0.031 ms total=    0.153 ms   1.06%
total                     14.383 ms
```
