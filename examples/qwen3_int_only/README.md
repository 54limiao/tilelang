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

Run perplexity on the bundled Declaration of Independence text:

```bash
/root/venv/bin/python examples/qwen3_int_only/ppl.py \
  --backend int-only \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-int-only-r12 \
  --max-tokens 2049 \
  --cache-prompt "你是一个有用而无害的聊天助手。" \
  --use-r2 \
  --use-r3
```

Reference backends:

```bash
/root/venv/bin/python examples/qwen3_int_only/ppl.py \
  --backend hf \
  --model-dir /code/Qwen3-0.6B \
  --max-tokens 2049 \
  --cache-prompt "你是一个有用而无害的聊天助手。"
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
  --warmup 2 \
  --repeat 5
```

Run the split attention profile that materializes `softmax_i16` and uses the `int16 x int8` PV GEMM path:

```bash
/root/venv/bin/python examples/qwen3_int_only/profile_kernels.py \
  --model-dir /code/Qwen3-0.6B \
  --max-tokens 2049 \
  --layers 1 \
  --warmup 2 \
  --repeat 5 \
  --split-attn
```

`--split-attn` is wired for both no-cache and prefix/cache PPL paths.

Recent 2048-token-class Declaration PPL record:

```text
backend=hf tokens=1902 loss=3.106583 ppl=22.344565
backend=int-only tokens=1902 loss=3.317960 ppl=27.603982
backend=int-only --split-attn tokens=1902 loss=3.361184 ppl=28.823308
```

Current same-machine single-layer 2048-token-class profile baseline (`--warmup 1 --repeat 5`):

```text
default attention:
attention_i8_fixed     avg=   1.767 ms total=    8.834 ms  54.14%
gate_up_proj_i8        avg=   0.187 ms total=    0.934 ms   5.73%
qkv_proj_i8            avg=   0.171 ms total=    0.856 ms   5.24%
down_proj_i8           avg=   0.121 ms total=    0.603 ms   3.70%
o_proj_i8              avg=   0.107 ms total=    0.534 ms   3.27%
rms_q_q15              avg=   0.095 ms total=    0.474 ms   2.90%
residual_attn_rms_q15  avg=   0.087 ms total=    0.433 ms   2.65%
silu_mul_dq8_mid       avg=   0.079 ms total=    0.394 ms   2.42%
dq8_hidden_mlp         avg=   0.075 ms total=    0.376 ms   2.31%
dq8_attn               avg=   0.075 ms total=    0.374 ms   2.29%
attention_norm         avg=   0.074 ms total=    0.372 ms   2.28%
dq8_q_head             avg=   0.062 ms total=    0.312 ms   1.91%
rope_q                 avg=   0.059 ms total=    0.295 ms   1.81%
rms_k_q15              avg=   0.056 ms total=    0.280 ms   1.72%
rms_input_q15          avg=   0.051 ms total=    0.253 ms   1.55%
dq8_kv_head            avg=   0.044 ms total=    0.219 ms   1.34%
dq8_v_head             avg=   0.042 ms total=    0.208 ms   1.27%
rope_k                 avg=   0.040 ms total=    0.199 ms   1.22%
dq8_hidden             avg=   0.039 ms total=    0.197 ms   1.20%
residual_mlp           avg=   0.034 ms total=    0.172 ms   1.06%
total                     16.318 ms

split attention:
attention_softmax_i16  avg=   0.522 ms total=    2.608 ms  24.95%
attention_i16v8        avg=   0.361 ms total=    1.804 ms  17.26%
gate_up_proj_i8        avg=   0.171 ms total=    0.856 ms   8.19%
qkv_proj_i8            avg=   0.166 ms total=    0.829 ms   7.93%
down_proj_i8           avg=   0.117 ms total=    0.586 ms   5.60%
rms_q_q15              avg=   0.092 ms total=    0.462 ms   4.42%
o_proj_i8              avg=   0.089 ms total=    0.445 ms   4.26%
silu_mul_dq8_mid       avg=   0.072 ms total=    0.362 ms   3.46%
dq8_q_head             avg=   0.057 ms total=    0.283 ms   2.71%
rope_q                 avg=   0.055 ms total=    0.277 ms   2.65%
rms_k_q15              avg=   0.052 ms total=    0.262 ms   2.50%
rms_input_q15          avg=   0.046 ms total=    0.228 ms   2.18%
dq8_kv_head            avg=   0.041 ms total=    0.203 ms   1.94%
residual_attn_rms_q15  avg=   0.040 ms total=    0.198 ms   1.89%
dq8_v_head             avg=   0.038 ms total=    0.192 ms   1.84%
dq8_attn               avg=   0.037 ms total=    0.187 ms   1.79%
rope_k                 avg=   0.037 ms total=    0.184 ms   1.76%
dq8_hidden             avg=   0.035 ms total=    0.175 ms   1.67%
dq8_hidden_mlp         avg=   0.033 ms total=    0.164 ms   1.57%
residual_mlp           avg=   0.030 ms total=    0.149 ms   1.43%
total                     10.453 ms
```
