# Qwen3 0.6B Integer-Only Prototype

This example runs a Qwen3-0.6B inference path using Q15.16 activations, per-token dynamic quantization, int8 per-channel weights, int8 attention, and integer TileLang kernels for the default path.

The committed TileLang path is W8A8: dynamic quant, fused q/k/v int8 GEMM, fused Q15 RMSNorm, RoPE/R3, SiLU+mul+dynamic-quant, fixed-point attention with GQA/cache, fused attention residual+RMSNorm, and paired gate/up projection.

The expected local model path is:

```bash
/code/Qwen3-0.6B
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

Recent 2048-token-class Declaration PPL record:

```text
backend=hf tokens=1902 loss=3.106583 ppl=22.344565
backend=int-only tokens=1902 loss=3.317960 ppl=27.603982
```

Current single-layer 2048-token-class profile baseline after the RMSNorm/residual fusions:

```text
attention_i8_fixed     avg=102.163 ms total=510.814 ms 98.46%
gate_up_proj_i8        avg=0.306 ms   total=1.528 ms   0.29%
down_proj_i8           avg=0.168 ms   total=0.838 ms   0.16%
o_proj_i8              avg=0.124 ms   total=0.618 ms   0.12%
qkv_proj_i8            avg=0.263 ms   total=1.315 ms   0.25%
rms_q_q15              avg=0.088 ms   total=0.441 ms   0.08%
silu_mul_dq8_mid       avg=0.080 ms   total=0.402 ms   0.08%
dq8_attn               avg=0.070 ms   total=0.349 ms   0.07%
total                  518.792 ms
```
