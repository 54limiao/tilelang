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

Current same-machine single-layer 2048-token-class profile baseline:

```text
default attention:
attention_i8_fixed     avg=1.757 ms
attention_norm         avg=0.049 ms
total                  8.947 ms

split attention:
attention_softmax_i16  avg=0.510 ms
attention_i16v8        avg=0.358 ms
total                  6.168 ms
```
