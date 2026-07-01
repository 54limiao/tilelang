# Qwen3 0.6B Integer-Only Prototype

This example runs a Qwen3-0.6B inference path using Q15.16 activations, per-token dynamic quantization, int8 per-channel weights, int12 QKV attention, and integer TileLang kernels for the default path.

The expected local model path is:

```bash
/code/Qwen3-0.6B
```

Run perplexity on the bundled Declaration of Independence text:

```bash
/root/venv/bin/python examples/qwen3_int_only/ppl.py \
  --backend int-only \
  --max-tokens 2049
```

Reference backends:

```bash
/root/venv/bin/python examples/qwen3_int_only/ppl.py --backend local-float --max-tokens 2049
/root/venv/bin/python examples/qwen3_int_only/ppl.py --backend hf --max-tokens 2049
```

Run the focused tests:

```bash
/root/venv/bin/python -m pytest -q examples/qwen3_int_only/test_example_qwen3_int_only.py -q
```

Recent 2048-token-class Declaration run:

```text
backend=int-only tokens=1902 loss=3.172896 ppl=23.876540
```
