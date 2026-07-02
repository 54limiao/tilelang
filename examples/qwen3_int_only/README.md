# Qwen3 0.6B Static Int-Only Path

This example keeps one main path: per-channel static int8 weights, calibrated static activation scales, fused int8 attention, Q15.16 residual activations, static MLP input int8, static gated MLP int16, and TileLang integer kernels.

Expected roots:

```bash
/publicdata/huggingface.co/Qwen/Qwen3-0.6B/
/publicdata/huggingface.co/datasets/
```

Pack weights and calibration scales:

```bash
/root/venv/bin/python examples/qwen3_int_only/prepack.py \
  --model-dir /publicdata/huggingface.co/Qwen/Qwen3-0.6B \
  --out-dir /tmp/Qwen3-0.6B-static-calib-32x2048 \
  --use-r1 \
  --use-r2 \
  --use-r3 \
  --calib-dataset fineweb \
  --calib-column text \
  --calib-seq-len 2048 \
  --calib-batches 32 \
  --calib-prefix-tokens 512
```

Run the 2048-token FineWeb quality baseline against HF bf16:

```bash
/root/venv/bin/python examples/qwen3_int_only/ppl.py \
  --model-dir /publicdata/huggingface.co/Qwen/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-static-calib-32x2048 \
  --eval-dataset fineweb \
  --max-tokens 2049 \
  --batch-size 1 \
  --num-batches 1
```

Short static MLP gated-i16 check from the current implementation:

```text
tokens=256 loss=4.439738 ppl=84.752757 compare=hf cos=0.99380889 mse=1.00682883e-01 rel_mse=1.23538492e-02
```

Profile the same static path:

```bash
/root/venv/bin/python examples/qwen3_int_only/profile_kernels.py \
  --model-dir /publicdata/huggingface.co/Qwen/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-static-calib-32x2048 \
  --max-tokens 2049 \
  --layers 28 \
  --warmup 1 \
  --repeat 3
```

Trace a block against the dequantized packed-weight float reference:

```bash
/root/venv/bin/python examples/qwen3_int_only/trace_block.py \
  --model-dir /publicdata/huggingface.co/Qwen/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-static-calib-32x2048 \
  --eval-dataset fineweb \
  --max-tokens 65 \
  --layer 0 \
  --jsonl-out /tmp/qwen_trace_metrics.jsonl
```

The showcase files are `model.py` and `kernels.py`. Packing, QuaRot helpers, and torch reference utilities live in `utils.py`; `prepack.py`, `ppl.py`, `profile_kernels.py`, and `trace_block.py` are tools.
