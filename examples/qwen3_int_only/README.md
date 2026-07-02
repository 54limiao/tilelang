# Qwen3 0.6B Static Int-Only Path

This example keeps one inference path: packed static attention scales, R1/R2/R3 QuaRot metadata, fused static int8 attention, Q15.16 residual activations, per-token runtime quantization before GEMMs, and TileLang integer kernels.

Expected model and calibration data roots:

```bash
/code/Qwen3-0.6B
/publicdata/huggingface.co/Qwen/Qwen3-0.6B/
/publicdata/huggingface.co/datasets/
```

Pack weights and static attention scales:

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
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-static-calib-32x2048 \
  --eval-dataset fineweb \
  --max-tokens 2049 \
  --batch-size 1 \
  --num-batches 1 \
  --mlp-i16-layers 20-27
```

Current single-window prefix results:

```text
hf prefix tokens=2048 loss=3.801437 ppl=44.765483
static W8A8 tokens=2048 loss=3.835020 ppl=46.294334 compare=hf cos=0.98933302 mse=2.42206294e-01 rel_mse=2.12312901e-02
static --mlp-i16-layers 20-27 tokens=2048 loss=3.827422 ppl=45.943948 compare=hf cos=0.99145196 mse=1.97211390e-01 rel_mse=1.72871323e-02
static --mlp-i16-layers 12-27 tokens=2048 loss=3.824328 ppl=45.802009 compare=hf cos=0.99238650 mse=1.74848317e-01 rel_mse=1.53268327e-02
static --mlp-i16 all tokens=2048 loss=3.817359 ppl=45.483925 compare=hf cos=0.99371496 mse=1.44382801e-01 rel_mse=1.26562902e-02
```

`--mlp-i16-layers 20-27` is the current speed/quality recommendation: it crosses the `cos > 0.99` target while keeping most layers on the W8A8 path. Layer indices are zero-based; `--mlp-i16` enables all layers.

Profile the static path:

```bash
/root/venv/bin/python examples/qwen3_int_only/profile_kernels.py \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-static-calib-32x2048 \
  --max-tokens 2049 \
  --layers 28 \
  --warmup 1 \
  --repeat 3 \
  --mlp-i16-layers 20-27
```

Recent 28-layer prefix/cache profile baselines:

```text
static --mlp-i16-layers 20-27 total=105.981 ms
static --mlp-i16-layers 12-27 total=108.670 ms
```

Trace a block against the dequantized packed-weight float reference:

```bash
/root/venv/bin/python examples/qwen3_int_only/trace_block.py \
  --model-dir /code/Qwen3-0.6B \
  --packed-dir /tmp/Qwen3-0.6B-static-calib-32x2048 \
  --eval-dataset fineweb \
  --max-tokens 65 \
  --layer 0 \
  --jsonl-out /tmp/qwen_trace_metrics.jsonl
```

The static pack stores per-head `q_pre_rope_i16`, `k_pre_rope_i16`, `q_post_rope_i8`, `k_post_rope_i8`, and `v_i8` scales. Runtime attention kernels read these scales from the pack; calibration is not done at inference time.
