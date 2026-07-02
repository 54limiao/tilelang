# Qwen3 0.6B Static Int-Only Path

Run the standard 2048-token quality check:

```bash
examples/qwen3_int_only/test_static_path.sh
```

Default inputs:

```text
model:   /publicdata/huggingface.co/Qwen/Qwen3-0.6B
data:    /publicdata/huggingface.co/datasets
pack:    /tmp/Qwen3-0.6B-static-calib-32x2048
prefix:  你是一个有用而无害的聊天助手。
```

Current 2048-token FineWeb result against HF bf16:

```text
backend=int-only tokens=2048 loss=3.821411 ppl=45.668586 compare=hf cos=0.99346597 mse=1.50167644e-01 mae=2.89471441e-01 max_abs=1.05577879e+01 rel_mse=1.31633772e-02
```

Kernel profile for the int-only LLM block path. This is 2048 sequence length, 28 layers, prefix KV cache enabled, 1 warmup + 3 measured repeats:

```text
total block kernels: 166.520 ms / 3 repeats = 55.507 ms per 28-layer pass

kernel                           avg ms   total ms    pct     TOPS
attention_cache_i8v8_fused_static 0.951    79.899   47.98    72.60
down_residual_static              0.185    15.523    9.32   139.45
qkv_proj_i8                       0.140    11.758    7.06   122.74
gate_up_proj_static               0.134    11.264    6.76   192.18
rope_sq8_q_attn_hadamard          0.089     7.474    4.49    12.07
rms_q_q15                         0.068     5.738    3.45
silu_mul_sq16_mid_fast            0.067     5.601    3.36
o_proj_i8                         0.059     4.947    2.97   145.84
rope_sq8_k_attn_hadamard          0.056     4.705    2.83     9.59
rms_k_q15                         0.045     3.778    2.27
rms_input_dq8_fast                0.040     3.394    2.04
residual_attn_rms_q15             0.040     3.332    2.00
sq8_v_attn_noscale                0.039     3.254    1.95
sq8_hidden_static                 0.035     2.979    1.79
dq8_attn                          0.034     2.873    1.73
```

The first run writes `qwen3_int_only.safetensors` and `timestamp`; later runs skip packing when the pack matches the current static schema. Use `FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh` to rebuild.

Main quantization path:

- weights: per-channel static int8 for all linear weights
- activations: calibrated static scales from FineWeb, 32 batches x 2048 tokens, ignoring the first 512 prefix tokens
- residual stream: Q15.16 int32
- attention: q/k/v int8, r2/r3 QuaRot head rotations, prefix KV cache, fused static int8 attention path
- MLP: static int8 input, static int16 gated activation, int-only TileLang kernels
- comparison: int-only logits are compared with HF bf16 logits using PPL, cosine, MSE, MAE, max_abs, and rel_mse

Useful knobs:

```bash
EVAL_TOKENS=2048 NUM_BATCHES=1 examples/qwen3_int_only/test_static_path.sh
FORCE_PACK=1 examples/qwen3_int_only/test_static_path.sh
PACKED_DIR=/tmp/custom-pack examples/qwen3_int_only/test_static_path.sh
/root/venv/bin/python examples/qwen3_int_only/utils/profile_kernels.py --max-tokens 2048 --layers 28 --warmup 1 --repeat 3
```

The showcase implementation is in `model.py` and `kernels.py`; packing, QuaRot, PPL, profiling, and trace helpers live under `utils/`.
