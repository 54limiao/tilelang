# XP5 Int-Only Qwen3 Inference — Development Log & Plan

Living document. Goal: fully fixed-point (XP5-compatible) LLM inference kernels that
reach **total tensorcore util > 50%** while keeping **cos > 0.99** and **ppl loss < 5%**
vs HF bf16. No floating point anywhere in the int-only path. XP5 supports int32 and
int16-and-below only; **no general i64**.

## Hard constraints (verified)
- XP5 vquant instruction internally uses a 64-bit multiply accumulator
  (`cmodel .../veu_core.cpp:3747 quant()`): `mul_out = ((int64)temp_s * m_scale) >> r_shift`,
  round-half-up, saturate to signed `post_w`. So `T.fix.quant`'s int64 `_wide_round_shift`
  is hardware-faithful and must NOT be "simplified away". General kernel math must stay int32/int16.
- Static quant nodes are FIXED: qkv input and every projection input are static quant. Do not change.
- LUT is 10-bit indexed (1024 entries), `T.fix.lut_10bit`.
- vquant dtype pairs: i8->i32, i32->i8, i16->i32, i32->i10, i32->i16.

## Environment
- Python: `/root/venv/bin/python` (system python lacks tvm_ffi).
- GPU: 2x A100-80GB, int8 TC peak 624 TOPS.
- Run: `examples/qwen3_int_only/test_static_path.sh` (Qwen3-0.6B default, BACKEND=int-only|hybrid).
- Model at `/publicdata/huggingface.co/Qwen/Qwen3-0.6B`; packed at `/tmp/Qwen3-0.6B-static-calib-32x2048`.
- Rebuild C++: `cmake --build build -j$(nproc)`; PYTHONPATH = repo root.

## Baseline (from README, 2048 tokens)
### Qwen3-0.6B int-only quality: loss 3.821986, ppl 45.69, cos 0.99019, mse 0.2236 (PASSES cos>0.99, ppl loss ~+2%)
### Qwen3-0.6B int-only kernel profile (28 layers, total TC util 11.14%):
| kernel | total ms | pct | tc util |
| --- | ---: | ---: | ---: |
| attention_i8 | 268.4 | 35.84% | 13.85% |
| qk_norm_rope_i8 | 207.9 | 27.76% | (no TC) |
| linear_i8 | 184.8 | 24.67% | 25.03% |
| rms_sq8 | 37.0 | 4.94% | (no TC) |
| silu_hadamard_i8 | 35.0 | 4.68% | (no TC) |
| quant_v_i8 | 15.8 | 2.11% | (no TC) |
| total | 748.9 | 100% | 11.14% |

### Qwen3-14B int-only total TC util 20.29%; linear_i8 dominates (75.9%).

## Bottleneck analysis
- On 0.6B, non-GEMM scalar kernels (qk_norm_rope, rms, silu_hadamard, quant_v) = ~40% of
  time and produce ZERO tensorcore work -> they crush total TC util.
- attention_i8 is 36% of time but low TC util (13.85%): PV done as int16 x int8, online
  softmax LUT is scalar-heavy; QK/PV GEMM shapes are small (head_dim=128, seq tiles).
- linear_i8 is efficient (25% util) but only 25% of time on 0.6B.
- => Model size matters: larger models make linear_i8 dominate and lift total util (14B: 20%).
  BUT to break 50% we must ALSO (a) raise linear_i8 util toward peak, (b) cut scalar-kernel
  time via fusion, (c) make attention issue more TC-dense work / larger tiles.

## Plan (ordered)
1. [ ] Establish reproducible fresh baseline profile on current HEAD (0.6B + 14B if feasible),
       save raw logs under /tmp/qwen3_int_only_logs. Confirm README numbers.
2. [ ] Study tilelang GEMM examples (examples/gemm, gemm_int4, deepseek_deepgemm) for
       int8 tiling, pipelining, swizzle, TMA/cp.async to raise linear_i8 util.
3. [ ] Optimize linear_i8: better block/warp tiling + software pipeline; target >40% util.
4. [ ] Fuse scalar kernels into neighbors to remove standalone launches:
       - rms_sq8 -> fuse into producer linear (epilogue) or into qk_norm.
       - quant_v_i8 -> fuse into qkv epilogue.
       - qk_norm_rope_i8 -> reduce LUT passes / vectorize.
5. [ ] Attention: improve tiling, KV-cache layout (contiguous int8), and count PV honestly;
       consider flash-style block softmax in fixed point to raise TC density.
6. [ ] Down proj: fold R4 Hadamard fully into linear_i8 weights (verify already done) so
       down is a plain int8 GEMM; if any runtime hadamard remains, convert to GEMM.
7. [ ] Optionally add (input, mul, shift, out_dtype) overload to T.fix.quant for dev ergonomics.
8. [ ] Evaluate on 7B/14B to confirm util>50% is reachable with larger GEMMs; update README.

## Session log
- 2026-07-03: Set up env, confirmed /root/venv python, verified vquant HW semantics (int64
  internal is legal). Created this doc + memory notes. Next: fresh baseline profile.
