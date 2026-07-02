import pytest

torch = pytest.importorskip("torch")

import tilelang.testing
from tilelang.language.fix import pack_scale

from examples.qwen3_int_only.kernels import (
    Q15_16,
    MASK,
    Q_MULTIPLIER_WIDTH,
    add_rmsnorm_dynamic_quant_q15_16_weighted_fast,
    add_rmsnorm_q15_16_weighted,
    add_q15_16,
    attention_i8v8_q15_16_gqa_fused_static,
    attention_i8v8_q15_16_gqa_cache_fused_static_current,
    compile_kernel,
    dynamic_quant_q15_16,
    exp_lut_neg,
    linear_dynamic_int16_residual_q15_16,
    linear_dynamic_int8_pair_q15_16,
    linear_dynamic_int8_qkv_q15_16,
    linear_dynamic_int8_residual_q15_16,
    linear_dynamic_int8_q15_16,
    linear_static_int8_pair_q15_16,
    linear_static_int16_residual_q15_16,
    rope_rotate_q15_16_heads,
    rope_rotate_static_quant_q15_16_attn,
    rope_rotate_static_quant_q15_16_attn_hadamard_approx,
    rope_rotate_static_quant_q15_16_attn_noscale,
    rope_q15_16_heads,
    rmsnorm_dynamic_quant_q15_16_weighted_fast,
    rmsnorm_q15_16_grouped_weighted_rowwise,
    rmsnorm_q15_16_weighted,
    rsqrt_lut,
    sigmoid_lut,
    static_quant_q15_16,
    silu_mul_dynamic_quant_q15_16,
    silu_mul_dynamic_quant_q15_16_fast,
    silu_mul_dynamic_quant_q15_16_i16_fast,
    silu_mul_static_quant_q15_16_fast,
    silu_mul_static_quant_q15_16_i16_fast,
    static_quant_q15_16_per_head_attn,
    static_quant_q15_16_per_head_attn_noscale,
)


def q15(x):
    return torch.clamp(torch.round(x * Q15_16), -(1 << 31), (1 << 31) - 1).to(torch.int32)


def ceil_scale(x, qmax, dim):
    return torch.div(x.abs().amax(dim=dim) + qmax - 1, qmax, rounding_mode="floor").clamp(min=1).to(torch.uint32)


def quantize_with_scale(x, scale, qmax):
    q = torch.div(x.abs() + (scale.int()[..., None] >> 1), scale.int()[..., None], rounding_mode="floor")
    q = torch.where(x < 0, -q, q)
    return q.clamp(-qmax - 1, qmax)


def fix_quant_i32(x, scale):
    mul = (scale & MASK).to(torch.int64)
    shift = (scale >> Q_MULTIPLIER_WIDTH).to(torch.int64)
    prod = x.to(torch.int64) * mul
    out = prod >> shift
    out += torch.where(shift >= 1, (prod >> torch.clamp(shift - 1, min=0)) & 1, torch.zeros_like(out))
    return out.to(torch.int32)


def fix_lut_10bit(x, lut, scale):
    scale_qt = torch.as_tensor(pack_scale(scale) if isinstance(scale, float) else scale, device=x.device, dtype=torch.int64)
    idx = fix_quant_i32(x.to(torch.int32), scale_qt).clamp(-512, 511) + 512
    return lut[idx.long()]


def hadamard(dim, device):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < dim:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h / (dim**0.5)


@tilelang.testing.requires_cuda
def test_dynamic_quant_i12():
    rows, cols = 5, 64
    x = torch.linspace(-2.5, 2.7, rows * cols, device="cuda").reshape(rows, cols)
    xq = torch.round(x * Q15_16).to(torch.int32)
    kernel = compile_kernel(dynamic_quant_q15_16(rows, cols, "int16", 2047), [1, 2])
    y, s = kernel(xq)
    ref_s = ceil_scale(xq, 2047, 1)
    ref_y = quantize_with_scale(xq, ref_s, 2047).to(torch.int16)
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_dynamic_quant_i16():
    rows, cols = 5, 64
    xq = torch.randint(-180000, 180001, (rows, cols), device="cuda", dtype=torch.int32)
    kernel = compile_kernel(dynamic_quant_q15_16(rows, cols, "int16"), [1, 2])
    y, s = kernel(xq)
    ref_s = ceil_scale(xq, 32767, 1)
    ref_y = quantize_with_scale(xq, ref_s, 32767).to(torch.int16)
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_static_quant_i8():
    rows, cols = 5, 64
    xq = torch.randint(-180000, 180001, (rows, cols), device="cuda", dtype=torch.int32)
    scale = torch.tensor([777], device="cuda", dtype=torch.uint32)
    y, s = compile_kernel(static_quant_q15_16(rows, cols, "int8"), [2, 3])(xq, scale)
    ref_y = quantize_with_scale(xq, scale, 127).to(torch.int8)
    torch.testing.assert_close(s, scale.expand(rows), rtol=0, atol=0)
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_static_quant_per_head_attn_layout_matches_permute():
    torch.manual_seed(0)
    tokens, heads, head_dim = 9, 4, 16
    x = torch.randint(-70000, 70001, (tokens, heads, head_dim), device="cuda", dtype=torch.int32)
    scale = torch.randint(300, 900, (heads,), device="cuda", dtype=torch.uint32)
    ref = quantize_with_scale(x.permute(1, 0, 2).reshape(heads, -1), scale, 127).reshape(heads, tokens, head_dim).to(torch.int8)
    y, sy = compile_kernel(static_quant_q15_16_per_head_attn(tokens, heads, head_dim, "int8"), [2, 3])(x, scale)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)
    torch.testing.assert_close(sy, scale[:, None].expand(heads, tokens).contiguous(), rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_static_quant_per_head_attn_noscale_matches_scaled():
    torch.manual_seed(0)
    tokens, heads, head_dim = 9, 4, 16
    x = torch.randint(-70000, 70001, (tokens, heads, head_dim), device="cuda", dtype=torch.int32)
    scale = torch.randint(300, 900, (heads,), device="cuda", dtype=torch.uint32)
    ref, _ = compile_kernel(static_quant_q15_16_per_head_attn(tokens, heads, head_dim, "int8"), [2, 3])(x, scale)
    y = compile_kernel(static_quant_q15_16_per_head_attn_noscale(tokens, heads, head_dim, "int8"), [2])(x, scale)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_add_rmsnorm_q15_matches_unfused():
    torch.manual_seed(0)
    rows, cols = 5, 128
    a = torch.randint(-180000, 180001, (rows, cols), device="cuda", dtype=torch.int32)
    b = torch.randint(-180000, 180001, (rows, cols), device="cuda", dtype=torch.int32)
    wq = q15((torch.randn(cols, device="cuda") * 0.03 + 1.0).clamp(0.8, 1.2))
    lut = torch.from_numpy(rsqrt_lut()).cuda()
    y, n = compile_kernel(add_rmsnorm_q15_16_weighted(rows, cols), [4, 5])(a, b, wq, lut)
    ref_y = compile_kernel(add_q15_16(rows, cols), [2])(a, b)
    ref_n = compile_kernel(rmsnorm_q15_16_weighted(rows, cols), [3])(ref_y, wq, lut)
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
    torch.testing.assert_close(n, ref_n, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_add_rmsnorm_dynamic_quant_fast_matches_two_step():
    torch.manual_seed(0)
    rows, cols = 5, 128
    a = torch.randint(-180000, 180001, (rows, cols), device="cuda", dtype=torch.int32)
    b = torch.randint(-180000, 180001, (rows, cols), device="cuda", dtype=torch.int32)
    wq = q15((torch.randn(cols, device="cuda") * 0.03 + 1.0).clamp(0.8, 1.2))
    lut = torch.from_numpy(rsqrt_lut()).cuda()
    ref_y, ref_n = compile_kernel(add_rmsnorm_q15_16_weighted(rows, cols), [4, 5])(a, b, wq, lut)
    ref_q, ref_s = compile_kernel(dynamic_quant_q15_16(rows, cols, "int8"), [1, 2])(ref_n)
    y, q, s = compile_kernel(add_rmsnorm_dynamic_quant_q15_16_weighted_fast(rows, cols), [4, 5, 6])(a, b, wq, lut)
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_rmsnorm_dynamic_quant_fast_matches_two_step():
    torch.manual_seed(0)
    rows, cols = 5, 128
    x = torch.randint(-180000, 180001, (rows, cols), device="cuda", dtype=torch.int32)
    wq = q15((torch.randn(cols, device="cuda") * 0.03 + 1.0).clamp(0.8, 1.2))
    lut = torch.from_numpy(rsqrt_lut()).cuda()
    ref_n = compile_kernel(rmsnorm_q15_16_weighted(rows, cols), [3])(x, wq, lut)
    ref_q, ref_s = compile_kernel(dynamic_quant_q15_16(rows, cols, "int8"), [1, 2])(ref_n)
    q, s = compile_kernel(rmsnorm_dynamic_quant_q15_16_weighted_fast(rows, cols), [3, 4])(x, wq, lut)
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_linear_i8_tiled():
    torch.manual_seed(0)
    rows, in_features, out_features = 16, 64, 32
    x = torch.randint(-128, 127, (rows, in_features), device="cuda", dtype=torch.int8)
    w = torch.randint(-128, 127, (out_features, in_features), device="cuda", dtype=torch.int8)
    xs = torch.randint(1, 256, (rows,), device="cuda", dtype=torch.uint32)
    ws = torch.randint(1, 512, (out_features,), device="cuda", dtype=torch.uint32)
    kernel = compile_kernel(linear_dynamic_int8_q15_16(rows, in_features, out_features), [4])
    y = kernel(x, xs, w, ws)
    acc = (x.float() @ w.float().T).int()
    ref = (acc >> 8) * ((xs[:, None].int() * ws[None, :].int()) >> 8)
    torch.testing.assert_close(y.to(torch.int64), ref.to(torch.int64), rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_linear_i8_residual_matches_linear_add():
    torch.manual_seed(0)
    rows, in_features, out_features = 16, 64, 32
    x = torch.randint(-128, 127, (rows, in_features), device="cuda", dtype=torch.int8)
    w = torch.randint(-128, 127, (out_features, in_features), device="cuda", dtype=torch.int8)
    xs = torch.randint(1, 256, (rows,), device="cuda", dtype=torch.uint32)
    ws = torch.randint(1, 512, (out_features,), device="cuda", dtype=torch.uint32)
    residual = torch.randint(-100000, 100001, (rows, out_features), device="cuda", dtype=torch.int32)
    ref = compile_kernel(linear_dynamic_int8_q15_16(rows, in_features, out_features), [4])(x, xs, w, ws) + residual
    y = compile_kernel(linear_dynamic_int8_residual_q15_16(rows, in_features, out_features), [5])(x, xs, w, ws, residual)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_linear_i16_residual_matches_torch_ref():
    torch.manual_seed(0)
    rows, in_features, out_features = 16, 64, 32
    x = torch.randint(-32768, 32767, (rows, in_features), device="cuda", dtype=torch.int16)
    w = torch.randint(-128, 127, (out_features, in_features), device="cuda", dtype=torch.int8)
    xs = torch.randint(1, 256, (rows,), device="cuda", dtype=torch.uint32)
    ws = torch.randint(1, 512, (out_features,), device="cuda", dtype=torch.uint32)
    residual = torch.randint(-100000, 100001, (rows, out_features), device="cuda", dtype=torch.int32)
    y = compile_kernel(linear_dynamic_int16_residual_q15_16(rows, in_features, out_features), [5])(x, xs, w, ws, residual)
    acc = (x.cpu().to(torch.int64) @ w.cpu().to(torch.int64).T).cuda()
    ref = residual + (((acc >> 8) * xs[:, None].long() * ws[None, :].long()) >> 8).int()
    rel_mse = torch.mean((y.float() - ref.float()) ** 2) / torch.mean(ref.float() ** 2)
    assert float(rel_mse) < 1e-6


@tilelang.testing.requires_cuda
def test_linear_i8_pair_tiled():
    torch.manual_seed(0)
    rows, in_features, out_features = 16, 64, 32
    x = torch.randint(-128, 127, (rows, in_features), device="cuda", dtype=torch.int8)
    w0 = torch.randint(-128, 127, (out_features, in_features), device="cuda", dtype=torch.int8)
    w1 = torch.randint(-128, 127, (out_features, in_features), device="cuda", dtype=torch.int8)
    xs = torch.randint(1, 256, (rows,), device="cuda", dtype=torch.uint32)
    ws0 = torch.randint(1, 512, (out_features,), device="cuda", dtype=torch.uint32)
    ws1 = torch.randint(1, 512, (out_features,), device="cuda", dtype=torch.uint32)
    y0, y1 = compile_kernel(linear_dynamic_int8_pair_q15_16(rows, in_features, out_features), [6, 7])(x, xs, w0, ws0, w1, ws1)
    acc0 = (x.float() @ w0.float().T).int()
    acc1 = (x.float() @ w1.float().T).int()
    ref0 = (acc0 >> 8) * ((xs[:, None].int() * ws0[None, :].int()) >> 8)
    ref1 = (acc1 >> 8) * ((xs[:, None].int() * ws1[None, :].int()) >> 8)
    torch.testing.assert_close(y0.to(torch.int64), ref0.to(torch.int64), rtol=0, atol=0)
    torch.testing.assert_close(y1.to(torch.int64), ref1.to(torch.int64), rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_linear_i8_qkv_matches_unfused():
    torch.manual_seed(0)
    rows, in_features, q_features, kv_features = 16, 64, 64, 32
    x = torch.randint(-128, 127, (rows, in_features), device="cuda", dtype=torch.int8)
    xs = torch.randint(1, 256, (rows,), device="cuda", dtype=torch.uint32)
    wq = torch.randint(-128, 127, (q_features, in_features), device="cuda", dtype=torch.int8)
    wk = torch.randint(-128, 127, (kv_features, in_features), device="cuda", dtype=torch.int8)
    wv = torch.randint(-128, 127, (kv_features, in_features), device="cuda", dtype=torch.int8)
    wsq = torch.randint(1, 512, (q_features,), device="cuda", dtype=torch.uint32)
    wsk = torch.randint(1, 512, (kv_features,), device="cuda", dtype=torch.uint32)
    wsv = torch.randint(1, 512, (kv_features,), device="cuda", dtype=torch.uint32)
    q, k, v = compile_kernel(linear_dynamic_int8_qkv_q15_16(rows, in_features, q_features, kv_features), [8, 9, 10])(x, xs, wq, wsq, wk, wsk, wv, wsv)
    ref_q = compile_kernel(linear_dynamic_int8_q15_16(rows, in_features, q_features), [4])(x, xs, wq, wsq)
    ref_k, ref_v = compile_kernel(linear_dynamic_int8_pair_q15_16(rows, in_features, kv_features), [6, 7])(x, xs, wk, wsk, wv, wsv)
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(k, ref_k, rtol=0, atol=0)
    torch.testing.assert_close(v, ref_v, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_linear_static_mlp_kernels_match_torch_ref():
    torch.manual_seed(0)
    rows, in_features, out_features = 16, 64, 32
    x = torch.randint(-128, 127, (rows, in_features), device="cuda", dtype=torch.int8)
    mid = torch.randint(-32768, 32767, (rows, in_features), device="cuda", dtype=torch.int16)
    w0 = torch.randint(-128, 127, (out_features, in_features), device="cuda", dtype=torch.int8)
    w1 = torch.randint(-128, 127, (out_features, in_features), device="cuda", dtype=torch.int8)
    xs = torch.tensor([321], device="cuda", dtype=torch.uint32)
    mids = torch.tensor([17], device="cuda", dtype=torch.uint32)
    ws0 = torch.randint(1, 512, (out_features,), device="cuda", dtype=torch.uint32)
    ws1 = torch.randint(1, 512, (out_features,), device="cuda", dtype=torch.uint32)
    residual = torch.randint(-100000, 100001, (rows, out_features), device="cuda", dtype=torch.int32)
    y0, y1 = compile_kernel(linear_static_int8_pair_q15_16(rows, in_features, out_features), [6, 7])(x, xs, w0, ws0, w1, ws1)
    y = compile_kernel(linear_static_int16_residual_q15_16(rows, in_features, out_features), [5])(mid, mids, w0, ws0, residual)
    acc0 = (x.float() @ w0.float().T).int()
    acc1 = (x.float() @ w1.float().T).int()
    ref0 = (acc0 >> 8) * ((xs[0].int() * ws0[None, :].int()) >> 8)
    ref1 = (acc1 >> 8) * ((xs[0].int() * ws1[None, :].int()) >> 8)
    mid_i32 = mid.int()
    acc_hi = ((mid_i32 >> 8).float() @ w0.float().T).int()
    acc_mid = (((mid_i32 - ((mid_i32 >> 8) << 8)) >> 1).float() @ w0.float().T).int()
    ref_y = residual + (((acc_hi + (acc_mid >> 7)).to(torch.int64) * mids[0].to(torch.int64) * ws0[None, :].to(torch.int64)) >> 8).to(torch.int32)
    torch.testing.assert_close(y0.to(torch.int64), ref0.to(torch.int64), rtol=0, atol=0)
    torch.testing.assert_close(y1.to(torch.int64), ref1.to(torch.int64), rtol=0, atol=0)
    torch.testing.assert_close(y.to(torch.int64), ref_y.to(torch.int64), rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_rmsnorm_q15_matches_float_reference():
    torch.manual_seed(0)
    rows, cols = 6, 128
    x = (torch.randn(rows, cols, device="cuda") * 0.08).clamp(-0.4, 0.4)
    w = (torch.randn(cols, device="cuda") * 0.03 + 1.0).clamp(0.8, 1.2)
    xq = q15(x)
    y = compile_kernel(rmsnorm_q15_16_weighted(rows, cols), [3])(xq, q15(w), torch.from_numpy(rsqrt_lut()).cuda())
    scale = ceil_scale(xq, 32767, -1)
    q = quantize_with_scale(xq, scale, 32767)
    ref = q15(q.float() / torch.sqrt(torch.mean(q.float() * q.float(), dim=-1, keepdim=True).clamp_min(1.0)) * w)
    rel = torch.sqrt(torch.mean((y.float() - ref.float()) ** 2)) / torch.sqrt(torch.mean(ref.float() ** 2))
    assert float(rel) < 0.012


@tilelang.testing.requires_cuda
def test_rmsnorm_q15_matches_dynamic_i16():
    torch.manual_seed(0)
    rows, cols = 6, 128
    xq = torch.randint(-220000, 220001, (rows, cols), device="cuda", dtype=torch.int32)
    wq = q15((torch.randn(cols, device="cuda") * 0.03 + 1.0).clamp(0.8, 1.2))
    lut = torch.from_numpy(rsqrt_lut()).cuda()
    y = compile_kernel(rmsnorm_q15_16_weighted(rows, cols), [3])(xq, wq, lut)
    x16, _ = compile_kernel(dynamic_quant_q15_16(rows, cols, "int16"), [1, 2])(xq)
    ref = compile_kernel(rmsnorm_q15_16_weighted(rows, cols), [3])(x16.to(torch.int32), wq, lut)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_rmsnorm_q15_grouped_rowwise_matches_dynamic_i16_groups():
    torch.manual_seed(0)
    rows, groups, cols = 7, 4, 128
    xq = torch.randint(-220000, 220001, (rows, groups * cols), device="cuda", dtype=torch.int32)
    wq = q15((torch.randn(cols, device="cuda") * 0.03 + 1.0).clamp(0.8, 1.2))
    lut = torch.from_numpy(rsqrt_lut()).cuda()
    x16, _ = compile_kernel(dynamic_quant_q15_16(rows, groups * cols, "int16"), [1, 2])(xq)
    ref = compile_kernel(rmsnorm_q15_16_weighted(rows * groups, cols), [3])(x16.reshape(rows * groups, cols).to(torch.int32), wq, lut)
    y = compile_kernel(rmsnorm_q15_16_grouped_weighted_rowwise(rows, groups, cols), [3])(xq, wq, lut)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_rope_rotate_q15_16_heads():
    torch.manual_seed(0)
    seq_len, heads, dim = 7, 2, 32
    x = torch.randn((seq_len * heads, dim), device="cuda") * 0.2
    cos = torch.cos(torch.randn((seq_len, dim // 2), device="cuda") * 0.1)
    sin = torch.sin(torch.randn((seq_len, dim // 2), device="cuda") * 0.1)
    h = hadamard(dim, "cuda")
    xq, cq, sq, hq = q15(x), q15(cos), q15(sin), q15(h)
    y = compile_kernel(rope_rotate_q15_16_heads(seq_len, heads, dim), [4])(xq, cq, sq, hq)
    cexp = cq[:, None, :].expand(seq_len, heads, dim // 2).reshape(seq_len * heads, dim // 2)
    sexp = sq[:, None, :].expand(seq_len, heads, dim // 2).reshape(seq_len * heads, dim // 2)
    lo = ((xq[:, : dim // 2] >> 8) * (cexp >> 8)) - ((xq[:, dim // 2 :] >> 8) * (sexp >> 8))
    hi = ((xq[:, : dim // 2] >> 8) * (sexp >> 8)) + ((xq[:, dim // 2 :] >> 8) * (cexp >> 8))
    rope = torch.cat((lo, hi), dim=-1)
    ref = ((rope[:, :, None] >> 8) * (hq[None, :, :] >> 8)).sum(dim=1).to(torch.int32)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_rope_head_kernels_match_expanded_tables():
    torch.manual_seed(0)
    seq_len, heads, dim = 5, 4, 128
    x = torch.randn((seq_len * heads, dim), device="cuda") * 0.2
    cos = torch.cos(torch.randn((seq_len, dim // 2), device="cuda") * 0.1)
    sin = torch.sin(torch.randn((seq_len, dim // 2), device="cuda") * 0.1)
    h = hadamard(dim, "cuda")
    xq, cq, sq, hq = q15(x), q15(cos), q15(sin), q15(h)
    got_rope = compile_kernel(rope_q15_16_heads(seq_len, heads, dim), [3])(xq, cq, sq)
    got_rot = compile_kernel(rope_rotate_q15_16_heads(seq_len, heads, dim), [4])(xq, cq, sq, hq)
    cexp = cq[:, None, :].expand(seq_len, heads, dim // 2).reshape(seq_len * heads, dim // 2)
    sexp = sq[:, None, :].expand(seq_len, heads, dim // 2).reshape(seq_len * heads, dim // 2)
    lo = ((xq[:, : dim // 2] >> 8) * (cexp >> 8)) - ((xq[:, dim // 2 :] >> 8) * (sexp >> 8))
    hi = ((xq[:, : dim // 2] >> 8) * (sexp >> 8)) + ((xq[:, dim // 2 :] >> 8) * (cexp >> 8))
    ref_rope = torch.cat((lo, hi), dim=-1).to(torch.int32)
    ref_rot = ((ref_rope[:, :, None] >> 8) * (hq[None, :, :] >> 8)).sum(dim=1).to(torch.int32)
    torch.testing.assert_close(got_rope, ref_rope, rtol=0, atol=0)
    torch.testing.assert_close(got_rot, ref_rot, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_rope_rotate_static_quant_attn_matches_unfused():
    torch.manual_seed(0)
    seq_len, heads, dim = 5, 3, 32
    x = torch.randn((seq_len * heads, dim), device="cuda") * 0.2
    cos = torch.cos(torch.randn((seq_len, dim // 2), device="cuda") * 0.1)
    sin = torch.sin(torch.randn((seq_len, dim // 2), device="cuda") * 0.1)
    h = hadamard(dim, "cuda")
    xq, cq, sq, hq = q15(x), q15(cos), q15(sin), q15(h)
    scale = torch.randint(300, 900, (heads,), device="cuda", dtype=torch.uint32)
    rot = compile_kernel(rope_rotate_q15_16_heads(seq_len, heads, dim), [4])(xq, cq, sq, hq)
    ref, ref_s = compile_kernel(static_quant_q15_16_per_head_attn(seq_len, heads, dim, "int8"), [2, 3])(
        rot.reshape(seq_len, heads, dim), scale
    )
    y, sy = compile_kernel(rope_rotate_static_quant_q15_16_attn(seq_len, heads, dim), [5, 6])(xq, cq, sq, hq, scale)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)
    torch.testing.assert_close(sy, ref_s, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_rope_rotate_static_quant_attn_noscale_matches_scaled():
    torch.manual_seed(0)
    seq_len, heads, dim = 5, 3, 32
    x = torch.randn((seq_len * heads, dim), device="cuda") * 0.2
    cos = torch.cos(torch.randn((seq_len, dim // 2), device="cuda") * 0.1)
    sin = torch.sin(torch.randn((seq_len, dim // 2), device="cuda") * 0.1)
    h = hadamard(dim, "cuda")
    xq, cq, sq, hq = q15(x), q15(cos), q15(sin), q15(h)
    scale = torch.randint(300, 900, (heads,), device="cuda", dtype=torch.uint32)
    ref, _ = compile_kernel(rope_rotate_static_quant_q15_16_attn(seq_len, heads, dim), [5, 6])(xq, cq, sq, hq, scale)
    y = compile_kernel(rope_rotate_static_quant_q15_16_attn_noscale(seq_len, heads, dim), [5])(xq, cq, sq, hq, scale)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_rope_rotate_static_quant_attn_hadamard_approx_close():
    torch.manual_seed(0)
    seq_len, heads, dim = 64, 16, 128
    x = torch.randn((seq_len * heads, dim), device="cuda") * 0.2
    cos = torch.cos(torch.randn((seq_len, dim // 2), device="cuda") * 0.1)
    sin = torch.sin(torch.randn((seq_len, dim // 2), device="cuda") * 0.1)
    h = hadamard(dim, "cuda")
    xq, cq, sq, hq = q15(x), q15(cos), q15(sin), q15(h)
    scale = torch.randint(300, 900, (heads,), device="cuda", dtype=torch.uint32)
    ref = compile_kernel(rope_rotate_static_quant_q15_16_attn_noscale(seq_len, heads, dim), [5])(xq, cq, sq, hq, scale)
    y = compile_kernel(rope_rotate_static_quant_q15_16_attn_hadamard_approx(seq_len, heads, dim), [5])(xq, cq, sq, hq, scale)
    rf, yf = ref.float().flatten(), y.float().flatten()
    diff = yf - rf
    cos_sim = torch.nn.functional.cosine_similarity(yf, rf, dim=0)
    rel_mse = (diff * diff).sum() / (rf * rf).sum().clamp_min(1e-30)
    assert float(cos_sim) > 0.999
    assert float(rel_mse) < 0.003


@tilelang.testing.requires_cuda
def test_attention_i8v8_q15_16_gqa_fused_static_smoke():
    torch.manual_seed(0)
    q_heads, kv_heads, seqlen, dim, block_n = 4, 2, 32, 64, 32
    q = torch.randint(-127, 128, (q_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    k = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    v = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    qs = torch.round((torch.rand((q_heads,), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    ks = torch.round((torch.rand((kv_heads,), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    vs = torch.round((torch.rand((kv_heads,), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    lut = torch.from_numpy(exp_lut_neg()).cuda()
    y0 = compile_kernel(attention_i8v8_q15_16_gqa_fused_static(q_heads, kv_heads, seqlen, dim, block_n=block_n), [7])(q, k, v, qs, ks, vs, lut)
    y1 = compile_kernel(attention_i8v8_q15_16_gqa_fused_static(q_heads, kv_heads, seqlen, dim, block_n=block_n), [7])(q, k, v, qs, ks, vs, lut)
    assert y0.shape == (seqlen, q_heads * dim)
    assert y0.dtype == torch.int32
    assert torch.count_nonzero(y0) > 0
    torch.testing.assert_close(y0, y1, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_attention_i8v8_q15_16_gqa_cache_fused_static_current_smoke():
    torch.manual_seed(0)
    q_heads, kv_heads, cache_len, seqlen, dim, block_n = 4, 2, 9, 17, 64, 32
    q = torch.randint(-127, 128, (q_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    k = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    v = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    ck = torch.randint(-127, 128, (kv_heads, cache_len, dim), device="cuda", dtype=torch.int8)
    cv = torch.randint(-127, 128, (kv_heads, cache_len, dim), device="cuda", dtype=torch.int8)
    qs = torch.round((torch.rand((q_heads,), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    ks = torch.round((torch.rand((kv_heads,), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    vs = torch.round((torch.rand((kv_heads,), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    cks = torch.round((torch.rand((kv_heads, cache_len), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    cvs = torch.round((torch.rand((kv_heads, cache_len), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    lut = torch.from_numpy(exp_lut_neg()).cuda()
    y0 = compile_kernel(attention_i8v8_q15_16_gqa_cache_fused_static_current(q_heads, kv_heads, seqlen, cache_len, dim, block_n=block_n), [11])(
        q, ck, cv, k, v, qs, cks, cvs, ks, vs, lut
    )
    y1 = compile_kernel(attention_i8v8_q15_16_gqa_cache_fused_static_current(q_heads, kv_heads, seqlen, cache_len, dim, block_n=block_n), [11])(
        q, ck, cv, k, v, qs, cks, cvs, ks, vs, lut
    )
    assert y0.shape == (seqlen, q_heads * dim)
    assert y0.dtype == torch.int32
    assert torch.count_nonzero(y0) > 0
    torch.testing.assert_close(y0, y1, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_silu_mul_dynamic_quant_matches_unfused():
    torch.manual_seed(0)
    rows, cols = 3, 64
    gate = torch.randint(-300000, 300001, (rows, cols), device="cuda", dtype=torch.int32)
    up = torch.randint(-300000, 300001, (rows, cols), device="cuda", dtype=torch.int32)
    lut = torch.from_numpy(sigmoid_lut()).cuda()
    y, q, s = compile_kernel(silu_mul_dynamic_quant_q15_16(rows, cols), [3, 4, 5])(gate, up, lut)
    ref_y = ((((gate >> 10) * fix_lut_10bit(gate, lut, 1.0 / 1024.0)) >> 8) * (up >> 8)).to(torch.int32)
    ref_q, ref_s = compile_kernel(dynamic_quant_q15_16(rows, cols, "int8"), [1, 2])(ref_y)
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_silu_mul_dynamic_quant_fast_matches_full():
    torch.manual_seed(0)
    rows, cols = 3, 64
    gate = torch.randint(-300000, 300001, (rows, cols), device="cuda", dtype=torch.int32)
    up = torch.randint(-300000, 300001, (rows, cols), device="cuda", dtype=torch.int32)
    lut = torch.from_numpy(sigmoid_lut()).cuda()
    _y, ref_q, ref_s = compile_kernel(silu_mul_dynamic_quant_q15_16(rows, cols), [3, 4, 5])(gate, up, lut)
    q, s = compile_kernel(silu_mul_dynamic_quant_q15_16_fast(rows, cols), [3, 4])(gate, up, lut)
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_silu_mul_dynamic_quant_i16_fast_matches_dynamic_i16():
    torch.manual_seed(0)
    rows, cols = 3, 64
    gate = torch.randint(-300000, 300001, (rows, cols), device="cuda", dtype=torch.int32)
    up = torch.randint(-300000, 300001, (rows, cols), device="cuda", dtype=torch.int32)
    lut = torch.from_numpy(sigmoid_lut()).cuda()
    y, _q8, _s8 = compile_kernel(silu_mul_dynamic_quant_q15_16(rows, cols), [3, 4, 5])(gate, up, lut)
    ref_q, ref_s = compile_kernel(dynamic_quant_q15_16(rows, cols, "int16"), [1, 2])(y)
    q, s = compile_kernel(silu_mul_dynamic_quant_q15_16_i16_fast(rows, cols), [3, 4])(gate, up, lut)
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_silu_mul_static_quant_matches_static_quant():
    torch.manual_seed(0)
    rows, cols = 3, 64
    gate = torch.randint(-300000, 300001, (rows, cols), device="cuda", dtype=torch.int32)
    up = torch.randint(-300000, 300001, (rows, cols), device="cuda", dtype=torch.int32)
    scale = torch.tensor([100000], device="cuda", dtype=torch.uint32)
    lut = torch.from_numpy(sigmoid_lut()).cuda()
    q, s = compile_kernel(silu_mul_static_quant_q15_16_fast(rows, cols), [4, 5])(gate, up, lut, scale)
    y = ((((gate >> 10) * fix_lut_10bit(gate, lut, 1.0 / 1024.0)) >> 8) * (up >> 8)).to(torch.int32)
    ref_q = quantize_with_scale(y, scale, 127).to(torch.int8)
    torch.testing.assert_close(s, scale.expand(rows), rtol=0, atol=0)
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_silu_mul_static_quant_i16_matches_static_quant():
    torch.manual_seed(0)
    rows, cols = 3, 64
    gate = torch.randint(-300000, 300001, (rows, cols), device="cuda", dtype=torch.int32)
    up = torch.randint(-300000, 300001, (rows, cols), device="cuda", dtype=torch.int32)
    scale = torch.tensor([800], device="cuda", dtype=torch.uint32)
    lut = torch.from_numpy(sigmoid_lut()).cuda()
    q, s = compile_kernel(silu_mul_static_quant_q15_16_i16_fast(rows, cols), [4, 5])(gate, up, lut, scale)
    y = ((((gate >> 10) * fix_lut_10bit(gate, lut, 1.0 / 1024.0)) >> 8) * (up >> 8)).to(torch.int32)
    ref_q = quantize_with_scale(y, scale, 32767).to(torch.int16)
    torch.testing.assert_close(s, scale.expand(rows), rtol=0, atol=0)
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
