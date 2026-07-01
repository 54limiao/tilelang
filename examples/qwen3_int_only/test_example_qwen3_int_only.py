import pytest

torch = pytest.importorskip("torch")

import tilelang.testing

from examples.qwen3_int_only.kernels import (
    Q15_16,
    ATTN_VALUE_SHIFT,
    MASK,
    Q_MULTIPLIER_WIDTH,
    compile_kernel,
    dynamic_quant_q15_16,
    exp_lut_neg,
    flash_attention_i12_q15_16_per_scale,
    linear_dynamic_int16_q15_16,
    packed_scale_matrix,
    recip_lut_i12,
    recip_lut_i16_norm,
    rmsnorm_i16_q15_16_weighted,
    rsqrt_lut,
)


def q15(x):
    return torch.clamp(torch.round(x * Q15_16), -(1 << 31), (1 << 31) - 1).to(torch.int32)


def fix_quant_i32(x, scale):
    mul = (scale & MASK).to(torch.int64)
    shift = (scale >> Q_MULTIPLIER_WIDTH).to(torch.int64)
    prod = x.to(torch.int64) * mul
    out = prod >> shift
    out += torch.where(shift >= 1, (prod >> torch.clamp(shift - 1, min=0)) & 1, torch.zeros_like(out))
    return out.to(torch.int32)


@tilelang.testing.requires_cuda
def test_dynamic_quant_i12():
    rows, cols = 5, 64
    x = torch.linspace(-2.5, 2.7, rows * cols, device="cuda").reshape(rows, cols)
    xq = torch.round(x * Q15_16).to(torch.int32)
    kernel = compile_kernel(dynamic_quant_q15_16(rows, cols, "int16", 2047), [2, 3])
    y, s = kernel(xq, torch.from_numpy(recip_lut_i12()).cuda())
    ref_s = torch.div(xq.abs().amax(dim=1), 2047, rounding_mode="floor").clamp(min=1).to(torch.uint32)
    ref_y = torch.div(xq, ref_s.int()[:, None], rounding_mode="floor").clamp(-2048, 2047).to(torch.int16)
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_dynamic_quant_i16():
    rows, cols = 5, 64
    xq = torch.randint(-180000, 180001, (rows, cols), device="cuda", dtype=torch.int32)
    kernel = compile_kernel(dynamic_quant_q15_16(rows, cols, "int16"), [2, 3])
    y, s = kernel(xq, torch.empty((4096,), device="cuda", dtype=torch.uint32))
    ref_s = torch.div(xq.abs().amax(dim=1), 32767, rounding_mode="floor").clamp(min=1).to(torch.uint32)
    ref_y = torch.div(xq, ref_s.int()[:, None], rounding_mode="floor").clamp(-32768, 32767).to(torch.int16)
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_linear_i16_chunk():
    torch.manual_seed(0)
    rows, in_features, out_features, chunk = 3, 256, 5, 64
    x = torch.randint(-2048, 2048, (rows, in_features), device="cuda", dtype=torch.int16)
    w = torch.randint(-128, 127, (out_features, in_features), device="cuda", dtype=torch.int8)
    xs = torch.randint(1, 128, (rows,), device="cuda", dtype=torch.uint32)
    ws = torch.randint(1, 512, (out_features,), device="cuda", dtype=torch.uint32)
    kernel = compile_kernel(linear_dynamic_int16_q15_16(rows, in_features, out_features), [4])
    y = kernel(x, xs, w, ws)
    acc = (x[:, None, :].int() * w[None, :, :].int()).reshape(rows, out_features, -1, chunk).sum(dim=-1).sum(dim=-1)
    ref = ((acc >> 12) * ((xs[:, None].int() * ws[None, :].int()) >> 2)) >> 2
    torch.testing.assert_close(y.to(torch.int64), ref.to(torch.int64), rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_rmsnorm_i16():
    torch.manual_seed(0)
    rows, cols = 6, 128
    x = (torch.randn(rows, cols, device="cuda") * 0.08).clamp(-0.4, 0.4)
    w = (torch.randn(cols, device="cuda") * 0.03 + 1.0).clamp(0.8, 1.2)
    xq = q15(x)
    x16, _ = compile_kernel(dynamic_quant_q15_16(rows, cols, "int16", 4095), [2, 3])(xq, torch.from_numpy(recip_lut_i16_norm()).cuda())
    y = compile_kernel(rmsnorm_i16_q15_16_weighted(rows, cols), [3])(x16, q15(w), torch.from_numpy(rsqrt_lut()).cuda())
    scale = torch.div(xq.abs().amax(dim=-1), 4095, rounding_mode="floor").clamp_min(1)
    q = torch.div(xq, scale[:, None], rounding_mode="floor").clamp(-4096, 4095)
    ref = q15(q.float() / torch.sqrt(torch.mean(q.float() * q.float(), dim=-1, keepdim=True).clamp_min(1.0)) * w)
    rel = torch.sqrt(torch.mean((y.float() - ref.float()) ** 2)) / torch.sqrt(torch.mean(ref.float() ** 2))
    assert float(rel) < 0.012


@tilelang.testing.requires_cuda
def test_attention_i12_q15_16_per_scale():
    torch.manual_seed(0)
    batch, seqlen, dim = 2, 32, 16
    q = torch.randint(-1500, 1501, (batch, seqlen, dim), device="cuda", dtype=torch.int16)
    k = torch.randint(-1500, 1501, (batch, seqlen, dim), device="cuda", dtype=torch.int16)
    v = torch.randint(-1800, 1801, (batch, seqlen, dim), device="cuda", dtype=torch.int16)
    vs = torch.randint(8, 256, (batch, seqlen), device="cuda", dtype=torch.uint32)
    qs = torch.linspace(0.00025, 0.00070, batch * seqlen, device="cuda").reshape(batch, seqlen)
    ks = torch.linspace(0.00030, 0.00080, batch * seqlen, device="cuda").reshape(batch, seqlen)
    score = qs[:, :, None] * ks[:, None, :] / (dim**0.5) * 64.0
    score_q = torch.from_numpy(packed_scale_matrix(score)).cuda()
    kernel = compile_kernel(flash_attention_i12_q15_16_per_scale(batch, seqlen, dim), [6])
    y = kernel(q, k, v, torch.from_numpy(exp_lut_neg()).cuda(), score_q.reshape(-1).contiguous(), vs)
    red = (q.float() @ k.float().transpose(-1, -2)).to(torch.int32)
    hi = red >> 14
    lo = red - (hi << 14)
    score_i64 = score_q.to(torch.int64)
    hi_scale = (((score_i64 >> Q_MULTIPLIER_WIDTH) - 14) << Q_MULTIPLIER_WIDTH) | (score_i64 & MASK)
    sc = fix_quant_i32(hi, hi_scale)
    sc += fix_quant_i32(lo, score_i64)
    mask = torch.ones((seqlen, seqlen), device="cuda", dtype=torch.bool).tril()
    sc = sc.masked_fill(~mask, -32768)
    lut = torch.round(torch.exp((torch.arange(4097, device="cuda").float() - 4096.0) / 64.0) * 1023.0).clamp(0, 1023).int()
    ex = lut[(sc - sc.max(dim=-1, keepdim=True).values).clamp(-4096, 0) + 4096].to(torch.int64)
    val = ((v.to(torch.int64) * vs.to(torch.int64)[:, :, None]) >> ATTN_VALUE_SHIFT)
    acc = (ex[:, :, :, None] * val[:, None, :, :]).sum(dim=2)
    ref = torch.div(acc, ex.sum(dim=-1, keepdim=True), rounding_mode="trunc").to(torch.int32) << ATTN_VALUE_SHIFT
    torch.testing.assert_close(y, ref, rtol=0, atol=128)
