import pytest

torch = pytest.importorskip("torch")

import tilelang.testing
from tilelang.language.fix import pack_scale

from examples.qwen3_int_only.kernels import (
    Q15_16,
    ATTN_VALUE_SHIFT,
    MASK,
    Q_MULTIPLIER_WIDTH,
    add_dynamic_quant_q15_16,
    add_rmsnorm_q15_16_weighted,
    add_q15_16,
    attention_i16v8_q15_16_gqa_cache,
    attention_i8_q15_16_gqa_softmax_i16,
    attention_i8_q15_16_gqa_cache_softmax_i16,
    attention_i16v8_q15_16_gqa,
    attention_normalize_q15_16,
    compile_kernel,
    dynamic_quant_q15_16,
    exp_lut_neg,
    flash_attention_i8_q15_16,
    flash_attention_i8_q15_16_cache,
    flash_attention_i8_q15_16_gqa,
    flash_attention_i8_q15_16_gqa_cache,
    flash_attention_i8_q15_16_gqa_tiled,
    gemm_int16_int8_split,
    linear_dynamic_int8_pair_q15_16,
    linear_dynamic_int8_qkv_q15_16,
    linear_dynamic_int8_q15_16,
    rope_rotate_q15_16,
    rmsnorm_q15_16_grouped_weighted,
    rmsnorm_q15_16_weighted,
    rsqrt_lut,
    sigmoid_lut,
    silu_mul_dynamic_quant_q15_16,
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
    ref_s = torch.div(xq.abs().amax(dim=1), 2047, rounding_mode="floor").clamp(min=1).to(torch.uint32)
    ref_y = torch.div(xq, ref_s.int()[:, None], rounding_mode="floor").clamp(-2048, 2047).to(torch.int16)
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_dynamic_quant_i16():
    rows, cols = 5, 64
    xq = torch.randint(-180000, 180001, (rows, cols), device="cuda", dtype=torch.int32)
    kernel = compile_kernel(dynamic_quant_q15_16(rows, cols, "int16"), [1, 2])
    y, s = kernel(xq)
    ref_s = torch.div(xq.abs().amax(dim=1), 32767, rounding_mode="floor").clamp(min=1).to(torch.uint32)
    ref_y = torch.div(xq, ref_s.int()[:, None], rounding_mode="floor").clamp(-32768, 32767).to(torch.int16)
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_add_dynamic_quant_i16_matches_unfused():
    torch.manual_seed(0)
    rows, cols = 5, 64
    a = torch.randint(-180000, 180001, (rows, cols), device="cuda", dtype=torch.int32)
    b = torch.randint(-180000, 180001, (rows, cols), device="cuda", dtype=torch.int32)
    y, q, s = compile_kernel(add_dynamic_quant_q15_16(rows, cols), [2, 3, 4])(a, b)
    ref_y = compile_kernel(add_q15_16(rows, cols), [2])(a, b)
    ref_q, ref_s = compile_kernel(dynamic_quant_q15_16(rows, cols, "int16", 4095), [1, 2])(ref_y)
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(s, ref_s, rtol=0, atol=0)


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
def test_gemm_int16_int8():
    torch.manual_seed(0)
    rows, in_features, out_features = 64, 128, 64
    x = torch.randint(-2048, 2048, (rows, in_features), device="cuda", dtype=torch.int16)
    w = torch.randint(-127, 128, (out_features, in_features), device="cuda", dtype=torch.int8)
    y = compile_kernel(gemm_int16_int8_split(rows, in_features, out_features), [2])(x, w)
    ref = torch.matmul(x.float(), w.t().float())
    yf = y.float()
    cos = torch.nn.functional.cosine_similarity(yf.flatten(), ref.flatten(), dim=0)
    rel_mse = torch.mean((yf - ref) ** 2) / torch.mean(ref**2)
    assert cos > 0.99999
    assert rel_mse < 1e-6


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
def test_rmsnorm_q15_matches_float_reference():
    torch.manual_seed(0)
    rows, cols = 6, 128
    x = (torch.randn(rows, cols, device="cuda") * 0.08).clamp(-0.4, 0.4)
    w = (torch.randn(cols, device="cuda") * 0.03 + 1.0).clamp(0.8, 1.2)
    xq = q15(x)
    y = compile_kernel(rmsnorm_q15_16_weighted(rows, cols), [3])(xq, q15(w), torch.from_numpy(rsqrt_lut()).cuda())
    scale = torch.div(xq.abs().amax(dim=-1), 32767, rounding_mode="floor").clamp_min(1)
    q = torch.div(xq, scale[:, None], rounding_mode="floor").clamp(-32768, 32767)
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
def test_rmsnorm_q15_grouped_matches_dynamic_i16():
    torch.manual_seed(0)
    rows, groups, cols = 5, 4, 128
    xq = torch.randint(-220000, 220001, (rows, groups * cols), device="cuda", dtype=torch.int32)
    wq = q15((torch.randn(cols, device="cuda") * 0.03 + 1.0).clamp(0.8, 1.2))
    lut = torch.from_numpy(rsqrt_lut()).cuda()
    y = compile_kernel(rmsnorm_q15_16_grouped_weighted(rows, groups, cols), [3])(xq, wq, lut)
    x16, _ = compile_kernel(dynamic_quant_q15_16(rows, groups * cols, "int16"), [1, 2])(xq)
    ref = compile_kernel(rmsnorm_q15_16_weighted(rows * groups, cols), [3])(x16.reshape(rows * groups, cols).to(torch.int32), wq, lut)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_rope_rotate_q15_16():
    torch.manual_seed(0)
    rows, dim = 7, 32
    x = torch.randn((rows, dim), device="cuda") * 0.2
    cos = torch.cos(torch.randn((rows, dim // 2), device="cuda") * 0.1)
    sin = torch.sin(torch.randn((rows, dim // 2), device="cuda") * 0.1)
    h = hadamard(dim, "cuda")
    xq, cq, sq, hq = q15(x), q15(cos), q15(sin), q15(h)
    y = compile_kernel(rope_rotate_q15_16(rows, dim), [4])(xq, cq, sq, hq)
    lo = ((xq[:, : dim // 2] >> 8) * (cq >> 8)) - ((xq[:, dim // 2 :] >> 8) * (sq >> 8))
    hi = ((xq[:, : dim // 2] >> 8) * (sq >> 8)) + ((xq[:, dim // 2 :] >> 8) * (cq >> 8))
    rope = torch.cat((lo, hi), dim=-1)
    ref = ((rope[:, :, None] >> 8) * (hq[None, :, :] >> 8)).sum(dim=1).to(torch.int32)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_attention_i8_q15_16_fixed_point():
    torch.manual_seed(0)
    batch, seqlen, dim, block_n = 2, 127, 128, 64
    q = torch.randint(-127, 128, (batch, seqlen, dim), device="cuda", dtype=torch.int8)
    k = torch.randint(-127, 128, (batch, seqlen, dim), device="cuda", dtype=torch.int8)
    v = torch.randint(-127, 128, (batch, seqlen, dim), device="cuda", dtype=torch.int8)
    qs = torch.round((torch.rand((batch, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    ks = torch.round((torch.rand((batch, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    vs = torch.round((torch.rand((batch, seqlen), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    lut = torch.from_numpy(exp_lut_neg()).cuda()
    y = compile_kernel(flash_attention_i8_q15_16(batch, seqlen, dim, block_n), [7])(q, k, v, qs, ks, vs, lut)
    red = (q.float() @ k.float().transpose(-1, -2)).to(torch.int32)
    scale = (((qs[:, :, None].int() >> 4) * (ks[:, None, :].int() >> 4)) >> 8) * 5793
    score = ((red >> 8) * scale) >> 18
    score = score.masked_fill(~torch.ones((seqlen, seqlen), device="cuda", dtype=torch.bool).tril(), -32768)
    ref = torch.empty_like(y)
    for b in range(batch):
        for i in range(seqlen):
            score_max = torch.tensor(-(1 << 31), device="cuda", dtype=torch.int32)
            denom = torch.tensor(0, device="cuda", dtype=torch.int64)
            acc_o = torch.zeros((dim,), device="cuda", dtype=torch.int64)
            for nb in range((i // block_n) + 1):
                block = score[b, i, nb * block_n : (nb + 1) * block_n]
                new_max = torch.maximum(block.max(), score_max)
                old = fix_lut_10bit(score_max - new_max, lut, 0.125).to(torch.int64)
                ex = fix_lut_10bit(block - new_max, lut, 0.125).to(torch.int64)
                denom = ((denom * old) >> 10) + ex.sum()
                vv = (v[b, nb * block_n : (nb + 1) * block_n].to(torch.int64) * vs[b, nb * block_n : (nb + 1) * block_n].to(torch.int64)[:, None]) >> ATTN_VALUE_SHIFT
                acc_o = (((acc_o >> 7) * old) >> 3) + (ex[:, None] * vv).sum(dim=0)
                score_max = new_max
            ref[b, i] = (acc_o // denom).to(torch.int32) << ATTN_VALUE_SHIFT
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_attention_i8_q15_16_cache_fixed_point():
    torch.manual_seed(0)
    batch, cache_len, seqlen, dim, block_n = 2, 17, 73, 128, 64
    q = torch.randint(-127, 128, (batch, seqlen, dim), device="cuda", dtype=torch.int8)
    k = torch.randint(-127, 128, (batch, seqlen, dim), device="cuda", dtype=torch.int8)
    v = torch.randint(-127, 128, (batch, seqlen, dim), device="cuda", dtype=torch.int8)
    ck = torch.randint(-127, 128, (batch, cache_len, dim), device="cuda", dtype=torch.int8)
    cv = torch.randint(-127, 128, (batch, cache_len, dim), device="cuda", dtype=torch.int8)
    qs = torch.round((torch.rand((batch, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    ks = torch.round((torch.rand((batch, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    vs = torch.round((torch.rand((batch, seqlen), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    cks = torch.round((torch.rand((batch, cache_len), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    cvs = torch.round((torch.rand((batch, cache_len), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    lut = torch.from_numpy(exp_lut_neg()).cuda()
    y = compile_kernel(flash_attention_i8_q15_16_cache(batch, seqlen, cache_len, dim, block_n), [11])(q, ck, cv, k, v, qs, cks, cvs, ks, vs, lut)
    k_all = torch.cat((ck, k), dim=1)
    v_all = torch.cat((cv, v), dim=1)
    ks_all = torch.cat((cks, ks), dim=1)
    vs_all = torch.cat((cvs, vs), dim=1)
    red = (q.float() @ k_all.float().transpose(-1, -2)).to(torch.int32)
    scale = (((qs[:, :, None].int() >> 4) * (ks_all[:, None, :].int() >> 4)) >> 8) * 5793
    score = ((red >> 8) * scale) >> 18
    q_pos = cache_len + torch.arange(seqlen, device="cuda")
    k_pos = torch.arange(cache_len + seqlen, device="cuda")
    score = score.masked_fill(k_pos[None, :] > q_pos[:, None], -32768)
    ref = torch.empty_like(y)
    for b in range(batch):
        for i in range(seqlen):
            score_max = torch.tensor(-(1 << 31), device="cuda", dtype=torch.int32)
            denom = torch.tensor(0, device="cuda", dtype=torch.int64)
            acc_o = torch.zeros((dim,), device="cuda", dtype=torch.int64)
            for nb in range(((cache_len + i) // block_n) + 1):
                block = score[b, i, nb * block_n : (nb + 1) * block_n]
                if block.numel() < block_n:
                    block = torch.cat((block, torch.full((block_n - block.numel(),), -32768, device="cuda", dtype=torch.int32)))
                new_max = torch.maximum(block.max(), score_max)
                old = fix_lut_10bit(score_max - new_max, lut, 0.125).to(torch.int64)
                ex = fix_lut_10bit(block - new_max, lut, 0.125).to(torch.int64)
                denom = ((denom * old) >> 10) + ex.sum()
                vv = torch.zeros((block_n, dim), device="cuda", dtype=torch.int64)
                end = min((nb + 1) * block_n, cache_len + seqlen)
                valid = end - nb * block_n
                vv[:valid] = (v_all[b, nb * block_n : end].to(torch.int64) * vs_all[b, nb * block_n : end].to(torch.int64)[:, None]) >> ATTN_VALUE_SHIFT
                acc_o = (((acc_o >> 7) * old) >> 3) + (ex[:, None] * vv).sum(dim=0)
                score_max = new_max
            ref[b, i] = (acc_o // denom).to(torch.int32) << ATTN_VALUE_SHIFT
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_attention_i8_q15_16_gqa_matches_repeated_kv():
    torch.manual_seed(0)
    q_heads, kv_heads, seqlen, dim = 16, 8, 127, 128
    group = q_heads // kv_heads
    q = torch.randint(-127, 128, (q_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    k = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    v = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    qs = torch.round((torch.rand((q_heads, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    ks = torch.round((torch.rand((kv_heads, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    vs = torch.round((torch.rand((kv_heads, seqlen), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    lut = torch.from_numpy(exp_lut_neg()).cuda()
    y = compile_kernel(flash_attention_i8_q15_16_gqa(q_heads, kv_heads, seqlen, dim), [7])(q, k, v, qs, ks, vs, lut)
    ref = compile_kernel(flash_attention_i8_q15_16(q_heads, seqlen, dim), [7])(
        q,
        k.repeat_interleave(group, dim=0).contiguous(),
        v.repeat_interleave(group, dim=0).contiguous(),
        qs,
        ks.repeat_interleave(group, dim=0).contiguous(),
        vs.repeat_interleave(group, dim=0).contiguous(),
        lut,
    )
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_attention_i8_q15_16_gqa_tiled_matches_gqa():
    torch.manual_seed(0)
    q_heads, kv_heads, seqlen, dim = 16, 8, 128, 128
    q = torch.randint(-127, 128, (q_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    k = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    v = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    qs = torch.round((torch.rand((q_heads, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    ks = torch.round((torch.rand((kv_heads, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    vs = torch.round((torch.rand((kv_heads, seqlen), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    lut = torch.from_numpy(exp_lut_neg()).cuda()
    num, den = compile_kernel(flash_attention_i8_q15_16_gqa_tiled(q_heads, kv_heads, seqlen, dim), [7, 8])(q, k, v, qs, ks, vs, lut)
    y = compile_kernel(attention_normalize_q15_16(q_heads, seqlen, dim), [2])(num, den)
    ref = compile_kernel(flash_attention_i8_q15_16_gqa(q_heads, kv_heads, seqlen, dim), [7])(q, k, v, qs, ks, vs, lut)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_attention_i8_q15_16_gqa_softmax_i16v8_matches_torch_pv():
    torch.manual_seed(0)
    q_heads, kv_heads, seqlen, dim, block_n = 16, 8, 128, 128, 64
    group = q_heads // kv_heads
    q = torch.randint(-127, 128, (q_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    k = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    v = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    qs = torch.round((torch.rand((q_heads, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    ks = torch.round((torch.rand((kv_heads, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    vs = torch.round((torch.rand((kv_heads, seqlen), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    lut = torch.from_numpy(exp_lut_neg()).cuda()
    p = compile_kernel(attention_i8_q15_16_gqa_softmax_i16(q_heads, kv_heads, seqlen, dim, block_n=block_n), [5])(q, k, qs, ks, lut)
    y = compile_kernel(attention_i16v8_q15_16_gqa(q_heads, kv_heads, seqlen, dim, block_n=block_n), [3])(p, v, vs)
    v_rep = v.repeat_interleave(group, dim=0).to(torch.float64)
    vs_rep = vs.repeat_interleave(group, dim=0).to(torch.float64)[:, :, None]
    ref = torch.zeros((q_heads, seqlen, dim), device="cuda", dtype=torch.float64)
    for row_base in range(0, seqlen, 32):
        row_end = min(row_base + 32, seqlen)
        for nb in range(row_base // block_n + 1):
            s = nb * block_n
            e = min(s + block_n, seqlen)
            ref[:, row_base:row_end] += torch.matmul(p[:, row_base:row_end, s:e].to(torch.float64) / 16383.0, v_rep[:, s:e] * vs_rep[:, s:e] / Q15_16)
    ref = ref.permute(1, 0, 2).reshape(seqlen, q_heads * dim) * Q15_16
    cos = torch.nn.functional.cosine_similarity(y.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.99999


@tilelang.testing.requires_cuda
def test_attention_i8_q15_16_gqa_cache_matches_repeated_kv():
    torch.manual_seed(0)
    q_heads, kv_heads, cache_len, seqlen, dim = 16, 8, 17, 73, 128
    group = q_heads // kv_heads
    q = torch.randint(-127, 128, (q_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    k = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    v = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    ck = torch.randint(-127, 128, (kv_heads, cache_len, dim), device="cuda", dtype=torch.int8)
    cv = torch.randint(-127, 128, (kv_heads, cache_len, dim), device="cuda", dtype=torch.int8)
    qs = torch.round((torch.rand((q_heads, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    ks = torch.round((torch.rand((kv_heads, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    vs = torch.round((torch.rand((kv_heads, seqlen), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    cks = torch.round((torch.rand((kv_heads, cache_len), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    cvs = torch.round((torch.rand((kv_heads, cache_len), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    lut = torch.from_numpy(exp_lut_neg()).cuda()
    y = compile_kernel(flash_attention_i8_q15_16_gqa_cache(q_heads, kv_heads, seqlen, cache_len, dim), [11])(q, ck, cv, k, v, qs, cks, cvs, ks, vs, lut)
    ref = compile_kernel(flash_attention_i8_q15_16_cache(q_heads, seqlen, cache_len, dim), [11])(
        q,
        ck.repeat_interleave(group, dim=0).contiguous(),
        cv.repeat_interleave(group, dim=0).contiguous(),
        k.repeat_interleave(group, dim=0).contiguous(),
        v.repeat_interleave(group, dim=0).contiguous(),
        qs,
        cks.repeat_interleave(group, dim=0).contiguous(),
        cvs.repeat_interleave(group, dim=0).contiguous(),
        ks.repeat_interleave(group, dim=0).contiguous(),
        vs.repeat_interleave(group, dim=0).contiguous(),
        lut,
    )
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_attention_i8_q15_16_gqa_cache_softmax_i16v8_matches_cache_attention():
    torch.manual_seed(0)
    q_heads, kv_heads, cache_len, seqlen, dim, block_n = 16, 8, 17, 73, 128, 64
    q = torch.randint(-127, 128, (q_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    k = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    v = torch.randint(-127, 128, (kv_heads, seqlen, dim), device="cuda", dtype=torch.int8)
    ck = torch.randint(-127, 128, (kv_heads, cache_len, dim), device="cuda", dtype=torch.int8)
    cv = torch.randint(-127, 128, (kv_heads, cache_len, dim), device="cuda", dtype=torch.int8)
    qs = torch.round((torch.rand((q_heads, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    ks = torch.round((torch.rand((kv_heads, seqlen), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    vs = torch.round((torch.rand((kv_heads, seqlen), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    cks = torch.round((torch.rand((kv_heads, cache_len), device="cuda") * 0.02 + 0.005) * Q15_16).to(torch.uint32)
    cvs = torch.round((torch.rand((kv_heads, cache_len), device="cuda") * 0.03 + 0.004) * Q15_16).to(torch.uint32)
    lut = torch.from_numpy(exp_lut_neg()).cuda()
    ref = compile_kernel(flash_attention_i8_q15_16_gqa_cache(q_heads, kv_heads, seqlen, cache_len, dim, block_n=block_n), [11])(q, ck, cv, k, v, qs, cks, cvs, ks, vs, lut)
    p = compile_kernel(attention_i8_q15_16_gqa_cache_softmax_i16(q_heads, kv_heads, seqlen, cache_len, dim, block_n=block_n), [7])(q, ck, k, qs, cks, ks, lut)
    y = compile_kernel(attention_i16v8_q15_16_gqa_cache(q_heads, kv_heads, seqlen, cache_len, dim, block_n=block_n), [5])(p, cv, v, cvs, vs)
    y = y.reshape(seqlen, q_heads, dim).permute(1, 0, 2).contiguous()
    cos = torch.nn.functional.cosine_similarity(y.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.999


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
