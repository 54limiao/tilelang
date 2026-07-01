import math

import numpy as np
import pytest

from tilelang.language.fix import Q_MULTIPLIER_WIDTH, pack_scale

torch = pytest.importorskip("torch")

try:
    import tilelang
    import tilelang.testing
    import tilelang.language as T
except AssertionError as err:
    pytest.skip(f"TileLang native libraries are not built: {err}", allow_module_level=True)

MASK = (1 << Q_MULTIPLIER_WIDTH) - 1
PROB_SHIFT = 11
SOFTMAX_PROB_NUM = (1 << (15 + PROB_SHIFT)) - 1
EXP_TO_Q7 = 1.0 / 8.0
Q15_16 = 1 << 16


def exp_lut():
    return np.array([np.clip(round(math.exp(i / 64.0 - 8.0) * 1023.0), 0, 1023) for i in range(1024)], dtype=np.int16)


def rsqrt_lut():
    return np.array([0 if i < 640 else np.clip(round(1024.0 / math.sqrt(i / 128.0 - 4.0)), 0, 1023) for i in range(1024)], dtype=np.int16)


def silu_lut():
    return np.array([round((x / 256.0) / (1.0 + math.exp(-(x / 256.0))) * Q15_16) for x in range(-2048, 2048)], dtype=np.int32)


def fake_quant(x, scale, dtype=torch.int16):
    info = torch.iinfo(dtype)
    scale = torch.as_tensor(scale, dtype=torch.float32, device=x.device)
    return torch.clamp(torch.round(x / scale), info.min, info.max).to(dtype)


def pack(scales):
    return torch.tensor([pack_scale(float(s)) for s in scales], dtype=torch.uint32, device="cuda")


def recip_lut():
    return np.array([0 if i == 0 else min((127 << 9) // i, MASK) for i in range(4096)], dtype=np.uint32)


def quant_kernel(n, scalar_scale):
    @T.prim_func
    def main(
        X: T.Tensor((n,), "int32"),
        S0: T.Tensor((1,), "uint32"),
        SV: T.Tensor((n,), "uint32"),
        A: T.Tensor((n,), "int16"),
        B: T.Tensor((n,), "int16"),
        C: T.Tensor((n,), "int16"),
    ):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(n):
                A[i] = T.fix.quant(X[i], scale=scalar_scale, out_dtype="int16")
                B[i] = T.fix.quant(X[i], scale=S0[0], out_dtype="int16")
                C[i] = T.fix.quant(X[i], scale=SV[i], out_dtype="int16")

    return main


def dynamic_quant_kernel(rows, cols):
    @T.prim_func
    def main(
        X: T.Tensor((rows, cols), "int32"),
        LUT: T.Tensor((4096,), "uint32"),
        Y: T.Tensor((rows, cols), "int8"),
        S: T.Tensor((rows,), "uint32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            xa = T.alloc_fragment((1, cols), "int32")
            amax = T.alloc_fragment((1,), "int32")
            idx = T.alloc_fragment((1,), "int32")
            qt = T.alloc_fragment((1,), "int32")
            for c in T.Parallel(cols):
                xa[0, c] = T.abs(X[r, c])
            T.reduce_max(xa, amax, dim=1, clear=True)
            idx[0] = amax[0] >> T.int32(10)
            if idx[0] > T.int32(4095):
                idx[0] = T.int32(4095)
            if idx[0] < T.int32(1):
                idx[0] = T.int32(1)
            qt[0] = (T.int32(19) << T.int32(Q_MULTIPLIER_WIDTH)) | (T.cast(LUT[idx[0]], "int32") & T.int32(MASK))
            S[r] = T.cast(amax[0] // T.int32(127), "uint32")
            for c in T.Parallel(cols):
                Y[r, c] = T.fix.quant(X[r, c], scale=qt[0], out_dtype="int8")

    return main


def rope_q15_16_kernel(rows, dim):
    @T.prim_func
    def main(
        X: T.Tensor((rows, dim), "int32"),
        COS: T.Tensor((rows, dim // 2), "int32"),
        SIN: T.Tensor((rows, dim // 2), "int32"),
        Y: T.Tensor((rows, dim), "int32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            a = T.alloc_fragment((dim // 2,), "int32")
            b = T.alloc_fragment((dim // 2,), "int32")
            for d in T.Parallel(dim // 2):
                a[d] = (X[r, d] >> T.int32(8)) * (COS[r, d] >> T.int32(8))
                b[d] = (X[r, d + dim // 2] >> T.int32(8)) * (SIN[r, d] >> T.int32(8))
                Y[r, d] = a[d] - b[d]
                a[d] = (X[r, d] >> T.int32(8)) * (SIN[r, d] >> T.int32(8))
                b[d] = (X[r, d + dim // 2] >> T.int32(8)) * (COS[r, d] >> T.int32(8))
                Y[r, d + dim // 2] = a[d] + b[d]

    return main


def rms_q15_16_kernel(rows, cols):
    mean_shift = int(math.log2(cols))

    @T.prim_func
    def main(X: T.Tensor((rows, cols), "int32"), LUT: T.Tensor((1024,), "int16"), Y: T.Tensor((rows, cols), "int32")):
        with T.Kernel(rows, threads=128) as r:
            xx = T.alloc_fragment((1, cols), "int32")
            ss = T.alloc_fragment((1,), "int32")
            ns = T.alloc_fragment((1,), "int32")
            wk = T.alloc_fragment((1,), "int32")
            inv = T.alloc_fragment((1,), "int32")
            fold = T.alloc_fragment((1,), "int32")
            qt = T.alloc_fragment((1,), "int32")
            for c in T.Parallel(cols):
                xx[0, c] = ((X[r, c] >> T.int32(8)) * (X[r, c] >> T.int32(8))) >> T.int32(mean_shift)
            T.reduce_sum(xx, ss, dim=1, clear=True)
            ss[0] += T.int32(1)
            ns[0] = T.int32(0)
            wk[0] = ss[0]
            if (wk[0] & T.int32(-65536)) != T.int32(0):
                ns[0] += T.int32(16)
                wk[0] = wk[0] >> T.int32(16)
            if (wk[0] & T.int32(0xFF00)) != T.int32(0):
                ns[0] += T.int32(8)
                wk[0] = wk[0] >> T.int32(8)
            if (wk[0] & T.int32(0xF0)) != T.int32(0):
                ns[0] += T.int32(4)
                wk[0] = wk[0] >> T.int32(4)
            if (wk[0] & T.int32(0xC)) != T.int32(0):
                ns[0] += T.int32(2)
            inv[0] = T.fix.quant_lut(ss[0], LUT, scale=(ns[0] << T.int32(Q_MULTIPLIER_WIDTH)) | T.int32(128), index_dtype="int10", out_dtype="int32")
            fold[0] = T.fix.quant(inv[0], scale=1024.0, out_dtype="int32")
            qt[0] = ((T.int32(6) + (ns[0] >> T.int32(1))) << T.int32(Q_MULTIPLIER_WIDTH)) | ((fold[0] >> T.int32(4)) & T.int32(MASK))
            for c in T.Parallel(cols):
                Y[r, c] = T.fix.quant(X[r, c] >> T.int32(8), scale=qt[0], out_dtype="int32") << T.int32(6)

    return main


def silu_q15_16_kernel(rows, cols):
    @T.prim_func
    def main(X: T.Tensor((rows, cols), "int32"), LUT: T.Tensor((4096,), "int32"), Y: T.Tensor((rows, cols), "int32")):
        with T.Kernel(rows, threads=128) as r:
            idx = T.alloc_fragment((1, cols), "int32")
            for c in T.Parallel(cols):
                idx[0, c] = (X[r, c] >> T.int32(8)) + T.int32(2048)
                if idx[0, c] < T.int32(0):
                    idx[0, c] = T.int32(0)
                if idx[0, c] > T.int32(4095):
                    idx[0, c] = T.int32(4095)
                Y[r, c] = LUT[idx[0, c]]

    return main


def softmax_kernel(rows, cols, scale):
    @T.prim_func
    def main(X: T.Tensor((rows, cols), "int16"), LUT: T.Tensor((1024,), "int16"), Y: T.Tensor((rows, cols), "int16")):
        with T.Kernel(rows, threads=128) as r:
            xi = T.alloc_fragment((1, cols), "int32")
            e = T.alloc_fragment((1, cols), "int32")
            mx = T.alloc_fragment((1,), "int32")
            se = T.alloc_fragment((1,), "int32")
            qt = T.alloc_fragment((1,), "int32")
            mx[0] = T.int32(-2147483648)
            for c in T.Parallel(cols):
                xi[0, c] = T.cast(X[r, c], "int32")
            T.reduce_max(xi, mx, dim=1)
            for c in T.Parallel(cols):
                e[0, c] = T.fix.quant_lut(xi[0, c] - mx[0], LUT, scale=scale, index_dtype="int10", out_dtype="int32")
            T.reduce_sum(e, se, dim=1, clear=True)
            qt[0] = (T.int32(PROB_SHIFT) << T.int32(Q_MULTIPLIER_WIDTH)) | ((T.int32(SOFTMAX_PROB_NUM) // se[0]) & T.int32(MASK))
            for c in T.Parallel(cols):
                Y[r, c] = T.fix.quant(e[0, c], scale=qt[0], out_dtype="int16")

    return main


def rms_kernel(rows, cols):
    mean_shift = int(math.log2(cols))

    @T.prim_func
    def main(X: T.Tensor((rows, cols), "int16"), LUT: T.Tensor((1024,), "int16"), Y: T.Tensor((rows, cols), "int16")):
        with T.Kernel(rows, threads=128) as r:
            xx = T.alloc_fragment((1, cols), "int32")
            ss = T.alloc_fragment((1,), "int32")
            ns = T.alloc_fragment((1,), "int32")
            wk = T.alloc_fragment((1,), "int32")
            inv = T.alloc_fragment((1,), "int32")
            fold = T.alloc_fragment((1,), "int32")
            qt = T.alloc_fragment((1,), "int32")
            for c in T.Parallel(cols):
                xx[0, c] = (T.cast(X[r, c], "int32") * T.cast(X[r, c], "int32")) >> T.int32(mean_shift)
            T.reduce_sum(xx, ss, dim=1)
            ss[0] += T.int32(1)
            ns[0] = T.int32(0)
            wk[0] = ss[0]
            if (wk[0] & T.int32(-65536)) != T.int32(0):
                ns[0] += T.int32(16)
                wk[0] = wk[0] >> T.int32(16)
            if (wk[0] & T.int32(0xFF00)) != T.int32(0):
                ns[0] += T.int32(8)
                wk[0] = wk[0] >> T.int32(8)
            if (wk[0] & T.int32(0xF0)) != T.int32(0):
                ns[0] += T.int32(4)
                wk[0] = wk[0] >> T.int32(4)
            if (wk[0] & T.int32(0xC)) != T.int32(0):
                ns[0] += T.int32(2)
            inv[0] = T.fix.quant_lut(ss[0], LUT, scale=(ns[0] << T.int32(Q_MULTIPLIER_WIDTH)) | T.int32(128), index_dtype="int10", out_dtype="int32")
            fold[0] = T.fix.quant(inv[0], scale=1024.0, out_dtype="int32")
            qt[0] = ((T.int32(6) + (ns[0] >> T.int32(1))) << T.int32(Q_MULTIPLIER_WIDTH)) | ((fold[0] >> T.int32(4)) & T.int32(MASK))
            for c in T.Parallel(cols):
                Y[r, c] = T.fix.quant(T.cast(X[r, c], "int32"), scale=qt[0], out_dtype="int16")

    return main


def attn_kernel(batch, seqlen, dim, score_scale, block_n=32):
    @T.prim_func
    def main(
        Q: T.Tensor((batch, seqlen, dim), "int16"),
        K: T.Tensor((batch, seqlen, dim), "int16"),
        V: T.Tensor((batch, seqlen, dim), "int16"),
        LUT: T.Tensor((1024,), "int16"),
        O: T.Tensor((batch, seqlen, dim), "int16"),
    ):
        with T.Kernel(batch * seqlen, threads=128) as blk:
            b = blk // seqlen
            i = blk - b * seqlen
            dot = T.alloc_fragment((block_n, dim), "int32")
            red = T.alloc_fragment((block_n,), "int32")
            out = T.alloc_fragment((dim,), "int32")
            sc = T.alloc_fragment((1, block_n), "int32")
            ex = T.alloc_fragment((1, block_n), "int32")
            pv = T.alloc_fragment((dim, block_n), "int32")
            acc = T.alloc_fragment((dim,), "int32")
            mx = T.alloc_fragment((1,), "int32")
            bm = T.alloc_fragment((1,), "int32")
            nm = T.alloc_fragment((1,), "int32")
            os = T.alloc_fragment((1,), "int32")
            bs = T.alloc_fragment((1,), "int32")
            sm = T.alloc_fragment((1,), "int32")
            qt = T.alloc_fragment((1,), "int32")
            mx[0] = T.int32(-2147483648)
            sm[0] = T.int32(0)
            for d in T.Parallel(dim):
                acc[d] = T.int32(0)
            for nb in T.Pipelined(seqlen // block_n):
                for j, d in T.Parallel(block_n, dim):
                    dot[j, d] = T.cast(Q[b, i, d], "int32") * T.cast(K[b, nb * block_n + j, d], "int32")
                T.reduce_sum(dot, red, dim=1, clear=True)
                for j in T.Parallel(block_n):
                    sc[0, j] = T.fix.quant(red[j], scale=score_scale, out_dtype="int32")
                T.reduce_max(sc, bm, dim=1, clear=True)
                nm[0] = bm[0]
                if mx[0] > nm[0]:
                    nm[0] = mx[0]
                os[0] = T.fix.quant_lut(mx[0] - nm[0], LUT, scale=1.0, index_dtype="int10", out_dtype="int32")
                os[0] = T.fix.quant(os[0], scale=EXP_TO_Q7, out_dtype="int32")
                for j in T.Parallel(block_n):
                    ex[0, j] = T.fix.quant_lut(sc[0, j] - nm[0], LUT, scale=1.0, index_dtype="int10", out_dtype="int32")
                    ex[0, j] = T.fix.quant(ex[0, j], scale=EXP_TO_Q7, out_dtype="int32")
                T.reduce_sum(ex, bs, dim=1, clear=True)
                sm[0] = T.fix.quant(sm[0], scale=(T.int32(7) << T.int32(Q_MULTIPLIER_WIDTH)) | os[0], out_dtype="int32") + bs[0]
                for d, j in T.Parallel(dim, block_n):
                    pv[d, j] = ex[0, j] * T.cast(V[b, nb * block_n + j, d], "int32")
                T.reduce_sum(pv, out, dim=1, clear=True)
                for d in T.Parallel(dim):
                    acc[d] = T.fix.quant(acc[d], scale=(T.int32(7) << T.int32(Q_MULTIPLIER_WIDTH)) | os[0], out_dtype="int32") + out[d]
                mx[0] = nm[0]
            qt[0] = (T.int32(16) << T.int32(Q_MULTIPLIER_WIDTH)) | ((T.int32(MASK) // sm[0]) & T.int32(MASK))
            for d in T.Parallel(dim):
                O[b, i, d] = T.fix.quant(acc[d], scale=qt[0], out_dtype="int16")

    return main


@tilelang.testing.requires_cuda
def test_fix_quant_qdq_cuda():
    x = torch.tensor([-40.5, -8.25, -0.75, 0.0, 0.75, 8.25, 40.5], device="cuda")
    in_scale, out_scale = 0.25, 0.4
    qx = fake_quant(x, in_scale, torch.int32)
    scalar = in_scale / out_scale
    vec = torch.tensor([0.5, 0.625, 0.75, 1.0, 1.25, 0.375, 0.875], device="cuda")
    a, b, c = tilelang.compile(quant_kernel(x.numel(), scalar), out_idx=[3, 4, 5], target="cuda")(qx, pack([scalar]), pack(vec))
    vec_out_scale = in_scale / vec
    torch.testing.assert_close(a.float() * out_scale, x, rtol=0, atol=out_scale)
    torch.testing.assert_close(b.float() * out_scale, x, rtol=0, atol=out_scale)
    torch.testing.assert_close(c.float() * vec_out_scale, x, rtol=0, atol=float(vec_out_scale.max()))


@tilelang.testing.requires_cuda
def test_dynamic_quant_q15_16_cuda():
    x = torch.stack((
        torch.linspace(-1.2, 1.1, 128, device="cuda"),
        torch.cos(torch.arange(128, device="cuda").float() * 0.09) * 0.8,
    ))
    qx = fake_quant(x, 1.0 / Q15_16, torch.int32)
    y, s = tilelang.compile(dynamic_quant_kernel(*qx.shape), out_idx=[2, 3], target="cuda")(qx, torch.from_numpy(recip_lut()).cuda())
    row_scale = qx.abs().amax(dim=-1).clamp(min=1).float() / 127.0 / Q15_16
    golden = torch.round(x / row_scale[:, None]).clamp(-128, 127).to(torch.int8).float() * row_scale[:, None]
    torch.testing.assert_close(y.float() * row_scale[:, None], golden, rtol=0, atol=float(row_scale.max() * 2.01))
    src = tilelang.compile(dynamic_quant_kernel(*qx.shape), out_idx=[2, 3], target="cuda").get_kernel_source()
    assert "int64_t)X" not in src and "int64_t)amax" not in src and "int64_t)qt" not in src


@tilelang.testing.requires_cuda
def test_rope_q15_16_cuda():
    x = torch.linspace(-0.8, 0.9, 64, device="cuda").reshape(2, 32)
    theta = torch.arange(16, device="cuda").float()[None, :] * torch.tensor([[0.05], [0.11]], device="cuda")
    cos, sin = torch.cos(theta), torch.sin(theta)
    qx, qcos, qsin = fake_quant(x, 1.0 / Q15_16, torch.int32), fake_quant(cos, 1.0 / Q15_16, torch.int32), fake_quant(sin, 1.0 / Q15_16, torch.int32)
    y = tilelang.compile(rope_q15_16_kernel(*qx.shape), out_idx=[3], target="cuda")(qx, qcos, qsin)
    golden = torch.cat((x[:, :16] * cos - x[:, 16:] * sin, x[:, :16] * sin + x[:, 16:] * cos), dim=-1)
    torch.testing.assert_close(y.float() / Q15_16, golden, rtol=0, atol=0.012)


@tilelang.testing.requires_cuda
def test_rmsnorm_q15_16_cuda():
    x = torch.tensor([[-1.2, -0.3, 0.2, 0.5, 1.1, 0.7, -0.8, 0.1], [0.6, -1.0, 1.3, -1.5, 0.2, -0.4, 0.3, -0.2]], device="cuda")
    qx = fake_quant(x, 1.0 / Q15_16, torch.int32)
    y = tilelang.compile(rms_q15_16_kernel(*qx.shape), out_idx=[2], target="cuda")(qx, torch.from_numpy(rsqrt_lut()).cuda())
    golden = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True))
    torch.testing.assert_close(y.float() / Q15_16, golden, rtol=0, atol=0.1)


@tilelang.testing.requires_cuda
def test_silu_q15_16_cuda():
    x = torch.linspace(-6.0, 6.0, 256, device="cuda").reshape(2, 128)
    qx = fake_quant(x, 1.0 / Q15_16, torch.int32)
    y = tilelang.compile(silu_q15_16_kernel(*qx.shape), out_idx=[2], target="cuda")(qx, torch.from_numpy(silu_lut()).cuda())
    torch.testing.assert_close(y.float() / Q15_16, torch.nn.functional.silu(x), rtol=0, atol=0.018)


@tilelang.testing.requires_cuda
def test_softmax_cuda():
    logits = torch.stack((
        torch.linspace(-1.2, 1.1, 128, device="cuda"),
        torch.cos(torch.arange(128, device="cuda").float() * 0.07),
    ))
    in_scale, out_scale = 1.0 / 64.0, 1.0 / 32768.0
    x = fake_quant(logits, in_scale)
    y = tilelang.compile(softmax_kernel(*x.shape, in_scale * 64.0), out_idx=[2], target="cuda")(x, torch.from_numpy(exp_lut()).cuda())
    torch.testing.assert_close(y.float() * out_scale, torch.softmax(logits, dim=-1), rtol=0, atol=0.025)


@tilelang.testing.requires_cuda
def test_rmsnorm_cuda():
    x = torch.tensor([[-1.2, -0.3, 0.2, 0.5, 1.1, 0.7, -0.8, 0.1], [0.6, -1.0, 1.3, -1.5, 0.2, -0.4, 0.3, -0.2]], device="cuda")
    in_scale, out_scale = 1.0 / 256.0, 1.0 / 1024.0
    qx = fake_quant(x, in_scale)
    y = tilelang.compile(rms_kernel(*qx.shape), out_idx=[2], target="cuda")(qx, torch.from_numpy(rsqrt_lut()).cuda())
    golden = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True))
    torch.testing.assert_close(y.float() * out_scale, golden, rtol=0, atol=0.08)


@tilelang.testing.requires_cuda
def test_flash_attention_cuda():
    q = torch.linspace(-0.45, 0.55, 512, device="cuda").reshape(1, 64, 8)
    k = torch.cos(torch.arange(512, device="cuda").float() * 0.17).reshape(1, 64, 8) * 0.45
    v = torch.sin(torch.arange(512, device="cuda").float() * 0.11).reshape(1, 64, 8) * 0.25
    q_scale = k_scale = 1.0 / 128.0
    v_scale = 1.0 / 512.0
    attn_scale = 1.0 / math.sqrt(8.0)
    qx, kx, vx = fake_quant(q, q_scale), fake_quant(k, k_scale), fake_quant(v, v_scale)
    score_scale = q_scale * k_scale * attn_scale * 64.0
    y = tilelang.compile(attn_kernel(1, 64, 8, score_scale, block_n=32), out_idx=[4], target="cuda")(
        qx, kx, vx, torch.from_numpy(exp_lut()).cuda()
    )
    golden = torch.softmax(q @ k.transpose(-1, -2) * attn_scale, dim=-1) @ v
    torch.testing.assert_close(y.float() * v_scale, golden, rtol=0, atol=0.045)
