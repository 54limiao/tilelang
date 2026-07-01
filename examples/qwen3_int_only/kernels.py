import math

import numpy as np
import tilelang
import tilelang.language as T

from tilelang.language.fix import Q_MULTIPLIER_WIDTH, pack_scale

MASK = (1 << Q_MULTIPLIER_WIDTH) - 1
Q15_16 = 1 << 16
EXP_TO_Q7 = 1.0 / 8.0
ATTN_VALUE_SHIFT = 7


def exp_lut_neg():
    return np.array([np.clip(round(math.exp((i - 4096) / 64.0) * 1023.0), 0, 1023) for i in range(4097)], dtype=np.int16)


def rsqrt_lut():
    return np.array([0 if i < 640 else np.clip(round(1024.0 / math.sqrt(i / 128.0 - 4.0)), 0, 1023) for i in range(1024)], dtype=np.int16)


def silu_lut():
    return np.array([round((x / 256.0) / (1.0 + math.exp(-(x / 256.0))) * Q15_16) for x in range(-2048, 2048)], dtype=np.int32)


def recip_lut_i8():
    return np.array([0 if i == 0 else min((127 << 9) // i, MASK) for i in range(4096)], dtype=np.uint32)


def recip_lut_i16():
    return np.array([0 if i == 0 else min((32767 << 4) // i, MASK) for i in range(4096)], dtype=np.uint32)


def recip_lut_i12():
    return np.array([0 if i == 0 else min((2047 << 5) // i, MASK) for i in range(4096)], dtype=np.uint32)


def recip_lut_i16_norm():
    return np.array([0 if i == 0 else min((4095 << 4) // i, MASK) for i in range(4096)], dtype=np.uint32)


def dynamic_quant_q15_16(rows, cols, out_dtype="int8", qmax_override=None):
    qmax = 127 if out_dtype == "int8" else 32767
    if qmax_override is not None:
        qmax = qmax_override
    frac_shift = 9 if out_dtype == "int8" else 4
    if out_dtype == "int16" and qmax == 2047:
        frac_shift = 5

    @T.prim_func
    def main(
        X: T.Tensor((rows, cols), "int32"),
        LUT: T.Tensor((4096,), "uint32"),
        Y: T.Tensor((rows, cols), out_dtype),
        S: T.Tensor((rows,), "uint32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            xa = T.alloc_fragment((1, cols), "int32")
            amax = T.alloc_fragment((1,), "int32")
            idx = T.alloc_fragment((1,), "int32")
            idx_shift = T.alloc_fragment((1,), "int32")
            qt = T.alloc_fragment((1,), "int32")
            for c in T.Parallel(cols):
                xa[0, c] = X[r, c]
                if xa[0, c] < T.int32(0):
                    xa[0, c] = T.int32(0) - xa[0, c]
            T.reduce_max(xa, amax, dim=1, clear=True)
            idx_shift[0] = T.int32(10)
            if amax[0] > T.int32(4193280):
                idx_shift[0] = T.int32(11)
            if amax[0] > T.int32(8386560):
                idx_shift[0] = T.int32(12)
            if amax[0] > T.int32(16773120):
                idx_shift[0] = T.int32(13)
            if amax[0] > T.int32(33546240):
                idx_shift[0] = T.int32(14)
            if amax[0] > T.int32(67092480):
                idx_shift[0] = T.int32(15)
            if amax[0] > T.int32(134184960):
                idx_shift[0] = T.int32(16)
            if amax[0] > T.int32(268369920):
                idx_shift[0] = T.int32(17)
            if amax[0] > T.int32(536739840):
                idx_shift[0] = T.int32(18)
            if amax[0] > T.int32(1073479680):
                idx_shift[0] = T.int32(19)
            idx[0] = amax[0] >> idx_shift[0]
            if idx[0] > T.int32(4095):
                idx[0] = T.int32(4095)
            if idx[0] < T.int32(1):
                idx[0] = T.int32(1)
            qt[0] = ((T.int32(frac_shift) + idx_shift[0]) << T.int32(Q_MULTIPLIER_WIDTH)) | (T.cast(LUT[idx[0]], "int32") & T.int32(MASK))
            idx[0] = amax[0] // T.int32(qmax)
            if idx[0] < T.int32(1):
                idx[0] = T.int32(1)
            S[r] = T.cast(idx[0], "uint32")
            for c in T.Parallel(cols):
                if out_dtype == "int16":
                    xa[0, c] = X[r, c] // idx[0]
                    if xa[0, c] > T.int32(qmax):
                        xa[0, c] = T.int32(qmax)
                    if xa[0, c] < T.int32(0 - qmax - 1):
                        xa[0, c] = T.int32(0 - qmax - 1)
                    Y[r, c] = T.cast(xa[0, c], out_dtype)
                elif qmax_override is None or qmax_override >= 32767:
                    Y[r, c] = T.fix.quant(X[r, c], scale=qt[0], out_dtype=out_dtype)
                else:
                    xa[0, c] = T.fix.quant(X[r, c], scale=qt[0], out_dtype="int32")
                    if xa[0, c] > T.int32(qmax):
                        xa[0, c] = T.int32(qmax)
                    if xa[0, c] < T.int32(0 - qmax - 1):
                        xa[0, c] = T.int32(0 - qmax - 1)
                    Y[r, c] = T.cast(xa[0, c], out_dtype)

    return main


def rope_q15_16(rows, dim):
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


def rmsnorm_i16_q15_16_weighted(rows, cols):
    mean_shift = int(math.log2(cols))

    @T.prim_func
    def main(
        X: T.Tensor((rows, cols), "int16"),
        W: T.Tensor((cols,), "int32"),
        RLUT: T.Tensor((1024,), "int16"),
        Y: T.Tensor((rows, cols), "int32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            xx = T.alloc_fragment((1, cols), "int32")
            ss = T.alloc_fragment((1,), "int32")
            ns = T.alloc_fragment((1,), "int32")
            wk = T.alloc_fragment((1,), "int32")
            inv = T.alloc_fragment((1,), "int32")
            fold = T.alloc_fragment((1,), "int32")
            qt = T.alloc_fragment((1,), "int32")
            norm = T.alloc_fragment((cols,), "int32")
            for c in T.Parallel(cols):
                xx[0, c] = (T.cast(X[r, c], "int32") * T.cast(X[r, c], "int32")) >> T.int32(mean_shift)
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
            inv[0] = T.fix.quant_lut(ss[0], RLUT, scale=(ns[0] << T.int32(Q_MULTIPLIER_WIDTH)) | T.int32(128), index_dtype="int10", out_dtype="int32")
            fold[0] = T.fix.quant(inv[0], scale=1024.0, out_dtype="int32")
            qt[0] = ((T.int32(6) + (ns[0] >> T.int32(1))) << T.int32(Q_MULTIPLIER_WIDTH)) | ((fold[0] >> T.int32(4)) & T.int32(MASK))
            for c in T.Parallel(cols):
                norm[c] = T.fix.quant(T.cast(X[r, c], "int32"), scale=qt[0], out_dtype="int32")
                Y[r, c] = (norm[c] * (W[c] >> T.int32(8))) >> T.int32(2)

    return main


def silu_q15_16(rows, cols):
    @T.prim_func
    def main(X: T.Tensor((rows, cols), "int32"), LUT: T.Tensor((4096,), "int32"), Y: T.Tensor((rows, cols), "int32")):
        with T.Kernel(rows, threads=128) as r:
            idx = T.alloc_fragment((1, cols), "int32")
            for c in T.Parallel(cols):
                idx[0, c] = (X[r, c] >> T.int32(8)) + T.int32(2048)
                if idx[0, c] < T.int32(0):
                    Y[r, c] = T.int32(0)
                elif idx[0, c] > T.int32(4095):
                    Y[r, c] = X[r, c]
                else:
                    Y[r, c] = LUT[idx[0, c]]

    return main


def add_q15_16(rows, cols):
    @T.prim_func
    def main(A: T.Tensor((rows, cols), "int32"), B: T.Tensor((rows, cols), "int32"), Y: T.Tensor((rows, cols), "int32")):
        with T.Kernel(rows, threads=128) as r:
            for c in T.Parallel(cols):
                Y[r, c] = A[r, c] + B[r, c]

    return main


def mul_q15_16(rows, cols):
    @T.prim_func
    def main(A: T.Tensor((rows, cols), "int32"), B: T.Tensor((rows, cols), "int32"), Y: T.Tensor((rows, cols), "int32")):
        with T.Kernel(rows, threads=128) as r:
            for c in T.Parallel(cols):
                Y[r, c] = ((A[r, c] >> T.int32(8)) * (B[r, c] >> T.int32(8)))

    return main


def linear_dynamic_q15_16(rows, in_features, out_features, x_dtype="int8"):
    if x_dtype == "int16":
        chunk_size = 64
        while in_features % chunk_size != 0:
            chunk_size //= 2
        chunks = in_features // chunk_size

        @T.prim_func
        def main(
            X: T.Tensor((rows, in_features), x_dtype),
            XS: T.Tensor((rows,), "uint32"),
            W: T.Tensor((out_features, in_features), "int8"),
            WS: T.Tensor((out_features,), "uint32"),
            Y: T.Tensor((rows, out_features), "int32"),
        ):
            with T.Kernel(rows, out_features, threads=128) as (r, o):
                prod = T.alloc_fragment((chunks, chunk_size), "int32")
                acc = T.alloc_fragment((chunks,), "int32")
                total = T.alloc_fragment((1,), "int32")
                scale = T.alloc_fragment((1,), "int32")
                for g, k in T.Parallel(chunks, chunk_size):
                    prod[g, k] = T.cast(X[r, g * chunk_size + k], "int32") * T.cast(W[o, g * chunk_size + k], "int32")
                T.reduce_sum(prod, acc, dim=1, clear=True)
                T.reduce_sum(acc, total, dim=0, clear=True)
                scale[0] = T.cast((XS[r] * WS[o]) >> T.int32(2), "int32")
                Y[r, o] = ((total[0] >> T.int32(12)) * scale[0]) >> T.int32(2)

        return main

    scale_shift = 8
    out_shift = 8

    @T.prim_func
    def main(
        X: T.Tensor((rows, in_features), x_dtype),
        XS: T.Tensor((rows,), "uint32"),
        W: T.Tensor((out_features, in_features), "int8"),
        WS: T.Tensor((out_features,), "uint32"),
        Y: T.Tensor((rows, out_features), "int32"),
    ):
        with T.Kernel(rows, out_features, threads=128) as (r, o):
            prod = T.alloc_fragment((1, in_features), "int32")
            acc = T.alloc_fragment((1,), "int32")
            scale = T.alloc_fragment((1,), "int32")
            for k in T.Parallel(in_features):
                prod[0, k] = T.cast(X[r, k], "int32") * T.cast(W[o, k], "int32")
            T.reduce_sum(prod, acc, dim=1, clear=True)
            scale[0] = T.cast((XS[r] * WS[o]) >> T.int32(scale_shift), "int32")
            Y[r, o] = (acc[0] * scale[0]) >> T.int32(out_shift)

    return main


def linear_dynamic_int8_q15_16(rows, in_features, out_features):
    return linear_dynamic_q15_16(rows, in_features, out_features, "int8")


def linear_dynamic_int16_q15_16(rows, in_features, out_features):
    return linear_dynamic_q15_16(rows, in_features, out_features, "int16")


def flash_attention_i12_q15_16_per_scale(batch, seqlen, dim, block_n=32):
    @T.prim_func
    def main(
        Q: T.Tensor((batch, seqlen, dim), "int16"),
        K: T.Tensor((batch, seqlen, dim), "int16"),
        V: T.Tensor((batch, seqlen, dim), "int16"),
        LUT: T.Tensor((4097,), "int16"),
        QS: T.Tensor((batch * seqlen * seqlen,), "uint32"),
        VS: T.Tensor((batch, seqlen), "uint32"),
        O: T.Tensor((batch, seqlen, dim), "int32"),
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
            hi = T.alloc_fragment((block_n,), "int32")
            lo = T.alloc_fragment((block_n,), "int32")
            mx = T.alloc_fragment((1,), "int32")
            bm = T.alloc_fragment((1,), "int32")
            nm = T.alloc_fragment((1,), "int32")
            os = T.alloc_fragment((1,), "int32")
            bs = T.alloc_fragment((1,), "int32")
            sm = T.alloc_fragment((1,), "int32")
            ei = T.alloc_fragment((1, block_n), "int32")
            idx = T.alloc_fragment((1,), "int32")
            mx[0] = T.int32(-2147483648)
            sm[0] = T.int32(0)
            for d in T.Parallel(dim):
                acc[d] = T.int32(0)
            for nb in T.Pipelined(seqlen // block_n):
                for j, d in T.Parallel(block_n, dim):
                    dot[j, d] = T.cast(Q[b, i, d], "int32") * T.cast(K[b, nb * block_n + j, d], "int32")
                T.reduce_sum(dot, red, dim=1, clear=True)
                for j in T.Parallel(block_n):
                    hi[j] = red[j] >> T.int32(14)
                    lo[j] = red[j] - (hi[j] << T.int32(14))
                    sc[0, j] = T.fix.quant(
                        hi[j],
                        scale=(((T.cast(QS[(b * seqlen + i) * seqlen + nb * block_n + j], "int32") >> T.int32(Q_MULTIPLIER_WIDTH)) - T.int32(14)) << T.int32(Q_MULTIPLIER_WIDTH))
                        | (T.cast(QS[(b * seqlen + i) * seqlen + nb * block_n + j], "int32") & T.int32(MASK)),
                        out_dtype="int32",
                    )
                    sc[0, j] += T.fix.quant(lo[j], scale=T.cast(QS[(b * seqlen + i) * seqlen + nb * block_n + j], "int32"), out_dtype="int32")
                    if nb * block_n + j > i:
                        sc[0, j] = T.int32(-32768)
                T.reduce_max(sc, bm, dim=1, clear=True)
                nm[0] = bm[0]
                if mx[0] > nm[0]:
                    nm[0] = mx[0]
                os[0] = mx[0] - nm[0]
                if os[0] < T.int32(-4096):
                    os[0] = T.int32(-4096)
                if os[0] > T.int32(0):
                    os[0] = T.int32(0)
                os[0] = T.cast(LUT[os[0] + T.int32(4096)], "int32")
                for j in T.Parallel(block_n):
                    ei[0, j] = sc[0, j] - nm[0]
                    if ei[0, j] < T.int32(-4096):
                        ei[0, j] = T.int32(-4096)
                    if ei[0, j] > T.int32(0):
                        ei[0, j] = T.int32(0)
                    ex[0, j] = T.cast(LUT[ei[0, j] + T.int32(4096)], "int32")
                T.reduce_sum(ex, bs, dim=1, clear=True)
                sm[0] = ((sm[0] * os[0]) >> T.int32(10)) + bs[0]
                for d, j in T.Parallel(dim, block_n):
                    pv[d, j] = ex[0, j] * ((T.cast(V[b, nb * block_n + j, d], "int32") * T.cast(VS[b, nb * block_n + j], "int32")) >> T.int32(ATTN_VALUE_SHIFT))
                T.reduce_sum(pv, out, dim=1, clear=True)
                for d in T.Parallel(dim):
                    acc[d] = (((acc[d] >> T.int32(7)) * os[0]) >> T.int32(3)) + out[d]
                mx[0] = nm[0]
            for d in T.Parallel(dim):
                O[b, i, d] = (acc[d] // sm[0]) << T.int32(ATTN_VALUE_SHIFT)

    return main


def packed_scale_matrix(scales):
    x = np.maximum(scales.detach().cpu().numpy().astype(np.float64), 1e-12)
    frac, exp = np.frexp(x)
    shift = (Q_MULTIPLIER_WIDTH - exp).astype(np.uint32)
    mul = np.rint(frac * MASK).astype(np.uint32)
    return ((shift.astype(np.uint32) << np.uint32(Q_MULTIPLIER_WIDTH)) | (mul & np.uint32(MASK))).reshape(tuple(scales.shape))


def compile_kernel(func, out_idx):
    return tilelang.compile(func, out_idx=out_idx, target="cuda")
