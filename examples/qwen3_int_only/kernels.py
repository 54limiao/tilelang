import math

import numpy as np
import tilelang
import tilelang.language as T

from tilelang.language.fix import Q_MULTIPLIER_WIDTH

MASK = (1 << Q_MULTIPLIER_WIDTH) - 1
Q15_16 = 1 << 16
ATTN_VALUE_SHIFT = 7
I32_MIN = -2147483648


def exp_lut_neg():
    return np.array([np.clip(round(math.exp(min(i - 512, 0) / 8.0) * 1023.0), 0, 1023) for i in range(1024)], dtype=np.int16)


def rsqrt_lut():
    return np.array([0 if i < 640 else np.clip(round(1024.0 / math.sqrt(i / 128.0 - 4.0)), 0, 1023) for i in range(1024)], dtype=np.int16)


def sigmoid_lut():
    def sigmoid(x):
        if x <= -7.0:
            return 0.0
        if x >= 7.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    return np.array([round(sigmoid((i - 512) / 64.0) * Q15_16) for i in range(1024)], dtype=np.int32)


def dynamic_quant_q15_16(rows, cols, out_dtype="int8", qmax_override=None):
    qmax = 127 if out_dtype == "int8" else 32767
    if qmax_override is not None:
        qmax = qmax_override

    @T.prim_func
    def main(
        X: T.Tensor((rows, cols), "int32"),
        Y: T.Tensor((rows, cols), out_dtype),
        S: T.Tensor((rows,), "uint32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            abs_x = T.alloc_fragment((1, cols), "int32")
            amax = T.alloc_fragment((1,), "int32")
            scale = T.alloc_fragment((1,), "int32")
            for c in T.Parallel(cols):
                abs_x[0, c] = X[r, c]
                if abs_x[0, c] < T.int32(0):
                    abs_x[0, c] = T.int32(0) - abs_x[0, c]
            T.reduce_max(abs_x, amax, dim=1, clear=True)
            scale[0] = T.max(amax[0] // T.int32(qmax), T.int32(1))
            S[r] = T.cast(scale[0], "uint32")
            for c in T.Parallel(cols):
                abs_x[0, c] = T.min(T.max(X[r, c] // scale[0], T.int32(0 - qmax - 1)), T.int32(qmax))
                Y[r, c] = T.cast(abs_x[0, c], out_dtype)

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
            inv[0] = T.fix.lut_10bit(ss[0], RLUT, scale=(ns[0] << T.int32(Q_MULTIPLIER_WIDTH)) | T.int32(128), out_dtype="int32")
            fold[0] = T.fix.quant(inv[0], scale=1024.0, out_dtype="int32")
            qt[0] = ((T.int32(6) + (ns[0] >> T.int32(1))) << T.int32(Q_MULTIPLIER_WIDTH)) | ((fold[0] >> T.int32(4)) & T.int32(MASK))
            for c in T.Parallel(cols):
                norm[c] = T.fix.quant(T.cast(X[r, c], "int32"), scale=qt[0], out_dtype="int32")
                Y[r, c] = (norm[c] * (W[c] >> T.int32(8))) >> T.int32(2)

    return main


def silu_q15_16(rows, cols):
    @T.prim_func
    def main(X: T.Tensor((rows, cols), "int32"), LUT: T.Tensor((1024,), "int32"), Y: T.Tensor((rows, cols), "int32")):
        with T.Kernel(rows, threads=128) as r:
            sig = T.alloc_fragment((1, cols), "int32")
            for c in T.Parallel(cols):
                sig[0, c] = T.fix.lut_10bit(X[r, c], LUT, scale=1.0 / 1024.0, out_dtype="int32")
                Y[r, c] = (X[r, c] >> T.int32(8)) * (sig[0, c] >> T.int32(8))

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


def linear_dynamic_int8_q15_16(rows, in_features, out_features, block_m=16, block_n=16, block_k=64):
    local_m = 2
    local_n = 2
    threads = (block_m // local_m) * (block_n // local_n)

    @T.prim_func
    def main(
        X: T.Tensor((rows, in_features), "int8"),
        XS: T.Tensor((rows,), "uint32"),
        W: T.Tensor((out_features, in_features), "int8"),
        WS: T.Tensor((out_features,), "uint32"),
        Y: T.Tensor((rows, out_features), "int32"),
    ):
        with T.Kernel(T.ceildiv(out_features, block_n), T.ceildiv(rows, block_m), threads=threads) as (bo, br):
            x_shared = T.alloc_shared((block_m, block_k), "int8")
            w_shared = T.alloc_shared((block_n, block_k), "int8")
            x_local = T.alloc_local((local_m, 4), "int8")
            w_local = T.alloc_local((local_n, 4), "int8")
            acc = T.alloc_local((local_m, local_n), "int32")
            tid = T.get_thread_binding()
            tm = tid % (block_m // local_m)
            tn = tid // (block_m // local_m)

            T.clear(acc)
            for ko in T.Pipelined(in_features // block_k, num_stages=2):
                T.copy(X[br * block_m, ko * block_k], x_shared)
                T.copy(W[bo * block_n, ko * block_k], w_shared)
                for ki in T.serial(block_k // 4):
                    for mi in T.serial(local_m):
                        for kk in T.vectorized(4):
                            x_local[mi, kk] = x_shared[tm * local_m + mi, ki * 4 + kk]
                    for ni in T.serial(local_n):
                        for kk in T.vectorized(4):
                            w_local[ni, kk] = w_shared[tn * local_n + ni, ki * 4 + kk]
                    for mi, ni in T.grid(local_m, local_n):
                        T.dp4a(x_local[mi, 0], w_local[ni, 0], acc[mi, ni])

            for mi, ni in T.grid(local_m, local_n):
                Y[br * block_m + tm * local_m + mi, bo * block_n + tn * local_n + ni] = (
                    acc[mi, ni] * T.cast((XS[br * block_m + tm * local_m + mi] * WS[bo * block_n + tn * local_n + ni]) >> T.int32(8), "int32")
                ) >> T.int32(8)

    return main


def linear_dynamic_int16_q15_16(rows, in_features, out_features):
    chunk_size = 64
    while in_features % chunk_size != 0:
        chunk_size //= 2
    chunks = in_features // chunk_size

    @T.prim_func
    def main(
        X: T.Tensor((rows, in_features), "int16"),
        XS: T.Tensor((rows,), "uint32"),
        W: T.Tensor((out_features, in_features), "int8"),
        WS: T.Tensor((out_features,), "uint32"),
        Y: T.Tensor((rows, out_features), "int32"),
    ):
        with T.Kernel(rows, out_features, threads=128) as (r, o):
            prod = T.alloc_fragment((chunks, chunk_size), "int32")
            partial = T.alloc_fragment((chunks,), "int32")
            acc = T.alloc_fragment((1,), "int32")
            for g, k in T.Parallel(chunks, chunk_size):
                prod[g, k] = T.cast(X[r, g * chunk_size + k], "int32") * T.cast(W[o, g * chunk_size + k], "int32")
            T.reduce_sum(prod, partial, dim=1, clear=True)
            T.reduce_sum(partial, acc, dim=0, clear=True)
            Y[r, o] = ((acc[0] >> T.int32(12)) * T.cast((XS[r] * WS[o]) >> T.int32(2), "int32")) >> T.int32(2)

    return main


def flash_attention_i12_q15_16_per_scale(batch, seqlen, dim, block_n=32):
    @T.prim_func
    def main(
        Q: T.Tensor((batch, seqlen, dim), "int16"),
        K: T.Tensor((batch, seqlen, dim), "int16"),
        V: T.Tensor((batch, seqlen, dim), "int16"),
        LUT: T.Tensor((1024,), "int16"),
        QS: T.Tensor((batch * seqlen * seqlen,), "uint32"),
        VS: T.Tensor((batch, seqlen), "uint32"),
        O: T.Tensor((batch, seqlen, dim), "int32"),
    ):
        with T.Kernel(batch * seqlen, threads=128) as blk:
            b = blk // seqlen
            i = blk - b * seqlen
            qk = T.alloc_fragment((block_n, dim), "int32")
            qk_sum = T.alloc_fragment((block_n,), "int32")
            value_part = T.alloc_fragment((dim,), "int32")
            score = T.alloc_fragment((1, block_n), "int32")
            score_exp = T.alloc_fragment((1, block_n), "int32")
            weighted_value = T.alloc_fragment((dim, block_n), "int32")
            acc_o = T.alloc_fragment((dim,), "int32")
            score_hi = T.alloc_fragment((block_n,), "int32")
            score_lo = T.alloc_fragment((block_n,), "int32")
            score_max = T.alloc_fragment((1,), "int32")
            block_max = T.alloc_fragment((1,), "int32")
            new_max = T.alloc_fragment((1,), "int32")
            old_scale = T.alloc_fragment((1,), "int32")
            block_sum = T.alloc_fragment((1,), "int32")
            denom = T.alloc_fragment((1,), "int32")
            score_max[0] = T.int32(I32_MIN)
            denom[0] = T.int32(0)
            for d in T.Parallel(dim):
                acc_o[d] = T.int32(0)
            for nb in T.Pipelined(seqlen // block_n):
                for j, d in T.Parallel(block_n, dim):
                    qk[j, d] = T.cast(Q[b, i, d], "int32") * T.cast(K[b, nb * block_n + j, d], "int32")
                T.reduce_sum(qk, qk_sum, dim=1, clear=True)
                for j in T.Parallel(block_n):
                    score_hi[j] = qk_sum[j] >> T.int32(14)
                    score_lo[j] = qk_sum[j] - (score_hi[j] << T.int32(14))
                    score[0, j] = T.fix.quant(
                        score_hi[j],
                        scale=(((T.cast(QS[(b * seqlen + i) * seqlen + nb * block_n + j], "int32") >> T.int32(Q_MULTIPLIER_WIDTH)) - T.int32(14)) << T.int32(Q_MULTIPLIER_WIDTH))
                        | (T.cast(QS[(b * seqlen + i) * seqlen + nb * block_n + j], "int32") & T.int32(MASK)),
                        out_dtype="int32",
                    )
                    score[0, j] += T.fix.quant(score_lo[j], scale=T.cast(QS[(b * seqlen + i) * seqlen + nb * block_n + j], "int32"), out_dtype="int32")
                    score[0, j] = T.if_then_else(nb * block_n + j > i, T.int32(-32768), score[0, j])

                T.reduce_max(score, block_max, dim=1, clear=True)
                new_max[0] = T.max(block_max[0], score_max[0])
                old_scale[0] = T.fix.lut_10bit(score_max[0] - new_max[0], LUT, scale=1.0 / 8.0, out_dtype="int32")
                for j in T.Parallel(block_n):
                    score_exp[0, j] = T.fix.lut_10bit(score[0, j] - new_max[0], LUT, scale=1.0 / 8.0, out_dtype="int32")
                T.reduce_sum(score_exp, block_sum, dim=1, clear=True)
                denom[0] = ((denom[0] * old_scale[0]) >> T.int32(10)) + block_sum[0]
                for d, j in T.Parallel(dim, block_n):
                    weighted_value[d, j] = score_exp[0, j] * ((T.cast(V[b, nb * block_n + j, d], "int32") * T.cast(VS[b, nb * block_n + j], "int32")) >> T.int32(ATTN_VALUE_SHIFT))
                T.reduce_sum(weighted_value, value_part, dim=1, clear=True)
                for d in T.Parallel(dim):
                    acc_o[d] = (((acc_o[d] >> T.int32(7)) * old_scale[0]) >> T.int32(3)) + value_part[d]
                score_max[0] = new_max[0]
            for d in T.Parallel(dim):
                O[b, i, d] = (acc_o[d] // denom[0]) << T.int32(ATTN_VALUE_SHIFT)

    return main


def packed_scale_matrix(scales):
    x = np.maximum(scales.detach().cpu().numpy().astype(np.float64), 1e-12)
    frac, exp = np.frexp(x)
    shift = (Q_MULTIPLIER_WIDTH - exp).astype(np.uint32)
    mul = np.rint(frac * MASK).astype(np.uint32)
    return ((shift.astype(np.uint32) << np.uint32(Q_MULTIPLIER_WIDTH)) | (mul & np.uint32(MASK))).reshape(tuple(scales.shape))


def compile_kernel(func, out_idx):
    return tilelang.compile(func, out_idx=out_idx, target="cuda")
