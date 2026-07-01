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
        x = min(max(x, -7.0), 7.0)
        return 1.0 / (1.0 + math.exp(-x))

    return np.array([round(sigmoid((i - 512) / 64.0) * 1024.0) for i in range(1024)], dtype=np.int32)


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


def add_dynamic_quant_q15_16(rows, cols, qmax=4095):
    @T.prim_func
    def main(
        A: T.Tensor((rows, cols), "int32"),
        B: T.Tensor((rows, cols), "int32"),
        Y: T.Tensor((rows, cols), "int32"),
        Q: T.Tensor((rows, cols), "int16"),
        S: T.Tensor((rows,), "uint32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            vals = T.alloc_fragment((1, cols), "int32")
            abs_x = T.alloc_fragment((1, cols), "int32")
            amax = T.alloc_fragment((1,), "int32")
            scale = T.alloc_fragment((1,), "int32")
            for c in T.Parallel(cols):
                vals[0, c] = A[r, c] + B[r, c]
                abs_x[0, c] = vals[0, c]
                if abs_x[0, c] < T.int32(0):
                    abs_x[0, c] = T.int32(0) - abs_x[0, c]
            T.reduce_max(abs_x, amax, dim=1, clear=True)
            scale[0] = T.max(amax[0] // T.int32(qmax), T.int32(1))
            S[r] = T.cast(scale[0], "uint32")
            for c in T.Parallel(cols):
                Y[r, c] = vals[0, c]
                vals[0, c] = T.min(T.max(vals[0, c] // scale[0], T.int32(0 - qmax - 1)), T.int32(qmax))
                Q[r, c] = T.cast(vals[0, c], "int16")

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


def rope_rotate_q15_16(rows, dim):
    @T.prim_func
    def main(
        X: T.Tensor((rows, dim), "int32"),
        COS: T.Tensor((rows, dim // 2), "int32"),
        SIN: T.Tensor((rows, dim // 2), "int32"),
        R: T.Tensor((dim, dim), "int32"),
        Y: T.Tensor((rows, dim), "int32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            rope_lo = T.alloc_fragment((dim // 2,), "int32")
            rope_hi = T.alloc_fragment((dim // 2,), "int32")
            acc = T.alloc_fragment((dim,), "int32")
            for d in T.Parallel(dim // 2):
                rope_lo[d] = ((X[r, d] >> T.int32(8)) * (COS[r, d] >> T.int32(8))) - (
                    (X[r, d + dim // 2] >> T.int32(8)) * (SIN[r, d] >> T.int32(8))
                )
                rope_hi[d] = ((X[r, d] >> T.int32(8)) * (SIN[r, d] >> T.int32(8))) + (
                    (X[r, d + dim // 2] >> T.int32(8)) * (COS[r, d] >> T.int32(8))
                )
            for o in T.Parallel(dim):
                acc[o] = T.int32(0)
                for k in T.serial(dim // 2):
                    acc[o] += (rope_lo[k] >> T.int32(8)) * (R[k, o] >> T.int32(8))
                    acc[o] += (rope_hi[k] >> T.int32(8)) * (R[k + dim // 2, o] >> T.int32(8))
                Y[r, o] = acc[o]

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


def rmsnorm_q15_16_weighted(rows, cols, qmax=4095):
    mean_shift = int(math.log2(cols))

    @T.prim_func
    def main(
        X: T.Tensor((rows, cols), "int32"),
        W: T.Tensor((cols,), "int32"),
        RLUT: T.Tensor((1024,), "int16"),
        Y: T.Tensor((rows, cols), "int32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            q = T.alloc_fragment((1, cols), "int32")
            xx = T.alloc_fragment((1, cols), "int32")
            amax = T.alloc_fragment((1,), "int32")
            scale = T.alloc_fragment((1,), "int32")
            ss = T.alloc_fragment((1,), "int32")
            ns = T.alloc_fragment((1,), "int32")
            wk = T.alloc_fragment((1,), "int32")
            inv = T.alloc_fragment((1,), "int32")
            fold = T.alloc_fragment((1,), "int32")
            qt = T.alloc_fragment((1,), "int32")
            norm = T.alloc_fragment((cols,), "int32")
            for c in T.Parallel(cols):
                q[0, c] = X[r, c]
                if q[0, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
            T.reduce_max(q, amax, dim=1, clear=True)
            scale[0] = T.max(amax[0] // T.int32(qmax), T.int32(1))
            for c in T.Parallel(cols):
                q[0, c] = T.min(T.max(X[r, c] // scale[0], T.int32(0 - qmax - 1)), T.int32(qmax))
                xx[0, c] = (q[0, c] * q[0, c]) >> T.int32(mean_shift)
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
                norm[c] = T.fix.quant(q[0, c], scale=qt[0], out_dtype="int32")
                Y[r, c] = (norm[c] * (W[c] >> T.int32(8))) >> T.int32(2)

    return main


def silu_q15_16(rows, cols):
    @T.prim_func
    def main(X: T.Tensor((rows, cols), "int32"), LUT: T.Tensor((1024,), "int32"), Y: T.Tensor((rows, cols), "int32")):
        with T.Kernel(rows, threads=128) as r:
            sig = T.alloc_fragment((1, cols), "int32")
            for c in T.Parallel(cols):
                sig[0, c] = T.fix.lut_10bit(X[r, c], LUT, scale=1.0 / 1024.0, out_dtype="int32")
                Y[r, c] = (X[r, c] >> T.int32(10)) * sig[0, c]

    return main


def silu_mul_dynamic_quant_q15_16(rows, cols):
    @T.prim_func
    def main(
        Gate: T.Tensor((rows, cols), "int32"),
        Up: T.Tensor((rows, cols), "int32"),
        LUT: T.Tensor((1024,), "int32"),
        Y: T.Tensor((rows, cols), "int32"),
        Q: T.Tensor((rows, cols), "int8"),
        S: T.Tensor((rows,), "uint32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            vals = T.alloc_fragment((1, cols), "int32")
            abs_x = T.alloc_fragment((1, cols), "int32")
            amax = T.alloc_fragment((1,), "int32")
            scale = T.alloc_fragment((1,), "int32")
            for c in T.Parallel(cols):
                vals[0, c] = (((Gate[r, c] >> T.int32(10)) * T.fix.lut_10bit(Gate[r, c], LUT, scale=1.0 / 1024.0, out_dtype="int32")) >> T.int32(8)) * (Up[r, c] >> T.int32(8))
                abs_x[0, c] = vals[0, c]
                if abs_x[0, c] < T.int32(0):
                    abs_x[0, c] = T.int32(0) - abs_x[0, c]
            T.reduce_max(abs_x, amax, dim=1, clear=True)
            scale[0] = T.max(amax[0] // T.int32(127), T.int32(1))
            S[r] = T.cast(scale[0], "uint32")
            for c in T.Parallel(cols):
                Y[r, c] = vals[0, c]
                vals[0, c] = T.min(T.max(vals[0, c] // scale[0], T.int32(-128)), T.int32(127))
                Q[r, c] = T.cast(vals[0, c], "int8")

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


def linear_dynamic_int8_q15_16(rows, in_features, out_features, block_m=16, block_n=32, block_k=64):
    @T.prim_func
    def main(
        X: T.Tensor((rows, in_features), "int8"),
        XS: T.Tensor((rows,), "uint32"),
        W: T.Tensor((out_features, in_features), "int8"),
        WS: T.Tensor((out_features,), "uint32"),
        Y: T.Tensor((rows, out_features), "int32"),
    ):
        with T.Kernel(T.ceildiv(out_features, block_n), T.ceildiv(rows, block_m), threads=128) as (bo, br):
            x_shared = T.alloc_shared((block_m, block_k), "int8")
            w_shared = T.alloc_shared((block_n, block_k), "int8")
            acc = T.alloc_fragment((block_m, block_n), "int32")

            T.clear(acc)
            for ko in T.Pipelined(in_features // block_k, num_stages=2):
                T.copy(X[br * block_m, ko * block_k], x_shared)
                T.copy(W[bo * block_n, ko * block_k], w_shared)
                T.gemm(x_shared, w_shared, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for m, n in T.Parallel(block_m, block_n):
                Y[br * block_m + m, bo * block_n + n] = (
                    (acc[m, n] >> T.int32(8)) * T.cast((XS[br * block_m + m] * WS[bo * block_n + n]) >> T.int32(8), "int32")
                )

    return main


def linear_dynamic_int8_pair_q15_16(rows, in_features, out_features, block_m=16, block_n=32, block_k=64):
    @T.prim_func
    def main(
        X: T.Tensor((rows, in_features), "int8"),
        XS: T.Tensor((rows,), "uint32"),
        W0: T.Tensor((out_features, in_features), "int8"),
        WS0: T.Tensor((out_features,), "uint32"),
        W1: T.Tensor((out_features, in_features), "int8"),
        WS1: T.Tensor((out_features,), "uint32"),
        Y0: T.Tensor((rows, out_features), "int32"),
        Y1: T.Tensor((rows, out_features), "int32"),
    ):
        with T.Kernel(T.ceildiv(out_features, block_n), T.ceildiv(rows, block_m), threads=128) as (bo, br):
            x_shared = T.alloc_shared((block_m, block_k), "int8")
            w0_shared = T.alloc_shared((block_n, block_k), "int8")
            w1_shared = T.alloc_shared((block_n, block_k), "int8")
            acc0 = T.alloc_fragment((block_m, block_n), "int32")
            acc1 = T.alloc_fragment((block_m, block_n), "int32")

            T.clear(acc0)
            T.clear(acc1)
            for ko in T.Pipelined(in_features // block_k, num_stages=2):
                T.copy(X[br * block_m, ko * block_k], x_shared)
                T.copy(W0[bo * block_n, ko * block_k], w0_shared)
                T.copy(W1[bo * block_n, ko * block_k], w1_shared)
                T.gemm(x_shared, w0_shared, acc0, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(x_shared, w1_shared, acc1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for m, n in T.Parallel(block_m, block_n):
                Y0[br * block_m + m, bo * block_n + n] = (
                    (acc0[m, n] >> T.int32(8)) * T.cast((XS[br * block_m + m] * WS0[bo * block_n + n]) >> T.int32(8), "int32")
                )
                Y1[br * block_m + m, bo * block_n + n] = (
                    (acc1[m, n] >> T.int32(8)) * T.cast((XS[br * block_m + m] * WS1[bo * block_n + n]) >> T.int32(8), "int32")
                )

    return main


def flash_attention_i8_q15_16(batch, seqlen, dim, block_n=64, score_shift=26, lut_scale=0.125):
    @T.prim_func
    def main(
        Q: T.Tensor((batch, seqlen, dim), "int8"),
        K: T.Tensor((batch, seqlen, dim), "int8"),
        V: T.Tensor((batch, seqlen, dim), "int8"),
        QS: T.Tensor((batch, seqlen), "uint32"),
        KS: T.Tensor((batch, seqlen), "uint32"),
        VS: T.Tensor((batch, seqlen), "uint32"),
        LUT: T.Tensor((1024,), "int16"),
        O: T.Tensor((batch, seqlen, dim), "int32"),
    ):
        with T.Kernel(batch * seqlen, threads=128) as blk:
            b = blk // seqlen
            i = blk - b * seqlen
            qk = T.alloc_fragment((block_n, dim), "int32")
            qk_sum = T.alloc_fragment((block_n,), "int32")
            score = T.alloc_fragment((1, block_n), "int32")
            score_exp = T.alloc_fragment((1, block_n), "int32")
            weighted_value = T.alloc_fragment((dim, block_n), "int32")
            value_part = T.alloc_fragment((dim,), "int32")
            acc_o = T.alloc_fragment((dim,), "int32")
            block_max = T.alloc_fragment((1,), "int32")
            new_max = T.alloc_fragment((1,), "int32")
            old_scale = T.alloc_fragment((1,), "int32")
            block_sum = T.alloc_fragment((1,), "int32")
            denom = T.alloc_fragment((1,), "int32")
            score_max = T.alloc_fragment((1,), "int32")
            scale = T.alloc_fragment((block_n,), "int32")
            score_max[0] = T.int32(I32_MIN)
            denom[0] = T.int32(0)
            for d in T.Parallel(dim):
                acc_o[d] = T.int32(0)
            for nb in T.Pipelined(i // block_n + 1):
                for j, d in T.Parallel(block_n, dim):
                    qk[j, d] = T.cast(Q[b, i, d], "int32") * T.cast(T.if_then_else(nb * block_n + j < seqlen, K[b, nb * block_n + j, d], T.int8(0)), "int32")
                T.reduce_sum(qk, qk_sum, dim=1, clear=True)
                for j in T.Parallel(block_n):
                    scale[j] = (
                        ((T.cast(QS[b, i], "int32") >> T.int32(4)) * (T.cast(T.if_then_else(nb * block_n + j < seqlen, KS[b, nb * block_n + j], T.uint32(1)), "int32") >> T.int32(4)))
                        >> T.int32(8)
                    ) * T.int32(5793)
                    score[0, j] = ((qk_sum[j] >> T.int32(8)) * scale[j]) >> T.int32(score_shift - 8)
                    score[0, j] = T.if_then_else((nb * block_n + j > i) or (nb * block_n + j >= seqlen), T.int32(-32768), score[0, j])
                T.reduce_max(score, block_max, dim=1, clear=True)
                new_max[0] = T.max(block_max[0], score_max[0])
                old_scale[0] = T.fix.lut_10bit(score_max[0] - new_max[0], LUT, scale=lut_scale, out_dtype="int32")
                for j in T.Parallel(block_n):
                    score_exp[0, j] = T.fix.lut_10bit(score[0, j] - new_max[0], LUT, scale=lut_scale, out_dtype="int32")
                T.reduce_sum(score_exp, block_sum, dim=1, clear=True)
                denom[0] = ((denom[0] * old_scale[0]) >> T.int32(10)) + block_sum[0]
                for d, j in T.Parallel(dim, block_n):
                    weighted_value[d, j] = score_exp[0, j] * (
                        (
                            T.cast(T.if_then_else(nb * block_n + j < seqlen, V[b, nb * block_n + j, d], T.int8(0)), "int32")
                            * T.cast(T.if_then_else(nb * block_n + j < seqlen, VS[b, nb * block_n + j], T.uint32(0)), "int32")
                        )
                        >> T.int32(ATTN_VALUE_SHIFT)
                    )
                T.reduce_sum(weighted_value, value_part, dim=1, clear=True)
                for d in T.Parallel(dim):
                    acc_o[d] = (((acc_o[d] >> T.int32(7)) * old_scale[0]) >> T.int32(3)) + value_part[d]
                score_max[0] = new_max[0]
            for d in T.Parallel(dim):
                O[b, i, d] = (acc_o[d] // denom[0]) << T.int32(ATTN_VALUE_SHIFT)

    return main


def flash_attention_i8_q15_16_gqa(q_heads, kv_heads, seqlen, dim, block_n=64, score_shift=26, lut_scale=0.125):
    group = q_heads // kv_heads

    @T.prim_func
    def main(
        Q: T.Tensor((q_heads, seqlen, dim), "int8"),
        K: T.Tensor((kv_heads, seqlen, dim), "int8"),
        V: T.Tensor((kv_heads, seqlen, dim), "int8"),
        QS: T.Tensor((q_heads, seqlen), "uint32"),
        KS: T.Tensor((kv_heads, seqlen), "uint32"),
        VS: T.Tensor((kv_heads, seqlen), "uint32"),
        LUT: T.Tensor((1024,), "int16"),
        O: T.Tensor((q_heads, seqlen, dim), "int32"),
    ):
        with T.Kernel(q_heads * seqlen, threads=128) as blk:
            h = blk // seqlen
            kh = h // group
            i = blk - h * seqlen
            qk = T.alloc_fragment((block_n, dim), "int32")
            qk_sum = T.alloc_fragment((block_n,), "int32")
            score = T.alloc_fragment((1, block_n), "int32")
            score_exp = T.alloc_fragment((1, block_n), "int32")
            weighted_value = T.alloc_fragment((dim, block_n), "int32")
            value_part = T.alloc_fragment((dim,), "int32")
            acc_o = T.alloc_fragment((dim,), "int32")
            block_max = T.alloc_fragment((1,), "int32")
            new_max = T.alloc_fragment((1,), "int32")
            old_scale = T.alloc_fragment((1,), "int32")
            block_sum = T.alloc_fragment((1,), "int32")
            denom = T.alloc_fragment((1,), "int32")
            score_max = T.alloc_fragment((1,), "int32")
            scale = T.alloc_fragment((block_n,), "int32")
            score_max[0] = T.int32(I32_MIN)
            denom[0] = T.int32(0)
            for d in T.Parallel(dim):
                acc_o[d] = T.int32(0)
            for nb in T.Pipelined(i // block_n + 1):
                for j, d in T.Parallel(block_n, dim):
                    qk[j, d] = T.cast(Q[h, i, d], "int32") * T.cast(T.if_then_else(nb * block_n + j < seqlen, K[kh, nb * block_n + j, d], T.int8(0)), "int32")
                T.reduce_sum(qk, qk_sum, dim=1, clear=True)
                for j in T.Parallel(block_n):
                    scale[j] = (
                        ((T.cast(QS[h, i], "int32") >> T.int32(4)) * (T.cast(T.if_then_else(nb * block_n + j < seqlen, KS[kh, nb * block_n + j], T.uint32(1)), "int32") >> T.int32(4)))
                        >> T.int32(8)
                    ) * T.int32(5793)
                    score[0, j] = ((qk_sum[j] >> T.int32(8)) * scale[j]) >> T.int32(score_shift - 8)
                    score[0, j] = T.if_then_else((nb * block_n + j > i) or (nb * block_n + j >= seqlen), T.int32(-32768), score[0, j])
                T.reduce_max(score, block_max, dim=1, clear=True)
                new_max[0] = T.max(block_max[0], score_max[0])
                old_scale[0] = T.fix.lut_10bit(score_max[0] - new_max[0], LUT, scale=lut_scale, out_dtype="int32")
                for j in T.Parallel(block_n):
                    score_exp[0, j] = T.fix.lut_10bit(score[0, j] - new_max[0], LUT, scale=lut_scale, out_dtype="int32")
                T.reduce_sum(score_exp, block_sum, dim=1, clear=True)
                denom[0] = ((denom[0] * old_scale[0]) >> T.int32(10)) + block_sum[0]
                for d, j in T.Parallel(dim, block_n):
                    weighted_value[d, j] = score_exp[0, j] * (
                        (
                            T.cast(T.if_then_else(nb * block_n + j < seqlen, V[kh, nb * block_n + j, d], T.int8(0)), "int32")
                            * T.cast(T.if_then_else(nb * block_n + j < seqlen, VS[kh, nb * block_n + j], T.uint32(0)), "int32")
                        )
                        >> T.int32(ATTN_VALUE_SHIFT)
                    )
                T.reduce_sum(weighted_value, value_part, dim=1, clear=True)
                for d in T.Parallel(dim):
                    acc_o[d] = (((acc_o[d] >> T.int32(7)) * old_scale[0]) >> T.int32(3)) + value_part[d]
                score_max[0] = new_max[0]
            for d in T.Parallel(dim):
                O[h, i, d] = (acc_o[d] // denom[0]) << T.int32(ATTN_VALUE_SHIFT)

    return main



def flash_attention_i8_q15_16_gqa_cache(q_heads, kv_heads, seqlen, cache_len, dim, block_n=64, score_shift=26, lut_scale=0.125):
    group = q_heads // kv_heads
    kv_len = cache_len + seqlen

    @T.prim_func
    def main(
        Q: T.Tensor((q_heads, seqlen, dim), "int8"),
        CACHE_K: T.Tensor((kv_heads, cache_len, dim), "int8"),
        CACHE_V: T.Tensor((kv_heads, cache_len, dim), "int8"),
        K: T.Tensor((kv_heads, seqlen, dim), "int8"),
        V: T.Tensor((kv_heads, seqlen, dim), "int8"),
        QS: T.Tensor((q_heads, seqlen), "uint32"),
        CACHE_KS: T.Tensor((kv_heads, cache_len), "uint32"),
        CACHE_VS: T.Tensor((kv_heads, cache_len), "uint32"),
        KS: T.Tensor((kv_heads, seqlen), "uint32"),
        VS: T.Tensor((kv_heads, seqlen), "uint32"),
        LUT: T.Tensor((1024,), "int16"),
        O: T.Tensor((q_heads, seqlen, dim), "int32"),
    ):
        with T.Kernel(q_heads * seqlen, threads=128) as blk:
            h = blk // seqlen
            kh = h // group
            i = blk - h * seqlen
            qk = T.alloc_fragment((block_n, dim), "int32")
            qk_sum = T.alloc_fragment((block_n,), "int32")
            score = T.alloc_fragment((1, block_n), "int32")
            score_exp = T.alloc_fragment((1, block_n), "int32")
            weighted_value = T.alloc_fragment((dim, block_n), "int32")
            value_part = T.alloc_fragment((dim,), "int32")
            acc_o = T.alloc_fragment((dim,), "int32")
            block_max = T.alloc_fragment((1,), "int32")
            new_max = T.alloc_fragment((1,), "int32")
            old_scale = T.alloc_fragment((1,), "int32")
            block_sum = T.alloc_fragment((1,), "int32")
            denom = T.alloc_fragment((1,), "int32")
            score_max = T.alloc_fragment((1,), "int32")
            q_scale = T.alloc_fragment((1,), "int32")
            k_scale = T.alloc_fragment((block_n,), "int32")
            v_scale = T.alloc_fragment((block_n,), "int32")
            score_max[0] = T.int32(I32_MIN)
            denom[0] = T.int32(0)
            q_scale[0] = T.cast(QS[h, i], "int32") >> T.int32(4)
            for d in T.Parallel(dim):
                acc_o[d] = T.int32(0)
            for nb in T.Pipelined((cache_len + i) // block_n + 1):
                for j, d in T.Parallel(block_n, dim):
                    pos = nb * block_n + j
                    qk[j, d] = T.cast(Q[h, i, d], "int32") * T.cast(
                        T.if_then_else(
                            pos < cache_len,
                            CACHE_K[kh, pos, d],
                            T.if_then_else(pos < kv_len, K[kh, pos - cache_len, d], T.int8(0)),
                        ),
                        "int32",
                    )
                T.reduce_sum(qk, qk_sum, dim=1, clear=True)
                for j in T.Parallel(block_n):
                    pos = nb * block_n + j
                    k_scale[j] = T.if_then_else(
                        pos < cache_len,
                        T.cast(CACHE_KS[kh, pos], "int32"),
                        T.if_then_else(pos < kv_len, T.cast(KS[kh, pos - cache_len], "int32"), T.int32(1)),
                    )
                    score[0, j] = (
                        ((qk_sum[j] >> T.int32(8)) * (((q_scale[0] * (k_scale[j] >> T.int32(4))) >> T.int32(8)) * T.int32(5793)))
                        >> T.int32(score_shift - 8)
                    )
                    score[0, j] = T.if_then_else(pos > cache_len + i, T.int32(-32768), score[0, j])
                T.reduce_max(score, block_max, dim=1, clear=True)
                new_max[0] = T.max(block_max[0], score_max[0])
                old_scale[0] = T.fix.lut_10bit(score_max[0] - new_max[0], LUT, scale=lut_scale, out_dtype="int32")
                for j in T.Parallel(block_n):
                    score_exp[0, j] = T.fix.lut_10bit(score[0, j] - new_max[0], LUT, scale=lut_scale, out_dtype="int32")
                T.reduce_sum(score_exp, block_sum, dim=1, clear=True)
                denom[0] = ((denom[0] * old_scale[0]) >> T.int32(10)) + block_sum[0]
                for j in T.Parallel(block_n):
                    pos = nb * block_n + j
                    v_scale[j] = T.if_then_else(
                        pos < cache_len,
                        T.cast(CACHE_VS[kh, pos], "int32"),
                        T.if_then_else(pos < kv_len, T.cast(VS[kh, pos - cache_len], "int32"), T.int32(0)),
                    )
                for d, j in T.Parallel(dim, block_n):
                    pos = nb * block_n + j
                    weighted_value[d, j] = score_exp[0, j] * (
                        (
                            T.cast(
                                T.if_then_else(
                                    pos < cache_len,
                                    CACHE_V[kh, pos, d],
                                    T.if_then_else(pos < kv_len, V[kh, pos - cache_len, d], T.int8(0)),
                                ),
                                "int32",
                            )
                            * v_scale[j]
                        )
                        >> T.int32(ATTN_VALUE_SHIFT)
                    )
                T.reduce_sum(weighted_value, value_part, dim=1, clear=True)
                for d in T.Parallel(dim):
                    acc_o[d] = (((acc_o[d] >> T.int32(7)) * old_scale[0]) >> T.int32(3)) + value_part[d]
                score_max[0] = new_max[0]
            for d in T.Parallel(dim):
                O[h, i, d] = (acc_o[d] // denom[0]) << T.int32(ATTN_VALUE_SHIFT)

    return main


def flash_attention_i8_q15_16_cache(batch, seqlen, cache_len, dim, block_n=64, score_shift=26, lut_scale=0.125):
    kv_len = cache_len + seqlen

    @T.prim_func
    def main(
        Q: T.Tensor((batch, seqlen, dim), "int8"),
        CACHE_K: T.Tensor((batch, cache_len, dim), "int8"),
        CACHE_V: T.Tensor((batch, cache_len, dim), "int8"),
        K: T.Tensor((batch, seqlen, dim), "int8"),
        V: T.Tensor((batch, seqlen, dim), "int8"),
        QS: T.Tensor((batch, seqlen), "uint32"),
        CACHE_KS: T.Tensor((batch, cache_len), "uint32"),
        CACHE_VS: T.Tensor((batch, cache_len), "uint32"),
        KS: T.Tensor((batch, seqlen), "uint32"),
        VS: T.Tensor((batch, seqlen), "uint32"),
        LUT: T.Tensor((1024,), "int16"),
        O: T.Tensor((batch, seqlen, dim), "int32"),
    ):
        with T.Kernel(batch * seqlen, threads=128) as blk:
            b = blk // seqlen
            i = blk - b * seqlen
            qk = T.alloc_fragment((block_n, dim), "int32")
            qk_sum = T.alloc_fragment((block_n,), "int32")
            score = T.alloc_fragment((1, block_n), "int32")
            score_exp = T.alloc_fragment((1, block_n), "int32")
            weighted_value = T.alloc_fragment((dim, block_n), "int32")
            value_part = T.alloc_fragment((dim,), "int32")
            acc_o = T.alloc_fragment((dim,), "int32")
            block_max = T.alloc_fragment((1,), "int32")
            new_max = T.alloc_fragment((1,), "int32")
            old_scale = T.alloc_fragment((1,), "int32")
            block_sum = T.alloc_fragment((1,), "int32")
            denom = T.alloc_fragment((1,), "int32")
            score_max = T.alloc_fragment((1,), "int32")
            q_scale = T.alloc_fragment((1,), "int32")
            k_scale = T.alloc_fragment((block_n,), "int32")
            v_scale = T.alloc_fragment((block_n,), "int32")
            score_max[0] = T.int32(I32_MIN)
            denom[0] = T.int32(0)
            q_scale[0] = T.cast(QS[b, i], "int32") >> T.int32(4)
            for d in T.Parallel(dim):
                acc_o[d] = T.int32(0)
            for nb in T.Pipelined((cache_len + i) // block_n + 1):
                for j, d in T.Parallel(block_n, dim):
                    pos = nb * block_n + j
                    qk[j, d] = T.cast(Q[b, i, d], "int32") * T.cast(
                        T.if_then_else(
                            pos < cache_len,
                            CACHE_K[b, pos, d],
                            T.if_then_else(pos < kv_len, K[b, pos - cache_len, d], T.int8(0)),
                        ),
                        "int32",
                    )
                T.reduce_sum(qk, qk_sum, dim=1, clear=True)
                for j in T.Parallel(block_n):
                    pos = nb * block_n + j
                    k_scale[j] = T.if_then_else(
                        pos < cache_len,
                        T.cast(CACHE_KS[b, pos], "int32"),
                        T.if_then_else(pos < kv_len, T.cast(KS[b, pos - cache_len], "int32"), T.int32(1)),
                    )
                    score[0, j] = (
                        ((qk_sum[j] >> T.int32(8)) * (((q_scale[0] * (k_scale[j] >> T.int32(4))) >> T.int32(8)) * T.int32(5793)))
                        >> T.int32(score_shift - 8)
                    )
                    score[0, j] = T.if_then_else(pos > cache_len + i, T.int32(-32768), score[0, j])
                T.reduce_max(score, block_max, dim=1, clear=True)
                new_max[0] = T.max(block_max[0], score_max[0])
                old_scale[0] = T.fix.lut_10bit(score_max[0] - new_max[0], LUT, scale=lut_scale, out_dtype="int32")
                for j in T.Parallel(block_n):
                    score_exp[0, j] = T.fix.lut_10bit(score[0, j] - new_max[0], LUT, scale=lut_scale, out_dtype="int32")
                T.reduce_sum(score_exp, block_sum, dim=1, clear=True)
                denom[0] = ((denom[0] * old_scale[0]) >> T.int32(10)) + block_sum[0]
                for j in T.Parallel(block_n):
                    pos = nb * block_n + j
                    v_scale[j] = T.if_then_else(
                        pos < cache_len,
                        T.cast(CACHE_VS[b, pos], "int32"),
                        T.if_then_else(pos < kv_len, T.cast(VS[b, pos - cache_len], "int32"), T.int32(0)),
                    )
                for d, j in T.Parallel(dim, block_n):
                    pos = nb * block_n + j
                    weighted_value[d, j] = score_exp[0, j] * (
                        (
                            T.cast(
                                T.if_then_else(
                                    pos < cache_len,
                                    CACHE_V[b, pos, d],
                                    T.if_then_else(pos < kv_len, V[b, pos - cache_len, d], T.int8(0)),
                                ),
                                "int32",
                            )
                            * v_scale[j]
                        )
                        >> T.int32(ATTN_VALUE_SHIFT)
                    )
                T.reduce_sum(weighted_value, value_part, dim=1, clear=True)
                for d in T.Parallel(dim):
                    acc_o[d] = (((acc_o[d] >> T.int32(7)) * old_scale[0]) >> T.int32(3)) + value_part[d]
                score_max[0] = new_max[0]
            for d in T.Parallel(dim):
                O[b, i, d] = (acc_o[d] // denom[0]) << T.int32(ATTN_VALUE_SHIFT)

    return main


def compile_kernel(func, out_idx):
    return tilelang.compile(func, out_idx=out_idx, target="cuda")
