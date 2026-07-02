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
            scale[0] = T.max((amax[0] + T.int32(qmax - 1)) // T.int32(qmax), T.int32(1))
            S[r] = T.cast(scale[0], "uint32")
            for c in T.Parallel(cols):
                abs_x[0, c] = X[r, c]
                if abs_x[0, c] < T.int32(0):
                    abs_x[0, c] = T.int32(0) - abs_x[0, c]
                abs_x[0, c] = (abs_x[0, c] + (scale[0] >> T.int32(1))) // scale[0]
                if X[r, c] < T.int32(0):
                    abs_x[0, c] = T.int32(0) - abs_x[0, c]
                abs_x[0, c] = T.min(T.max(abs_x[0, c], T.int32(0 - qmax - 1)), T.int32(qmax))
                Y[r, c] = T.cast(abs_x[0, c], out_dtype)

    return main


def static_quant_q15_16_per_head_attn(tokens, heads, head_dim, out_dtype="int8", qmax_override=None):
    qmax = 127 if out_dtype == "int8" else 32767
    if qmax_override is not None:
        qmax = qmax_override

    @T.prim_func
    def main(
        X: T.Tensor((tokens, heads, head_dim), "int32"),
        S: T.Tensor((heads,), "uint32"),
        Y: T.Tensor((heads, tokens, head_dim), out_dtype),
        SY: T.Tensor((heads, tokens), "uint32"),
    ):
        with T.Kernel(tokens, heads, threads=128) as (t, h):
            scale = T.alloc_fragment((1,), "int32")
            vals = T.alloc_fragment((head_dim,), "int32")
            scale[0] = T.max(T.cast(S[h], "int32"), T.int32(1))
            SY[h, t] = T.cast(scale[0], "uint32")
            for d in T.Parallel(head_dim):
                vals[d] = X[t, h, d]
                if vals[d] < T.int32(0):
                    vals[d] = T.int32(0) - vals[d]
                vals[d] = (vals[d] + (scale[0] >> T.int32(1))) // scale[0]
                if X[t, h, d] < T.int32(0):
                    vals[d] = T.int32(0) - vals[d]
                vals[d] = T.min(T.max(vals[d], T.int32(0 - qmax - 1)), T.int32(qmax))
                Y[h, t, d] = T.cast(vals[d], out_dtype)

    return main


def static_quant_q15_16_per_head_attn_noscale(tokens, heads, head_dim, out_dtype="int8", qmax_override=None):
    qmax = 127 if out_dtype == "int8" else 32767
    if qmax_override is not None:
        qmax = qmax_override

    @T.prim_func
    def main(
        X: T.Tensor((tokens, heads, head_dim), "int32"),
        S: T.Tensor((heads,), "uint32"),
        Y: T.Tensor((heads, tokens, head_dim), out_dtype),
    ):
        with T.Kernel(tokens, heads, threads=128) as (t, h):
            scale = T.alloc_fragment((1,), "int32")
            vals = T.alloc_fragment((head_dim,), "int32")
            scale[0] = T.max(T.cast(S[h], "int32"), T.int32(1))
            for d in T.Parallel(head_dim):
                vals[d] = X[t, h, d]
                if vals[d] < T.int32(0):
                    vals[d] = T.int32(0) - vals[d]
                vals[d] = (vals[d] + (scale[0] >> T.int32(1))) // scale[0]
                if X[t, h, d] < T.int32(0):
                    vals[d] = T.int32(0) - vals[d]
                vals[d] = T.min(T.max(vals[d], T.int32(0 - qmax - 1)), T.int32(qmax))
                Y[h, t, d] = T.cast(vals[d], out_dtype)

    return main


def add_rmsnorm_q15_16_weighted(rows, cols, qmax=32767):
    mean_shift = int(math.log2(cols))

    @T.prim_func
    def main(
        A: T.Tensor((rows, cols), "int32"),
        B: T.Tensor((rows, cols), "int32"),
        W: T.Tensor((cols,), "int32"),
        RLUT: T.Tensor((1024,), "int16"),
        Y: T.Tensor((rows, cols), "int32"),
        N: T.Tensor((rows, cols), "int32"),
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
                Y[r, c] = A[r, c] + B[r, c]
                q[0, c] = Y[r, c]
                if q[0, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
            T.reduce_max(q, amax, dim=1, clear=True)
            scale[0] = T.max((amax[0] + T.int32(qmax - 1)) // T.int32(qmax), T.int32(1))
            for c in T.Parallel(cols):
                q[0, c] = Y[r, c]
                if q[0, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
                q[0, c] = (q[0, c] + (scale[0] >> T.int32(1))) // scale[0]
                if Y[r, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
                q[0, c] = T.min(T.max(q[0, c], T.int32(0 - qmax - 1)), T.int32(qmax))
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
            inv[0] = T.fix.lut_10bit(ss[0], RLUT, scale=((ns[0] - T.int32(7)) << T.int32(Q_MULTIPLIER_WIDTH)) | T.int32(1), out_dtype="int32")
            fold[0] = T.fix.quant(inv[0], scale=1024.0, out_dtype="int32")
            qt[0] = ((T.int32(6) + (ns[0] >> T.int32(1))) << T.int32(Q_MULTIPLIER_WIDTH)) | ((fold[0] >> T.int32(4)) & T.int32(MASK))
            for c in T.Parallel(cols):
                norm[c] = T.fix.quant(q[0, c], scale=qt[0], out_dtype="int32")
                N[r, c] = (norm[c] * (W[c] >> T.int32(8))) >> T.int32(2)

    return main


def add_rmsnorm_dynamic_quant_q15_16_weighted_fast(rows, cols, qmax=127):
    mean_shift = int(math.log2(cols))

    @T.prim_func
    def main(
        A: T.Tensor((rows, cols), "int32"),
        B: T.Tensor((rows, cols), "int32"),
        W: T.Tensor((cols,), "int32"),
        RLUT: T.Tensor((1024,), "int16"),
        Y: T.Tensor((rows, cols), "int32"),
        Q: T.Tensor((rows, cols), "int8"),
        S: T.Tensor((rows,), "uint32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            x = T.alloc_fragment((1, cols), "int32")
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
            post_abs = T.alloc_fragment((1, cols), "int32")
            post_scale = T.alloc_fragment((1,), "int32")
            for c in T.Parallel(cols):
                x[0, c] = A[r, c] + B[r, c]
                Y[r, c] = x[0, c]
                q[0, c] = x[0, c]
                if q[0, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
            T.reduce_max(q, amax, dim=1, clear=True)
            scale[0] = T.max((amax[0] + T.int32(32766)) // T.int32(32767), T.int32(1))
            for c in T.Parallel(cols):
                q[0, c] = x[0, c]
                if q[0, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
                q[0, c] = (q[0, c] + (scale[0] >> T.int32(1))) // scale[0]
                if x[0, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
                q[0, c] = T.min(T.max(q[0, c], T.int32(-32768)), T.int32(32767))
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
            inv[0] = T.fix.lut_10bit(ss[0], RLUT, scale=((ns[0] - T.int32(7)) << T.int32(Q_MULTIPLIER_WIDTH)) | T.int32(1), out_dtype="int32")
            fold[0] = T.fix.quant(inv[0], scale=1024.0, out_dtype="int32")
            qt[0] = ((T.int32(6) + (ns[0] >> T.int32(1))) << T.int32(Q_MULTIPLIER_WIDTH)) | ((fold[0] >> T.int32(4)) & T.int32(MASK))
            for c in T.Parallel(cols):
                norm[c] = T.fix.quant(q[0, c], scale=qt[0], out_dtype="int32")
                x[0, c] = (norm[c] * (W[c] >> T.int32(8))) >> T.int32(2)
                post_abs[0, c] = x[0, c]
                if post_abs[0, c] < T.int32(0):
                    post_abs[0, c] = T.int32(0) - post_abs[0, c]
            T.reduce_max(post_abs, amax, dim=1, clear=True)
            post_scale[0] = T.max((amax[0] + T.int32(qmax - 1)) // T.int32(qmax), T.int32(1))
            S[r] = T.cast(post_scale[0], "uint32")
            for c in T.Parallel(cols):
                post_abs[0, c] = x[0, c]
                if post_abs[0, c] < T.int32(0):
                    post_abs[0, c] = T.int32(0) - post_abs[0, c]
                post_abs[0, c] = (post_abs[0, c] + (post_scale[0] >> T.int32(1))) // post_scale[0]
                if x[0, c] < T.int32(0):
                    post_abs[0, c] = T.int32(0) - post_abs[0, c]
                post_abs[0, c] = T.min(T.max(post_abs[0, c], T.int32(0 - qmax - 1)), T.int32(qmax))
                Q[r, c] = T.cast(post_abs[0, c], "int8")

    return main


def rope_q15_16_heads(seq_len, heads, dim):
    @T.prim_func
    def main(
        X: T.Tensor((seq_len * heads, dim), "int32"),
        COS: T.Tensor((seq_len, dim // 2), "int32"),
        SIN: T.Tensor((seq_len, dim // 2), "int32"),
        Y: T.Tensor((seq_len * heads, dim), "int32"),
    ):
        with T.Kernel(seq_len * heads, threads=128) as r:
            t = r // heads
            a = T.alloc_fragment((dim // 2,), "int32")
            b = T.alloc_fragment((dim // 2,), "int32")
            for d in T.Parallel(dim // 2):
                a[d] = (X[r, d] >> T.int32(8)) * (COS[t, d] >> T.int32(8))
                b[d] = (X[r, d + dim // 2] >> T.int32(8)) * (SIN[t, d] >> T.int32(8))
                Y[r, d] = a[d] - b[d]
                a[d] = (X[r, d] >> T.int32(8)) * (SIN[t, d] >> T.int32(8))
                b[d] = (X[r, d + dim // 2] >> T.int32(8)) * (COS[t, d] >> T.int32(8))
                Y[r, d + dim // 2] = a[d] + b[d]

    return main


def rope_rotate_q15_16_heads(seq_len, heads, dim):
    @T.prim_func
    def main(
        X: T.Tensor((seq_len * heads, dim), "int32"),
        COS: T.Tensor((seq_len, dim // 2), "int32"),
        SIN: T.Tensor((seq_len, dim // 2), "int32"),
        R: T.Tensor((dim, dim), "int32"),
        Y: T.Tensor((seq_len * heads, dim), "int32"),
    ):
        with T.Kernel(seq_len * heads, threads=128) as r:
            t = r // heads
            rope_lo = T.alloc_fragment((dim // 2,), "int32")
            rope_hi = T.alloc_fragment((dim // 2,), "int32")
            acc = T.alloc_fragment((dim,), "int32")
            for d in T.Parallel(dim // 2):
                rope_lo[d] = ((X[r, d] >> T.int32(8)) * (COS[t, d] >> T.int32(8))) - (
                    (X[r, d + dim // 2] >> T.int32(8)) * (SIN[t, d] >> T.int32(8))
                )
                rope_hi[d] = ((X[r, d] >> T.int32(8)) * (SIN[t, d] >> T.int32(8))) + (
                    (X[r, d + dim // 2] >> T.int32(8)) * (COS[t, d] >> T.int32(8))
                )
            for o in T.Parallel(dim):
                acc[o] = T.int32(0)
                for k in T.serial(dim // 2):
                    acc[o] += (rope_lo[k] >> T.int32(8)) * (R[k, o] >> T.int32(8))
                    acc[o] += (rope_hi[k] >> T.int32(8)) * (R[k + dim // 2, o] >> T.int32(8))
                Y[r, o] = acc[o]

    return main


def rope_rotate_static_quant_q15_16_attn(seq_len, heads, dim, qmax=127):
    @T.prim_func
    def main(
        X: T.Tensor((seq_len * heads, dim), "int32"),
        COS: T.Tensor((seq_len, dim // 2), "int32"),
        SIN: T.Tensor((seq_len, dim // 2), "int32"),
        R: T.Tensor((dim, dim), "int32"),
        S: T.Tensor((heads,), "uint32"),
        Y: T.Tensor((heads, seq_len, dim), "int8"),
        SY: T.Tensor((heads, seq_len), "uint32"),
    ):
        with T.Kernel(seq_len * heads, threads=128) as r:
            t = r // heads
            h = r - t * heads
            rope_lo = T.alloc_fragment((dim // 2,), "int32")
            rope_hi = T.alloc_fragment((dim // 2,), "int32")
            acc = T.alloc_fragment((dim,), "int32")
            scale = T.alloc_fragment((1,), "int32")
            vals = T.alloc_fragment((dim,), "int32")
            scale[0] = T.max(T.cast(S[h], "int32"), T.int32(1))
            SY[h, t] = T.cast(scale[0], "uint32")
            for d in T.Parallel(dim // 2):
                rope_lo[d] = ((X[r, d] >> T.int32(8)) * (COS[t, d] >> T.int32(8))) - (
                    (X[r, d + dim // 2] >> T.int32(8)) * (SIN[t, d] >> T.int32(8))
                )
                rope_hi[d] = ((X[r, d] >> T.int32(8)) * (SIN[t, d] >> T.int32(8))) + (
                    (X[r, d + dim // 2] >> T.int32(8)) * (COS[t, d] >> T.int32(8))
                )
            for o in T.Parallel(dim):
                acc[o] = T.int32(0)
                for k in T.serial(dim // 2):
                    acc[o] += (rope_lo[k] >> T.int32(8)) * (R[k, o] >> T.int32(8))
                    acc[o] += (rope_hi[k] >> T.int32(8)) * (R[k + dim // 2, o] >> T.int32(8))
                vals[o] = acc[o]
                if vals[o] < T.int32(0):
                    vals[o] = T.int32(0) - vals[o]
                vals[o] = (vals[o] + (scale[0] >> T.int32(1))) // scale[0]
                if acc[o] < T.int32(0):
                    vals[o] = T.int32(0) - vals[o]
                vals[o] = T.min(T.max(vals[o], T.int32(0 - qmax - 1)), T.int32(qmax))
                Y[h, t, o] = T.cast(vals[o], "int8")

    return main


def rope_rotate_static_quant_q15_16_attn_noscale(seq_len, heads, dim, qmax=127):
    @T.prim_func
    def main(
        X: T.Tensor((seq_len * heads, dim), "int32"),
        COS: T.Tensor((seq_len, dim // 2), "int32"),
        SIN: T.Tensor((seq_len, dim // 2), "int32"),
        R: T.Tensor((dim, dim), "int32"),
        S: T.Tensor((heads,), "uint32"),
        Y: T.Tensor((heads, seq_len, dim), "int8"),
    ):
        with T.Kernel(seq_len * heads, threads=128) as r:
            t = r // heads
            h = r - t * heads
            rope_lo = T.alloc_fragment((dim // 2,), "int32")
            rope_hi = T.alloc_fragment((dim // 2,), "int32")
            acc = T.alloc_fragment((dim,), "int32")
            scale = T.alloc_fragment((1,), "int32")
            vals = T.alloc_fragment((dim,), "int32")
            scale[0] = T.max(T.cast(S[h], "int32"), T.int32(1))
            for d in T.Parallel(dim // 2):
                rope_lo[d] = ((X[r, d] >> T.int32(8)) * (COS[t, d] >> T.int32(8))) - (
                    (X[r, d + dim // 2] >> T.int32(8)) * (SIN[t, d] >> T.int32(8))
                )
                rope_hi[d] = ((X[r, d] >> T.int32(8)) * (SIN[t, d] >> T.int32(8))) + (
                    (X[r, d + dim // 2] >> T.int32(8)) * (COS[t, d] >> T.int32(8))
                )
            for o in T.Parallel(dim):
                acc[o] = T.int32(0)
                for k in T.serial(dim // 2):
                    acc[o] += (rope_lo[k] >> T.int32(8)) * (R[k, o] >> T.int32(8))
                    acc[o] += (rope_hi[k] >> T.int32(8)) * (R[k + dim // 2, o] >> T.int32(8))
                vals[o] = acc[o]
                if vals[o] < T.int32(0):
                    vals[o] = T.int32(0) - vals[o]
                vals[o] = (vals[o] + (scale[0] >> T.int32(1))) // scale[0]
                if acc[o] < T.int32(0):
                    vals[o] = T.int32(0) - vals[o]
                vals[o] = T.min(T.max(vals[o], T.int32(0 - qmax - 1)), T.int32(qmax))
                Y[h, t, o] = T.cast(vals[o], "int8")

    return main


@T.macro
def _warp_hadamard_i32(local, buf, thread_elem, warp_size, rounds):
    tx = T.get_thread_binding(0)
    for i in T.serial(rounds):
        stride = 1 << i
        other = tx ^ stride
        sign = (tx >> i) & 1
        for j in T.serial(thread_elem):
            buf[j] = T.tvm_warp_shuffle(0xFFFFFFFF, local[j], other % warp_size, warp_size, warp_size)
            local[j] = T.if_then_else(sign == 0, local[j] + buf[j], buf[j] - local[j])


def rope_rotate_static_quant_q15_16_attn_hadamard_approx(seq_len, heads, dim, qmax=127):
    thread_elem = 8
    threads = 16
    thread_round = 3
    warp_round = 4

    @T.prim_func
    def main(
        X: T.Tensor((seq_len * heads, dim), "int32"),
        COS: T.Tensor((seq_len, dim // 2), "int32"),
        SIN: T.Tensor((seq_len, dim // 2), "int32"),
        R: T.Tensor((dim, dim), "int32"),
        S: T.Tensor((heads,), "uint32"),
        Y: T.Tensor((heads, seq_len, dim), "int8"),
    ):
        with T.Kernel(seq_len * heads, threads=threads) as r:
            tx = T.get_thread_binding(0)
            t = r // heads
            h = r - t * heads
            local = T.alloc_local((thread_elem,), "int32")
            other_val = T.alloc_local((thread_elem,), "int32")
            scale = T.alloc_local((1,), "int32")
            scale[0] = T.max(T.cast(S[h], "int32"), T.int32(1))
            for i in T.serial(thread_elem):
                d = tx * thread_elem + i
                src_d = T.if_then_else(d < T.int32(dim // 2), d, d - T.int32(dim // 2))
                x0 = X[r, src_d]
                x1 = X[r, src_d + dim // 2]
                c = COS[t, src_d]
                s = SIN[t, src_d]
                lo = ((x0 >> T.int32(8)) * (c >> T.int32(8))) - ((x1 >> T.int32(8)) * (s >> T.int32(8)))
                hi = ((x0 >> T.int32(8)) * (s >> T.int32(8))) + ((x1 >> T.int32(8)) * (c >> T.int32(8)))
                v0 = T.if_then_else(d < T.int32(dim // 2), lo >> T.int32(8), hi >> T.int32(8))
                local[i] = T.if_then_else(R[d, 0] >= T.int32(0), v0, T.int32(0) - v0)
            for i in T.serial(thread_round):
                chunksize = 1 << (i + 1)
                chunknum = thread_elem // chunksize
                for j in T.serial(chunknum):
                    chunkbase = j * chunksize
                    for k in T.serial(chunksize // 2):
                        a = local[chunkbase + k]
                        b = local[chunkbase + k + chunksize // 2]
                        local[chunkbase + k] = a + b
                        local[chunkbase + k + chunksize // 2] = a - b
            _warp_hadamard_i32(local, other_val, thread_elem, threads, warp_round)
            for i in T.serial(thread_elem):
                v = local[i] * T.int32(22)
                qv = T.alloc_local((1,), "int32")
                qv[0] = v
                if qv[0] < T.int32(0):
                    qv[0] = T.int32(0) - qv[0]
                qv[0] = (qv[0] + (scale[0] >> T.int32(1))) // scale[0]
                if v < T.int32(0):
                    qv[0] = T.int32(0) - qv[0]
                qv[0] = T.min(T.max(qv[0], T.int32(0 - qmax - 1)), T.int32(qmax))
                Y[h, t, tx * thread_elem + i] = T.cast(qv[0], "int8")

    return main


def rmsnorm_q15_16_weighted(rows, cols, qmax=32767):
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
            scale[0] = T.max((amax[0] + T.int32(qmax - 1)) // T.int32(qmax), T.int32(1))
            for c in T.Parallel(cols):
                q[0, c] = X[r, c]
                if q[0, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
                q[0, c] = (q[0, c] + (scale[0] >> T.int32(1))) // scale[0]
                if X[r, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
                q[0, c] = T.min(T.max(q[0, c], T.int32(0 - qmax - 1)), T.int32(qmax))
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
            inv[0] = T.fix.lut_10bit(ss[0], RLUT, scale=((ns[0] - T.int32(7)) << T.int32(Q_MULTIPLIER_WIDTH)) | T.int32(1), out_dtype="int32")
            fold[0] = T.fix.quant(inv[0], scale=1024.0, out_dtype="int32")
            qt[0] = ((T.int32(6) + (ns[0] >> T.int32(1))) << T.int32(Q_MULTIPLIER_WIDTH)) | ((fold[0] >> T.int32(4)) & T.int32(MASK))
            for c in T.Parallel(cols):
                norm[c] = T.fix.quant(q[0, c], scale=qt[0], out_dtype="int32")
                Y[r, c] = (norm[c] * (W[c] >> T.int32(8))) >> T.int32(2)

    return main


def rmsnorm_dynamic_quant_q15_16_weighted_fast(rows, cols, qmax=127):
    mean_shift = int(math.log2(cols))

    @T.prim_func
    def main(
        X: T.Tensor((rows, cols), "int32"),
        W: T.Tensor((cols,), "int32"),
        RLUT: T.Tensor((1024,), "int16"),
        Q: T.Tensor((rows, cols), "int8"),
        S: T.Tensor((rows,), "uint32"),
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
            post_abs = T.alloc_fragment((1, cols), "int32")
            post_scale = T.alloc_fragment((1,), "int32")
            for c in T.Parallel(cols):
                q[0, c] = X[r, c]
                if q[0, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
            T.reduce_max(q, amax, dim=1, clear=True)
            scale[0] = T.max((amax[0] + T.int32(32766)) // T.int32(32767), T.int32(1))
            for c in T.Parallel(cols):
                q[0, c] = X[r, c]
                if q[0, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
                q[0, c] = (q[0, c] + (scale[0] >> T.int32(1))) // scale[0]
                if X[r, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
                q[0, c] = T.min(T.max(q[0, c], T.int32(-32768)), T.int32(32767))
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
            inv[0] = T.fix.lut_10bit(ss[0], RLUT, scale=((ns[0] - T.int32(7)) << T.int32(Q_MULTIPLIER_WIDTH)) | T.int32(1), out_dtype="int32")
            fold[0] = T.fix.quant(inv[0], scale=1024.0, out_dtype="int32")
            qt[0] = ((T.int32(6) + (ns[0] >> T.int32(1))) << T.int32(Q_MULTIPLIER_WIDTH)) | ((fold[0] >> T.int32(4)) & T.int32(MASK))
            for c in T.Parallel(cols):
                norm[c] = T.fix.quant(q[0, c], scale=qt[0], out_dtype="int32")
                post_abs[0, c] = (norm[c] * (W[c] >> T.int32(8))) >> T.int32(2)
                if post_abs[0, c] < T.int32(0):
                    post_abs[0, c] = T.int32(0) - post_abs[0, c]
            T.reduce_max(post_abs, amax, dim=1, clear=True)
            post_scale[0] = T.max((amax[0] + T.int32(qmax - 1)) // T.int32(qmax), T.int32(1))
            S[r] = T.cast(post_scale[0], "uint32")
            for c in T.Parallel(cols):
                norm[c] = (norm[c] * (W[c] >> T.int32(8))) >> T.int32(2)
                post_abs[0, c] = norm[c]
                if post_abs[0, c] < T.int32(0):
                    post_abs[0, c] = T.int32(0) - post_abs[0, c]
                post_abs[0, c] = (post_abs[0, c] + (post_scale[0] >> T.int32(1))) // post_scale[0]
                if norm[c] < T.int32(0):
                    post_abs[0, c] = T.int32(0) - post_abs[0, c]
                post_abs[0, c] = T.min(T.max(post_abs[0, c], T.int32(0 - qmax - 1)), T.int32(qmax))
                Q[r, c] = T.cast(post_abs[0, c], "int8")

    return main


def rmsnorm_q15_16_grouped_weighted_rowwise(rows, groups, group_cols, qmax=32767):
    cols = groups * group_cols
    mean_shift = int(math.log2(group_cols))

    @T.prim_func
    def main(
        X: T.Tensor((rows, cols), "int32"),
        W: T.Tensor((group_cols,), "int32"),
        RLUT: T.Tensor((1024,), "int16"),
        Y: T.Tensor((rows * groups, group_cols), "int32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            abs_x = T.alloc_fragment((1, cols), "int32")
            q = T.alloc_fragment((1, group_cols), "int32")
            xx = T.alloc_fragment((1, group_cols), "int32")
            amax = T.alloc_fragment((1,), "int32")
            scale = T.alloc_fragment((1,), "int32")
            ss = T.alloc_fragment((1,), "int32")
            ns = T.alloc_fragment((1,), "int32")
            wk = T.alloc_fragment((1,), "int32")
            inv = T.alloc_fragment((1,), "int32")
            fold = T.alloc_fragment((1,), "int32")
            qt = T.alloc_fragment((1,), "int32")
            norm = T.alloc_fragment((group_cols,), "int32")
            for c in T.Parallel(cols):
                abs_x[0, c] = X[r, c]
                if abs_x[0, c] < T.int32(0):
                    abs_x[0, c] = T.int32(0) - abs_x[0, c]
            T.reduce_max(abs_x, amax, dim=1, clear=True)
            scale[0] = T.max((amax[0] + T.int32(qmax - 1)) // T.int32(qmax), T.int32(1))
            for g in T.serial(groups):
                for c in T.Parallel(group_cols):
                    q[0, c] = X[r, g * group_cols + c]
                    if q[0, c] < T.int32(0):
                        q[0, c] = T.int32(0) - q[0, c]
                    q[0, c] = (q[0, c] + (scale[0] >> T.int32(1))) // scale[0]
                    if X[r, g * group_cols + c] < T.int32(0):
                        q[0, c] = T.int32(0) - q[0, c]
                    q[0, c] = T.min(T.max(q[0, c], T.int32(0 - qmax - 1)), T.int32(qmax))
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
                inv[0] = T.fix.lut_10bit(ss[0], RLUT, scale=((ns[0] - T.int32(7)) << T.int32(Q_MULTIPLIER_WIDTH)) | T.int32(1), out_dtype="int32")
                fold[0] = T.fix.quant(inv[0], scale=1024.0, out_dtype="int32")
                qt[0] = ((T.int32(6) + (ns[0] >> T.int32(1))) << T.int32(Q_MULTIPLIER_WIDTH)) | ((fold[0] >> T.int32(4)) & T.int32(MASK))
                for c in T.Parallel(group_cols):
                    norm[c] = T.fix.quant(q[0, c], scale=qt[0], out_dtype="int32")
                    Y[r * groups + g, c] = (norm[c] * (W[c] >> T.int32(8))) >> T.int32(2)

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
            scale[0] = T.max((amax[0] + T.int32(126)) // T.int32(127), T.int32(1))
            S[r] = T.cast(scale[0], "uint32")
            for c in T.Parallel(cols):
                Y[r, c] = vals[0, c]
                abs_x[0, c] = vals[0, c]
                if abs_x[0, c] < T.int32(0):
                    abs_x[0, c] = T.int32(0) - abs_x[0, c]
                abs_x[0, c] = (abs_x[0, c] + (scale[0] >> T.int32(1))) // scale[0]
                if vals[0, c] < T.int32(0):
                    abs_x[0, c] = T.int32(0) - abs_x[0, c]
                vals[0, c] = T.min(T.max(abs_x[0, c], T.int32(-128)), T.int32(127))
                Q[r, c] = T.cast(vals[0, c], "int8")

    return main


def silu_mul_dynamic_quant_q15_16_fast(rows, cols):
    @T.prim_func
    def main(
        Gate: T.Tensor((rows, cols), "int32"),
        Up: T.Tensor((rows, cols), "int32"),
        LUT: T.Tensor((1024,), "int32"),
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
            scale[0] = T.max((amax[0] + T.int32(126)) // T.int32(127), T.int32(1))
            S[r] = T.cast(scale[0], "uint32")
            for c in T.Parallel(cols):
                abs_x[0, c] = vals[0, c]
                if abs_x[0, c] < T.int32(0):
                    abs_x[0, c] = T.int32(0) - abs_x[0, c]
                abs_x[0, c] = (abs_x[0, c] + (scale[0] >> T.int32(1))) // scale[0]
                if vals[0, c] < T.int32(0):
                    abs_x[0, c] = T.int32(0) - abs_x[0, c]
                vals[0, c] = T.min(T.max(abs_x[0, c], T.int32(-128)), T.int32(127))
                Q[r, c] = T.cast(vals[0, c], "int8")

    return main


def silu_mul_dynamic_quant_q15_16_i16_fast(rows, cols):
    @T.prim_func
    def main(
        Gate: T.Tensor((rows, cols), "int32"),
        Up: T.Tensor((rows, cols), "int32"),
        LUT: T.Tensor((1024,), "int32"),
        Q: T.Tensor((rows, cols), "int16"),
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
            scale[0] = T.max((amax[0] + T.int32(32766)) // T.int32(32767), T.int32(1))
            S[r] = T.cast(scale[0], "uint32")
            for c in T.Parallel(cols):
                abs_x[0, c] = vals[0, c]
                if abs_x[0, c] < T.int32(0):
                    abs_x[0, c] = T.int32(0) - abs_x[0, c]
                abs_x[0, c] = (abs_x[0, c] + (scale[0] >> T.int32(1))) // scale[0]
                if vals[0, c] < T.int32(0):
                    abs_x[0, c] = T.int32(0) - abs_x[0, c]
                abs_x[0, c] = T.min(T.max(abs_x[0, c], T.int32(-32768)), T.int32(32767))
                Q[r, c] = T.cast(abs_x[0, c], "int16")

    return main


def add_q15_16(rows, cols):
    @T.prim_func
    def main(A: T.Tensor((rows, cols), "int32"), B: T.Tensor((rows, cols), "int32"), Y: T.Tensor((rows, cols), "int32")):
        with T.Kernel(rows, threads=128) as r:
            for c in T.Parallel(cols):
                Y[r, c] = A[r, c] + B[r, c]

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


def linear_dynamic_int8_residual_q15_16(rows, in_features, out_features, block_m=16, block_n=32, block_k=64):
    @T.prim_func
    def main(
        X: T.Tensor((rows, in_features), "int8"),
        XS: T.Tensor((rows,), "uint32"),
        W: T.Tensor((out_features, in_features), "int8"),
        WS: T.Tensor((out_features,), "uint32"),
        RES: T.Tensor((rows, out_features), "int32"),
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
                Y[br * block_m + m, bo * block_n + n] = RES[br * block_m + m, bo * block_n + n] + (
                    (acc[m, n] >> T.int32(8)) * T.cast((XS[br * block_m + m] * WS[bo * block_n + n]) >> T.int32(8), "int32")
                )

    return main


def linear_dynamic_int16_residual_q15_16(rows, in_features, out_features, block_m=16, block_n=32, block_k=64):
    @T.prim_func
    def main(
        X: T.Tensor((rows, in_features), "int16"),
        XS: T.Tensor((rows,), "uint32"),
        W: T.Tensor((out_features, in_features), "int8"),
        WS: T.Tensor((out_features,), "uint32"),
        RES: T.Tensor((rows, out_features), "int32"),
        Y: T.Tensor((rows, out_features), "int32"),
    ):
        with T.Kernel(T.ceildiv(out_features, block_n), T.ceildiv(rows, block_m), threads=128) as (bo, br):
            x_hi = T.alloc_shared((block_m, block_k), "int8")
            x_mid = T.alloc_shared((block_m, block_k), "int8")
            w_shared = T.alloc_shared((block_n, block_k), "int8")
            acc_hi = T.alloc_fragment((block_m, block_n), "int32")
            acc_mid = T.alloc_fragment((block_m, block_n), "int32")
            x_val = T.alloc_fragment((block_m, block_k), "int32")

            T.clear(acc_hi)
            T.clear(acc_mid)
            for ko in T.Pipelined(in_features // block_k, num_stages=2):
                for m, k in T.Parallel(block_m, block_k):
                    x_val[m, k] = T.cast(X[br * block_m + m, ko * block_k + k], "int32")
                    x_hi[m, k] = T.cast(x_val[m, k] >> T.int32(8), "int8")
                    x_mid[m, k] = T.cast((x_val[m, k] - ((x_val[m, k] >> T.int32(8)) << T.int32(8))) >> T.int32(1), "int8")
                T.copy(W[bo * block_n, ko * block_k], w_shared)
                T.gemm(x_hi, w_shared, acc_hi, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(x_mid, w_shared, acc_mid, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for m, n in T.Parallel(block_m, block_n):
                Y[br * block_m + m, bo * block_n + n] = RES[br * block_m + m, bo * block_n + n] + (
                    T.cast(
                        (T.cast(acc_hi[m, n] + (acc_mid[m, n] >> T.int32(7)), "int64") * T.cast(XS[br * block_m + m], "int64") * T.cast(WS[bo * block_n + n], "int64"))
                        >> T.int32(8),
                        "int32",
                    )
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


def linear_dynamic_int8_qkv_q15_16(rows, in_features, q_features, kv_features, block_m=16, block_n=32, block_k=64):
    @T.prim_func
    def main(
        X: T.Tensor((rows, in_features), "int8"),
        XS: T.Tensor((rows,), "uint32"),
        WQ: T.Tensor((q_features, in_features), "int8"),
        WSQ: T.Tensor((q_features,), "uint32"),
        WK: T.Tensor((kv_features, in_features), "int8"),
        WSK: T.Tensor((kv_features,), "uint32"),
        WV: T.Tensor((kv_features, in_features), "int8"),
        WSV: T.Tensor((kv_features,), "uint32"),
        Q: T.Tensor((rows, q_features), "int32"),
        K: T.Tensor((rows, kv_features), "int32"),
        V: T.Tensor((rows, kv_features), "int32"),
    ):
        with T.Kernel(T.ceildiv(q_features, block_n), T.ceildiv(rows, block_m), threads=128) as (bo, br):
            x_shared = T.alloc_shared((block_m, block_k), "int8")
            wq_shared = T.alloc_shared((block_n, block_k), "int8")
            wk_shared = T.alloc_shared((block_n, block_k), "int8")
            wv_shared = T.alloc_shared((block_n, block_k), "int8")
            acc_q = T.alloc_fragment((block_m, block_n), "int32")
            acc_k = T.alloc_fragment((block_m, block_n), "int32")
            acc_v = T.alloc_fragment((block_m, block_n), "int32")
            T.clear(acc_q)
            T.clear(acc_k)
            T.clear(acc_v)
            for ko in T.Pipelined(in_features // block_k, num_stages=2):
                T.copy(X[br * block_m, ko * block_k], x_shared)
                T.copy(WQ[bo * block_n, ko * block_k], wq_shared)
                if bo * block_n < kv_features:
                    T.copy(WK[bo * block_n, ko * block_k], wk_shared)
                    T.copy(WV[bo * block_n, ko * block_k], wv_shared)
                T.gemm(x_shared, wq_shared, acc_q, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                if bo * block_n < kv_features:
                    T.gemm(x_shared, wk_shared, acc_k, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(x_shared, wv_shared, acc_v, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for m, n in T.Parallel(block_m, block_n):
                Q[br * block_m + m, bo * block_n + n] = (
                    (acc_q[m, n] >> T.int32(8)) * T.cast((XS[br * block_m + m] * WSQ[bo * block_n + n]) >> T.int32(8), "int32")
                )
                if bo * block_n + n < kv_features:
                    K[br * block_m + m, bo * block_n + n] = (
                        (acc_k[m, n] >> T.int32(8)) * T.cast((XS[br * block_m + m] * WSK[bo * block_n + n]) >> T.int32(8), "int32")
                    )
                    V[br * block_m + m, bo * block_n + n] = (
                        (acc_v[m, n] >> T.int32(8)) * T.cast((XS[br * block_m + m] * WSV[bo * block_n + n]) >> T.int32(8), "int32")
                    )

    return main


def attention_i8v8_q15_16_gqa_fused_static(q_heads, kv_heads, seqlen, dim, block_m=32, block_n=64, score_shift=26, lut_scale=0.125):
    group = q_heads // kv_heads
    q_size = q_heads * dim

    @T.prim_func
    def main(
        Q: T.Tensor((q_heads, seqlen, dim), "int8"),
        K: T.Tensor((kv_heads, seqlen, dim), "int8"),
        V: T.Tensor((kv_heads, seqlen, dim), "int8"),
        QS: T.Tensor((q_heads,), "uint32"),
        KS: T.Tensor((kv_heads,), "uint32"),
        VS: T.Tensor((kv_heads,), "uint32"),
        LUT: T.Tensor((1024,), "int16"),
        O: T.Tensor((seqlen, q_size), "int32"),
    ):
        with T.Kernel(q_heads, T.ceildiv(seqlen, block_m), threads=128) as (h, bm):
            kh = h // group
            row_base = bm * block_m
            q_shared = T.alloc_shared((block_m, dim), "int8")
            k_shared = T.alloc_shared((block_n, dim), "int8")
            v_shared = T.alloc_shared((block_n, dim), "int8")
            p2 = T.alloc_shared((block_m, block_n), "int8")
            p1 = T.alloc_shared((block_m, block_n), "int8")
            p0 = T.alloc_shared((block_m, block_n), "int8")
            qk = T.alloc_fragment((block_m, block_n), "int32")
            score = T.alloc_fragment((block_m, block_n), "int32")
            block_max = T.alloc_fragment((block_m,), "int32")
            score_max = T.alloc_fragment((block_m,), "int32")
            new_max = T.alloc_fragment((block_m,), "int32")
            old_scale = T.alloc_fragment((block_m,), "int32")
            block_sum = T.alloc_fragment((block_m,), "int32")
            denom = T.alloc_fragment((block_m,), "int32")
            q_scale = T.alloc_fragment((1,), "int32")
            k_scale = T.alloc_fragment((1,), "int32")
            v_scale = T.alloc_fragment((1,), "int32")
            score_scale = T.alloc_fragment((1,), "int32")
            prob = T.alloc_fragment((block_m, block_n), "int32")
            scaled = T.alloc_fragment((block_m, block_n), "int32")
            pv = T.alloc_fragment((block_m, dim), "int32")
            acc = T.alloc_fragment((block_m, dim), "int32")
            q_scale[0] = T.cast(QS[h], "int32") >> T.int32(4)
            k_scale[0] = T.cast(KS[kh], "int32") >> T.int32(4)
            v_scale[0] = T.cast(VS[kh], "int32")
            score_scale[0] = ((q_scale[0] * k_scale[0]) >> T.int32(8)) * T.int32(5793)
            for m in T.Parallel(block_m):
                score_max[m] = T.int32(I32_MIN)
                denom[m] = T.int32(0)
            for m, d in T.Parallel(block_m, dim):
                q_shared[m, d] = Q[h, row_base + m, d]
                acc[m, d] = T.int32(0)
            for nb in T.Pipelined(row_base // block_n + 1):
                for j, d in T.Parallel(block_n, dim):
                    k_shared[j, d] = K[kh, nb * block_n + j, d]
                T.clear(qk)
                T.gemm(q_shared, k_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for m, j in T.Parallel(block_m, block_n):
                    score[m, j] = ((qk[m, j] >> T.int32(8)) * score_scale[0]) >> T.int32(score_shift - 8)
                    score[m, j] = T.if_then_else(row_base + m >= nb * block_n + j, score[m, j], T.int32(-32768))
                T.reduce_max(score, block_max, dim=1, clear=True)
                for m in T.Parallel(block_m):
                    new_max[m] = T.max(score_max[m], block_max[m])
                    old_scale[m] = T.fix.lut_10bit(score_max[m] - new_max[m], LUT, scale=lut_scale, out_dtype="int32")
                for m, j in T.Parallel(block_m, block_n):
                    score[m, j] = T.if_then_else(row_base + m >= nb * block_n + j, T.fix.lut_10bit(score[m, j] - new_max[m], LUT, scale=lut_scale, out_dtype="int32"), T.int32(0))
                T.reduce_sum(score, block_sum, dim=1, clear=True)
                for m in T.Parallel(block_m):
                    denom[m] = ((denom[m] * old_scale[m]) >> T.int32(10)) + block_sum[m]
                    score_max[m] = new_max[m]
            for nb in T.Pipelined(row_base // block_n + 1):
                for j, d in T.Parallel(block_n, dim):
                    k_shared[j, d] = K[kh, nb * block_n + j, d]
                    v_shared[j, d] = V[kh, nb * block_n + j, d]
                T.clear(qk)
                T.gemm(q_shared, k_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for m, j in T.Parallel(block_m, block_n):
                    score[m, j] = ((qk[m, j] >> T.int32(8)) * score_scale[0]) >> T.int32(score_shift - 8)
                    score[m, j] = T.if_then_else(row_base + m >= nb * block_n + j, T.fix.lut_10bit(score[m, j] - score_max[m], LUT, scale=lut_scale, out_dtype="int32"), T.int32(0))
                    prob[m, j] = T.min(T.truncdiv((score[m, j] * T.int32(16383)) + (denom[m] >> T.int32(1)), denom[m]), T.int32(16383))
                    scaled[m, j] = T.truncdiv((prob[m, j] * v_scale[0]) + T.int32(8191), T.int32(16383))
                    p2[m, j] = T.cast(scaled[m, j] >> T.int32(14), "int8")
                    p1[m, j] = T.cast((scaled[m, j] >> T.int32(7)) - ((scaled[m, j] >> T.int32(14)) << T.int32(7)), "int8")
                    p0[m, j] = T.cast(scaled[m, j] - ((scaled[m, j] >> T.int32(7)) << T.int32(7)), "int8")
                T.clear(pv)
                T.gemm(p2, v_shared, pv, policy=T.GemmWarpPolicy.FullRow)
                for m, d in T.Parallel(block_m, dim):
                    acc[m, d] += pv[m, d] << T.int32(14)
                T.clear(pv)
                T.gemm(p1, v_shared, pv, policy=T.GemmWarpPolicy.FullRow)
                for m, d in T.Parallel(block_m, dim):
                    acc[m, d] += pv[m, d] << T.int32(7)
                T.clear(pv)
                T.gemm(p0, v_shared, pv, policy=T.GemmWarpPolicy.FullRow)
                for m, d in T.Parallel(block_m, dim):
                    acc[m, d] += pv[m, d]
            for m, d in T.Parallel(block_m, dim):
                O[row_base + m, h * dim + d] = acc[m, d]

    return main


def attention_i8v8_q15_16_gqa_cache_fused_static_current(q_heads, kv_heads, seqlen, cache_len, dim, block_m=16, block_n=64, score_shift=26, lut_scale=0.125):
    group = q_heads // kv_heads
    kv_len = cache_len + seqlen
    q_size = q_heads * dim

    @T.prim_func
    def main(
        Q: T.Tensor((q_heads, seqlen, dim), "int8"),
        CACHE_K: T.Tensor((kv_heads, cache_len, dim), "int8"),
        CACHE_V: T.Tensor((kv_heads, cache_len, dim), "int8"),
        K: T.Tensor((kv_heads, seqlen, dim), "int8"),
        V: T.Tensor((kv_heads, seqlen, dim), "int8"),
        QS: T.Tensor((q_heads,), "uint32"),
        CACHE_KS: T.Tensor((kv_heads, cache_len), "uint32"),
        CACHE_VS: T.Tensor((kv_heads, cache_len), "uint32"),
        KS: T.Tensor((kv_heads,), "uint32"),
        VS: T.Tensor((kv_heads,), "uint32"),
        LUT: T.Tensor((1024,), "int16"),
        O: T.Tensor((seqlen, q_size), "int32"),
    ):
        with T.Kernel(q_heads, T.ceildiv(seqlen, block_m), threads=128) as (h, bm):
            kh = h // group
            row_base = bm * block_m
            q_shared = T.alloc_shared((block_m, dim), "int8")
            k_shared = T.alloc_shared((block_n, dim), "int8")
            v_shared = T.alloc_shared((block_n, dim), "int8")
            p2 = T.alloc_shared((block_m, block_n), "int8")
            p1 = T.alloc_shared((block_m, block_n), "int8")
            p0 = T.alloc_shared((block_m, block_n), "int8")
            qk = T.alloc_fragment((block_m, block_n), "int32")
            score = T.alloc_fragment((block_m, block_n), "int32")
            block_max = T.alloc_fragment((block_m,), "int32")
            score_max = T.alloc_fragment((block_m,), "int32")
            new_max = T.alloc_fragment((block_m,), "int32")
            old_scale = T.alloc_fragment((block_m,), "int32")
            block_sum = T.alloc_fragment((block_m,), "int32")
            denom = T.alloc_fragment((block_m,), "int32")
            q_scale = T.alloc_fragment((1,), "int32")
            k_scale = T.alloc_fragment((block_n,), "int32")
            v_scale = T.alloc_fragment((block_n,), "int32")
            prob = T.alloc_fragment((block_m, block_n), "int32")
            scaled = T.alloc_fragment((block_m, block_n), "int32")
            pv = T.alloc_fragment((block_m, dim), "int32")
            acc = T.alloc_fragment((block_m, dim), "int32")
            q_scale[0] = T.cast(QS[h], "int32") >> T.int32(4)
            for m in T.Parallel(block_m):
                score_max[m] = T.int32(I32_MIN)
                denom[m] = T.int32(0)
            for m, d in T.Parallel(block_m, dim):
                q_shared[m, d] = Q[h, row_base + m, d]
                acc[m, d] = T.int32(0)
            for nb in T.Pipelined((cache_len + row_base + block_m - 1) // block_n + 1):
                for j, d in T.Parallel(block_n, dim):
                    pos = nb * block_n + j
                    k_shared[j, d] = T.if_then_else(pos < cache_len, CACHE_K[kh, pos, d], T.if_then_else(pos < kv_len, K[kh, pos - cache_len, d], T.int8(0)))
                for j in T.Parallel(block_n):
                    pos = nb * block_n + j
                    k_scale[j] = T.if_then_else(pos < cache_len, T.cast(CACHE_KS[kh, pos], "int32"), T.if_then_else(pos < kv_len, T.cast(KS[kh], "int32"), T.int32(1)))
                T.clear(qk)
                T.gemm(q_shared, k_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for m, j in T.Parallel(block_m, block_n):
                    pos = nb * block_n + j
                    score[m, j] = ((qk[m, j] >> T.int32(8)) * (((q_scale[0] * (k_scale[j] >> T.int32(4))) >> T.int32(8)) * T.int32(5793))) >> T.int32(score_shift - 8)
                    score[m, j] = T.if_then_else(pos <= cache_len + row_base + m, score[m, j], T.int32(-32768))
                T.reduce_max(score, block_max, dim=1, clear=True)
                for m in T.Parallel(block_m):
                    new_max[m] = T.max(score_max[m], block_max[m])
                    old_scale[m] = T.fix.lut_10bit(score_max[m] - new_max[m], LUT, scale=lut_scale, out_dtype="int32")
                for m, j in T.Parallel(block_m, block_n):
                    pos = nb * block_n + j
                    score[m, j] = T.if_then_else(pos <= cache_len + row_base + m, T.fix.lut_10bit(score[m, j] - new_max[m], LUT, scale=lut_scale, out_dtype="int32"), T.int32(0))
                T.reduce_sum(score, block_sum, dim=1, clear=True)
                for m in T.Parallel(block_m):
                    denom[m] = ((denom[m] * old_scale[m]) >> T.int32(10)) + block_sum[m]
                    score_max[m] = new_max[m]
            for nb in T.Pipelined((cache_len + row_base + block_m - 1) // block_n + 1):
                for j, d in T.Parallel(block_n, dim):
                    pos = nb * block_n + j
                    k_shared[j, d] = T.if_then_else(pos < cache_len, CACHE_K[kh, pos, d], T.if_then_else(pos < kv_len, K[kh, pos - cache_len, d], T.int8(0)))
                    v_shared[j, d] = T.if_then_else(pos < cache_len, CACHE_V[kh, pos, d], T.if_then_else(pos < kv_len, V[kh, pos - cache_len, d], T.int8(0)))
                for j in T.Parallel(block_n):
                    pos = nb * block_n + j
                    k_scale[j] = T.if_then_else(pos < cache_len, T.cast(CACHE_KS[kh, pos], "int32"), T.if_then_else(pos < kv_len, T.cast(KS[kh], "int32"), T.int32(1)))
                    v_scale[j] = T.if_then_else(pos < cache_len, T.cast(CACHE_VS[kh, pos], "int32"), T.if_then_else(pos < kv_len, T.cast(VS[kh], "int32"), T.int32(0)))
                T.clear(qk)
                T.gemm(q_shared, k_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for m, j in T.Parallel(block_m, block_n):
                    pos = nb * block_n + j
                    score[m, j] = ((qk[m, j] >> T.int32(8)) * (((q_scale[0] * (k_scale[j] >> T.int32(4))) >> T.int32(8)) * T.int32(5793))) >> T.int32(score_shift - 8)
                    score[m, j] = T.if_then_else(pos <= cache_len + row_base + m, T.fix.lut_10bit(score[m, j] - score_max[m], LUT, scale=lut_scale, out_dtype="int32"), T.int32(0))
                    prob[m, j] = T.min(T.truncdiv((score[m, j] * T.int32(16383)) + (denom[m] >> T.int32(1)), denom[m]), T.int32(16383))
                    scaled[m, j] = T.truncdiv((prob[m, j] * v_scale[j]) + T.int32(8191), T.int32(16383))
                    p2[m, j] = T.cast(scaled[m, j] >> T.int32(14), "int8")
                    p1[m, j] = T.cast((scaled[m, j] >> T.int32(7)) - ((scaled[m, j] >> T.int32(14)) << T.int32(7)), "int8")
                    p0[m, j] = T.cast(scaled[m, j] - ((scaled[m, j] >> T.int32(7)) << T.int32(7)), "int8")
                T.clear(pv)
                T.gemm(p2, v_shared, pv, policy=T.GemmWarpPolicy.FullRow)
                for m, d in T.Parallel(block_m, dim):
                    acc[m, d] += pv[m, d] << T.int32(14)
                T.clear(pv)
                T.gemm(p1, v_shared, pv, policy=T.GemmWarpPolicy.FullRow)
                for m, d in T.Parallel(block_m, dim):
                    acc[m, d] += pv[m, d] << T.int32(7)
                T.clear(pv)
                T.gemm(p0, v_shared, pv, policy=T.GemmWarpPolicy.FullRow)
                for m, d in T.Parallel(block_m, dim):
                    acc[m, d] += pv[m, d]
            for m, d in T.Parallel(block_m, dim):
                O[row_base + m, h * dim + d] = acc[m, d]

    return main


def compile_kernel(func, out_idx):
    return tilelang.compile(func, out_idx=out_idx, target="cuda")
