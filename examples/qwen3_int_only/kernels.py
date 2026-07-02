import math

import tilelang
import tilelang.language as T

from tilelang.language.fix import Q_MULTIPLIER_WIDTH, pack_scale

MASK = (1 << Q_MULTIPLIER_WIDTH) - 1
ATTN_VALUE_SHIFT = 7
I32_MIN = -2147483648
SCALE_1024 = pack_scale(1024.0)
SCALE_INV_1024 = pack_scale(1.0 / 1024.0)
SCALE_EXP_LUT = pack_scale(0.125)
DYN_SCALE_SHIFT = 25
DYN_SCALE_ONE = 1 << DYN_SCALE_SHIFT


def quant_v_i8(tokens, heads, head_dim):
    @T.prim_func
    def main(
        X: T.Tensor((tokens, heads, head_dim), "int32"),
        QT: T.Tensor((heads,), "uint32"),
        Y: T.Tensor((heads, tokens, head_dim), "int8"),
    ):
        with T.Kernel(tokens, heads, threads=128) as (t, h):
            qt = T.alloc_local((1,), "uint32")
            qt[0] = QT[h]
            for d in T.Parallel(head_dim):
                Y[h, t, d] = T.fix.quant(X[t, h, d], scale=qt[0], out_dtype="int8")

    return main


def rms_q15(rows, cols, qmax=32767):
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
            row_qt = T.alloc_fragment((1,), "int32")
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
            row_qt[0] = (T.int32(DYN_SCALE_SHIFT) << T.int32(Q_MULTIPLIER_WIDTH)) | T.min(
                (T.int32(DYN_SCALE_ONE) + (scale[0] >> T.int32(1))) // scale[0],
                T.int32(MASK),
            )
            for c in T.Parallel(cols):
                q[0, c] = T.fix.quant(Y[r, c], scale=row_qt[0], out_dtype="int16")
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
            fold[0] = T.fix.quant(inv[0], scale=SCALE_1024, out_dtype="int32")
            qt[0] = ((T.int32(6) + (ns[0] >> T.int32(1))) << T.int32(Q_MULTIPLIER_WIDTH)) | ((fold[0] >> T.int32(4)) & T.int32(MASK))
            for c in T.Parallel(cols):
                norm[c] = T.fix.quant(q[0, c], scale=qt[0], out_dtype="int32")
                N[r, c] = (norm[c] * (W[c] >> T.int32(8))) >> T.int32(2)

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


def rope_sq8(seq_len, heads, dim, qmax=127):
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
        QT: T.Tensor((heads,), "uint32"),
        Y: T.Tensor((heads, seq_len, dim), "int8"),
    ):
        with T.Kernel(seq_len * heads, threads=threads) as r:
            tx = T.get_thread_binding(0)
            t = r // heads
            h = r - t * heads
            local = T.alloc_local((thread_elem,), "int32")
            other_val = T.alloc_local((thread_elem,), "int32")
            qt = T.alloc_local((1,), "int32")
            qt[0] = T.cast(QT[h], "int32")
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
                Y[h, t, tx * thread_elem + i] = T.fix.quant(v, scale=qt[0], out_dtype="int8")

    return main


def rms_sq8(rows, cols, qmax=127):
    mean_shift = int(math.log2(cols))

    @T.prim_func
    def main(
        A: T.Tensor((rows, cols), "int32"),
        B: T.Tensor((rows, cols), "int32"),
        W: T.Tensor((cols,), "int32"),
        RLUT: T.Tensor((1024,), "int16"),
        QT: T.Tensor((1,), "uint32"),
        Y: T.Tensor((rows, cols), "int32"),
        Q: T.Tensor((rows, cols), "int8"),
        S: T.Tensor((rows,), "uint32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            q = T.alloc_fragment((1, cols), "int32")
            xx = T.alloc_fragment((1, cols), "int32")
            amax = T.alloc_fragment((1,), "int32")
            scale = T.alloc_fragment((1,), "int32")
            row_qt = T.alloc_fragment((1,), "int32")
            ss = T.alloc_fragment((1,), "int32")
            ns = T.alloc_fragment((1,), "int32")
            wk = T.alloc_fragment((1,), "int32")
            inv = T.alloc_fragment((1,), "int32")
            fold = T.alloc_fragment((1,), "int32")
            qt = T.alloc_fragment((1,), "int32")
            norm = T.alloc_fragment((cols,), "int32")
            post = T.alloc_fragment((1, cols), "int32")
            post_qt = T.alloc_fragment((1,), "int32")
            post_qt[0] = T.cast(QT[0], "int32")
            S[r] = QT[0]
            for c in T.Parallel(cols):
                Y[r, c] = A[r, c] + B[r, c]
                q[0, c] = Y[r, c]
                if q[0, c] < T.int32(0):
                    q[0, c] = T.int32(0) - q[0, c]
            T.reduce_max(q, amax, dim=1, clear=True)
            scale[0] = T.max((amax[0] + T.int32(32766)) // T.int32(32767), T.int32(1))
            row_qt[0] = (T.int32(DYN_SCALE_SHIFT) << T.int32(Q_MULTIPLIER_WIDTH)) | T.min(
                (T.int32(DYN_SCALE_ONE) + (scale[0] >> T.int32(1))) // scale[0],
                T.int32(MASK),
            )
            for c in T.Parallel(cols):
                q[0, c] = T.fix.quant(Y[r, c], scale=row_qt[0], out_dtype="int16")
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
            fold[0] = T.fix.quant(inv[0], scale=SCALE_1024, out_dtype="int32")
            qt[0] = ((T.int32(6) + (ns[0] >> T.int32(1))) << T.int32(Q_MULTIPLIER_WIDTH)) | ((fold[0] >> T.int32(4)) & T.int32(MASK))
            for c in T.Parallel(cols):
                norm[c] = T.fix.quant(q[0, c], scale=qt[0], out_dtype="int32")
                post[0, c] = (norm[c] * (W[c] >> T.int32(8))) >> T.int32(2)
                Q[r, c] = T.fix.quant(post[0, c], scale=post_qt[0], out_dtype="int8")

    return main


def silu_i16(rows, cols):
    @T.prim_func
    def main(
        Gate: T.Tensor((rows, cols), "int32"),
        Up: T.Tensor((rows, cols), "int32"),
        LUT: T.Tensor((1024,), "int32"),
        QT: T.Tensor((1,), "uint32"),
        Q: T.Tensor((rows, cols), "int16"),
        S: T.Tensor((rows,), "uint32"),
    ):
        with T.Kernel(rows, threads=128) as r:
            vals = T.alloc_fragment((1, cols), "int32")
            qt = T.alloc_fragment((1,), "int32")
            qt[0] = T.cast(QT[0], "int32")
            S[r] = QT[0]
            for c in T.Parallel(cols):
                vals[0, c] = (((Gate[r, c] >> T.int32(10)) * T.fix.lut_10bit(Gate[r, c], LUT, scale=SCALE_INV_1024, out_dtype="int32")) >> T.int32(8)) * (Up[r, c] >> T.int32(8))
                Q[r, c] = T.fix.quant(vals[0, c], scale=qt[0], out_dtype="int16")

    return main


def linear_i8(rows, in_features, out_features, block_m=16, block_n=32, block_k=64):
    @T.prim_func
    def main(
        X: T.Tensor((rows, in_features), "int8"),
        W: T.Tensor((out_features, in_features), "int8"),
        QT: T.Tensor((out_features,), "uint32"),
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
                Y[br * block_m + m, bo * block_n + n] = T.fix.quant(acc[m, n], scale=QT[bo * block_n + n], out_dtype="int32")

    return main


def linear_i16(rows, in_features, out_features, block_m=16, block_n=32, block_k=64):
    @T.prim_func
    def main(
        X: T.Tensor((rows, in_features), "int16"),
        W: T.Tensor((out_features, in_features), "int8"),
        QT: T.Tensor((out_features,), "uint32"),
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
                Y[br * block_m + m, bo * block_n + n] = T.fix.quant(acc_hi[m, n] + (acc_mid[m, n] >> T.int32(7)), scale=QT[bo * block_n + n], out_dtype="int32")

    return main


def attention_i8(q_heads, kv_heads, seqlen, cache_len, dim, block_m=16, block_n=64, score_shift=26, lut_scale=SCALE_EXP_LUT):
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
        KS: T.Tensor((kv_heads,), "uint32"),
        LUT: T.Tensor((1024,), "int16"),
        OQT: T.Tensor((kv_heads,), "uint32"),
        O: T.Tensor((seqlen, q_size), "int8"),
    ):
        with T.Kernel(q_heads, T.ceildiv(seqlen, block_m), threads=128) as (h, bm):
            kh = h // group
            row_base = bm * block_m
            q_shared = T.alloc_shared((block_m, dim), "int8")
            k_shared = T.alloc_shared((block_n, dim), "int8")
            v_shared = T.alloc_shared((block_n, dim), "int8")
            p_hi = T.alloc_shared((block_m, block_n), "int8")
            p_mid = T.alloc_shared((block_m, block_n), "int8")
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
            out_qt = T.alloc_fragment((1,), "int32")
            out = T.alloc_fragment((block_m, dim), "int32")
            prob = T.alloc_fragment((block_m, block_n), "int32")
            pv = T.alloc_fragment((block_m, dim), "int32")
            acc = T.alloc_fragment((block_m, dim), "int32")
            q_scale[0] = T.cast(QS[h], "int32") >> T.int32(4)
            out_qt[0] = T.cast(OQT[kh], "int32")
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
                    k_scale[j] = T.max(T.cast(KS[kh], "int32"), T.int32(1))
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
                    k_scale[j] = T.max(T.cast(KS[kh], "int32"), T.int32(1))
                T.clear(qk)
                T.gemm(q_shared, k_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for m, j in T.Parallel(block_m, block_n):
                    pos = nb * block_n + j
                    score[m, j] = ((qk[m, j] >> T.int32(8)) * (((q_scale[0] * (k_scale[j] >> T.int32(4))) >> T.int32(8)) * T.int32(5793))) >> T.int32(score_shift - 8)
                    score[m, j] = T.if_then_else(pos <= cache_len + row_base + m, T.fix.lut_10bit(score[m, j] - score_max[m], LUT, scale=lut_scale, out_dtype="int32"), T.int32(0))
                    prob[m, j] = T.min(T.truncdiv((score[m, j] * T.int32(32767)) + (denom[m] >> T.int32(1)), denom[m]), T.int32(32767))
                    p_hi[m, j] = T.cast(prob[m, j] >> T.int32(8), "int8")
                    p_mid[m, j] = T.cast((prob[m, j] - ((prob[m, j] >> T.int32(8)) << T.int32(8))) >> T.int32(1), "int8")
                T.clear(pv)
                T.gemm(p_hi, v_shared, pv, policy=T.GemmWarpPolicy.FullRow)
                for m, d in T.Parallel(block_m, dim):
                    acc[m, d] += pv[m, d] << T.int32(8)
                T.clear(pv)
                T.gemm(p_mid, v_shared, pv, policy=T.GemmWarpPolicy.FullRow)
                for m, d in T.Parallel(block_m, dim):
                    acc[m, d] += pv[m, d] << T.int32(1)
            for m, d in T.Parallel(block_m, dim):
                out[m, d] = T.fix.quant(acc[m, d], scale=out_qt[0], out_dtype="int32")
                out[m, d] = T.min(T.max(out[m, d], T.int32(-128)), T.int32(127))
                O[row_base + m, h * dim + d] = T.cast(out[m, d], "int8")

    return main
