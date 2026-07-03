import math

import tilelang.language as T


Q15_16_F = 65536.0
LOG2_E = 1.4426950408889634
NEG_INF = -3.402823e38


@T.macro
def _warp_hadamard_f32(local, buf, thread_elem, warp_size, rounds):
    tx = T.get_thread_binding(0)
    for i in T.serial(rounds):
        stride = 1 << i
        other = tx ^ stride
        sign = (tx >> i) & 1
        for j in T.Pipelined(thread_elem, num_stages=1):
            buf[j] = T.tvm_warp_shuffle(0xFFFFFFFF, local[j], other % warp_size, warp_size, warp_size)
            local[j] = T.if_then_else(sign == 0, local[j] + buf[j], buf[j] - local[j])


@T.macro
def _quant_i8_f32(x, scale):
    q = T.cast(T.round(x / T.max(T.cast(scale, "float32"), T.float32(1.0e-8)), rounding_mode="ties-away-from-zero"), "int32")
    return T.cast(T.min(T.max(q, T.int32(-128)), T.int32(127)), "int8")


def rms_quant_hybrid(rows, cols):
    inv_cols = 1.0 / cols

    @T.prim_func
    def main(
        A: T.Tensor((rows, cols), "float32"),
        B: T.Tensor((rows, cols), "int32"),
        W: T.Tensor((cols,), "float32"),
        SCALE: T.Tensor((1,), "float32"),
        Y: T.Tensor((rows, cols), "float32"),
        Q: T.Tensor((rows, cols), "int8"),
    ):
        with T.Kernel(rows, threads=128) as r:
            x = T.alloc_fragment((1, cols), "float32")
            xx = T.alloc_fragment((1, cols), "float32")
            ss = T.alloc_fragment((1,), "float32")
            inv = T.alloc_fragment((1,), "float32")
            for c in T.Parallel(cols):
                x[0, c] = A[r, c] + T.cast(B[r, c], "float32") / T.float32(Q15_16_F)
                Y[r, c] = x[0, c]
                xx[0, c] = x[0, c] * x[0, c] * T.float32(inv_cols)
            T.reduce_sum(xx, ss, dim=1, clear=True)
            inv[0] = T.rsqrt(ss[0] + T.float32(1.0e-6))
            for c in T.Parallel(cols):
                Q[r, c] = _quant_i8_f32(x[0, c] * inv[0] * W[c], SCALE[0])

    return main


def qk_norm_rope_quant_hybrid(seq_len, heads, dim, gpb=8):
    thread_elem = 8
    lanes = 16
    thread_round = 3
    warp_round = 4
    half_dim = dim // 2
    inv_dim = 1.0 / dim
    inv_sqrt_dim = 1.0 / math.sqrt(dim)
    total = seq_len * heads
    while total % gpb != 0:
        gpb //= 2
    threads = lanes * gpb

    @T.prim_func
    def main(
        X: T.Tensor((seq_len * heads, dim), "int32"),
        W: T.Tensor((dim,), "float32"),
        COS: T.Tensor((seq_len, half_dim), "float32"),
        SIN: T.Tensor((seq_len, half_dim), "float32"),
        SCALE: T.Tensor((heads,), "float32"),
        Y: T.Tensor((heads, seq_len, dim), "int8"),
    ):
        # Pack `gpb` independent (token, head) rows per block; each row uses its own
        # 16-lane group with a hand-rolled warp-shuffle sum reduction. Numerically
        # identical to the one-row-per-block version, ~10x higher occupancy.
        with T.Kernel(total // gpb, threads=threads) as blk:
            tx = T.get_thread_binding(0)
            grp = tx // lanes
            lane = tx - grp * lanes
            r = blk * gpb + grp
            t = r // heads
            h = r - t * heads
            xv = T.alloc_local((thread_elem,), "float32")
            pv = T.alloc_local((thread_elem,), "float32")
            local = T.alloc_local((thread_elem,), "float32")
            other = T.alloc_local((thread_elem,), "float32")
            ss = T.alloc_local((1,), "float32")
            inv = T.alloc_local((1,), "float32")
            ss[0] = T.float32(0.0)
            for i in T.serial(thread_elem):
                xv[i] = T.cast(X[r, lane * T.int32(thread_elem) + i], "float32") / T.float32(Q15_16_F)
                ss[0] += xv[i] * xv[i] * T.float32(inv_dim)
            ss[0] += T.tvm_warp_shuffle(0xFFFFFFFF, ss[0], lane ^ T.int32(8), lanes, lanes)
            ss[0] += T.tvm_warp_shuffle(0xFFFFFFFF, ss[0], lane ^ T.int32(4), lanes, lanes)
            ss[0] += T.tvm_warp_shuffle(0xFFFFFFFF, ss[0], lane ^ T.int32(2), lanes, lanes)
            ss[0] += T.tvm_warp_shuffle(0xFFFFFFFF, ss[0], lane ^ T.int32(1), lanes, lanes)
            inv[0] = T.rsqrt(ss[0] + T.float32(1.0e-6))
            for i in T.serial(thread_elem):
                pv[i] = T.tvm_warp_shuffle(0xFFFFFFFF, xv[i], lane ^ T.int32(8), lanes, lanes)
            for i in T.serial(thread_elem):
                d = lane * T.int32(thread_elem) + i
                src = (lane & T.int32(7)) * T.int32(thread_elem) + i
                x0src = T.if_then_else(lane < T.int32(8), xv[i], pv[i])
                x1src = T.if_then_else(lane < T.int32(8), pv[i], xv[i])
                x0 = x0src * inv[0] * W[src]
                x1 = x1src * inv[0] * W[src + T.int32(half_dim)]
                c = T.cast(COS[t, src], "float32")
                s = T.cast(SIN[t, src], "float32")
                local[i] = T.if_then_else(d < T.int32(half_dim), x0 * c - x1 * s, x0 * s + x1 * c)
            for i in T.serial(thread_round):
                chunksize = 1 << (i + 1)
                chunknum = thread_elem // chunksize
                for j in T.serial(chunknum):
                    chunkbase = j * chunksize
                    for k in T.serial(chunksize // 2):
                        a = local[chunkbase + k]
                        b = local[chunkbase + k + chunksize // 2]
                        local[chunkbase + k] = a + b
                        local[chunkbase + k + chunksize // 2] = local[chunkbase + k] - T.float32(2.0) * b
            _warp_hadamard_f32(local, other, thread_elem, lanes, warp_round)
            for i in T.serial(thread_elem):
                Y[h, t, lane * T.int32(thread_elem) + i] = _quant_i8_f32(local[i] * T.float32(inv_sqrt_dim), SCALE[h])

    return main


def silu_hadamard_quant_hybrid(rows, cols, block_dim=128):
    thread_elem = 8
    lanes = 16
    thread_round = 3
    warp_round = 4
    groups = cols // block_dim
    inv_sqrt_block = 1.0 / math.sqrt(block_dim)
    # Pack several independent 128-wide groups per block (16 lanes each). No
    # cross-group reduction, so this matches the one-group-per-block version.
    gpb = 8 if groups % 8 == 0 else (4 if groups % 4 == 0 else (2 if groups % 2 == 0 else 1))
    threads = lanes * gpb

    @T.prim_func
    def main(
        Gate: T.Tensor((rows, cols), "int32"),
        Up: T.Tensor((rows, cols), "int32"),
        SCALE: T.Tensor((1,), "float32"),
        Q: T.Tensor((rows, cols), "int8"),
    ):
        with T.Kernel(rows, groups // gpb, threads=threads) as (r, gb):
            tx = T.get_thread_binding(0)
            grp = tx // lanes
            lane = tx - grp * lanes
            g = gb * gpb + grp
            local = T.alloc_local((thread_elem,), "float32")
            other = T.alloc_local((thread_elem,), "float32")
            for i in T.serial(thread_elem):
                c = g * T.int32(block_dim) + lane * T.int32(thread_elem) + i
                gate = T.cast(Gate[r, c], "float32") / T.float32(Q15_16_F)
                up = T.cast(Up[r, c], "float32") / T.float32(Q15_16_F)
                sig = T.sigmoid(gate)
                local[i] = T.if_then_else(
                    gate < T.float32(-7.0),
                    T.float32(0.0),
                    T.if_then_else(gate > T.float32(7.0), gate * up, gate * sig * up),
                )
            for i in T.serial(thread_round):
                chunksize = 1 << (i + 1)
                chunknum = thread_elem // chunksize
                for j in T.serial(chunknum):
                    chunkbase = j * chunksize
                    for k in T.serial(chunksize // 2):
                        a = local[chunkbase + k]
                        b = local[chunkbase + k + chunksize // 2]
                        local[chunkbase + k] = a + b
                        local[chunkbase + k + chunksize // 2] = local[chunkbase + k] - T.float32(2.0) * b
            _warp_hadamard_f32(local, other, thread_elem, lanes, warp_round)
            for i in T.serial(thread_elem):
                Q[r, g * T.int32(block_dim) + lane * T.int32(thread_elem) + i] = _quant_i8_f32(local[i] * T.float32(inv_sqrt_block), SCALE[0])

    return main


def attention_hybrid(q_heads, kv_heads, seqlen, cache_len, dim, block_m=16, block_n=128):
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
        SCORE_SCALE: T.Tensor((q_heads,), "float32"),
        VS: T.Tensor((kv_heads,), "float32"),
        OS: T.Tensor((1,), "float32"),
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
            score = T.alloc_fragment((block_m, block_n), "float32")
            block_max = T.alloc_fragment((block_m,), "float32")
            score_max = T.alloc_fragment((block_m,), "float32")
            new_max = T.alloc_fragment((block_m,), "float32")
            old_scale = T.alloc_fragment((block_m,), "float32")
            block_sum = T.alloc_fragment((block_m,), "float32")
            denom = T.alloc_fragment((block_m,), "float32")
            prob = T.alloc_fragment((block_m, block_n), "int32")
            pv = T.alloc_fragment((block_m, dim), "int32")
            tile_acc = T.alloc_fragment((block_m, dim), "int32")
            acc = T.alloc_fragment((block_m, dim), "float32")
            score_scale = T.alloc_fragment((1,), "float32")
            score_scale[0] = SCORE_SCALE[h]
            for m in T.Parallel(block_m):
                score_max[m] = T.float32(NEG_INF)
                denom[m] = T.float32(0.0)
            for m, d in T.Parallel(block_m, dim):
                q_shared[m, d] = Q[h, row_base + m, d]
                acc[m, d] = T.float32(0.0)
            for nb in T.Pipelined((cache_len + row_base + block_m - 1) // block_n + 1):
                for j, d in T.Parallel(block_n, dim):
                    pos = nb * block_n + j
                    k_shared[j, d] = T.if_then_else(pos < cache_len, CACHE_K[kh, pos, d], T.if_then_else(pos < kv_len, K[kh, pos - cache_len, d], T.int8(0)))
                    v_shared[j, d] = T.if_then_else(pos < cache_len, CACHE_V[kh, pos, d], T.if_then_else(pos < kv_len, V[kh, pos - cache_len, d], T.int8(0)))
                T.clear(qk)
                T.gemm(q_shared, k_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for m, j in T.Parallel(block_m, block_n):
                    pos = nb * block_n + j
                    score[m, j] = T.if_then_else(
                        pos <= cache_len + row_base + m,
                        T.cast(qk[m, j], "float32") * score_scale[0],
                        T.float32(NEG_INF),
                    )
                T.reduce_max(score, block_max, dim=1, clear=True)
                for m in T.Parallel(block_m):
                    new_max[m] = T.max(score_max[m], block_max[m])
                    old_scale[m] = T.exp2((score_max[m] - new_max[m]) * T.float32(LOG2_E))
                for m, j in T.Parallel(block_m, block_n):
                    pos = nb * block_n + j
                    score[m, j] = T.if_then_else(pos <= cache_len + row_base + m, T.exp2((score[m, j] - new_max[m]) * T.float32(LOG2_E)), T.float32(0.0))
                T.reduce_sum(score, block_sum, dim=1, clear=True)
                for m, j in T.Parallel(block_m, block_n):
                    pos = nb * block_n + j
                    prob[m, j] = T.cast(T.round(score[m, j] * T.float32(32767.0), rounding_mode="ties-away-from-zero"), "int32")
                    prob[m, j] = T.min(T.max(prob[m, j], T.int32(0)), T.int32(32767))
                    p_hi[m, j] = T.cast(prob[m, j] >> T.int32(8), "int8")
                    p_mid[m, j] = T.cast((prob[m, j] - ((prob[m, j] >> T.int32(8)) << T.int32(8))) >> T.int32(1), "int8")
                for m, d in T.Parallel(block_m, dim):
                    acc[m, d] *= old_scale[m]
                    tile_acc[m, d] = T.int32(0)
                T.clear(pv)
                T.gemm(p_hi, v_shared, pv, policy=T.GemmWarpPolicy.FullRow)
                for m, d in T.Parallel(block_m, dim):
                    tile_acc[m, d] += pv[m, d] << T.int32(8)
                T.clear(pv)
                T.gemm(p_mid, v_shared, pv, policy=T.GemmWarpPolicy.FullRow)
                for m, d in T.Parallel(block_m, dim):
                    tile_acc[m, d] += pv[m, d] << T.int32(1)
                    acc[m, d] += T.cast(tile_acc[m, d], "float32") / T.float32(32767.0)
                for m in T.Parallel(block_m):
                    denom[m] = denom[m] * old_scale[m] + block_sum[m]
                    score_max[m] = new_max[m]
            for m, d in T.Parallel(block_m, dim):
                O[row_base + m, h * dim + d] = _quant_i8_f32(acc[m, d] * VS[kh] / denom[m], OS[0])

    return main
