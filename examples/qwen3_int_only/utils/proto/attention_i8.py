import numpy as np

from examples.qwen3_int_only.utils.proto.common import exp_lut_neg_np, lut_10bit, metrics


def _kv_at(cache, cur, pos, cache_len):
    return cache[:, pos] if pos < cache_len else cur[:, pos - cache_len]


def proto(Q, CACHE_K, CACHE_V, K, V, QS, KS, VS, LUT=None, OS=None, block_m=16, block_n=64, score_shift=26, lut_scale=0.125):
    LUT = exp_lut_neg_np() if LUT is None else LUT
    q_heads, seqlen, dim = Q.shape
    kv_heads = K.shape[0]
    cache_len = CACHE_K.shape[1]
    group = q_heads // kv_heads
    kv_len = cache_len + seqlen
    OS = np.array([512], dtype=np.uint32) if OS is None else OS
    O = np.empty((seqlen, q_heads * dim), dtype=np.int8)
    for h in range(q_heads):
        kh = h // group
        q_scale = int(QS[h]) >> 4
        k_scale = max(int(KS[kh]), 1)
        v_scale = max(int(VS[kh]), 1)
        o_scale = max(int(OS[0]), 1)
        for row_base in range(0, seqlen, block_m):
            q_shared = Q[h, row_base : row_base + block_m].astype(np.int32)
            score_max = np.full((block_m,), -(1 << 31), dtype=np.int64)
            denom = np.zeros((block_m,), dtype=np.int64)
            acc = np.zeros((block_m, dim), dtype=np.int64)
            nblocks = (cache_len + row_base + block_m - 1) // block_n + 1
            for nb in range(nblocks):
                pos = nb * block_n + np.arange(block_n)
                k_shared = np.zeros((block_n, dim), dtype=np.int32)
                for j, p in enumerate(pos):
                    if p < kv_len:
                        k_shared[j] = _kv_at(CACHE_K, K, int(p), cache_len)[kh]
                qk = q_shared @ k_shared.T
                score = ((qk.astype(np.int64) >> 8) * (((q_scale * (k_scale >> 4)) >> 8) * 5793)) >> (score_shift - 8)
                for m in range(block_m):
                    score[m] = np.where(pos <= cache_len + row_base + m, score[m], -32768)
                block_max = np.max(score, axis=1)
                new_max = np.maximum(score_max, block_max)
                old_scale = lut_10bit(score_max - new_max, LUT, lut_scale, "int32").astype(np.int64)
                exp_score = np.zeros_like(score, dtype=np.int64)
                for m in range(block_m):
                    valid = pos <= cache_len + row_base + m
                    exp_score[m] = np.where(valid, lut_10bit(score[m] - new_max[m], LUT, lut_scale, "int32"), 0)
                denom = ((denom * old_scale) >> 10) + np.sum(exp_score, axis=1)
                score_max = new_max
            for nb in range(nblocks):
                pos = nb * block_n + np.arange(block_n)
                k_shared = np.zeros((block_n, dim), dtype=np.int32)
                v_shared = np.zeros((block_n, dim), dtype=np.int32)
                for j, p in enumerate(pos):
                    if p < kv_len:
                        k_shared[j] = _kv_at(CACHE_K, K, int(p), cache_len)[kh]
                        v_shared[j] = _kv_at(CACHE_V, V, int(p), cache_len)[kh]
                qk = q_shared @ k_shared.T
                score = ((qk.astype(np.int64) >> 8) * (((q_scale * (k_scale >> 4)) >> 8) * 5793)) >> (score_shift - 8)
                exp_score = np.zeros_like(score, dtype=np.int64)
                for m in range(block_m):
                    valid = pos <= cache_len + row_base + m
                    exp_score[m] = np.where(valid, lut_10bit(score[m] - score_max[m], LUT, lut_scale, "int32"), 0)
                prob = np.minimum(((exp_score * 32767) + (denom[:, None] >> 1)) // denom[:, None], 32767)
                p_hi = (prob >> 8).astype(np.int8).astype(np.int32)
                p_mid = ((prob - ((prob >> 8) << 8)) >> 1).astype(np.int8).astype(np.int32)
                acc += (p_hi @ v_shared) << 8
                acc += (p_mid @ v_shared) << 1
            out_abs = (np.abs(acc) * v_scale + ((o_scale * 32767) >> 1)) // (o_scale * 32767)
            out = np.where(acc < 0, -out_abs, out_abs)
            O[row_base : row_base + block_m, h * dim : (h + 1) * dim] = np.clip(out, -128, 127).astype(np.int8)
    return O


def check(rng):
    q_heads, kv_heads, seqlen, cache_len, dim = 4, 2, 16, 5, 16
    Q = rng.integers(-80, 80, size=(q_heads, seqlen, dim), dtype=np.int8)
    K = rng.integers(-80, 80, size=(kv_heads, seqlen, dim), dtype=np.int8)
    V = rng.integers(-80, 80, size=(kv_heads, seqlen, dim), dtype=np.int8)
    CK = rng.integers(-80, 80, size=(kv_heads, cache_len, dim), dtype=np.int8)
    CV = rng.integers(-80, 80, size=(kv_heads, cache_len, dim), dtype=np.int8)
    QS = rng.integers(300, 900, size=(q_heads,), dtype=np.uint32)
    KS = rng.integers(300, 900, size=(kv_heads,), dtype=np.uint32)
    VS = rng.integers(300, 900, size=(kv_heads,), dtype=np.uint32)
    out = []
    for h in range(q_heads):
        kh = h // (q_heads // kv_heads)
        q = Q[h].astype(np.float64) * (QS[h] / 65536.0)
        k = np.concatenate((CK[kh], K[kh]), axis=0).astype(np.float64) * (KS[kh] / 65536.0)
        v = np.concatenate((CV[kh], V[kh]), axis=0).astype(np.float64) * (VS[kh] / 65536.0)
        score = q @ k.T / np.sqrt(dim)
        mask = np.arange(cache_len + seqlen)[None, :] <= cache_len + np.arange(seqlen)[:, None]
        score = np.where(mask, score, -1e30)
        p = np.exp(score - np.max(score, axis=1, keepdims=True))
        p /= np.sum(p, axis=1, keepdims=True)
        out.append(p @ v)
    ref = np.concatenate(out, axis=1)
    ref_q15 = np.rint(ref * 65536.0).astype(np.int64)
    OS = np.array([max(int((np.max(np.abs(ref_q15)) + 126) // 127), 1)], dtype=np.uint32)
    O = proto(Q, CK, CV, K, V, QS, KS, VS, OS=OS, block_m=16, block_n=8)
    metrics("attention_i8", O.astype(np.float64) * (OS[0] / 65536.0), ref)
