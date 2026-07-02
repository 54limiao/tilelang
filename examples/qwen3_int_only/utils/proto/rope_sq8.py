import numpy as np

from examples.qwen3_int_only.utils.proto.common import Q15_16, fwht, metrics, q15, static_quant_signed


def proto(X, COS, SIN, R, S):
    rows, dim = X.shape
    heads = S.shape[0]
    seq_len = rows // heads
    Y = np.empty((heads, seq_len, dim), dtype=np.int8)
    sign = np.where(R[:, 0].astype(np.int64) >= 0, 1, -1)
    half = dim // 2
    for r in range(rows):
        t = r // heads
        h = r - t * heads
        x0 = X[r, :half].astype(np.int64)
        x1 = X[r, half:].astype(np.int64)
        c = COS[t].astype(np.int64)
        s = SIN[t].astype(np.int64)
        lo = ((x0 >> 8) * (c >> 8)) - ((x1 >> 8) * (s >> 8))
        hi = ((x0 >> 8) * (s >> 8)) + ((x1 >> 8) * (c >> 8))
        v = np.concatenate((lo >> 8, hi >> 8)) * sign
        v = fwht(v) * 22
        Y[h, t] = static_quant_signed(v, S[h], 127).astype(np.int8)
    return Y


def check(rng):
    seq_len, heads, dim = 8, 2, 128
    X = q15(rng.normal(size=(seq_len * heads, dim)))
    pos = np.arange(seq_len)[:, None]
    freq = np.arange(dim // 2)[None, :]
    theta = pos * (10000.0 ** (-(2.0 * freq) / dim))
    COS = q15(np.cos(theta))
    SIN = q15(np.sin(theta))
    R = np.eye(dim, dtype=np.int32)
    x = X.reshape(seq_len, heads, dim).astype(np.float64) / Q15_16
    rot = np.concatenate((x[..., : dim // 2] * np.cos(theta)[:, None, :] - x[..., dim // 2 :] * np.sin(theta)[:, None, :], x[..., : dim // 2] * np.sin(theta)[:, None, :] + x[..., dim // 2 :] * np.cos(theta)[:, None, :]), axis=-1)
    ref = np.transpose(fwht(rot) / np.sqrt(dim), (1, 0, 2))
    ref_q15 = q15(ref)
    S = np.maximum((np.max(np.abs(ref_q15.astype(np.int64)), axis=(1, 2)) + 126) // 127, 1).astype(np.uint32)
    Y = proto(X, COS, SIN, R, S)
    got = Y.astype(np.float64) * (S[:, None, None] / Q15_16)
    metrics("rope_sq8", got, ref)
