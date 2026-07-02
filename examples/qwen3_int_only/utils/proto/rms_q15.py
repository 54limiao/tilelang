import math

import numpy as np

from examples.qwen3_int_only.utils.proto.common import MASK, Q15_16, Q_MULTIPLIER_WIDTH, fix_quant, lut_10bit, metrics, q15, rsqrt_lut_np, torch_rmsnorm

DYN_SCALE_SHIFT = 25


def proto(A, B, W, RLUT=None, qmax=32767):
    RLUT = rsqrt_lut_np() if RLUT is None else RLUT
    rows, cols = A.shape
    mean_shift = int(math.log2(cols))
    Y = (A.astype(np.int64) + B.astype(np.int64)).astype(np.int32)
    N = np.empty_like(Y)
    for r in range(rows):
        amax = int(np.max(np.abs(Y[r].astype(np.int64))))
        scale = max((amax + qmax - 1) // qmax, 1)
        row_qt = (DYN_SCALE_SHIFT << Q_MULTIPLIER_WIDTH) | min(((1 << DYN_SCALE_SHIFT) + (scale >> 1)) // scale, MASK)
        q = fix_quant(Y[r], row_qt, "int16").astype(np.int32)
        ss = int(np.sum((q.astype(np.int64) * q.astype(np.int64)) >> mean_shift)) + 1
        ns = 0
        wk = ss
        if wk & -65536:
            ns += 16
            wk >>= 16
        if wk & 0xFF00:
            ns += 8
            wk >>= 8
        if wk & 0xF0:
            ns += 4
            wk >>= 4
        if wk & 0xC:
            ns += 2
        inv = int(lut_10bit(ss, RLUT, ((ns - 7) << Q_MULTIPLIER_WIDTH) | 1, "int32"))
        fold = int(fix_quant(inv, 1024.0, "int32"))
        qt = ((6 + (ns >> 1)) << Q_MULTIPLIER_WIDTH) | ((fold >> 4) & MASK)
        norm = fix_quant(q, qt, "int32").astype(np.int64)
        N[r] = ((norm * (W.astype(np.int64) >> 8)) >> 2).astype(np.int32)
    return Y, N


def check(rng):
    rows, cols = 6, 128
    A = q15(rng.normal(size=(rows, cols)))
    B = q15(rng.normal(scale=0.2, size=(rows, cols)))
    W = q15(rng.uniform(0.5, 1.5, size=(cols,)))
    Y, N = proto(A, B, W)
    ref = torch_rmsnorm((A.astype(np.float32) + B.astype(np.float32)) / Q15_16, W.astype(np.float32) / Q15_16).numpy()
    metrics("rms_q15", N.astype(np.float64) / Q15_16, ref)
