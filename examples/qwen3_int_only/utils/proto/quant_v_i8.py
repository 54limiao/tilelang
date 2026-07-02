import numpy as np

from examples.qwen3_int_only.utils.proto.common import Q15_16, metrics, static_quant_signed


def proto(X, S):
    tokens, heads, head_dim = X.shape
    Y = np.empty((heads, tokens, head_dim), dtype=np.int8)
    for t in range(tokens):
        for h in range(heads):
            Y[h, t] = static_quant_signed(X[t, h], S[h], 127).astype(np.int8)
    return Y


def check(rng):
    X = rng.integers(-3 * Q15_16, 3 * Q15_16, size=(8, 3, 16), dtype=np.int32)
    S = rng.integers(300, 1600, size=(3,), dtype=np.uint32)
    Y = proto(X, S)
    ref = np.clip(np.rint((X.astype(np.float64) / Q15_16) / (S[None, :, None].astype(np.float64) / Q15_16)), -128, 127).transpose(1, 0, 2)
    metrics("quant_v_i8", Y.astype(np.float64), ref)
