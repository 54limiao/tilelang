import numpy as np

from examples.qwen3_int_only.utils.proto.common import Q15_16, fix_quant, metrics, ratio_qt


def proto(X, XS, W, WS):
    x = X.astype(np.int32)
    x_hi = (x >> 8).astype(np.int8).astype(np.int32)
    x_mid = ((x - ((x >> 8) << 8)) >> 1).astype(np.int8).astype(np.int32)
    w = W.astype(np.int32)
    acc_hi = x_hi @ w.T
    acc_mid = x_mid @ w.T
    qt = ratio_qt(int(XS[0]) * WS.astype(np.int64), np.full_like(WS.astype(np.int64), 256))
    return fix_quant(acc_hi.astype(np.int64) + (acc_mid.astype(np.int64) >> 7), qt[None, :], "int32")


def check(rng):
    rows, in_features, out_features = 8, 64, 24
    X = rng.integers(-32768, 32768, size=(rows, in_features), dtype=np.int16)
    W = rng.integers(-128, 128, size=(out_features, in_features), dtype=np.int8)
    XS = np.array([384], dtype=np.uint32)
    WS = rng.integers(256, 2048, size=(out_features,), dtype=np.uint32)
    Y = proto(X, XS, W, WS)
    ref = (X.astype(np.float64) * (XS[0] / Q15_16)) @ (W.astype(np.float64) * (WS[:, None] / Q15_16)).T
    metrics("linear_i16", Y.astype(np.float64) / Q15_16, ref)
