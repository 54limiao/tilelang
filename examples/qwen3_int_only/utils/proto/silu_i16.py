import numpy as np

from examples.qwen3_int_only.utils.proto.common import Q15_16, lut_10bit, metrics, q15, sigmoid_lut_np, static_quant_signed


def proto(Gate, Up, LUT, SCALE):
    rows, cols = Gate.shape
    Q = np.empty((rows, cols), dtype=np.int16)
    S = np.empty((rows,), dtype=np.uint32)
    scale = max(int(SCALE[0]), 1)
    for r in range(rows):
        S[r] = scale
        sig = lut_10bit(Gate[r], LUT, 1.0 / 1024.0, "int32").astype(np.int64)
        vals = ((((Gate[r].astype(np.int64) >> 10) * sig) >> 8) * (Up[r].astype(np.int64) >> 8))
        Q[r] = static_quant_signed(vals, scale, 32767).astype(np.int16)
    return Q, S


def check(rng):
    rows, cols = 6, 128
    gate = q15(rng.normal(size=(rows, cols)))
    up = q15(rng.normal(size=(rows, cols)))
    scale = np.array([256], dtype=np.uint32)
    Q, _S = proto(gate, up, sigmoid_lut_np(), scale)
    x = gate.astype(np.float64) / Q15_16
    ref = x / (1.0 + np.exp(-np.clip(x, -7.0, 7.0))) * (up.astype(np.float64) / Q15_16)
    metrics("silu_i16", Q.astype(np.float64) * (scale[0] / Q15_16), ref)
