import math

import numpy as np

from examples.qwen3_int_only.utils.proto.common import Q15_16, fix_quant, fwht, lut_10bit, metrics, pack_scale, q15, sigmoid_lut_np


def proto(Gate, Up, LUT, QT, block_dim=128):
    rows, cols = Gate.shape
    vals = np.empty((rows, cols), dtype=np.int32)
    for r in range(rows):
        sig = lut_10bit(Gate[r], LUT, 1.0 / 1024.0, "int32").astype(np.int64)
        vals[r] = ((((Gate[r].astype(np.int64) >> 10) * sig) >> 8) * (Up[r].astype(np.int64) >> 8)).astype(np.int32)
    y = vals.reshape(rows, cols // block_dim, block_dim)
    return fix_quant(fwht(y).reshape(rows, cols), QT[0], "int8")


def check(rng):
    rows, cols = 4, 256
    gate = q15(rng.normal(size=(rows, cols)))
    up = q15(rng.normal(size=(rows, cols)))
    ref = gate.astype(np.float64) / Q15_16
    ref = ref / (1.0 + np.exp(-np.clip(ref, -7.0, 7.0))) * (up.astype(np.float64) / Q15_16)
    ref = (fwht(ref.reshape(rows, cols // 128, 128)) / math.sqrt(128)).reshape(rows, cols)
    scale = max(np.max(np.abs(ref)) / 127.0, 1.0 / Q15_16)
    q = proto(gate, up, sigmoid_lut_np(), np.array([pack_scale(1.0 / (scale * math.sqrt(128) * Q15_16))], dtype=np.uint32))
    metrics("silu_hadamard_i8", q.astype(np.float64) * scale, ref)
