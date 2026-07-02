import numpy as np

from examples.qwen3_int_only.utils.proto.common import Q15_16, metrics, q15, static_quant_signed, torch_rmsnorm
from examples.qwen3_int_only.utils.proto.rms_q15 import proto as rms_q15_proto


def proto(A, B, W, QS, RLUT=None):
    Y, N = rms_q15_proto(A, B, W, RLUT, qmax=32767)
    scale = max(int(QS[0]), 1)
    Q = np.empty(Y.shape, dtype=np.int8)
    S = np.empty((Y.shape[0],), dtype=np.uint32)
    for r in range(Y.shape[0]):
        S[r] = scale
        Q[r] = static_quant_signed(N[r], scale, 127).astype(np.int8)
    return Y, Q, S


def check(rng):
    rows, cols = 6, 128
    A = q15(rng.normal(size=(rows, cols)))
    B = q15(rng.normal(scale=0.2, size=(rows, cols)))
    W = q15(rng.uniform(0.5, 1.5, size=(cols,)))
    ref = torch_rmsnorm((A.astype(np.float32) + B.astype(np.float32)) / Q15_16, W.astype(np.float32) / Q15_16).numpy()
    ref_q15 = q15(ref)
    QS = np.array([max(int((np.max(np.abs(ref_q15.astype(np.int64))) + 126) // 127), 1)], dtype=np.uint32)
    _Y, Q, _S = proto(A, B, W, QS)
    got = Q.astype(np.float64) * (QS[0] / Q15_16)
    metrics("rms_sq8", got, ref)
