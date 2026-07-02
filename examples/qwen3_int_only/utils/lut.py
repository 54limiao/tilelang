import math

import numpy as np


def exp_lut_neg():
    return np.array([np.clip(round(math.exp(min(i - 512, 0) / 8.0) * 1023.0), 0, 1023) for i in range(1024)], dtype=np.int16)


def rsqrt_lut():
    return np.array([0 if i < 640 else np.clip(round(1024.0 / math.sqrt(i / 128.0 - 4.0)), 0, 1023) for i in range(1024)], dtype=np.int16)


def sigmoid_lut():
    def sigmoid(x):
        if x <= -7.0:
            return 0.0
        if x >= 7.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    return np.array([round(sigmoid((i - 512) / 64.0) * 1024.0) for i in range(1024)], dtype=np.int32)
