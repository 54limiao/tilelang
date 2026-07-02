import math

import numpy as np
import torch

Q15_16 = 1 << 16
Q_MULTIPLIER_WIDTH = 26
MASK = (1 << Q_MULTIPLIER_WIDTH) - 1
I32_MIN = -(1 << 31)


def i32(x):
    return np.asarray(x, dtype=np.int64).clip(-(1 << 31), (1 << 31) - 1).astype(np.int32)


def q15(x):
    return i32(np.rint(np.asarray(x, dtype=np.float64) * Q15_16))


def pack_scale(real_multiplier):
    real_multiplier = float(real_multiplier)
    if real_multiplier <= 0.0:
        raise ValueError("scale must be positive")
    best_mul, best_shift, best_err = 0, 0, float("inf")
    for shift in range(64):
        mul = int(round(real_multiplier * (1 << shift)))
        if 1 <= mul < (1 << Q_MULTIPLIER_WIDTH):
            err = abs(real_multiplier - (mul / float(1 << shift)))
            if err <= best_err:
                best_mul, best_shift, best_err = mul, shift, err
    if best_mul == 0:
        raise ValueError("scale cannot be represented")
    return (best_shift << Q_MULTIPLIER_WIDTH) | (best_mul & MASK)


def round_shift(x, shift):
    x = np.asarray(x, dtype=np.int64)
    shift = int(shift)
    out = x >> shift
    if shift >= 1:
        out += (x >> (shift - 1)) & 1
    return out


def fix_quant(x, scale, out_dtype="int32"):
    scale = pack_scale(scale) if isinstance(scale, float) else (int(scale) & 0xFFFFFFFF)
    mul = scale & MASK
    shift = (scale >> Q_MULTIPLIER_WIDTH) & 0x3F
    out = round_shift(np.asarray(x, dtype=np.int64) * mul, shift)
    if out_dtype == "int8":
        out = np.clip(out, -128, 127)
    elif out_dtype == "int16":
        out = np.clip(out, -32768, 32767)
    elif out_dtype == "int10":
        out = np.clip(out, -512, 511)
    return out.astype(np.dtype(out_dtype))


def lut_10bit(x, table, scale, out_dtype="int32"):
    q = fix_quant(x, scale, "int32").astype(np.int64)
    q = np.clip(q, -512, 511) + 512
    return np.asarray(table)[q].astype(np.dtype(out_dtype))


def static_quant_signed(x, scale, qmax):
    scale = max(int(scale), 1)
    x = np.asarray(x, dtype=np.int64)
    y = (np.abs(x) + (scale >> 1)) // scale
    y = np.where(x < 0, -y, y)
    return np.clip(y, -qmax - 1, qmax)


def metrics(name, got, ref):
    got = np.asarray(got, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    diff = got - ref
    cos = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    mse = float(np.mean(diff * diff))
    max_abs = float(np.max(np.abs(diff)))
    print(f"{name:14s} cos={cos:.8f} mse={mse:.8e} max_abs={max_abs:.8e}")


def rsqrt_lut_np():
    return np.array([0 if i < 640 else np.clip(round(1024.0 / math.sqrt(i / 128.0 - 4.0)), 0, 1023) for i in range(1024)], dtype=np.int16)


def sigmoid_lut_np():
    return np.array([round((1.0 / (1.0 + math.exp(-min(max((i - 512) / 64.0, -7.0), 7.0)))) * 1024.0) for i in range(1024)], dtype=np.int32)


def exp_lut_neg_np():
    return np.array([np.clip(round(math.exp(min(i - 512, 0) / 8.0) * 1023.0), 0, 1023) for i in range(1024)], dtype=np.int16)


def torch_rmsnorm(x, weight):
    x = torch.as_tensor(x, dtype=torch.float32)
    weight = torch.as_tensor(weight, dtype=torch.float32)
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6) * weight


def fwht(x):
    arr = np.asarray(x)
    y = arr.astype(np.float64 if np.issubdtype(arr.dtype, np.floating) else np.int64).copy()
    step = 1
    while step < y.shape[-1]:
        for base in range(0, y.shape[-1], step * 2):
            a = y[..., base : base + step].copy()
            b = y[..., base + step : base + step * 2].copy()
            y[..., base : base + step] = a + b
            y[..., base + step : base + step * 2] = a - b
        step *= 2
    return y
