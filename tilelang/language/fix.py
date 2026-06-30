from __future__ import annotations

from numbers import Integral, Real

from tvm import tirx
from tvm.tirx import IntImm, PrimExpr

from .tir import ir as T

Q_MULTIPLIER_WIDTH = 16
_Q_MULTIPLIER_MASK = (1 << Q_MULTIPLIER_WIDTH) - 1
_Q_SHIFT_MASK = 0x3F
_INT_RANGES = {
    "int8": (-128, 127),
    "int16": (-32768, 32767),
    "int32": (-(1 << 31), (1 << 31) - 1),
    "uint8": (0, 255),
    "uint16": (0, 65535),
    "uint32": (0, (1 << 32) - 1),
}
_SIGNED_INT_RANGES = {
    "int8": (-128, 127),
    "int10": (-512, 511),
    "int16": (-32768, 32767),
    "int32": (-(1 << 31), (1 << 31) - 1),
}


def quantize_multiplier_like_xprt(real_multiplier: float, precision: int = Q_MULTIPLIER_WIDTH) -> tuple[int, int]:
    real_multiplier = float(real_multiplier)
    if real_multiplier <= 0.0:
        raise ValueError("T.fix.quant scale must be positive")
    shift = precision
    while real_multiplier < 0.5:
        real_multiplier *= 2.0
        shift += 1
    while real_multiplier >= 1.0:
        real_multiplier /= 2.0
        shift -= 1
    return int(round(real_multiplier * ((1 << precision) - 1))), shift


def pack_scale(real_multiplier: float, precision: int = Q_MULTIPLIER_WIDTH) -> int:
    mul, shift = quantize_multiplier_like_xprt(real_multiplier, precision)
    return (int(shift) << Q_MULTIPLIER_WIDTH) | (int(mul) & _Q_MULTIPLIER_MASK)


def _as_i32(value: PrimExpr | int) -> PrimExpr:
    return T.cast(value, "int32")


def _i32(value: int) -> PrimExpr:
    return IntImm("int32", value)


def _is_python_real(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, Integral))


def _scale_to_qt(scale) -> PrimExpr | int:
    return pack_scale(float(scale)) if _is_python_real(scale) else scale


def _unpack_mul_i32(scale_qt) -> PrimExpr:
    return _as_i32(scale_qt) & _i32(_Q_MULTIPLIER_MASK)


def _unpack_shift_i32(scale_qt) -> PrimExpr:
    return (_as_i32(scale_qt) >> _i32(Q_MULTIPLIER_WIDTH)) & _i32(_Q_SHIFT_MASK)


def _saturate_i32(value: PrimExpr, dtype: str) -> PrimExpr:
    lo, hi = _INT_RANGES[str(dtype)]
    return tirx.min(tirx.max(value, _i32(lo)), _i32(hi))


def round_shift(x: PrimExpr, shift, *, rounding: str = "nearest") -> PrimExpr:
    shift_i32 = _as_i32(shift)
    x_i32 = _as_i32(x)
    out = x_i32 >> shift_i32
    if rounding == "nearest":
        bit_shift = tirx.max(shift_i32 - _i32(1), _i32(0))
        out += T.if_then_else(shift_i32 >= _i32(1), (x_i32 >> bit_shift) & _i32(1), _i32(0))
    return out


def saturate(x: PrimExpr, out_dtype: str) -> PrimExpr:
    return T.cast(_saturate_i32(_as_i32(x), str(out_dtype)), str(out_dtype))


def quant(
    x: PrimExpr,
    *,
    out_dtype: str,
    scale,
    rounding: str = "nearest",
    saturate: bool = True,
) -> PrimExpr:
    out_dtype = str(out_dtype)
    scale_qt = _scale_to_qt(scale)
    out = round_shift(_as_i32(x) * _unpack_mul_i32(scale_qt), _unpack_shift_i32(scale_qt), rounding=rounding)
    if saturate and out_dtype != "int32":
        out = _saturate_i32(out, out_dtype)
    return T.cast(out, out_dtype)


def signed_saturate(x: PrimExpr, dtype: str) -> PrimExpr:
    lo, hi = _SIGNED_INT_RANGES[str(dtype)]
    return T.cast(tirx.min(tirx.max(_as_i32(x), _i32(lo)), _i32(hi)), "int32")


def lut(x: PrimExpr, table, *, index_offset: PrimExpr | int = 0, out_dtype: str) -> PrimExpr:
    index = T.cast(_as_i32(x) + _as_i32(index_offset), "int32")
    return T.cast(table[index], str(out_dtype))


def quant_lut(
    x: PrimExpr,
    table,
    *,
    scale,
    index_dtype: str,
    out_dtype: str,
) -> PrimExpr:
    lo, _ = _SIGNED_INT_RANGES[str(index_dtype)]
    q = signed_saturate(quant(x, scale=scale, out_dtype="int32"), str(index_dtype))
    return lut(q, table, index_offset=-lo, out_dtype=str(out_dtype))
