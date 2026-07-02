from __future__ import annotations

from numbers import Integral, Real

from tvm import tirx
from tvm.tirx import IntImm, PrimExpr

from .tir import ir as T

Q_MULTIPLIER_WIDTH = 26
_Q_MULTIPLIER_MASK = (1 << Q_MULTIPLIER_WIDTH) - 1
_Q_SHIFT_MASK = 0x3F
_INT32_MIN = -(1 << 31)
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
_XP5_QUANT_DTYPES = {
    ("int8", "int32"),
    ("int16", "int32"),
    ("int32", "int32"),
    ("int32", "int8"),
    ("int32", "int16"),
}
_XP5_MUL_BITS = 26
_XP5_SHIFT_BITS = 6


def quantize_multiplier_like_xp5(real_multiplier: float) -> tuple[int, int]:
    real_multiplier = float(real_multiplier)
    if real_multiplier <= 0.0:
        raise ValueError("T.fix.quant scale must be positive")
    best_mul = 0
    best_shift = 0
    best_err = float("inf")
    for shift in range(1 << _XP5_SHIFT_BITS):
        mul = int(round(real_multiplier * (1 << shift)))
        if 1 <= mul < (1 << _XP5_MUL_BITS):
            err = abs(real_multiplier - (mul / float(1 << shift)))
            if err <= best_err:
                best_mul = mul
                best_shift = shift
                best_err = err
    if best_mul == 0:
        raise ValueError("T.fix.quant scale cannot be represented by XP5 mul/shift")
    return best_mul, best_shift


def pack_scale(real_multiplier: float) -> int:
    mul, shift = quantize_multiplier_like_xp5(real_multiplier)
    qt = (int(shift) << Q_MULTIPLIER_WIDTH) | (int(mul) & _Q_MULTIPLIER_MASK)
    return qt if qt < (1 << 31) else qt - (1 << 32)


def _as_i32(value: PrimExpr | int) -> PrimExpr:
    return T.cast(value, "int32")


def _i32(value: int) -> PrimExpr:
    return IntImm("int32", value)


def _i64(value: int) -> PrimExpr:
    return IntImm("int64", value)


def _dtype_of(value) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


def _check_int_range(name: str, value, lo: int, hi: int) -> None:
    if isinstance(value, Integral) and not (lo <= int(value) <= hi):
        raise ValueError(f"T.fix.quant {name} must be in [{lo}, {hi}]")


def _unpack_mul_i32(scale_qt) -> PrimExpr:
    return _as_i32(scale_qt) & _i32((1 << _XP5_MUL_BITS) - 1)


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


def div(
    x: PrimExpr,
    *,
    scale,
    out_dtype: str,
    saturate: bool = True,
) -> PrimExpr:
    scale_i32 = _as_i32(scale)
    x_i32 = _as_i32(x)
    scale_i64 = T.cast(scale_i32, "int64")
    x_i64 = T.cast(x_i32, "int64")
    out = T.if_then_else(
        scale_i32 == _i32(0),
        _i64(0),
        T.if_then_else(
            (x_i64 == _i64(_INT32_MIN)) & (scale_i64 == _i64(-1)),
            _i64(_INT32_MIN),
            T.truncdiv(x_i64, scale_i64),
        ),
    )
    out = T.cast(out, "int32")
    if saturate and str(out_dtype) != "int32":
        out = _saturate_i32(out, str(out_dtype))
    return T.cast(out, str(out_dtype))


def _wide_round_shift(x: PrimExpr, mul, shift, *, rounding: str) -> PrimExpr:
    shift_i32 = _as_i32(shift)
    x_i64 = T.cast(x, "int64")
    prod = x_i64 * T.cast(mul, "int64")
    out = prod >> shift_i32
    if rounding == "nearest":
        bit_shift = tirx.max(shift_i32 - _i32(1), _i32(0))
        out += T.if_then_else(shift_i32 >= _i32(1), T.cast((prod >> bit_shift) & T.cast(1, "int64"), "int64"), T.cast(0, "int64"))
    return T.cast(out, "int32")


def _check_scale(scale) -> None:
    if isinstance(scale, Real) and not isinstance(scale, (bool, Integral)):
        raise TypeError("T.fix.quant scale must be packed int32, not float")
    if isinstance(scale, Integral):
        scale = int(scale)
        if not (_INT32_MIN <= scale <= 0xFFFFFFFF):
            raise ValueError("T.fix.quant packed scale must fit in int32/uint32")
        qt = scale & 0xFFFFFFFF
        mul = qt & ((1 << Q_MULTIPLIER_WIDTH) - 1)
        shift = (qt >> Q_MULTIPLIER_WIDTH) & _Q_SHIFT_MASK
        _check_int_range("scale.mul", mul, 1, (1 << _XP5_MUL_BITS) - 1)
        _check_int_range("scale.shift", shift, 0, (1 << _XP5_SHIFT_BITS) - 1)


def _quant_scale(x: PrimExpr, scale, out_dtype: str, rounding: str, do_saturate: bool) -> PrimExpr:
    in_dtype = _dtype_of(x)
    if in_dtype is not None and (in_dtype, out_dtype) not in _XP5_QUANT_DTYPES:
        raise TypeError(f"T.fix.quant only supports XP5 vquant dtypes: {sorted(_XP5_QUANT_DTYPES)}")
    _check_scale(scale)
    out = _wide_round_shift(x, _unpack_mul_i32(scale), _unpack_shift_i32(scale), rounding=rounding)
    if do_saturate and out_dtype != "int32":
        out = _saturate_i32(out, out_dtype)
    return T.cast(out, out_dtype)


def quant(
    x: PrimExpr,
    *,
    out_dtype: str,
    scale,
    rounding: str = "nearest",
    saturate: bool = True,
) -> PrimExpr:
    out_dtype = str(out_dtype)
    return _quant_scale(x, scale, out_dtype, rounding, saturate)


def quant_i32(x: PrimExpr, *, scale, rounding: str = "nearest") -> PrimExpr:
    return _quant_scale(x, scale, "int32", rounding, False)


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
    return lut_10bit(x, table, scale=scale, out_dtype=out_dtype)


def lut_10bit(
    x: PrimExpr,
    table,
    *,
    scale,
    out_dtype: str,
) -> PrimExpr:
    q = signed_saturate(quant(x, scale=scale, out_dtype="int32"), "int10")
    return lut(q, table, index_offset=512, out_dtype=str(out_dtype))
