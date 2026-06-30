import math

import torch

from examples.qwen3_int_only.kernels import ATTN_VALUE_SHIFT
from examples.qwen3_int_only.model import Q15_16

MASK = (1 << 16) - 1


def q15(x):
    return torch.clamp(torch.round(x * Q15_16), -(1 << 31), (1 << 31) - 1).to(torch.int32)


def dynamic_quant_q15_torch(x, qmax):
    amax = x.abs().amax(dim=-1).clamp_min(1)
    scale = torch.div(amax, qmax, rounding_mode="floor").clamp_min(1)
    if qmax > 127:
        q = torch.div(x, scale[:, None], rounding_mode="floor").clamp(-32768, 32767).to(torch.int32)
        q = q.clamp(-qmax - 1, qmax)
        return q, scale.to(torch.int32)
    frac_shift = 9 if qmax == 127 else 5 if qmax == 2047 else 4
    idx_shift = torch.full_like(amax, 10)
    for threshold in [4193280, 8386560, 16773120, 33546240, 67092480, 134184960, 268369920, 536739840, 1073479680]:
        idx_shift += (amax > threshold).to(torch.int32)
    idx = torch.clamp(amax >> idx_shift, 1, 4095)
    lut = torch.clamp((qmax << frac_shift) // idx, max=MASK)
    shift = frac_shift + idx_shift
    prod = x.to(torch.int64) * lut[:, None].to(torch.int64)
    q = prod >> shift[:, None]
    q += torch.where(shift[:, None] >= 1, (prod >> torch.clamp(shift[:, None] - 1, min=0)) & 1, torch.zeros_like(q))
    q = q.clamp(-qmax - 1, qmax).to(torch.int32)
    return q, scale.to(torch.int32)


def round_shift_torch(x, shift):
    out = x >> shift
    out += torch.where(shift >= 1, (x >> torch.clamp(shift - 1, min=0)) & 1, torch.zeros_like(out))
    return out


def rmsnorm_i16_proto(x_q15, weight_q15):
    q, _ = dynamic_quant_q15_torch(x_q15, 4095)
    x = q.float()
    y = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True).clamp_min(1.0))
    return q15(y * (weight_q15.float() / Q15_16))


def rmsnorm_i16_kernel_proto(x_q15, weight_q15):
    q, _ = dynamic_quant_q15_torch(x_q15, 4095)
    ss = ((q.int() * q.int()) >> int(math.log2(q.shape[-1]))).sum(dim=-1) + 1
    ns = torch.zeros_like(ss)
    wk = ss.clone()
    for bits, mask in [(16, -65536), (8, 0xFF00), (4, 0xF0)]:
        take = (wk & mask) != 0
        ns += take.int() * bits
        wk = torch.where(take, wk >> bits, wk)
    ns += (((wk & 0xC) != 0).int() * 2)
    idx = torch.clamp(round_shift_torch(ss.to(torch.int64) * 128, ns).int(), -512, 511) + 512
    lut = torch.tensor([0 if i < 640 else min(round(1024.0 / math.sqrt(i / 128.0 - 4.0)), 1023) for i in range(1024)], device=x_q15.device, dtype=torch.int32)
    inv = lut[idx]
    fold = round_shift_torch(inv.to(torch.int64) * 32768, torch.full_like(inv, 5, dtype=torch.int64))
    qt_shift = 6 + (ns >> 1)
    qt_mul = (fold >> 4) & MASK
    prod = q.to(torch.int64) * qt_mul[:, None].to(torch.int64)
    norm = round_shift_torch(prod, qt_shift[:, None])
    return ((norm.int() * (weight_q15.int() >> 8)) >> 2).int()


def linear_dynamic_proto(x_q15, weight_i8, weight_scale_q15, qmax):
    q, xs = dynamic_quant_q15_torch(x_q15, qmax)
    acc = q.float() @ weight_i8.float().T
    scale = xs.float()[:, None] * weight_scale_q15.float()[None, :] / Q15_16
    return q15(acc * scale / Q15_16), q, xs


def linear_i16_kernel_proto(x_q15, weight_i8, weight_scale_q15):
    q, xs = dynamic_quant_q15_torch(x_q15, 32767)
    acc = (q.float() @ weight_i8.float().T).round().int()
    y = ((acc >> 12) * ((xs[:, None].int() * weight_scale_q15[None, :].int()) >> 2)) >> 2
    return y.int(), q, xs


def attention_i12_lut_proto(q_q15, k_q15, v_q15, cfg, out_int=False):
    q, qs = dynamic_quant_q15_torch(q_q15.reshape(-1, cfg.head_dim), 2047)
    k, ks = dynamic_quant_q15_torch(k_q15.reshape(-1, cfg.head_dim), 2047)
    v, vs = dynamic_quant_q15_torch(v_q15.reshape(-1, cfg.head_dim), 2047)
    qh = q.reshape(-1, cfg.num_attention_heads, cfg.head_dim).permute(1, 0, 2).float()
    kh = k.reshape(-1, cfg.num_attention_heads, cfg.head_dim).permute(1, 0, 2).float()
    vh = v.reshape(-1, cfg.num_attention_heads, cfg.head_dim).permute(1, 0, 2).float()
    qsf = qs.reshape(-1, cfg.num_attention_heads).permute(1, 0).double() / Q15_16
    ksf = ks.reshape(-1, cfg.num_attention_heads).permute(1, 0).double() / Q15_16
    score = (qh @ kh.transpose(-1, -2)).double() * qsf[:, :, None] * ksf[:, None, :] / math.sqrt(cfg.head_dim)
    mask = torch.ones(score.shape[-2:], device=score.device, dtype=torch.bool).tril()
    sc = torch.round(score * 64.0).int().masked_fill(~mask, -32768)
    idx = (sc - sc.max(dim=-1, keepdim=True).values).clamp(-4096, 0) + 4096
    lut = torch.round(torch.exp((torch.arange(4097, device=score.device).float() - 4096.0) / 64.0) * 1023.0).clamp(0, 1023).int()
    ex = lut[idx].float()
    denom = ex.sum(dim=-1, keepdim=True)
    if out_int:
        scale = vs.reshape(-1, cfg.num_attention_heads).permute(1, 0).int()
        out_scale = torch.cummax(scale, dim=1).values
        val = vh.float() * scale[:, :, None].float()
        out = torch.floor((ex @ val) / denom / out_scale[:, :, None].float()).clamp(-2048, 2047).int()
        return out.permute(1, 0, 2).reshape(-1, cfg.q_size), out_scale.permute(1, 0).contiguous()
    scale = vs.reshape(-1, cfg.num_attention_heads).permute(1, 0).int()
    val = ((vh.int() * scale[:, :, None]) >> ATTN_VALUE_SHIFT).float()
    out_q15 = torch.floor((ex @ val) / denom).int() << ATTN_VALUE_SHIFT
    return out_q15.permute(1, 0, 2).reshape(-1, cfg.q_size)


def mlp_proto(x_q15, w):
    post = rmsnorm_i16_proto(x_q15, w.post_attention_layernorm)
    gate, _, _ = linear_dynamic_proto(post, w.gate_proj.weight, w.gate_proj.scale, 32767)
    up, _, _ = linear_dynamic_proto(post, w.up_proj.weight, w.up_proj.scale, 32767)
    gated = q15(torch.nn.functional.silu(gate.float() / Q15_16) * (up.float() / Q15_16))
    down, _, _ = linear_dynamic_proto(gated, w.down_proj.weight, w.down_proj.scale, 32767)
    return x_q15 + down
