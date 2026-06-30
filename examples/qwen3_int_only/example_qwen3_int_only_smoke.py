import torch

from examples.qwen3_int_only.kernels import (
    Q15_16,
    compile_kernel,
    dynamic_quant_q15_16,
    exp_lut,
    flash_attention_i8_q15_16,
    linear_dynamic_int8_q15_16,
    linear_dynamic_int16_q15_16,
    mul_q15_16,
    packed_scale_tensor,
    recip_lut_i8,
    recip_lut_i16,
    rmsnorm_q15_16,
    rope_q15_16,
    rsqrt_lut,
    silu_lut,
    silu_q15_16,
)


def q15_16(x):
    return torch.clamp(torch.round(x * Q15_16), -(1 << 31), (1 << 31) - 1).to(torch.int32)


def per_channel_i8_weight(w):
    scale = w.abs().amax(dim=1).clamp(min=1e-6) / 127.0
    return torch.round(w / scale[:, None]).clamp(-128, 127).to(torch.int8), q15_16(scale).to(torch.uint32)


def main():
    rows, hidden, head_dim, intermediate = 32, 128, 32, 64
    token = torch.arange(rows, device="cuda").float()[:, None]
    base = torch.linspace(-1.25, 1.2, hidden, device="cuda")[None, :]
    x = base + 0.08 * torch.sin(token * 0.37)
    xq = q15_16(x)

    rms = compile_kernel(rmsnorm_q15_16(rows, hidden), out_idx=[2])
    dq16 = compile_kernel(dynamic_quant_q15_16(rows, hidden, out_dtype="int16"), out_idx=[2, 3])
    dq8_hidden = compile_kernel(dynamic_quant_q15_16(rows, hidden, out_dtype="int8"), out_idx=[2, 3])
    dq8_head = compile_kernel(dynamic_quant_q15_16(rows, head_dim, out_dtype="int8"), out_idx=[2, 3])
    q_proj = compile_kernel(linear_dynamic_int8_q15_16(rows, hidden, head_dim), out_idx=[4])
    k_proj = compile_kernel(linear_dynamic_int8_q15_16(rows, hidden, head_dim), out_idx=[4])
    v_proj = compile_kernel(linear_dynamic_int8_q15_16(rows, hidden, head_dim), out_idx=[4])
    o_proj = compile_kernel(linear_dynamic_int16_q15_16(rows, head_dim, hidden), out_idx=[4])
    gate_proj = compile_kernel(linear_dynamic_int16_q15_16(rows, hidden, intermediate), out_idx=[4])
    up_proj = compile_kernel(linear_dynamic_int16_q15_16(rows, hidden, intermediate), out_idx=[4])
    down_proj = compile_kernel(linear_dynamic_int16_q15_16(rows, intermediate, hidden), out_idx=[4])
    rope = compile_kernel(rope_q15_16(rows, head_dim), out_idx=[3])
    silu = compile_kernel(silu_q15_16(rows, hidden), out_idx=[2])
    silu_mid = compile_kernel(silu_q15_16(rows, intermediate), out_idx=[2])
    mul = compile_kernel(mul_q15_16(rows, intermediate), out_idx=[2])
    dq16_mid = compile_kernel(dynamic_quant_q15_16(rows, intermediate, out_dtype="int16"), out_idx=[2, 3])

    xr = rms(xq, torch.from_numpy(rsqrt_lut()).cuda())
    x16, s16 = dq16(xr, torch.from_numpy(recip_lut_i16()).cuda())
    x8, xs8 = dq8_hidden(xr, torch.from_numpy(recip_lut_i8()).cuda())
    wq = torch.sin(torch.arange(head_dim * hidden, device="cuda").float() * 0.013).reshape(head_dim, hidden) * 0.18
    wk = torch.cos(torch.arange(head_dim * hidden, device="cuda").float() * 0.011).reshape(head_dim, hidden) * 0.16
    wv = torch.sin(torch.arange(head_dim * hidden, device="cuda").float() * 0.007).reshape(head_dim, hidden) * 0.14
    wo = torch.cos(torch.arange(hidden * head_dim, device="cuda").float() * 0.009).reshape(hidden, head_dim) * 0.12
    wq8, wqs8 = per_channel_i8_weight(wq)
    wk8, wks8 = per_channel_i8_weight(wk)
    wv8, wvs8 = per_channel_i8_weight(wv)
    wo8, wos8 = per_channel_i8_weight(wo)
    q = q_proj(x8, xs8, wq8, wqs8)
    k = k_proj(x8, xs8, wk8, wks8)
    v = v_proj(x8, xs8, wv8, wvs8)
    q8, s8 = dq8_head(q, torch.from_numpy(recip_lut_i8()).cuda())
    k8, _ = dq8_head(k, torch.from_numpy(recip_lut_i8()).cuda())
    v8, vs8 = dq8_head(v, torch.from_numpy(recip_lut_i8()).cuda())

    pos = torch.arange(rows, device="cuda").float()[:, None]
    dim = torch.arange(head_dim // 2, device="cuda").float()[None, :]
    theta = pos * (0.01 + dim * 0.001)
    qr = rope(q, q15_16(torch.cos(theta)), q15_16(torch.sin(theta)))
    kr = rope(k, q15_16(torch.cos(theta)), q15_16(torch.sin(theta)))
    qr8, qrs8 = dq8_head(qr, torch.from_numpy(recip_lut_i8()).cuda())
    kr8, krs8 = dq8_head(kr, torch.from_numpy(recip_lut_i8()).cuda())
    score_scale = float((qrs8.to(torch.float64).mean() * krs8.to(torch.float64).mean()).item()) / (Q15_16 * Q15_16 * (head_dim**0.5)) * 64.0
    attn_scale = int(vs8.to(torch.int64).float().mean().item())
    attn = compile_kernel(flash_attention_i8_q15_16(1, rows, head_dim, block_n=32), out_idx=[6])
    attn_out = attn(
        qr8.reshape(1, rows, head_dim).to(torch.int16),
        kr8.reshape(1, rows, head_dim).to(torch.int16),
        v8.reshape(1, rows, head_dim).to(torch.int16),
        torch.from_numpy(exp_lut()).cuda(),
        torch.from_numpy(packed_scale_tensor(score_scale)).cuda(),
        torch.tensor([attn_scale], device="cuda", dtype=torch.uint32),
    ).reshape(rows, head_dim)
    attn16, attn_s16 = compile_kernel(dynamic_quant_q15_16(rows, head_dim, out_dtype="int16"), out_idx=[2, 3])(attn_out, torch.from_numpy(recip_lut_i16()).cuda())
    o = o_proj(attn16, attn_s16, wo8, wos8)
    act = silu(xr, torch.from_numpy(silu_lut()).cuda())
    wg = torch.cos(torch.arange(intermediate * hidden, device="cuda").float() * 0.017).reshape(intermediate, hidden) * 0.04
    wu = torch.sin(torch.arange(intermediate * hidden, device="cuda").float() * 0.019).reshape(intermediate, hidden) * 0.035
    wd = torch.cos(torch.arange(hidden * intermediate, device="cuda").float() * 0.023).reshape(hidden, intermediate) * 0.06
    wg8, wgs = per_channel_i8_weight(wg)
    wu8, wus = per_channel_i8_weight(wu)
    wd8, wds = per_channel_i8_weight(wd)
    gate = gate_proj(x16, s16, wg8, wgs)
    up = up_proj(x16, s16, wu8, wus)
    gated = mul(silu_mid(gate, torch.from_numpy(silu_lut()).cuda()), up)
    gated16, gs16 = dq16_mid(gated, torch.from_numpy(recip_lut_i16()).cuda())
    mlp = down_proj(gated16, gs16, wd8, wds)

    ref_rms = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True))
    ref_q = ref_rms @ wq.T
    ref_k = ref_rms @ wk.T
    ref_v = ref_rms @ wv.T
    ref_rope_q = torch.cat((ref_q[:, : head_dim // 2] * torch.cos(theta) - ref_q[:, head_dim // 2 :] * torch.sin(theta), ref_q[:, : head_dim // 2] * torch.sin(theta) + ref_q[:, head_dim // 2 :] * torch.cos(theta)), dim=-1)
    ref_rope_k = torch.cat((ref_k[:, : head_dim // 2] * torch.cos(theta) - ref_k[:, head_dim // 2 :] * torch.sin(theta), ref_k[:, : head_dim // 2] * torch.sin(theta) + ref_k[:, head_dim // 2 :] * torch.cos(theta)), dim=-1)
    ref_attn = torch.softmax((ref_rope_q @ ref_rope_k.T) / (head_dim**0.5), dim=-1) @ ref_v
    ref_o = ref_attn @ wo.T
    ref_mlp = (torch.nn.functional.silu(ref_rms @ wg.T) * (ref_rms @ wu.T)) @ wd.T
    torch.testing.assert_close(xr.float() / Q15_16, ref_rms, rtol=0, atol=0.12)
    torch.testing.assert_close(q.float() / Q15_16, ref_q, rtol=0, atol=0.35)
    torch.testing.assert_close(o.float() / Q15_16, ref_o, rtol=0, atol=1.2)
    torch.testing.assert_close(mlp.float() / Q15_16, ref_mlp, rtol=0, atol=0.75)
    print("qwen3 int-only smoke passed", x16.shape, q8.shape, qr.shape, o.shape, act.shape, mlp.shape)


if __name__ == "__main__":
    main()
