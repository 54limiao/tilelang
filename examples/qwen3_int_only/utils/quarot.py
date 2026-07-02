import math

import torch


ROTATE_SEED = 20260515


def random_hadamard_rotation(dim, seed=ROTATE_SEED, device="cuda"):
    base_dim = dim if dim & (dim - 1) == 0 else 128
    h = hadamard_rotation(base_dim, device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.where(torch.rand(base_dim, generator=generator) < 0.5, -1.0, 1.0).to(device)
    return (signs.reshape(-1, 1) * h).contiguous()


def hadamard_rotation(dim, device="cuda"):
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.size(0) < dim:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return (h / math.sqrt(dim)).contiguous()


def fast_hadamard(x, block_dim=None):
    shape = x.shape
    n = shape[-1] if block_dim is None else block_dim
    y = x.reshape(-1, n).to(torch.float32)
    step = 1
    while step < n:
        y = y.reshape(-1, n // (step * 2), step * 2)
        a = y[..., :step].clone()
        b = y[..., step:]
        y[..., :step] = a + b
        y[..., step:] = a - b
        y = y.reshape(-1, n)
        step *= 2
    return (y * (1.0 / math.sqrt(n))).reshape(shape).contiguous()


def _rotate_input(weight, rotation):
    if weight.shape[-1] == rotation.shape[0]:
        return weight.to(torch.float64) @ rotation.to(weight.device, torch.float64)
    block_dim = rotation.shape[0]
    shape = weight.shape
    w = weight.to(torch.float64).reshape(-1, shape[-1] // block_dim, block_dim)
    return (w @ rotation.to(weight.device, torch.float64)).reshape(shape)


def _rotate_output(weight, rotation):
    if weight.shape[0] == rotation.shape[0]:
        return rotation.to(weight.device, torch.float64).T @ weight.to(torch.float64)
    block_dim = rotation.shape[0]
    shape = weight.shape
    w = weight.to(torch.float64).reshape(shape[0] // block_dim, block_dim, -1)
    return torch.einsum("ab,nbc->nac", rotation.to(weight.device, torch.float64).T, w).reshape(shape)


def rotate_input(weight, rotation):
    return _rotate_input(weight, rotation).to(torch.float32)


def rotate_norm_input(weight, norm_weight, rotation):
    return rotate_input(weight * norm_weight.reshape(1, -1), rotation)


def rotate_output(weight, rotation):
    return _rotate_output(weight, rotation).to(torch.float32)


def rotate_head_output(weight, head_dim, rotation):
    w_t = weight.to(torch.float64).T
    shape = w_t.shape
    w_t = w_t.reshape(-1, shape[-1] // head_dim, head_dim)
    w_t = (w_t @ rotation.to(weight.device, torch.float64)).reshape(shape)
    return w_t.T.to(torch.float32)


def rotate_head_input(weight, head_dim, rotation):
    shape = weight.shape
    w = weight.to(torch.float64).reshape(-1, shape[-1] // head_dim, head_dim)
    return (w @ rotation.to(weight.device, torch.float64)).reshape(shape).to(torch.float32)


def rotate_block_input(weight, block_dim, rotation):
    shape = weight.shape
    w = weight.to(torch.float64).reshape(-1, shape[-1] // block_dim, block_dim)
    return (w @ rotation.to(weight.device, torch.float64)).reshape(shape).to(torch.float32)
