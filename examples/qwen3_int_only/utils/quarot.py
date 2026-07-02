import math

import torch


ROTATE_SEED = 20260515


def random_hadamard_rotation(dim, seed=ROTATE_SEED, device="cuda"):
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.size(0) < dim:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    h = h / math.sqrt(dim)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.where(torch.rand(dim, generator=generator) < 0.5, -1.0, 1.0).to(device)
    return (signs.reshape(-1, 1) * h).contiguous()


def hadamard_rotation(dim, device="cuda"):
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.size(0) < dim:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return (h / math.sqrt(dim)).contiguous()


def rotate_input(weight, rotation):
    return (weight.to(torch.float64) @ rotation.to(weight.device, torch.float64)).to(torch.float32)


def rotate_norm_input(weight, norm_weight, rotation):
    return rotate_input(weight * norm_weight.reshape(1, -1), rotation)


def rotate_output(weight, rotation):
    return (rotation.to(weight.device, torch.float64).T @ weight.to(torch.float64)).to(torch.float32)


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
