from __future__ import annotations
import torch
from torch import nn
from torch.func import functional_call

def make_mlp(in_dim: int, width: int, depth: int, dtype, device) -> nn.Sequential:
    layers, d = ([], in_dim)
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers).to(dtype=dtype, device=device)

class FlatModel:

    def __init__(self, model: nn.Module):
        self.model = model
        named = list(model.named_parameters())
        self.names = [n for n, _ in named]
        self.shapes = [p.shape for _, p in named]
        self.sizes = [p.numel() for _, p in named]
        self.flat0 = torch.cat([p.detach().reshape(-1) for _, p in named])
        self.D = self.flat0.numel()

    def unflatten(self, vec: torch.Tensor) -> dict[str, torch.Tensor]:
        out, i = ({}, 0)
        for n, s, k in zip(self.names, self.shapes, self.sizes):
            out[n] = vec[i:i + k].reshape(s)
            i += k
        return out

    def flatten(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([params[n].reshape(-1) for n in self.names])

    def u_point(self, params: dict, x: torch.Tensor) -> torch.Tensor:
        return functional_call(self.model, params, (x.reshape(1, -1),)).reshape(())

    def hidden_matrix_names(self) -> list[str]:
        mats = [n for n, s in zip(self.names, self.shapes) if len(s) == 2]
        return mats[1:-1]

def build(in_dim: int, width: int, depth: int, seed: int, dtype, device) -> FlatModel:
    torch.manual_seed(seed)
    return FlatModel(make_mlp(in_dim, width, depth, dtype, device))
