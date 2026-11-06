from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable
import torch
from torch.func import grad
PI = math.pi

def d(u: Callable, i: int) -> Callable:
    g = grad(u)
    return lambda x: g(x)[i]

def dn(u: Callable, i: int, n: int) -> Callable:
    for _ in range(n):
        u = d(u, i)
    return u

@dataclass
class Term:
    name: str
    points: torch.Tensor
    fn: Callable

@dataclass
class Problem:
    name: str
    in_dim: int
    terms: list[Term]
    exact: Callable | None
    eval_points: torch.Tensor
    knobs: dict = field(default_factory=dict)

    @property
    def n_residuals(self) -> int:
        return sum((t.points.shape[0] for t in self.terms))

    def term_slices(self) -> list[tuple[str, slice]]:
        out, i = ([], 0)
        for t in self.terms:
            n = t.points.shape[0]
            out.append((t.name, slice(i, i + n)))
            i += n
        return out

def grid(domain: list[tuple[float, float]], n: list[int], interior: bool, dtype, device) -> torch.Tensor:
    axes = []
    for (lo, hi), k in zip(domain, n):
        a = torch.linspace(lo, hi, k + 2 if interior else k, dtype=dtype, device=device)
        axes.append(a[1:-1] if interior else a)
    mesh = torch.meshgrid(*axes, indexing='ij')
    return torch.stack([m.reshape(-1) for m in mesh], dim=1)

def to_device(prob: Problem, device, dtype) -> Problem:
    for t in prob.terms:
        t.points = t.points.to(device=device, dtype=dtype)
    prob.eval_points = prob.eval_points.to(device=device, dtype=dtype)
    return prob
