"""PDE benchmark problems."""

from exact_gauss_newton.problems.base import Problem, Term, to_device
from exact_gauss_newton.problems.library import BUILDERS


def get_problem(name: str, device="cpu", dtype=None, **knobs) -> Problem:
    import torch

    prob = BUILDERS[name](**knobs)
    return to_device(prob, device, dtype or torch.float64)


__all__ = ["Problem", "Term", "BUILDERS", "get_problem"]
