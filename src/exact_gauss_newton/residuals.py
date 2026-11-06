"""Residual vector r(params) for a Problem, scaled so 0.5*||r||^2 = 0.5 * sum_terms mean(res^2)."""

from __future__ import annotations

import math

import torch
from torch.func import vmap

from exact_gauss_newton.models import FlatModel
from exact_gauss_newton.problems import Problem


class Residuals:
    CAUSAL_EPS = (1e-2, 1e-1, 1.0, 10.0, 100.0)  # annealed as in Wang, Sankaran & Perdikaris (2022)

    def __init__(self, fm: FlatModel, prob: Problem, causal: bool = False, pde_noise: float = 0.0, seed: int = 0):
        self.fm, self.prob = fm, prob
        self.slices = prob.term_slices()
        self.scales = [1.0 / math.sqrt(t.points.shape[0]) for t in prob.terms]
        self.n_evals = 0  # residual evaluations (one forward+derivative pass over all points)
        # noisy forcing: the PDE residual target is observed with N(0, sigma^2) noise at each collocation
        # point (fixed per seed), so the exact solution no longer has zero residual
        self.noise = {}
        if pde_noise > 0:
            g = torch.Generator().manual_seed(10_000 + seed)
            for t in prob.terms:
                if t.name == "pde":
                    self.noise[t.name] = (pde_noise * torch.randn(t.points.shape[0], generator=g)).to(t.points)
        # domain bounding box (from every term whose points are plain coordinates) for collocation resampling
        pts = torch.cat([t.points for t in prob.terms if t.points.shape[1] == prob.in_dim])
        self.lo, self.hi = pts.amin(0), pts.amax(0)
        self.pde_points0 = [t.points for t in prob.terms if t.name == "pde"]
        self.causal = causal and prob.in_dim == 2
        if self.causal:  # time is coordinate 1; bin the PDE points by t
            i = [k for k, t in enumerate(prob.terms) if t.name == "pde"][0]
            self._causal_slice = self.slices[i][1]
            uniq, inv = torch.unique(prob.terms[i].points[:, 1], return_inverse=True)
            self._causal_bins, self._causal_n = inv, len(uniq)
            self._causal_counts = torch.bincount(inv, minlength=len(uniq)).to(prob.terms[i].points.dtype)
            self.causal_stage = 0

    @torch.no_grad()
    def causal_weights(self, r: torch.Tensor) -> torch.Tensor:
        """Per-residual sqrt-weights: PDE residuals at time t_k get exp(-eps * sum_{j<k} L_j), L_j the
        mean squared raw residual at t_j. Ones for non-PDE terms and when causal weighting is off."""
        w = torch.ones_like(r)
        if not self.causal:
            return w
        rp = r[self._causal_slice]
        L = torch.zeros(self._causal_n, dtype=r.dtype, device=r.device).index_add_(0, self._causal_bins, rp * rp)
        L = L * rp.numel() / self._causal_counts  # undo the 1/sqrt(n) scaling -> mean raw residual^2 per slab
        cum = torch.cumsum(L, 0) - L
        wk = torch.exp(-self.CAUSAL_EPS[self.causal_stage] * cum)
        if wk.min() > 0.99 and self.causal_stage < len(self.CAUSAL_EPS) - 1:
            self.causal_stage += 1
        w[self._causal_slice] = wk[self._causal_bins].sqrt()
        return w

    def resample_pde(self, gen: torch.Generator | None = None):
        """Draw fresh uniform collocation points for every PDE term (same count); the bounding box is the domain.
        Not compatible with noisy forcing (the noise is tied to the fixed points)."""
        assert not self.noise, "resampling with fixed-point noise is undefined"
        for t in self.prob.terms:
            if t.name == "pde":
                u = torch.rand(t.points.shape, generator=gen, dtype=t.points.dtype, device=t.points.device)
                t.points = self.lo + (self.hi - self.lo) * u

    def restore_pde(self):
        k = 0
        for t in self.prob.terms:
            if t.name == "pde":
                t.points = self.pde_points0[k]
                k += 1

    def _term(self, params: dict, term, scale: float) -> torch.Tensor:
        def one(x):
            return term.fn(lambda y: self.fm.u_point(params, y), x)

        out = vmap(one)(term.points)
        if term.name in self.noise:
            out = out + self.noise[term.name]
        return out * scale

    def from_params(self, params: dict) -> torch.Tensor:
        self.n_evals += 1
        return torch.cat([self._term(params, t, s) for t, s in zip(self.prob.terms, self.scales)])

    def from_vec(self, vec: torch.Tensor) -> torch.Tensor:
        return self.from_params(self.fm.unflatten(vec))

    def loss_from_params(self, params: dict) -> torch.Tensor:
        """Training loss (causally weighted when enabled); logged losses stay unweighted."""
        r = self.from_params(params)
        self.last_raw_loss = 0.5 * (r * r).sum().detach()
        if self.causal:
            r = r * self.causal_weights(r.detach())
        return 0.5 * (r * r).sum()

    def term_losses(self, r: torch.Tensor) -> dict[str, float]:
        return {name: 0.5 * (r[sl] * r[sl]).sum().item() for name, sl in self.slices}

    def validation_residuals(self):
        """Residual function on PDE points shifted by half a grid step (detects collocation overfitting)."""
        import copy

        prob = copy.copy(self.prob)
        prob.terms = []
        for t in self.prob.terms:
            if t.name != "pde" or t.points.shape[0] < 4:
                continue
            pts = t.points
            h = (pts[1] - pts[0]) if prob.in_dim == 1 else (pts[1:] - pts[:-1]).abs().amax(0).amin() * torch.ones_like(pts[0])
            prob.terms.append(type(t)(t.name, (pts[:-1] + pts[1:]) / 2 if prob.in_dim == 1 else pts + h / 2, t.fn))
        if not prob.terms:
            return None
        other = Residuals(self.fm, prob)
        return lambda vec: other.from_vec(vec)

    @torch.no_grad()
    def rel_l2(self, params: dict) -> float:
        X = self.prob.eval_points
        u = vmap(lambda x: self.fm.u_point(params, x))(X)
        ue = self.prob.exact(X)
        return (torch.linalg.norm(u - ue) / torch.linalg.norm(ue)).item()
