"""PLM: Levenberg–Marquardt on the Gauss–Newton matrix of the PINN residual.

Step: (J^T J + mu * S) d = -J^T r, with S = I (identity damping) or diag(J^T J) (Marquardt).
Solvers:
  dense  form G = J^T J (D x D) and solve directly                       O(N D^2 + D^3)
  dual   d = -S^-1/2 J~^T (J~ J~^T + mu I)^-1 r, J~ = J S^-1/2 (N x N)   O(N^2 D + N^3)
Options: trust region on ||d||, NTK-style term weighting, Adam/Muon warm-up.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch.func import jacrev, linearize, vjp

from exact_gauss_newton.logging_ import Logger
from exact_gauss_newton.lstsq import output_layer_solve
from exact_gauss_newton.models import FlatModel
from exact_gauss_newton.problems.library import LINEAR
from exact_gauss_newton.residuals import Residuals


@dataclass
class PLMConfig:
    mu0: float = 1.0
    damping: str = "identity"  # identity | marquardt
    solver: str = "auto"  # dense | dual | auto (dual when N < D) | cg (matrix-free, never forms J)
    cg_iters: int = 50  # matrix-free: max PCG iterations per solve
    cg_tol: float = 1e-2  # matrix-free: relative residual tolerance (inexact Newton)
    probes: int = 4  # matrix-free: Hutchinson probes for diag(J^T J) and per-term ||J_k||_F^2
    trust_radius: float | None = None  # initial Delta; None disables
    term_weighting: bool = False
    weight_every: int = 10
    warmup: str = "none"  # none | adamw | muon
    warmup_frac: float = 0.1
    max_retries: int = 12
    mu_min: float = 1e-16
    lr: float = 1e-3  # warm-up lr
    muon_lr: float = 0.02
    marquardt_floor: float = 1e-3  # s_i >= floor * max(s): stops tiny-curvature directions exploding
    reuse_jac: int = 1  # inner LM steps per Jacobian (lagged Jacobian); 1 = classic
    val_points: bool = True  # log residual on a held-out shifted grid to detect collocation overfitting
    restart_after: int = 0  # restart (AdamW from the best iterate) after this many consecutive Jacobians with no
    #   accepted step, or with raw loss above restart_on_blowup x best (patience: LM's early transient is allowed)
    restart_on_blowup: float = 0.0  # 0 = off
    restart_frac: float = 0.05  # budget fraction for each restart's AdamW phase
    varpro: bool = False  # variable projection: output layer solved exactly (least squares / output-layer GN) at every
    #   trial point; LM acts on the hidden layers through the projected Jacobian (I - Q Q^T) J_h (Kaufman 1975)
    out_steps: int = 0  # output-layer GN steps per solve; 0 = auto (1 if the PDE is linear in u, else 6)
    final_solve: bool = False  # two-stage: after the LM budget, one output-layer solve (least squares / GN) on the learned features
    final_steps: int = 20  # output-layer GN steps for that solve (one is exact for a PDE linear in u)
    budget_frac: float = 1.0  # fraction of the job budget the LM stage may use (the rest is left unused / for the final solve)
    stop_loss: float = 1e-26  # stop once the raw loss is at fp64 roundoff (varpro reaches it in a few Jacobians)
    resample: bool = False  # fresh uniform collocation points for the PDE term at every Jacobian (the logged loss and
    #   the validation residual stay on the original fixed grid)

    @classmethod
    def from_dict(cls, d: dict) -> "PLMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class NotPositiveDefinite(RuntimeError):
    pass


def _solve(J: torch.Tensor, r: torch.Tensor, mu: float, s: torch.Tensor | None, dense: bool) -> torch.Tensor:
    """Return d solving (J^T J + mu S) d = -J^T r, S = diag(s) (s=None -> identity)."""
    N, D = J.shape
    if s is not None:
        Jt = J / torch.sqrt(s)  # J S^{-1/2}
    else:
        Jt = J
    if dense:
        G = Jt.T @ Jt
        G.diagonal().add_(mu)
        dt = torch.linalg.solve(G, -(Jt.T @ r))
    else:
        K = Jt @ Jt.T
        K.diagonal().add_(mu)
        L, info = torch.linalg.cholesky_ex(K)
        if info.item() != 0:  # K + mu I lost positive-definiteness in this precision (fp32: mu < eps*||K||)
            raise NotPositiveDefinite(mu)
        y = torch.cholesky_solve(r.reshape(-1, 1), L).reshape(-1)
        dt = -(Jt.T @ y)
    return dt / torch.sqrt(s) if s is not None else dt


class ExplicitJac:
    """Materialized N x D Jacobian; exact solve in dense (D x D) or dual (N x N) form."""

    def __init__(self, J: torch.Tensor, dense: bool):
        self.J, self.dense = J, dense
        self.cg_iters = 0

    def term_fro(self, slices):
        return torch.stack([(self.J[sl] ** 2).sum() for _, sl in slices])

    def weighted(self, wt):
        self.Jw = self.J * wt[:, None]

    def diag(self):
        return (self.Jw * self.Jw).sum(0)

    def solve(self, rw, mu, s):
        return _solve(self.Jw, rw, mu, s, self.dense)

    def lin(self, rw, d):
        return rw + self.Jw @ d


class MatrixFreeJac:
    """Never forms J: J v by forward-mode (linearize), J^T u by reverse-mode (vjp); the damped
    normal equations are solved by Jacobi-preconditioned CG. Memory O(D), time O(cg_iters) passes."""

    def __init__(self, fn, vec: torch.Tensor, log: Logger, cfg: PLMConfig):
        self.r, self._jvp = linearize(fn, vec)
        _, self._vjp = vjp(fn, vec)
        self.log, self.cfg = log, cfg
        self.N, self.D = self.r.shape[0], vec.shape[0]
        self.cg_iters = 0

    def Jv(self, v):
        self.log.count_grad()
        return self._jvp(v)

    def JTu(self, u):
        self.log.count_grad()
        return self._vjp(u)[0]

    def term_fro(self, slices):  # Hutchinson in parameter space: ||J_k||_F^2 = E ||J_k v||^2, v ~ N(0, I)
        acc = torch.zeros(len(slices), dtype=self.r.dtype, device=self.r.device)
        for _ in range(self.cfg.probes):
            Jv = self.Jv(torch.randn(self.D, dtype=self.r.dtype, device=self.r.device))
            acc += torch.stack([(Jv[sl] ** 2).sum() for _, sl in slices])
        return acc / self.cfg.probes

    def weighted(self, wt):
        self.wt = wt
        acc = torch.zeros(self.D, dtype=self.r.dtype, device=self.r.device)
        for _ in range(self.cfg.probes):  # Hutchinson in residual space: diag(Jw^T Jw) = E (Jw^T z)^2, z Rademacher
            z = torch.randn(self.N, dtype=self.r.dtype, device=self.r.device).sign() * wt
            acc += self.JTu(z) ** 2
        self.dhat = acc / self.cfg.probes

    def diag(self):
        return self.dhat

    def solve(self, rw, mu, s):
        sv = s if s is not None else torch.ones_like(self.dhat)
        b = -self.JTu(self.wt * rw)
        Minv = 1.0 / (self.dhat + mu * sv).clamp_min(1e-30)
        A = lambda v: self.JTu(self.wt * self.wt * self.Jv(v)) + mu * sv * v
        d = torch.zeros_like(b)
        rres = b.clone()
        z = Minv * rres
        pdir = z.clone()
        rz = rres @ z
        bnorm = b.norm().item()
        for k in range(self.cfg.cg_iters):
            Ap = A(pdir)
            alpha = rz / (pdir @ Ap)
            d = d + alpha * pdir
            rres = rres - alpha * Ap
            if rres.norm().item() <= self.cfg.cg_tol * bnorm:
                break
            z = Minv * rres
            rz_new = rres @ z
            pdir = z + (rz_new / rz) * pdir
            rz = rz_new
        self.cg_iters += k + 1
        return d

    def lin(self, rw, d):
        return rw + self.wt * self.Jv(d)


def run_plm(fm: FlatModel, res: Residuals, log: Logger, budget: float, cfg: PLMConfig) -> torch.Tensor:
    from exact_gauss_newton.first_order import run_first_order

    vec = fm.flat0.clone()
    if cfg.warmup != "none":
        vec = run_first_order(cfg.warmup, fm, res, log, budget * cfg.warmup_frac,
                              {"lr": cfg.lr, "muon_lr": cfg.muon_lr}, vec0=vec, stage=f"warmup_{cfg.warmup}")

    N, D = res.prob.n_residuals, fm.D
    dense = cfg.solver == "dense" or (cfg.solver == "auto" and N >= D)
    matrix_free = cfg.solver == "cg"
    jac = None if matrix_free else jacrev(res.from_vec)
    mu, delta = cfg.mu0, cfg.trust_radius
    slices = res.slices
    w = torch.ones(N, dtype=vec.dtype, device=vec.device)
    it, rejected, restarts, stalled, blown = 0, 0, 0, 0, 0
    best_loss, best_vec = float("inf"), vec.clone()

    val = res.validation_residuals() if cfg.val_points else None
    k_out = fm.sizes[-2] + fm.sizes[-1]  # output layer = last Linear (weight, bias)
    out_steps = cfg.out_steps or (1 if res.prob.name in LINEAR else 6)
    fixed_loss = None
    assert not (matrix_free and (cfg.varpro or cfg.resample)), "varpro/resample need the explicit-Jacobian solvers"
    if cfg.resample:
        import copy
        fixed = Residuals(fm, copy.copy(res.prob))  # shares the fixed point tensors; res.prob gets new ones
        fixed.prob.terms = [copy.copy(t) for t in res.prob.terms]
        fixed_loss = lambda v: 0.5 * (lambda rr: (rr @ rr).item())(fixed.from_vec(v))
        gen = torch.Generator(device=vec.device).manual_seed(777)

    def trial(v):
        """Residual at a trial point; with variable projection the output layer is re-solved first."""
        if cfg.varpro:
            return output_layer_solve(fm, res, v, max_steps=out_steps, log=log)
        return v, res.from_vec(v)

    budget_lm = budget * cfg.budget_frac
    log.start()
    with torch.no_grad():
        vec, r_raw = trial(vec)
        while log.elapsed() < budget_lm:
            if cfg.resample:
                res.resample_pde(gen)
                vec, r_raw = trial(vec)
            raw_loss = 0.5 * (r_raw @ r_raw).item()
            if raw_loss < cfg.stop_loss:
                break
            if raw_loss < best_loss:
                best_loss, best_vec = raw_loss, vec.clone()
            blown = blown + 1 if (cfg.restart_on_blowup > 0 and raw_loss > cfg.restart_on_blowup * best_loss) else 0
            if cfg.restart_after > 0 and max(stalled, blown) >= cfg.restart_after:
                log.pause()  # run_first_order compares against the cumulative clock, so pass a deadline
                with torch.enable_grad():  # the LM loop runs under no_grad
                    vec = run_first_order("adamw", fm, res, log, log.elapsed() + budget * cfg.restart_frac,
                                          {"lr": cfg.lr, "muon_lr": cfg.muon_lr}, vec0=best_vec.clone(), stage=f"restart{restarts}")
                log.start()
                mu, stalled, blown, restarts = cfg.mu0, 0, 0, restarts + 1
                vec, r_raw = trial(vec)
                if log.elapsed() >= budget:
                    break
            if matrix_free:
                lin_ = MatrixFreeJac(res.from_vec, vec, log, cfg)
            elif cfg.varpro:  # Kaufman's projected Jacobian: J_h - Q Q^T J_h, Q = orthonormal range of J_out
                J = jac(vec)
                Jo, Jh = J[:, D - k_out:], J[:, : D - k_out]
                U, S_, _ = torch.linalg.svd(Jo, full_matrices=False)
                Q = U[:, S_ > 1e-12 * S_.max()]
                lin_ = ExplicitJac(Jh - Q @ (Q.T @ Jh), cfg.solver == "dense" or (cfg.solver == "auto" and N >= D - k_out))
            else:
                lin_ = ExplicitJac(jac(vec), dense)
            log.count_jac()
            if cfg.term_weighting and it % cfg.weight_every == 0:
                # NTK-style: weight each term so its share of trace(J^T J) is equal
                fro = lin_.term_fro(slices).clamp_min(1e-30)
                lam = (fro.sum() / len(fro)) / fro
                for (_, sl), l in zip(slices, lam):
                    w[sl] = l.sqrt().clamp(1e-2, 1e2)
            wt = w * res.causal_weights(r_raw) if res.causal else w.clone()
            lin_.weighted(wt)
            if cfg.damping == "marquardt":
                s = lin_.diag()
                s = s.clamp_min(cfg.marquardt_floor * s.max())
            else:
                s = None
            # inner loop: up to `reuse_jac` LM steps with the same (lagged) Jacobian
            for inner in range(cfg.reuse_jac):
                rw = r_raw * wt
                loss_w = 0.5 * (rw @ rw).item()
                accepted = False
                for _ in range(cfg.max_retries):
                    try:
                        d = lin_.solve(rw, mu, s)
                    except NotPositiveDefinite:  # treat like a rejected step: more damping
                        mu *= 4.0
                        rejected += 1
                        continue
                    if delta is not None and torch.linalg.norm(d).item() > delta:
                        mu *= 4.0
                        rejected += 1
                        continue
                    if cfg.varpro:
                        d = torch.cat([d, torch.zeros(k_out, dtype=d.dtype, device=d.device)])
                    vec_new, r_new = trial(vec + d)
                    rw_new = r_new * wt
                    actual = loss_w - 0.5 * (rw_new @ rw_new).item()
                    lin = lin_.lin(rw, d[: D - k_out] if cfg.varpro else d)
                    predicted = loss_w - 0.5 * (lin @ lin).item()
                    rho = actual / predicted if predicted > 0 else -1.0
                    if rho > 1e-3 and torch.isfinite(r_new).all():
                        vec, r_raw = vec_new, r_new
                        nd = torch.linalg.norm(d).item()
                        if rho > 0.75:
                            mu = max(mu / 3.0, cfg.mu_min)
                            if delta is not None:
                                delta = max(delta, 2.0 * nd)
                        elif rho < 0.25 and delta is not None:
                            delta = nd / 4.0
                        accepted = True
                        break
                    mu *= 4.0
                    rejected += 1
                it += 1
                loss = fixed_loss(vec) if fixed_loss else 0.5 * (r_raw @ r_raw).item()
                extra = {"mu": mu, "rejected": rejected, "accepted": accepted, "inner": inner, "restarts": restarts, "cg_iters": lin_.cg_iters,
                         **{f"loss_{k}": v for k, v in res.term_losses(r_raw).items()}}
                if val is not None:
                    rv = val(vec)
                    extra["val_loss"] = 0.5 * (rv @ rv).item()
                log.log(it, vec, loss, "plm", res, extra)
                if not accepted or rho < 0.25:
                    break  # stale Jacobian no longer useful; recompute
            stalled = 0 if accepted else stalled + 1
            if not accepted and mu > 1e12 and cfg.restart_after == 0:
                break
    if cfg.resample:
        res.restore_pde()
    if cfg.final_solve:  # stage 2: exact output layer on the learned basis (on the original grid)
        with torch.no_grad():
            vec, r_fin = output_layer_solve(fm, res, vec, max_steps=cfg.final_steps, log=log)
            log.count_jac()
            loss = 0.5 * (r_fin @ r_fin).item()
            extra = {"mu": mu, "rejected": rejected, "accepted": True, "inner": 0, "restarts": restarts, "cg_iters": 0,
                     **{f"loss_{k}": v for k, v in res.term_losses(r_fin).items()}}
            if val is not None:
                rv = val(vec)
                extra["val_loss"] = 0.5 * (rv @ rv).item()
            log.log(it + 1, vec, loss, "final_solve", res, extra)
    log.pause()
    return vec
