from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.func import grad, jvp
from exact_gauss_newton.logging_ import Logger
from exact_gauss_newton.models import FlatModel
from exact_gauss_newton.residuals import Residuals

@dataclass
class NNCGConfig:
    mu: float = 0.01
    sketch_size: int = 60
    update_freq: int = 20
    cg_iters: int = 100
    cg_tol: float = 1e-06
    lr: float = 1.0
    armijo_alpha: float = 0.1
    armijo_beta: float = 0.5
    max_ls: int = 25
    warmup: str = 'adamw'
    warmup_frac: float = 0.1
    warmup_lr: float = 0.001

    @classmethod
    def from_dict(cls, d: dict) -> 'NNCGConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

def _nystrom_sketch(hvp_fn, D: int, s: int, dtype: torch.dtype, device: torch.device, gen: torch.Generator):
    s = min(s, D)
    S = torch.randn(D, s, dtype=dtype, device=device, generator=gen)
    Q, _ = torch.linalg.qr(S, mode='reduced')
    cols = [hvp_fn(Q[:, i]) for i in range(s)]
    Y = torch.stack(cols, dim=1)
    eps = torch.finfo(dtype).eps
    nu = D ** 0.5 * eps * torch.linalg.matrix_norm(Y, ord=2).clamp_min(1e-30)
    Y_nu = Y + nu * Q
    QtY = 0.5 * (Q.T @ Y_nu + (Q.T @ Y_nu).T)
    lam_shift = 0.0
    L, info = torch.linalg.cholesky_ex(QtY)
    if info.item() == 0:
        B = torch.linalg.solve_triangular(L.T, Y_nu.T, upper=True).T
        indefinite = False
    else:
        evals, W = torch.linalg.eigh(QtY)
        lam_min = evals.min().item()
        lam_shift = abs(min(lam_min, 0.0)) + 1e-12
        R = W @ torch.diag((evals + lam_shift).clamp_min(1e-12).rsqrt()) @ W.T
        B = Y_nu @ R
        indefinite = True
    U_hat, Sigma, _ = torch.linalg.svd(B, full_matrices=False)
    Lambda_hat = torch.clamp(Sigma ** 2 - (nu + lam_shift), min=0.0)
    return (U_hat, Lambda_hat, indefinite)

def _nystrom_pcg(hvp_fn, g: torch.Tensor, d0: torch.Tensor, U: torch.Tensor, Lam: torch.Tensor, mu: float, tol: float, max_iters: int):
    lam_s = Lam[-1].item()

    def P_inv(r: torch.Tensor) -> torch.Tensor:
        Ut_r = U.T @ r
        return (lam_s + mu) * (U @ (Ut_r / (Lam + mu))) + (r - U @ Ut_r)

    def A_mv(v: torch.Tensor) -> torch.Tensor:
        return hvp_fn(v) + mu * v
    x = d0.clone()
    r = g - A_mv(x) if x.norm().item() > 0 else g.clone()
    z = P_inv(r)
    p = z.clone()
    rz = (r @ z).item()
    neg_curv = False
    k = 0
    for k in range(max_iters):
        if r.norm().item() < tol:
            break
        v = A_mv(p)
        pv = (p @ v).item()
        if pv <= 1e-30:
            neg_curv = True
            if k == 0:
                x = z.clone()
            break
        alpha = rz / pv
        x = x + alpha * p
        r = r - alpha * v
        z = P_inv(r)
        rz_new = (r @ z).item()
        beta = rz_new / max(rz, 1e-30)
        p = z + beta * p
        rz = rz_new
    return (x, k + 1, neg_curv)

def run_nncg(fm: FlatModel, res: Residuals, log: Logger, budget: float, cfg: NNCGConfig) -> torch.Tensor:
    from exact_gauss_newton.first_order import run_first_order, run_two_stage
    vec = fm.flat0.clone()
    if cfg.warmup == 'adamw':
        vec = run_first_order('adamw', fm, res, log, budget * cfg.warmup_frac, {'lr': cfg.warmup_lr}, vec0=vec, stage='warmup_adamw')
    elif cfg.warmup == 'adamw_lbfgs':
        vec = run_two_stage('adamw', fm, res, log, budget * cfg.warmup_frac, {'lr': cfg.warmup_lr, 'switch_frac': 0.4})

    def loss_fn(v: torch.Tensor) -> torch.Tensor:
        r = res.from_vec(v)
        return 0.5 * (r @ r)
    grad_fn = grad(loss_fn)
    gen = torch.Generator(device=vec.device).manual_seed(1234)
    D = fm.D
    d_prev = torch.zeros_like(vec)
    U, Lam, sketch_indef = (None, None, False)
    it = 0
    log.start()
    while log.elapsed() < budget:
        with torch.no_grad():
            r_cur = res.from_vec(vec)
            loss_cur = 0.5 * (r_cur @ r_cur).item()
            g_cur = grad_fn(vec)
            log.count_grad()

            def hvp(v: torch.Tensor) -> torch.Tensor:
                log.count_grad()
                _, hv = jvp(grad_fn, (vec,), (v,))
                return hv
            if U is None or it % cfg.update_freq == 0:
                U, Lam, sketch_indef = _nystrom_sketch(hvp, D, cfg.sketch_size, vec.dtype, vec.device, gen)
            if log.elapsed() >= budget:
                log.log(it + 1, vec, loss_cur, 'nncg', res, {'cg_iters': 0, 'step_size': 0.0, 'sketch_indef': sketch_indef})
                break
            d, cg_iters, neg_curv = _nystrom_pcg(hvp, g_cur, d_prev, U, Lam, cfg.mu, cfg.cg_tol, cfg.cg_iters)
            slope = -(g_cur @ d).item()
            if slope >= 0:
                d = g_cur.clone()
                slope = -(g_cur @ g_cur).item()
            t = cfg.lr
            accepted = False
            for _ in range(cfg.max_ls):
                vec_try = vec - t * d
                r_try = res.from_vec(vec_try)
                loss_try = 0.5 * (r_try @ r_try).item()
                if torch.isfinite(r_try).all() and loss_try <= loss_cur + cfg.armijo_alpha * t * slope:
                    vec = vec_try
                    loss_cur = loss_try
                    d_prev = d
                    accepted = True
                    break
                t *= cfg.armijo_beta
            if not accepted:
                d_prev.zero_()
            it += 1
            log.log(it, vec, loss_cur, 'nncg', res, {'cg_iters': cg_iters, 'step_size': t if accepted else 0.0, 'accepted': accepted, 'sketch_indef': sketch_indef, 'neg_curv': neg_curv})
            if not accepted:
                break
    log.pause()
    return vec
