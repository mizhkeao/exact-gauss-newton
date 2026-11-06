from __future__ import annotations
import torch
from torch.func import jacrev
from exact_gauss_newton.logging_ import Logger
from exact_gauss_newton.models import FlatModel
from exact_gauss_newton.residuals import Residuals

def _minnorm_lstsq(J: torch.Tensor, b: torch.Tensor, rtol: float=1e-12) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    keep = S > rtol * S.max()
    return Vh[keep].T @ (U[:, keep].T @ b / S[keep])

def output_layer_solve(fm: FlatModel, res: Residuals, vec: torch.Tensor, r: torch.Tensor | None=None, max_steps: int=1, log: Logger | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    k = fm.sizes[-2] + fm.sizes[-1]
    hidden = vec[:-k]

    def r_of(theta):
        return res.from_vec(torch.cat([hidden, theta]))
    theta = vec[-k:]
    if r is None:
        r = r_of(theta)
    loss = 0.5 * (r @ r).item()
    for _ in range(max_steps):
        Jm = jacrev(r_of)(theta)
        if log is not None:
            log.count_grad(k)
        dvec = _minnorm_lstsq(Jm, -r)
        step, ok = (1.0, False)
        for _ in range(20):
            r_new = r_of(theta + step * dvec)
            loss_new = 0.5 * (r_new @ r_new).item()
            if torch.isfinite(r_new).all() and loss_new < loss * (1 - 1e-10):
                ok = True
                break
            step *= 0.5
        if not ok:
            break
        theta, r, loss = (theta + step * dvec, r_new, loss_new)
        if loss < 1e-30 or (step * dvec).norm() < 1e-09 * theta.norm():
            break
    return (torch.cat([hidden, theta]), r)

def run_lstsq(fm: FlatModel, res: Residuals, log: Logger, budget: float, cfg: dict) -> torch.Tensor:
    params = {k: v.clone() for k, v in fm.unflatten(fm.flat0).items()}
    scale = float(cfg.get('feat_scale', 1.0))
    if scale != 1.0:
        for n in fm.names[:2]:
            params[n] = params[n] * scale
    w_name, b_name = (fm.names[-2], fm.names[-1])
    k_w = params[w_name].numel()

    def r_of(theta):
        p = dict(params)
        p[w_name] = theta[:k_w].reshape(params[w_name].shape)
        p[b_name] = theta[k_w:].reshape(params[b_name].shape)
        return res.from_params(p)

    def full(theta):
        p = dict(params)
        p[w_name] = theta[:k_w].reshape(params[w_name].shape)
        p[b_name] = theta[k_w:].reshape(params[b_name].shape)
        return fm.flatten(p)
    theta = torch.cat([params[w_name].reshape(-1), params[b_name].reshape(-1)])
    J = jacrev(r_of)
    it, step = (0, 1.0)
    log.start()
    with torch.no_grad():
        r = r_of(theta)
        loss = loss0 = 0.5 * (r @ r).item()
        log.log(0, full(theta), loss, 'lstsq', res, {'step': 0.0})
        while log.elapsed() < budget:
            Jm = J(theta)
            log.count_jac()
            dvec = _minnorm_lstsq(Jm, -r)
            step, ok = (1.0, False)
            for _ in range(30):
                r_new = r_of(theta + step * dvec)
                loss_new = 0.5 * (r_new @ r_new).item()
                if torch.isfinite(r_new).all() and loss_new < loss * (1 - 1e-10):
                    ok = True
                    break
                step *= 0.5
            if not ok:
                break
            theta, r, prev = (theta + step * dvec, r_new, loss)
            loss = loss_new
            it += 1
            log.log(it, full(theta), loss, 'lstsq', res, {'step': step})
            if loss < 1e-20 * loss0 or (step * dvec).norm() < 1e-09 * theta.norm():
                break
    log.pause()
    return full(theta)
