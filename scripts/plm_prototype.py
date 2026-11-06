"""Quick check of PLM (dense Levenberg–Marquardt / Gauss-Newton) against Adam and Adam→L-BFGS.

Small 1D problems, width-50 depth-3 tanh MLP (~5k params), float64, equal wall-clock budget.
Evaluation time is excluded from the training clock.

    python plm_prototype.py --problem poisson --budget 60 --device cuda:3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.func import functional_call, grad, jacrev, vmap

PI = math.pi


# ----------------------------------------------------------------------------- problems
class Problem:
    """1D BVP on [0, 1]: operator(u)(x) = f(x), with Dirichlet-type boundary residuals."""

    def __init__(self, name: str):
        self.name = name
        if name == "poisson":  # -u'' = f, u = sin(pi x)
            self.order = 2
            self.exact = lambda x: torch.sin(PI * x)
            self.f = lambda x: PI**2 * torch.sin(PI * x)
            self.bc_orders = [0]
        elif name == "multiscale":  # -u'' = f, u = sin(pi x) + 0.1 sin(16 pi x)
            self.order = 2
            k = 16
            self.exact = lambda x: torch.sin(PI * x) + 0.1 * torch.sin(k * PI * x)
            self.f = lambda x: PI**2 * torch.sin(PI * x) + 0.1 * (k * PI) ** 2 * torch.sin(k * PI * x)
            self.bc_orders = [0]
        elif name == "beam":  # u'''' = q, u = sin(pi x); u = u'' = 0 at ends
            self.order = 4
            self.exact = lambda x: torch.sin(PI * x)
            self.f = lambda x: PI**4 * torch.sin(PI * x)
            self.bc_orders = [0, 2]
        else:
            raise ValueError(name)

    def operator(self, u_fn):
        """Return fn(params, x) giving the PDE operator applied to u."""
        d = u_fn
        for _ in range(self.order):
            d = grad(d, argnums=1)
        sign = -1.0 if self.order == 2 else 1.0
        return lambda p, x: sign * d(p, x)


# ----------------------------------------------------------------------------- model
def make_model(width: int, depth: int, dtype, device) -> torch.nn.Module:
    layers, d = [], 1
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
        d = width
    layers.append(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(*layers).to(dtype=dtype, device=device)


class FlatModel:
    """Model whose parameters live in one flat vector (needed for dense Gauss-Newton)."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.names = [n for n, _ in model.named_parameters()]
        self.shapes = [p.shape for _, p in model.named_parameters()]
        self.sizes = [p.numel() for _, p in model.named_parameters()]
        self.flat0 = torch.cat([p.detach().reshape(-1) for _, p in model.named_parameters()])

    def unflatten(self, vec: torch.Tensor) -> dict:
        out, i = {}, 0
        for n, s, k in zip(self.names, self.shapes, self.sizes):
            out[n] = vec[i : i + k].reshape(s)
            i += k
        return out

    def u(self, vec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Scalar-in, scalar-out network evaluation (vmap over x outside)."""
        return functional_call(self.model, self.unflatten(vec), (x.reshape(1, 1),)).reshape(())


# ----------------------------------------------------------------------------- residuals
def build_residual_fn(fm: FlatModel, prob: Problem, x_int: torch.Tensor, x_bc: torch.Tensor):
    op = prob.operator(fm.u)
    f_int = prob.f(x_int)
    n_int, n_bc = x_int.numel(), x_bc.numel()

    bc_fns = []
    for k in prob.bc_orders:
        d = fm.u
        for _ in range(k):
            d = grad(d, argnums=1)
        bc_fns.append(d)
    exact_bc = [prob.exact(x_bc) if k == 0 else torch.zeros_like(x_bc) for k in prob.bc_orders]

    def residuals(vec: torch.Tensor) -> torch.Tensor:
        """Scaled so that 0.5*||r||^2 = 0.5*(mean pde^2 + sum_k mean bc_k^2)."""
        r_pde = (vmap(op, in_dims=(None, 0))(vec, x_int) - f_int) / math.sqrt(n_int)
        parts = [r_pde]
        for d, g in zip(bc_fns, exact_bc):
            parts.append((vmap(d, in_dims=(None, 0))(vec, x_bc) - g) / math.sqrt(n_bc))
        return torch.cat(parts)

    return residuals


def loss_of(residuals, vec):
    r = residuals(vec)
    return 0.5 * (r * r).sum()


# ----------------------------------------------------------------------------- bookkeeping
class Logger:
    def __init__(self, fm: FlatModel, prob: Problem, device, dtype):
        self.fm = fm
        self.x_eval = torch.linspace(0, 1, 1001, dtype=dtype, device=device)
        self.u_exact = prob.exact(self.x_eval)
        self.rows = []
        self.train_time = 0.0
        self._t0 = None

    def start(self):
        self._t0 = time.perf_counter()

    def pause(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.train_time += time.perf_counter() - self._t0

    @torch.no_grad()
    def log(self, it: int, vec: torch.Tensor, loss: float, stage: str):
        self.pause()
        u = vmap(self.fm.u, in_dims=(None, 0))(vec, self.x_eval)
        err = (torch.linalg.norm(u - self.u_exact) / torch.linalg.norm(self.u_exact)).item()
        self.rows.append({"iter": it, "time": self.train_time, "loss": loss, "rel_l2": err, "stage": stage})
        self.start()
        return err


# ----------------------------------------------------------------------------- optimizers
def run_adam(fm, residuals, log: Logger, budget: float, lr: float, lbfgs_after: float | None):
    """Adam for the budget, or Adam for `lbfgs_after` fraction of it then L-BFGS."""
    vec = fm.flat0.clone().requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=lr)
    it = 0
    log.start()
    adam_budget = budget * lbfgs_after if lbfgs_after else budget
    while log.train_time + (time.perf_counter() - log._t0) < adam_budget:
        opt.zero_grad()
        loss = loss_of(residuals, vec)
        loss.backward()
        opt.step()
        it += 1
        if it % 50 == 0:
            log.log(it, vec.detach(), loss.item(), "adam")
    if lbfgs_after is None:
        log.pause()
        return vec.detach()

    opt = torch.optim.LBFGS([vec], lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=0, tolerance_change=0)

    def closure():
        opt.zero_grad()
        loss = loss_of(residuals, vec)
        loss.backward()
        return loss

    while log.train_time + (time.perf_counter() - log._t0) < budget:
        loss = opt.step(closure)
        it += 20
        log.log(it, vec.detach(), loss.item(), "lbfgs")
    log.pause()
    return vec.detach()


def run_plm(fm, residuals, log: Logger, budget: float, mu0: float):
    """Dense Levenberg–Marquardt on the Gauss-Newton matrix J^T J."""
    vec = fm.flat0.clone()
    D = vec.numel()
    eye = torch.eye(D, dtype=vec.dtype, device=vec.device)
    mu = mu0
    jac = jacrev(residuals)
    it = 0
    log.start()
    with torch.no_grad():
        r = residuals(vec)
        while log.train_time + (time.perf_counter() - log._t0) < budget:
            J = jac(vec)
            g = J.T @ r
            G = J.T @ J
            loss = 0.5 * (r @ r).item()
            for _ in range(12):  # retries with larger damping
                d = torch.linalg.solve(G + mu * eye, -g)
                r_new = residuals(vec + d)
                actual = loss - 0.5 * (r_new @ r_new).item()
                lin = r + J @ d
                predicted = loss - 0.5 * (lin @ lin).item()
                rho = actual / predicted if predicted > 0 else -1.0
                if rho > 1e-3:
                    vec = vec + d
                    r = r_new
                    if rho > 0.75:
                        mu = max(mu / 3.0, 1e-16)
                    break
                mu *= 4.0
            it += 1
            log.log(it, vec, 0.5 * (r @ r).item(), "plm")
    log.pause()
    return vec


# ----------------------------------------------------------------------------- main
def time_to(rows, thr):
    for row in rows:
        if row["rel_l2"] <= thr:
            return row["time"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="poisson", choices=["poisson", "multiscale", "beam"])
    ap.add_argument("--budget", type=float, default=60.0, help="training seconds per method")
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--n-int", type=int, default=512)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--adam-lr", type=float, default=1e-3)
    ap.add_argument("--mu0", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--out", type=Path, default=Path("plm_results"))
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    prob = Problem(args.problem)
    x_int = torch.linspace(0, 1, args.n_int + 2, dtype=dtype, device=device)[1:-1]
    x_bc = torch.tensor([0.0, 1.0], dtype=dtype, device=device)
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = make_model(args.width, args.depth, dtype, device)
        fm = FlatModel(model)
        residuals = build_residual_fn(fm, prob, x_int, x_bc)
        print(f"\n== {args.problem} seed {seed}  D={fm.flat0.numel()}  budget={args.budget}s  {device} {dtype}")

        runs = {
            "adam": lambda lg: run_adam(fm, residuals, lg, args.budget, args.adam_lr, None),
            "adam_lbfgs": lambda lg: run_adam(fm, residuals, lg, args.budget, args.adam_lr, 0.3),
            "plm": lambda lg: run_plm(fm, residuals, lg, args.budget, args.mu0),
        }
        for name, fn in runs.items():
            lg = Logger(fm, prob, device, dtype)
            # warm-up / compile the functorch graph outside the clock
            residuals(fm.flat0)
            vec = fn(lg)
            final = lg.rows[-1] if lg.rows else {"rel_l2": float("nan"), "loss": float("nan"), "iter": 0}
            best = min(r["rel_l2"] for r in lg.rows)
            s = {
                "final_rel_l2": final["rel_l2"], "best_rel_l2": best, "final_loss": final["loss"],
                "iters": final["iter"], "t_1e-3": time_to(lg.rows, 1e-3), "t_1e-5": time_to(lg.rows, 1e-5),
                "t_1e-7": time_to(lg.rows, 1e-7),
            }
            summary[f"{name}/seed{seed}"] = s
            print(f"  {name:11s} iters={s['iters']:7d} loss={s['final_loss']:.2e} relL2={s['final_rel_l2']:.2e} "
                  f"(best {best:.2e})  t@1e-3={s['t_1e-3']}  t@1e-5={s['t_1e-5']}  t@1e-7={s['t_1e-7']}")
            with open(args.out / f"{args.problem}_{name}_seed{seed}.json", "w") as f:
                json.dump(lg.rows, f)

    with open(args.out / f"{args.problem}_summary.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
