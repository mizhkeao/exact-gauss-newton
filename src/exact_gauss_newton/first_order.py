"""First-order and quasi-Newton runners: AdamW, Muon(+AdamW), L-BFGS, and X -> L-BFGS.

All runners share a wall-clock budget and a Logger; evaluation time is excluded.
"""

from __future__ import annotations

import time

import torch

from exact_gauss_newton.logging_ import Logger
from exact_gauss_newton.models import FlatModel
from exact_gauss_newton.residuals import Residuals


def _leaf_params(fm: FlatModel, vec: torch.Tensor) -> dict[str, torch.Tensor]:
    return {n: p.clone().requires_grad_(True) for n, p in fm.unflatten(vec).items()}


def make_first_order(name: str, fm: FlatModel, params: dict, cfg: dict) -> list[torch.optim.Optimizer]:
    lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 0.0)
    if name == "adamw":
        return [torch.optim.AdamW(list(params.values()), lr=lr, weight_decay=wd)]
    if name == "muon":
        hidden = set(fm.hidden_matrix_names())
        muon_p = [p for n, p in params.items() if n in hidden]
        other_p = [p for n, p in params.items() if n not in hidden]
        opts = [torch.optim.AdamW(other_p, lr=lr, weight_decay=wd)]
        if muon_p:
            opts.append(torch.optim.Muon(muon_p, lr=cfg.get("muon_lr", 0.02), weight_decay=wd,
                                         momentum=cfg.get("muon_momentum", 0.95)))
        return opts
    raise ValueError(name)


def run_first_order(name: str, fm: FlatModel, res: Residuals, log: Logger, budget: float, cfg: dict,
                    vec0: torch.Tensor | None = None, stage: str | None = None) -> torch.Tensor:
    params = _leaf_params(fm, fm.flat0 if vec0 is None else vec0)
    opts = make_first_order(name, fm, params, cfg)
    stage = stage or name
    it = 0
    log.start()
    while log.elapsed() < budget:
        for o in opts:
            o.zero_grad()
        loss = res.loss_from_params(params)
        loss.backward()
        for o in opts:
            o.step()
        it += 1
        log.count_grad()
        if it % cfg.get("log_every", 50) == 0:
            log.log(it, params, res.last_raw_loss.item(), stage, res)
    log.log(it, params, res.last_raw_loss.item(), stage, res)
    log.pause()
    return fm.flatten({n: p.detach() for n, p in params.items()})


def run_lbfgs(fm: FlatModel, res: Residuals, log: Logger, budget: float, cfg: dict,
              vec0: torch.Tensor | None = None) -> torch.Tensor:
    params = _leaf_params(fm, fm.flat0 if vec0 is None else vec0)
    plist = list(params.values())
    inner = cfg.get("lbfgs_inner", 20)
    opt = torch.optim.LBFGS(plist, lr=cfg.get("lbfgs_lr", 1.0), max_iter=inner,
                            history_size=cfg.get("history", 50), line_search_fn="strong_wolfe",
                            tolerance_grad=0.0, tolerance_change=0.0)

    def closure():
        opt.zero_grad()
        loss = res.loss_from_params(params)
        loss.backward()
        log.count_grad()
        return loss

    it = 0
    log.start()
    while log.elapsed() < budget:
        loss = opt.step(closure)
        it += inner
        log.log(it, params, res.last_raw_loss.item(), "lbfgs", res)
        if not torch.isfinite(loss):
            break
    log.pause()
    return fm.flatten({n: p.detach() for n, p in params.items()})


def run_two_stage(first: str, fm: FlatModel, res: Residuals, log: Logger, budget: float, cfg: dict) -> torch.Tensor:
    frac = cfg.get("switch_frac", 0.3)
    vec = run_first_order(first, fm, res, log, budget * frac, cfg)
    return run_lbfgs(fm, res, log, budget, cfg, vec0=vec)
