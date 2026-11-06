from __future__ import annotations
import json
import time
from pathlib import Path
import torch
from exact_gauss_newton.first_order import run_first_order, run_lbfgs, run_two_stage
from exact_gauss_newton.logging_ import Logger
from exact_gauss_newton.lstsq import run_lstsq
from exact_gauss_newton.models import build
from exact_gauss_newton.nncg import NNCGConfig, run_nncg
from exact_gauss_newton.plm import PLMConfig, run_plm
from exact_gauss_newton.problems import get_problem
from exact_gauss_newton.residuals import Residuals
METHODS = ['adamw', 'muon', 'lbfgs', 'adamw_lbfgs', 'muon_lbfgs', 'plm', 'lstsq', 'nncg']

def time_to(rows, thr):
    for row in rows:
        if row['rel_l2'] <= thr:
            return row['time']
    return None

def run_job(job: dict, out_path: Path) -> dict:
    device = torch.device(job.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    dtype = getattr(torch, job.get('dtype', 'float64'))
    prob = get_problem(job['problem'], device=device, dtype=dtype, **job.get('knobs', {}))
    fm = build(prob.in_dim, job['width'], job['depth'], job['seed'], dtype, device)
    knobs = job.get('knobs', {})
    res = Residuals(fm, prob, causal=bool(knobs.get('causal', False)), pde_noise=float(knobs.get('pde_noise', 0.0)), seed=job['seed'])
    res.from_vec(fm.flat0)
    log = Logger()
    method, cfg, budget = (job['method'], job.get('cfg', {}), job['budget'])
    t_wall = time.perf_counter()
    if method in ('adamw', 'muon'):
        vec = run_first_order(method, fm, res, log, budget, cfg)
    elif method == 'lbfgs':
        vec = run_lbfgs(fm, res, log, budget, cfg)
    elif method.endswith('_lbfgs'):
        vec = run_two_stage(method[:-len('_lbfgs')], fm, res, log, budget, cfg)
    elif method == 'plm':
        vec = run_plm(fm, res, log, budget, PLMConfig.from_dict(cfg))
    elif method == 'nncg':
        vec = run_nncg(fm, res, log, budget, NNCGConfig.from_dict(cfg))
    elif method == 'lstsq':
        vec = run_lstsq(fm, res, log, budget, cfg)
    else:
        raise ValueError(method)
    t_wall = time.perf_counter() - t_wall
    rows = log.rows
    final = rows[-1]
    summary = {'final_rel_l2': final['rel_l2'], 'best_rel_l2': min((r['rel_l2'] for r in rows)), 'final_loss': final['loss'], 'iters': final['iter'], 'train_time': log.train_time, 'wall_time_incl_eval': t_wall, 'grad_evals': log.grad_evals, 'jac_evals': log.jac_evals, 'peak_mem_mb': log.peak_memory_mb(), 'D': fm.D, 'N': prob.n_residuals, 'sec_per_iter': log.train_time / max(final['iter'], 1), **{f't_{thr:g}': time_to(rows, thr) for thr in (0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-07)}}
    result = {'job': job, 'summary': summary, 'rows': rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f)
    return result
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('job_json')
    ap.add_argument('out')
    a = ap.parse_args()
    job = json.loads(a.job_json)
    r = run_job(job, Path(a.out))
    s = r['summary']
    print(f"{job['problem']:11s} {job['method']:12s} w={job['width']:4d} seed={job['seed']} iters={s['iters']:7d} loss={s['final_loss']:.2e} relL2={s['final_rel_l2']:.2e} best={s['best_rel_l2']:.2e} mem={s['peak_mem_mb'] or 0:.0f}MB s/it={s['sec_per_iter']:.2e}")
