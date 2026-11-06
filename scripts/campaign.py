from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
BASE_CFG = {'lr': 0.001, 'muon_lr': 0.02, 'switch_frac': 0.3, 'history': 50}
PLM_REC = {'mu0': 1.0, 'damping': 'marquardt', 'term_weighting': True, 'warmup': 'adamw'}
PLM_REC2 = dict(PLM_REC, restart_after=10, restart_on_blowup=100.0)
METHODS = ['adamw', 'muon', 'lbfgs', 'adamw_lbfgs', 'muon_lbfgs', 'plm']

def job(problem, method, seed, width=50, depth=3, budget=60.0, knobs=None, cfg=None, dtype='float64', tag=''):
    c = dict(BASE_CFG)
    c.update(cfg or {})
    return {'problem': problem, 'method': method, 'seed': seed, 'width': width, 'depth': depth, 'budget': budget, 'knobs': knobs or {}, 'cfg': c, 'dtype': dtype, 'tag': tag}

def job_id(j: dict) -> str:
    h = hashlib.md5(json.dumps(j, sort_keys=True).encode()).hexdigest()[:8]
    knob = '_'.join((f'{k}{v}' for k, v in j['knobs'].items()))
    return f"{j['problem']}_{knob}_{j['method']}_{j['tag']}_w{j['width']}_s{j['seed']}_{j['dtype']}_{h}".replace('__', '_')

def phase_jobs(phase: str, seeds: list[int], budget: float) -> list[dict]:
    J = []
    if phase == 'A':
        probs = [('order1', {}), ('poisson', {}), ('beam', {}), ('multiscale', {'k': 16}), ('noisy', {'sigma': 0.01})]
        for (p, k), m, s in itertools.product(probs, METHODS, seeds):
            J.append(job(p, m, s, budget=budget, knobs=k))
    elif phase == 'B':
        variants = {'base_mu1e-2': {'mu0': 0.01}, 'base': {'mu0': 1.0}, 'marquardt': {'mu0': 1.0, 'damping': 'marquardt'}, 'trust': {'mu0': 1.0, 'trust_radius': 1.0}, 'weighted': {'mu0': 1.0, 'term_weighting': True}, 'warm_adamw': {'mu0': 1.0, 'warmup': 'adamw'}, 'warm_muon': {'mu0': 1.0, 'warmup': 'muon'}, 'dense': {'mu0': 1.0, 'solver': 'dense'}, 'all': {'mu0': 1.0, 'damping': 'marquardt', 'term_weighting': True, 'warmup': 'adamw'}}
        for (p, k), (tag, cfg), s in itertools.product([('beam', {}), ('multiscale', {'k': 16}), ('poisson', {})], variants.items(), seeds):
            J.append(job(p, 'plm', s, budget=budget, knobs=k, cfg=cfg, tag=tag))
    elif phase == 'C':
        widths = [25, 50, 100, 200, 400]
        for w, m, s in itertools.product(widths, ['adamw', 'muon', 'adamw_lbfgs', 'plm'], seeds):
            J.append(job('multiscale', m, s, width=w, budget=budget, knobs={'k': 16}))
            if m == 'plm' and w <= 100:
                J.append(job('multiscale', m, s, width=w, budget=budget, knobs={'k': 16}, cfg={'solver': 'dense'}, tag='dense'))
    elif phase == 'D':
        probs = [('helmholtz', {'a2': 4}), ('convection', {'beta': 30.0}), ('reaction', {'rho': 5.0}), ('wave', {})]
        for (p, k), m, s in itertools.product(probs, METHODS, seeds):
            J.append(job(p, m, s, budget=budget, knobs=k, cfg={'term_weighting': True} if m == 'plm' else None))
    elif phase == 'E':
        rec = lambda m: PLM_REC if m == 'plm' else None
        for k, m, s in itertools.product([1, 4, 16, 32], ['adamw', 'muon', 'adamw_lbfgs', 'muon_lbfgs', 'plm'], seeds):
            J.append(job('multiscale', m, s, budget=budget, knobs={'k': k}, cfg=rec(m), tag='rec' if m == 'plm' else ''))
        for sg, m, s in itertools.product([0.0, 0.001, 0.01, 0.1], ['adamw_lbfgs', 'muon_lbfgs', 'plm'], seeds):
            J.append(job('noisy', m, s, budget=budget, knobs={'sigma': sg}, cfg=rec(m), tag='rec' if m == 'plm' else ''))
        for (p, k), m, s in itertools.product([('order1', {}), ('poisson', {}), ('beam', {})], ['adamw_lbfgs', 'muon_lbfgs', 'plm'], seeds):
            J.append(job(p, m, s, budget=budget, knobs=k, cfg=rec(m), tag='rec' if m == 'plm' else ''))
    elif phase == 'E2':
        for b, m, s in itertools.product([1.0, 10.0, 30.0, 40.0], ['adamw_lbfgs', 'muon_lbfgs', 'plm'], seeds):
            J.append(job('convection', m, s, budget=budget, knobs={'beta': b}, cfg=dict(PLM_REC, reuse_jac=4) if m == 'plm' else None, tag='rec_lag4' if m == 'plm' else ''))
    elif phase == 'F':
        for dt, m, s in itertools.product(['float32', 'float64'], METHODS, seeds):
            J.append(job('poisson', m, s, budget=budget, dtype=dt, cfg=PLM_REC if m == 'plm' else None, tag='rec' if m == 'plm' else ''))
    elif phase == 'C2':
        variants = {'base': {}, 'trust': {'trust_radius': 1.0}, 'warm_adamw': {'warmup': 'adamw'}, 'lag4': {'reuse_jac': 4}, 'warm_trust_lag4': {'warmup': 'adamw', 'trust_radius': 1.0, 'reuse_jac': 4}}
        for w, (tag, cfg), s in itertools.product([100, 200, 400], variants.items(), seeds):
            J.append(job('multiscale', 'plm', s, width=w, budget=budget, knobs={'k': 16}, cfg=dict(mu0=1.0, **cfg), tag=tag))
    elif phase == 'B2':
        variants = {'marquardt_floor': {'damping': 'marquardt'}, 'lag4': {'reuse_jac': 4}, 'lag8': {'reuse_jac': 8}, 'marq_lag4_weighted': {'damping': 'marquardt', 'reuse_jac': 4, 'term_weighting': True}}
        for (p, k), (tag, cfg), s in itertools.product([('beam', {}), ('multiscale', {'k': 16}), ('poisson', {})], variants.items(), seeds):
            J.append(job(p, 'plm', s, budget=budget, knobs=k, cfg=dict(mu0=1.0, **cfg), tag=tag))
    elif phase == 'D2':
        probs = [('convection', {'beta': 30.0}), ('reaction', {'rho': 5.0}), ('wave', {}), ('helmholtz', {'a2': 4})]
        for (p, k), s in itertools.product(probs, seeds):
            for m in ['adamw_lbfgs', 'muon_lbfgs', 'lbfgs']:
                J.append(job(p, m, s, budget=budget, knobs=k))
            J.append(job(p, 'plm', s, budget=budget, knobs=k, cfg={'term_weighting': True}, tag='weighted'))
            J.append(job(p, 'plm', s, budget=budget, knobs=k, cfg={'term_weighting': True, 'reuse_jac': 4, 'warmup': 'adamw'}, tag='weighted_lag4_warm'))
    elif phase == 'G':
        probs = [('poisson', {}), ('beam', {}), ('multiscale', {'k': 16})]
        for (p, k), lr, s in itertools.product(probs, [0.0001, 0.0003, 0.001, 0.003, 0.01], seeds):
            J.append(job(p, 'adamw', s, budget=budget, knobs=k, cfg={'lr': lr}, tag=f'lr{lr:g}'))
            J.append(job(p, 'adamw_lbfgs', s, budget=budget, knobs=k, cfg={'lr': lr}, tag=f'lr{lr:g}'))
        for (p, k), lr, s in itertools.product(probs, [0.001, 0.003, 0.01, 0.03, 0.1], seeds):
            J.append(job(p, 'muon', s, budget=budget, knobs=k, cfg={'muon_lr': lr}, tag=f'mlr{lr:g}'))
            J.append(job(p, 'muon_lbfgs', s, budget=budget, knobs=k, cfg={'muon_lr': lr}, tag=f'mlr{lr:g}'))
        for (p, k), mu, s in itertools.product(probs, [0.0001, 0.01, 1.0, 100.0, 10000.0], seeds):
            J.append(job(p, 'plm', s, budget=budget, knobs=k, cfg={'mu0': mu}, tag=f'mu{mu:g}'))
    elif phase == 'H':
        probs = [('order1', {}), ('poisson', {}), ('beam', {}), ('multiscale', {'k': 16}), ('multiscale', {'k': 32})]
        for (p, k), m, s in itertools.product(probs, ['adamw_lbfgs', 'muon_lbfgs', 'plm'], seeds):
            J.append(job(p, m, s, budget=budget, knobs=k, cfg=PLM_REC2 if m == 'plm' else None, tag='rec2' if m == 'plm' else ''))
    elif phase == 'H2':
        probs = [('helmholtz', {'a2': 4}), ('reaction', {'rho': 5.0}), ('wave', {})]
        for (p, k), m, s in itertools.product(probs, ['adamw_lbfgs', 'muon_lbfgs', 'plm'], seeds):
            J.append(job(p, m, s, budget=budget, knobs=k, cfg=dict(PLM_REC2, reuse_jac=4) if m == 'plm' else None, tag='rec2_lag4' if m == 'plm' else ''))
    elif phase == 'H3':
        for b, m, s in itertools.product([30.0, 40.0], ['adamw', 'adamw_lbfgs', 'muon_lbfgs', 'plm'], seeds):
            J.append(job('convection', m, s, budget=budget, knobs={'beta': b, 'causal': 1}, cfg=dict(PLM_REC2, reuse_jac=4) if m == 'plm' else None, tag='rec2_lag4' if m == 'plm' else ''))
    elif phase == 'S':
        cg = dict(PLM_REC2, solver='cg')
        for w, s in itertools.product([100, 400, 800, 1600], seeds):
            J.append(job('multiscale', 'plm', s, width=w, budget=budget, knobs={'k': 16}, cfg=cg, tag='cg_rec2'))
            if w <= 400:
                J.append(job('multiscale', 'plm', s, width=w, budget=budget, knobs={'k': 16}, cfg=PLM_REC2, tag='dual_rec2'))
            if w >= 400:
                for m in ['adamw', 'muon', 'adamw_lbfgs']:
                    J.append(job('multiscale', m, s, width=w, budget=budget, knobs={'k': 16}))
    elif phase == 'N':
        for (p, k), sg, m, s in itertools.product([('poisson', {}), ('multiscale', {'k': 16})], [0.001, 0.01, 0.1], ['adamw_lbfgs', 'muon_lbfgs', 'plm'], seeds):
            J.append(job(p, m, s, budget=budget, knobs=dict(k, pde_noise=sg), cfg=PLM_REC2 if m == 'plm' else None, tag='rec2' if m == 'plm' else ''))
    elif phase == 'L':
        probs = [('order1', {}), ('poisson', {}), ('beam', {}), ('multiscale', {'k': 16}), ('multiscale', {'k': 32})]
        for (p, k), w, sc, s in itertools.product(probs, [50, 200, 800], [1.0, 4.0], seeds):
            J.append(job(p, 'lstsq', s, width=w, budget=budget, knobs=k, cfg={'feat_scale': sc}, tag=f's{sc:g}'))
        for w, sc, s in itertools.product([50, 200], [1.0, 4.0], seeds):
            J.append(job('helmholtz', 'lstsq', s, width=w, budget=budget, knobs={'a2': 4}, cfg={'feat_scale': sc}, tag=f's{sc:g}'))
        for (p, k), s in itertools.product([('multiscale', {'k': 16}), ('multiscale', {'k': 32}), ('helmholtz', {'a2': 4})], seeds):
            J.append(job(p, 'plm', s, width=200, budget=budget, knobs=k, cfg=PLM_REC2, tag='rec2'))
    elif phase == 'P':
        probs = [('nlpoisson', {}), ('burgers', {'nu': 0.05}), ('allen_cahn', {}), ('reaction', {'rho': 5.0})]
        for (p, k), s in itertools.product(probs, seeds):
            for m in ['adamw_lbfgs', 'muon_lbfgs']:
                J.append(job(p, m, s, budget=budget, knobs=k))
            lag = {} if p == 'nlpoisson' else {'reuse_jac': 4}
            J.append(job(p, 'plm', s, budget=budget, knobs=k, cfg=dict(PLM_REC2, **lag), tag='rec2_lag4' if lag else 'rec2'))
            for sc in [1.0, 4.0]:
                J.append(job(p, 'lstsq', s, budget=budget, knobs=k, cfg={'feat_scale': sc}, tag=f's{sc:g}'))
    elif phase == 'V':
        one_d = [('order1', {}), ('poisson', {}), ('beam', {}), ('multiscale', {'k': 16}), ('multiscale', {'k': 32})]
        two_d = [('helmholtz', {'a2': 4}), ('reaction', {'rho': 5.0}), ('nlpoisson', {}), ('burgers', {'nu': 0.05}), ('allen_cahn', {})]
        variants = {'vp': {'varpro': True}, 'rs': {'resample': True}, 'vprs': {'varpro': True, 'resample': True}}
        for (p, k), (tag, v), s in itertools.product(one_d, variants.items(), seeds):
            J.append(job(p, 'plm', s, budget=60.0, knobs=k, cfg=dict(PLM_REC2, **v), tag=f'rec2_{tag}'))
        for (p, k), (tag, v), s in itertools.product(two_d, variants.items(), seeds):
            lag = {} if p == 'nlpoisson' else {'reuse_jac': 4}
            J.append(job(p, 'plm', s, budget=300.0, knobs=k, cfg=dict(PLM_REC2, **lag, **v), tag=f"rec2{('_lag4' if lag else '')}_{tag}"))
    elif phase == 'W':
        one_d = [('order1', {}), ('poisson', {}), ('beam', {}), ('multiscale', {'k': 16}), ('multiscale', {'k': 32})]
        two_d = [('helmholtz', {'a2': 4}), ('reaction', {'rho': 5.0}), ('nlpoisson', {}), ('burgers', {'nu': 0.05}), ('allen_cahn', {})]
        for (p, k), fr, s in itertools.product(one_d, [0.1, 0.3, 1.0], seeds):
            J.append(job(p, 'plm', s, budget=60.0, knobs=k, cfg=dict(PLM_REC2, final_solve=True, budget_frac=fr), tag=f'rec2_fs{fr:g}'))
        for (p, k), fr, s in itertools.product(two_d, [0.1, 0.3, 1.0], seeds):
            extra = {} if p == 'nlpoisson' else {'reuse_jac': 4, 'resample': True}
            J.append(job(p, 'plm', s, budget=300.0, knobs=k, cfg=dict(PLM_REC2, **extra, final_solve=True, budget_frac=fr), tag=f"rec2{('_lag4_rs' if extra else '')}_fs{fr:g}"))
    else:
        raise ValueError(phase)
    return J

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--budget', type=float, default=60.0)
    ap.add_argument('--gpus', type=int, nargs='+', default=[0])
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'campaign')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    jobs = phase_jobs(a.phase, a.seeds, a.budget)
    out = a.out / a.phase
    pending = [(j, out / f'{job_id(j)}.json') for j in jobs]
    pending = [(j, p) for j, p in pending if not p.exists()]
    print(f'phase {a.phase}: {len(jobs)} jobs, {len(pending)} pending, {len(a.gpus)} gpus', flush=True)
    if a.list:
        for j, p in pending:
            print(' ', p.name)
        return
    running: dict[int, tuple[subprocess.Popen, dict]] = {}
    free = list(a.gpus)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out.parent / f'phase_{a.phase}.log', 'a')
    while pending or running:
        while pending and free:
            gpu = free.pop(0)
            j, p = pending.pop(0)
            j = dict(j, device=a.device)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT / 'src'))
            proc = subprocess.Popen([PY, '-m', 'exact_gauss_newton.run', json.dumps(j), str(p)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            running[gpu] = (proc, j)
        for gpu, (proc, j) in list(running.items()):
            if proc.poll() is not None:
                outp = proc.stdout.read()
                line = [l for l in outp.splitlines() if l.strip() and 'Warning' not in l and ('run_backward' not in l)]
                msg = line[-1] if line else '(no output)'
                if proc.returncode != 0:
                    msg = f"FAILED {j['problem']} {j['method']} w={j['width']} s={j['seed']}: {msg}"
                print(msg, flush=True)
                log.write(msg + '\n')
                log.flush()
                del running[gpu]
                free.append(gpu)
        time.sleep(1.0)
if __name__ == '__main__':
    main()
