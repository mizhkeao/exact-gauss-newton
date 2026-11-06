import torch
from torch.func import jacrev
from exact_gauss_newton.models import build
from exact_gauss_newton.plm import PLMConfig, _solve
from exact_gauss_newton.problems import BUILDERS, get_problem
from exact_gauss_newton.residuals import Residuals

def test_all_problems_build_and_residuals_run():
    for name in BUILDERS:
        prob = get_problem(name)
        fm = build(prob.in_dim, 8, 2, seed=0, dtype=torch.float64, device='cpu')
        r = Residuals(fm, prob).from_vec(fm.flat0)
        assert r.shape == (prob.n_residuals,) and torch.isfinite(r).all(), name

def test_exact_solution_has_zero_residual_poisson():
    import math
    prob = get_problem('beam')
    u_exact = lambda x: torch.sin(math.pi * x[0])
    for t in prob.terms:
        vals = torch.stack([t.fn(u_exact, x) for x in t.points[:5]])
        assert vals.abs().max() < 1e-08, t.name

def test_every_exact_solution_has_zero_residual():
    for name in BUILDERS:
        if name == 'noisy':
            continue
        prob = get_problem(name)
        u_exact = lambda x: prob.exact(x.reshape(1, -1))[0]
        for t in prob.terms:
            vals = torch.stack([t.fn(u_exact, x) for x in t.points[:7]])
            assert vals.abs().max() < 1e-07, (name, t.name, vals.abs().max().item())

def test_lstsq_solves_linear_pde_in_one_step():
    from exact_gauss_newton.logging_ import Logger
    from exact_gauss_newton.lstsq import run_lstsq
    prob = get_problem('poisson', n_int=64)
    fm = build(1, 64, 2, seed=0, dtype=torch.float64, device='cpu')
    res = Residuals(fm, prob)
    log = Logger()
    run_lstsq(fm, res, log, 30.0, {})
    assert log.rows[1]['loss'] < 1e-06 * log.rows[0]['loss'] and len(log.rows) <= 3

def test_paired_init():
    a = build(1, 16, 2, seed=3, dtype=torch.float64, device='cpu')
    b = build(1, 16, 2, seed=3, dtype=torch.float64, device='cpu')
    assert torch.equal(a.flat0, b.flat0)

def test_hidden_matrix_split():
    fm = build(2, 16, 3, seed=0, dtype=torch.float64, device='cpu')
    assert fm.hidden_matrix_names() == ['2.weight', '4.weight']

def test_dense_and_dual_solves_agree():
    torch.manual_seed(0)
    J = torch.randn(30, 80, dtype=torch.float64)
    r = torch.randn(30, dtype=torch.float64)
    s = (J * J).sum(0)
    for sv in (None, s):
        d1 = _solve(J, r, 0.1, sv, dense=True)
        d2 = _solve(J, r, 0.1, sv, dense=False)
        assert torch.allclose(d1, d2, atol=1e-08)

def test_plm_config_from_dict_ignores_unknown():
    c = PLMConfig.from_dict({'mu0': 2.0, 'lr': 0.001, 'history': 50})
    assert c.mu0 == 2.0

def test_matrix_free_step_matches_dual():
    from torch.func import jacrev
    from exact_gauss_newton.logging_ import Logger
    from exact_gauss_newton.plm import ExplicitJac, MatrixFreeJac, PLMConfig
    from exact_gauss_newton.problems import get_problem
    from exact_gauss_newton.residuals import Residuals
    prob = get_problem('poisson', n=16)
    fm = build(1, 8, 2, seed=0, dtype=torch.float64, device='cpu')
    res = Residuals(fm, prob)
    vec = fm.flat0.clone()
    cfg = PLMConfig(solver='cg', cg_iters=500, cg_tol=1e-10, probes=64)
    mf = MatrixFreeJac(res.from_vec, vec, Logger(), cfg)
    ex = ExplicitJac(jacrev(res.from_vec)(vec), dense=False)
    wt = torch.ones(res.prob.n_residuals, dtype=torch.float64)
    mf.weighted(wt)
    ex.weighted(wt)
    rw = res.from_vec(vec)
    d_mf, d_ex = (mf.solve(rw, 1.0, None), ex.solve(rw, 1.0, None))
    assert torch.allclose(d_mf, d_ex, rtol=1e-06, atol=1e-10)

def test_varpro_plm_reaches_machine_precision_on_linear_pde():
    import torch
    from exact_gauss_newton.logging_ import Logger
    from exact_gauss_newton.models import build
    from exact_gauss_newton.plm import PLMConfig, run_plm
    from exact_gauss_newton.problems import get_problem
    from exact_gauss_newton.residuals import Residuals
    prob = get_problem('poisson', n_int=64)
    fm = build(1, 32, 3, 0, torch.float64, 'cpu')
    res = Residuals(fm, prob)
    log = Logger()
    run_plm(fm, res, log, 30.0, PLMConfig(damping='marquardt', varpro=True))
    assert log.rows[0]['rel_l2'] < 1e-09, log.rows[0]
    assert log.rows[-1]['iter'] <= 40

def test_resample_restores_points_and_logs_fixed_loss():
    import torch
    from exact_gauss_newton.logging_ import Logger
    from exact_gauss_newton.models import build
    from exact_gauss_newton.plm import PLMConfig, run_plm
    from exact_gauss_newton.problems import get_problem
    from exact_gauss_newton.residuals import Residuals
    prob = get_problem('multiscale', n_int=64, k=4)
    pts0 = prob.terms[0].points.clone()
    fm = build(1, 32, 3, 0, torch.float64, 'cpu')
    res = Residuals(fm, prob)
    log = Logger()
    vec = run_plm(fm, res, log, 5.0, PLMConfig(damping='marquardt', resample=True, varpro=True))
    assert torch.equal(prob.terms[0].points, pts0)
    r = res.from_vec(vec)
    assert abs(log.rows[-1]['loss'] - 0.5 * (r @ r).item()) < 1e-12 * max(1.0, log.rows[-1]['loss'])
    assert log.rows[-1]['iter'] >= 3

def test_two_stage_final_solve_reaches_ls_floor():
    import torch
    from exact_gauss_newton.logging_ import Logger
    from exact_gauss_newton.models import build
    from exact_gauss_newton.plm import PLMConfig, run_plm
    from exact_gauss_newton.problems import get_problem
    from exact_gauss_newton.residuals import Residuals
    prob = get_problem('poisson', n_int=64)
    fm = build(1, 32, 3, 0, torch.float64, 'cpu')
    res = Residuals(fm, prob)
    log = Logger()
    run_plm(fm, res, log, 6.0, PLMConfig(damping='marquardt', final_solve=True, budget_frac=0.3))
    assert log.rows[-1]['stage'] == 'final_solve'
    assert log.rows[-1]['rel_l2'] < 1e-09 and log.rows[-2]['rel_l2'] > log.rows[-1]['rel_l2']
    assert log.train_time < 5.0

def test_nncg_decreases_loss_on_poisson():
    from exact_gauss_newton.logging_ import Logger
    from exact_gauss_newton.models import build
    from exact_gauss_newton.nncg import NNCGConfig, run_nncg
    from exact_gauss_newton.problems import get_problem
    from exact_gauss_newton.residuals import Residuals
    prob = get_problem('poisson', n_int=32)
    fm = build(1, 16, 2, seed=0, dtype=torch.float64, device='cpu')
    res = Residuals(fm, prob)
    log = Logger()
    run_nncg(fm, res, log, 2.0, NNCGConfig(sketch_size=8, cg_iters=15, warmup='none'))
    assert log.rows[-1]['loss'] < 0.001 * log.rows[0]['loss']
