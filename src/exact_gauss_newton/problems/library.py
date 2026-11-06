from __future__ import annotations
import math
import torch
from exact_gauss_newton.problems.base import PI, Problem, Term, d, dn, grid
F64 = torch.float64
CPU = torch.device('cpu')

def _bc_points_1d():
    return torch.tensor([[0.0], [1.0]], dtype=F64)

def _eval_1d(m=1001):
    return torch.linspace(0, 1, m, dtype=F64).reshape(-1, 1)

def order1(n_int: int=512, **_) -> Problem:
    exact = lambda X: torch.sin(PI * X[:, 0])
    f = lambda x: PI * torch.cos(PI * x[0])
    return Problem('order1', 1, [Term('pde', grid([(0, 1)], [n_int], True, F64, CPU), lambda u, x: d(u, 0)(x) - f(x)), Term('bc', torch.tensor([[0.0]], dtype=F64), lambda u, x: u(x) - torch.sin(PI * x[0]))], exact, _eval_1d())

def poisson(n_int: int=512, k: int=1, amp: float=0.0, **_) -> Problem:
    exact = lambda X: torch.sin(PI * X[:, 0]) + amp * torch.sin(k * PI * X[:, 0])
    f = lambda x: PI ** 2 * torch.sin(PI * x[0]) + amp * (k * PI) ** 2 * torch.sin(k * PI * x[0])
    return Problem('poisson' if amp == 0 else 'multiscale', 1, [Term('pde', grid([(0, 1)], [n_int], True, F64, CPU), lambda u, x: -dn(u, 0, 2)(x) - f(x)), Term('bc', _bc_points_1d(), lambda u, x: u(x))], exact, _eval_1d(), {'k': k, 'amp': amp})

def multiscale(k: int=16, **kw) -> Problem:
    return poisson(k=k, amp=0.1, **kw)

def beam(n_int: int=512, **_) -> Problem:
    exact = lambda X: torch.sin(PI * X[:, 0])
    q = lambda x: PI ** 4 * torch.sin(PI * x[0])
    return Problem('beam', 1, [Term('pde', grid([(0, 1)], [n_int], True, F64, CPU), lambda u, x: dn(u, 0, 4)(x) - q(x)), Term('bc_u', _bc_points_1d(), lambda u, x: u(x)), Term('bc_uxx', _bc_points_1d(), lambda u, x: dn(u, 0, 2)(x))], exact, _eval_1d())

def noisy(sigma: float=0.01, n_obs: int=64, n_int: int=512, seed: int=1234, **_) -> Problem:
    p = poisson(n_int=n_int)
    g = torch.Generator().manual_seed(seed)
    x_obs = torch.rand(n_obs, 1, generator=g, dtype=F64)
    u_obs = p.exact(x_obs) + sigma * torch.randn(n_obs, generator=g, dtype=F64)
    pts = torch.cat([x_obs, u_obs.reshape(-1, 1)], dim=1)
    p.terms.append(Term('data', pts, lambda u, x: u(x[:1]) - x[1]))
    p.name, p.knobs = ('noisy', {'sigma': sigma})
    return p

def helmholtz(a1: int=1, a2: int=4, n: int=32, **_) -> Problem:

    def exact(X):
        return torch.sin(a1 * PI * X[:, 0]) * torch.sin(a2 * PI * X[:, 1])

    def q(x):
        return (1 - (a1 * PI) ** 2 - (a2 * PI) ** 2) * torch.sin(a1 * PI * x[0]) * torch.sin(a2 * PI * x[1])
    interior = grid([(0, 1), (0, 1)], [n, n], True, F64, CPU)
    s = torch.linspace(0, 1, n, dtype=F64)
    z, o = (torch.zeros_like(s), torch.ones_like(s))
    bc = torch.cat([torch.stack([z, s], 1), torch.stack([o, s], 1), torch.stack([s, z], 1), torch.stack([s, o], 1)])
    return Problem('helmholtz', 2, [Term('pde', interior, lambda u, x: dn(u, 0, 2)(x) + dn(u, 1, 2)(x) + u(x) - q(x)), Term('bc', bc, lambda u, x: u(x))], exact, grid([(0, 1), (0, 1)], [64, 64], False, F64, CPU), {'a1': a1, 'a2': a2})

def _time_grids(x_dom, nx, nt):
    interior = grid([x_dom, (0, 1)], [nx, nt], True, F64, CPU)
    xs = torch.linspace(*x_dom, nx + 2, dtype=F64)
    ts = torch.linspace(0, 1, nt + 2, dtype=F64)
    ic = torch.stack([xs, torch.zeros_like(xs)], 1)
    left = torch.stack([torch.full_like(ts, x_dom[0]), ts], 1)
    right = torch.stack([torch.full_like(ts, x_dom[1]), ts], 1)
    ev = grid([x_dom, (0, 1)], [128, 64], False, F64, CPU)
    return (interior, ic, left, right, ev)

def convection(beta: float=30.0, nx: int=64, nt: int=32, **_) -> Problem:
    interior, ic, left, right, ev = _time_grids((0, 2 * PI), nx, nt)
    exact = lambda X: torch.sin(X[:, 0] - beta * X[:, 1])
    shift = torch.tensor([2 * PI, 0.0], dtype=F64)
    return Problem('convection', 2, [Term('pde', interior, lambda u, x: d(u, 1)(x) + beta * d(u, 0)(x)), Term('ic', ic, lambda u, x: u(x) - torch.sin(x[0])), Term('bc', left, lambda u, x: u(x) - u(x + shift.to(x)))], exact, ev, {'beta': beta})

def reaction(rho: float=5.0, nx: int=64, nt: int=32, **_) -> Problem:
    interior, ic, left, right, ev = _time_grids((0, 2 * PI), nx, nt)
    h = lambda x: torch.exp(-(x - PI) ** 2 / (2 * (PI / 4) ** 2))

    def exact(X):
        h0 = h(X[:, 0])
        return h0 * torch.exp(rho * X[:, 1]) / (h0 * torch.exp(rho * X[:, 1]) + 1 - h0)
    shift = torch.tensor([2 * PI, 0.0], dtype=F64)
    return Problem('reaction', 2, [Term('pde', interior, lambda u, x: d(u, 1)(x) - rho * u(x) * (1 - u(x))), Term('ic', ic, lambda u, x: u(x) - h(x[0])), Term('bc', left, lambda u, x: u(x) - u(x + shift.to(x)))], exact, ev, {'rho': rho})

def wave(c2: float=4.0, nx: int=64, nt: int=32, **_) -> Problem:
    interior, ic, left, right, ev = _time_grids((0, 1), nx, nt)
    c = math.sqrt(c2)

    def exact(X):
        x, t = (X[:, 0], X[:, 1])
        return torch.sin(PI * x) * torch.cos(c * PI * t) + 0.5 * torch.sin(4 * PI * x) * torch.cos(4 * c * PI * t)
    return Problem('wave', 2, [Term('pde', interior, lambda u, x: dn(u, 1, 2)(x) - c2 * dn(u, 0, 2)(x)), Term('ic', ic, lambda u, x: u(x) - torch.sin(PI * x[0]) - 0.5 * torch.sin(4 * PI * x[0])), Term('ic_t', ic, lambda u, x: d(u, 1)(x)), Term('bc', torch.cat([left, right]), lambda u, x: u(x))], exact, ev, {'c2': c2})

def nlpoisson(n_int: int=512, c: float=1.0, **_) -> Problem:
    exact = lambda X: torch.sin(PI * X[:, 0])
    f = lambda x: PI ** 2 * torch.sin(PI * x[0]) + c * torch.sin(PI * x[0]) ** 3
    return Problem('nlpoisson', 1, [Term('pde', grid([(0, 1)], [n_int], True, F64, CPU), lambda u, x: -dn(u, 0, 2)(x) + c * u(x) ** 3 - f(x)), Term('bc', _bc_points_1d(), lambda u, x: u(x))], exact, _eval_1d(), {'c': c})

def burgers(nu: float=0.05, nx: int=64, nt: int=32, **_) -> Problem:
    interior, ic, left, right, ev = _time_grids((-1.0, 1.0), nx, nt)

    def exact(X):
        return 1.0 / (1.0 + torch.exp((X[:, 0] + 0.25 - 0.5 * X[:, 1]) / (2 * nu)))
    ex1 = lambda x: 1.0 / (1.0 + torch.exp((x[0] + 0.25 - 0.5 * x[1]) / (2 * nu)))
    return Problem('burgers', 2, [Term('pde', interior, lambda u, x: d(u, 1)(x) + u(x) * d(u, 0)(x) - nu * dn(u, 0, 2)(x)), Term('ic', ic, lambda u, x: u(x) - ex1(x)), Term('bc', torch.cat([left, right]), lambda u, x: u(x) - ex1(x))], exact, ev, {'nu': nu})

def allen_cahn(diff: float=0.01, nx: int=64, nt: int=32, **_) -> Problem:
    interior, ic, left, right, ev = _time_grids((-1.0, 1.0), nx, nt)
    ex1 = lambda x: torch.exp(-x[1]) * torch.sin(PI * x[0])

    def exact(X):
        return torch.exp(-X[:, 1]) * torch.sin(PI * X[:, 0])

    def f(x):
        u = ex1(x)
        return -u + diff * PI ** 2 * u + 5.0 * (u ** 3 - u)
    return Problem('allen_cahn', 2, [Term('pde', interior, lambda u, x: d(u, 1)(x) - diff * dn(u, 0, 2)(x) + 5.0 * (u(x) ** 3 - u(x)) - f(x)), Term('ic', ic, lambda u, x: u(x) - ex1(x)), Term('bc', torch.cat([left, right]), lambda u, x: u(x))], exact, ev, {'diff': diff})
BUILDERS = {'order1': order1, 'poisson': poisson, 'beam': beam, 'multiscale': multiscale, 'noisy': noisy, 'helmholtz': helmholtz, 'convection': convection, 'reaction': reaction, 'wave': wave, 'nlpoisson': nlpoisson, 'burgers': burgers, 'allen_cahn': allen_cahn}
LINEAR = {'order1', 'poisson', 'beam', 'multiscale', 'noisy', 'helmholtz', 'convection', 'wave'}
