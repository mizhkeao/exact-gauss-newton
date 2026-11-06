# Taming Ill-Conditioning in PINNs: Exact Second-Order Descent via Gauss–Newton

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](docs/class_report.pdf)
[![Overleaf Project](https://img.shields.io/badge/Overleaf-Zip_Bundle-green.svg)](docs/overleaf_project.zip)
[![Code](https://img.shields.io/badge/GitHub-exact--gauss--newton-blue.svg)](https://github.com/mizhkeao/exact-gauss-newton)

Official implementation and benchmark suite for the CEE 292 project reproducing and extending Rathore et al. (*Challenges in Training PINNs: A Loss Landscape Perspective*, ICML 2024).

---

## Key Theses & Contributions

1. **The Need for Second-Order Descent:** Differential operators ($\partial_{xx}, \partial_{xxxx}$) stretch the eigenvalue spectrum of the loss Hessian $H_L(\theta)$ over $10^3\text{--}10^5+$ orders of magnitude. First-order optimizers (Adam) oscillate across canyon walls and stall; second-order curvature scaling is mathematically essential for isotropic descent.
2. **Hessian Convergence to Gauss–Newton:** In forward PINNs, deterministic residual minimization drives $r(\theta) \to 0$. As $\|r(\theta)\|_2 \to 0$, the loss Hessian asymptotically converges to the Gauss–Newton matrix:
   $$H_L(\theta) = J(\theta)^T J(\theta) + \sum_{i=1}^N r_i(\theta)\,\nabla^2 r_i(\theta) \;\xrightarrow{\|r\|_2 \to 0}\; J(\theta)^T J(\theta)$$
   - **Zero Approximation Error:** Discarding $\sum r_i \nabla^2 r_i$ preserves exact second-order curvature at the solution with zero truncation error.
   - **Guaranteed Descent:** $J^T J \succeq 0$ unconditionally eliminates the negative eigenvalues that render $H_L$ indefinite and derail Conjugate Gradient methods like Rathore et al.'s NNCG.
   - **Exact Dual $N \times N$ Cholesky:** By the push-through identity $(J^T J + \mu I_D)^{-1} J^T = J^T (J J^T + \mu I_N)^{-1}$, collocation-limited PINNs ($N < D$) are solved exactly via direct Cholesky factorization in the smaller dual dimension, bypassing iterative low-rank approximations.
   - **Noise Immunity:** $J^T J$ does not contain explicit residual terms $r_i$, remaining uncorrupted under additive noise and achieving $460\times\text{--}860\times$ lower error than NNCG.

---

## Results Summary

Head-to-head comparison across 10 clean PDEs and Gaussian-noisy PDEs on 3-layer width-50 $\tanh$ MLPs ($D \approx 5{,}250$, $N \approx 514\text{--}2{,}180$) in `float64` under identical MSE loss:

| PDE Problem | Regime / Challenge | AdamW $\to$ L-BFGS | AdamW(10%) $\to$ NNCG | AdamW $\to$ L-BFGS $\to$ NNCG | **Exact Gauss–Newton (EGN)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Poisson** | 1D linear ($-u'' = f$) | $2.92 \times 10^{-6}$ | $2.19 \times 10^{-6}$ (93 ms) | $1.63 \times 10^{-6}$ (92 ms) | **$5.16 \times 10^{-9}$** (**88 ms**) |
| **Nonlinear Poisson** | 1D nonlinear ($-u'' + u^3 = f$) | $1.72 \times 10^{-6}$ | $4.31 \times 10^{-6}$ (142 ms) | $2.39 \times 10^{-7}$ (75 ms) | **$4.42 \times 10^{-10}$** (**103 ms**) |
| **Beam** | 1D 4th-order ($u^{(4)} = f$, $\kappa > 10^{10}$) | $2.01 \times 10^{-5}$ | $1.88$ (*indef. $H_L$*) | $2.02 \times 10^{-5}$ (760 ms) | **$9.98 \times 10^{-8}$** (**406 ms**) |
| **Multiscale ($k=16$)** | 1D high-freq ($\sin 16\pi x$) | $1.42 \times 10^{-3}$ | $4.26$ (*indef. $H_L$*) | $2.36 \times 10^{-2}$ (610 ms) | **$9.90 \times 10^{-8}$** (**89 ms**) |
| **Helmholtz** | 2D oscillatory ($-\Delta u - k^2 u = f$) | $1.53 \times 10^{-3}$ | $18.6$ (*indef. $H_L$*) | $3.04 \times 10^{-1}$ (870 ms) | **$7.84 \times 10^{-5}$** (**185 ms**) |
| **Burgers** | 2D nonlinear shock ($u_t + u u_x = \nu u_{xx}$) | $7.44 \times 10^{-5}$ | $1.10 \times 10^{-2}$ (448 ms) | $5.40 \times 10^{-5}$ (337 ms) | **$2.82 \times 10^{-7}$** (**655 ms**) |
| **Reaction ($\rho=5$)** | 2D nonlinear front ($u_t = \rho u(1-u)$) | $1.92 \times 10^{-2}$ | — | $9.92 \times 10^{-3}$ (720 ms) | **$1.40 \times 10^{-4}$** (**328 ms**) |
| **Wave ($\beta=5$)** | 2D hyperbolic ($u_{tt} - c^2 u_{xx} = 0$) | $5.52 \times 10^{-2}$ | — | $1.27 \times 10^{-2}$ (29,000 ms) | **$3.08 \times 10^{-3}$** (**656 ms**) |
| **Multiscale + Noise** | Gaussian noise ($\sigma = 10^{-3}$) | $1.38 \times 10^{-3}$ | $3.96$ (*indef. $H_L$*) | $4.63 \times 10^{-3}$ (548 ms) | **$5.37 \times 10^{-6}$** (**$860\times$ lower vs. NNCG**) |
| **Multiscale + Noise** | Gaussian noise ($\sigma = 10^{-2}$) | $1.84 \times 10^{-3}$ | $3.97$ (*indef. $H_L$*) | $2.46 \times 10^{-2}$ (631 ms) | **$5.36 \times 10^{-5}$** (**$460\times$ lower vs. NNCG**) |
| **NL-Poisson + Noise** | Gaussian noise ($\sigma = 10^{-2}$) | $7.15 \times 10^{-5}$ | $7.19 \times 10^{-5}$ (139 ms) | $7.21 \times 10^{-5}$ (140 ms) | **$5.14 \times 10^{-5}$** (**104 ms/step, lowest error**) |

---

## Documents & Reports

- **Class Report (PDF, strictly 3 pages):** [`docs/class_report.pdf`](docs/class_report.pdf)
- **Overleaf Bundle:** [`docs/overleaf_project.zip`](docs/overleaf_project.zip) (contains `main.tex` and all publication figures)
- **LaTeX Source:** [`docs/class_report.tex`](docs/class_report.tex)
- **Markdown Version:** [`docs/class_report.md`](docs/class_report.md)

---

## Setup & Quickstart

```sh
# Clone repository
git clone https://github.com/mizhkeao/exact-gauss-newton.git
cd exact-gauss-newton

# Install dependencies and sync virtual environment
uv sync
uv run pytest
```

### Run an Exact Gauss–Newton (EGN) Job

```sh
uv run python -m exact_gauss_newton.run '{
  "problem": "multiscale",
  "method": "plm",
  "seed": 0,
  "width": 50,
  "depth": 3,
  "budget": 60,
  "knobs": {"k": 16},
  "cfg": {"mu0": 1.0, "damping": "marquardt", "term_weighting": true, "warmup": "adamw"},
  "dtype": "float64",
  "device": "cuda"
}' out.json
```

### Run Rathore et al.'s NysNewton-CG (NNCG)

```sh
uv run python -m exact_gauss_newton.run '{
  "problem": "multiscale",
  "method": "nncg",
  "seed": 0,
  "width": 50,
  "depth": 3,
  "budget": 60,
  "knobs": {"k": 16},
  "cfg": {"rank": 60, "cg_iters": 50, "warmup_frac": 0.1},
  "dtype": "float64",
  "device": "cuda"
}' out_nncg.json
```

---

## Repository Layout

```
src/exact_gauss_newton/
  plm.py               Exact Gauss–Newton (EGN): dual N×N solve, Marquardt damping, NTK term weighting
  nncg.py              NysNewton-CG (NNCG, Rathore et al. 2024): rank-60 Nyström preconditioner + CG
  first_order.py       AdamW, L-BFGS, and warm-up pipeline runners
  models.py            tanh MLPs with vectorized parameter operations
  residuals.py         Collocation points, boundary conditions, and residual vector r(θ)
  problems/            Library of 10 linear, nonlinear, 1D, 2D, and noisy PDE benchmarks
docs/
  class_report.pdf     3-page compiled paper report
  class_report.tex     LaTeX source
  overleaf_project.zip Overleaf-ready project bundle (main.tex + figures/)
  figures/             Publication figures (eigenvalue spectrum, convergence curves, 10-PDE suite)
scripts/
  campaign.py          Multi-GPU benchmark dispatcher
  analyze.py           Metric extraction and visualization suite
```
