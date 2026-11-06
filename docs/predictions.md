# Pre-registered predictions

Written 2026-11-05, before the campaign runs. Commit timestamp is the record.
Methods: AdamW (A), Muon+AdamW (M), L-BFGS from random init (L), Adam→L-BFGS (A→L),
Muon→L-BFGS (M→L), PLM (P: Levenberg–Marquardt on the Gauss–Newton matrix).

Tie rule: two methods tie if their ranges over seeds overlap. Rankings scored with Kendall's τ.

## Pilot evidence already in hand (60 s, width 50, seed 0, GPU)
| problem | A | A→L | P (μ₀=1e-2) |
|---|---|---|---|
| poisson | 4e-3 (best 4e-6) | 4e-6, plateaued | **2.6e-7**, still falling |
| beam (4th order) | 3e-2 | **2.4e-6**, plateaued | 3.3e-5, still falling, lost 25 s to overshoot |
| multiscale k=16 | 1e-2 | 2.5e-3 (best 2.4e-4) | **3.2e-4**, still falling |

## Predictions

**E1 ill-conditioning (derivative order 1, 2, 4).** P > A→L ≈ M→L > L > M ≈ A.
PLM's lead over A widens with order. M ≈ A: Muon's per-layer normalization does not
address operator-induced conditioning. Condition number of G grows ~ω^(2p).

**E2 noisy data (σ = 0, 1e-3, 1e-2, 1e-1).** At σ=0 P ≫ A→L. PLM's lead shrinks
with σ because Gauss–Newton ≠ Hessian when residuals stay nonzero; A→L may tie or win at σ=1e-1.

**E3 term balancing (2D Helmholtz, source scale knob).** P(weighted) > P(unweighted) > A→L > A
on boundary error; gaps grow with source scale. Muon: unpredicted.

**E4 precision (smooth Poisson, fp32 vs fp64).** fp64 floors: P ≤ 1e-8, A→L/M→L 1e-5–1e-6,
A/M 1e-3–1e-4. Local convergence slope: P ≈ 2, L-BFGS 1–2, Adam ≈ 1. In fp32 all floor at 1e-4–1e-5.

**E5 spectral bias (multiscale k = 1, 4, 16, 32).** On high-frequency error P > A→L > A;
Adam's time to fit mode k grows steeply with k, PLM's stays nearly flat. M vs A: unpredicted,
possibly Muon's first real advantage.

**E6 robustness (reaction ρ=5, main hyperparameter swept 6 decades).** Success rate
P > M > A→L > A > L. PLM nearly flat in μ₀; Muon's working lr range wider than Adam's.

**E7 cost (width 25…400).** Dense PLM time/iter scales ~D²–D³ vs ~D for others; dense PLM
loses on time-to-accuracy around width 100–200 and runs out of memory beyond. Dual-form PLM
(N×N solve) scales ~D and stays competitive. Peak memory: dense PLM ∝ D², others ∝ D.

**E8 badly-posed loss (convection β = 1, 10, 30, 40).** At high β, P reaches the lowest loss
but error does NOT follow (low-residual wrong solutions). Partly open.

**E9 Rathore benchmarks (convection, reaction, wave).** P ≥ A→L ≈ M→L > L > M ≈ A.

## PLM upgrade predictions (ablations on beam and multiscale)
- μ₀ = 1 vs 1e-2: removes most of the early overshoot; ≥20 s saved at 60 s budget.
- Marquardt scaling μ·diag(G): fewer rejected steps; helps most on beam (widest curvature spread).
- Trust region on ‖d‖: similar effect to larger μ₀; redundant with it, small additional gain.
- Term weighting: biggest effect where the operator null space is large (beam > poisson);
  reduces the early error spike, modest effect on final error.
- Adam warm-up (10% of budget): helps when PLM's early phase is wasteful (beam), hurts
  when PLM converges fast from scratch (poisson). Muon warm-up ≈ Adam warm-up.
- Dual (N×N) form: identical iterates to dense (up to rounding), ≥10× faster per iteration at width ≥100.

## Round 2 predictions (written 2026-11-05 after Phases A–G, before Phases H, S, N run)

**H restart rule.** The AdamW restart (3 stalled Jacobians or loss > 100× best) removes the k = 32
divergent seed; no change on problems that never stall. 5-seed medians stay within 2× of the 3-seed ones.

**H3 causal weighting (convection β = 30, 40).** With causal weighting at least one method gets
error < 0.1 at β = 30; PLM with causal weighting is the best of them at β = 30 and may still fail at
β = 40. The loss/error decoupling (E8) disappears where the error drops.

**S matrix-free PLM (CG on the damped normal equations, widths 100–1600, D = 20k–5.1M, 180 s).**
CG-PLM is within 3× of dual PLM at width 100–400 (inexact steps cost iterations, not accuracy);
peak memory O(D) — under 3 GB at D = 5.1M where dual needed 14 GB at 322k. CG-PLM beats
AdamW→L-BFGS at D = 1.3M and ties or loses at D = 5.1M, where each LM step costs ~100 passes
and fewer than 100 steps fit the budget. Per-step cost grows with the condition number (CG
iterations), so the lead over first-order methods shrinks with D.

**N noisy forcing (poisson, multiscale; σ = 1e-3, 1e-2, 1e-1).** Every method floors at an error
∝ σ because the least-squares solution interpolates the noisy forcing. PLM reaches the floor
fastest and shows the largest train/validation residual gap (it fits the collocation noise
exactly), but its *error* is no worse than L-BFGS's: P ≈ A→L ≈ M→L at σ ≥ 1e-2, P > A→L ≈ M→L
at σ = 1e-3. The inverse Laplacian damps white noise in the forcing, so the relative-error floor
is well below σ on Poisson (≈ 0.1 σ); on multiscale the solution's high-frequency part has
amplitude 0.1, so the same noise is relatively larger there and the floor is ≈ σ.

## Round 3 predictions (written 2026-11-06, before Phases L and P run)

The criticism to answer: every problem PLM wins by 3–4 orders is a *linear* PDE, where the residual
is affine in the output-layer weights of a fixed-feature network, so the whole problem is one
linear least-squares solve (random-feature / ELM PINN). New baseline `lstsq`: hidden layers frozen at
the paired init (first layer optionally scaled by `feat_scale` ∈ {1, 4}), output layer solved by
minimum-norm least squares; for a nonlinear PDE, Gauss–Newton on the output layer with backtracking.

Pilot evidence (CPU, seed 0, 1–60 s): LS reaches **1e-13** on poisson, beam and the cubic nlpoisson
at width 50 in ≤ 10 steps; multiscale k=16 needs feat_scale 4 (1e-5 at w=50, 3e-8 at w=800) and
k=32 fails at every width (8e-2); helmholtz 1.4e-3 at w=200; burgers 6e-2–1e-1; allen_cahn 4e-4 at w=50.

**L (linear problems, 60 s).** Frozen-feature LS *beats* PLM by 4–6 orders on order1, poisson and
beam at every width (machine precision in < 2 s). PLM beats LS at every width on multiscale k=32
(LS ≈ 0.1, PLM ≈ 1e-6) and on helmholtz at w=50 (LS ≥ 1, PLM 1e-5); on k=16 LS needs
feat_scale=4 and w ≥ 200 to tie PLM. So the PLM headline table overstates PLM: on smooth linear
problems the right comparison is a 1 s solve, and PLM loses it.

**P (nonlinear problems, 300 s).** Nonlinearity alone does *not* rescue PLM: on nlpoisson (smooth
solution, cubic term) output-layer Gauss–Newton ties or beats PLM (both ≤ 1e-7, LS at 1e-13).
What matters is whether frozen random features can represent the solution. Ranking:
- burgers (steep moving front, ν = 0.05): P (< 1e-3) > A→L ≈ M→L (1e-2) > LS (6e-2–1e-1).
- reaction (sharpening Gaussian front): P (3e-4) > M→L (1e-2) > LS (> 1e-1) ≈ A→L (1.0).
- allen_cahn (manufactured, smooth): LS (1e-4) ≈ A→L ≈ M→L, P best (≤ 1e-6).
- nlpoisson: LS (1e-13) > P (1e-8) > A→L ≈ M→L (1e-5).
Summary prediction: PLM is the only method that is competitive on *every* problem, but it is the
best method only where the solution has structure that random features at init do not span
(sharp fronts, high frequency, 2D). The honest claim becomes "PLM = feature learning with a
Gauss–Newton solve"; its edge over LS is the feature learning, not the solver.

## Round 4 predictions (written 2026-11-06, before Phase V runs)

Two PLM upgrades aimed at the two failure modes of rounds 1–3. **Variable projection (`vp`)**: the
output layer is solved exactly (minimum-norm least squares; output-layer Gauss–Newton for PDEs nonlinear
in u) at every trial point, and LM acts on the hidden layers through Kaufman's projected Jacobian
(I − QQᵀ)J_h. This folds the frozen-feature least-squares baseline into PLM. **Collocation resampling
(`rs`)**: fresh uniform PDE collocation points at every Jacobian; the logged loss and the validation
residual stay on the original fixed grid. Aimed at the low-loss wrong solutions (k = 32: 2/5 seeds,
reaction: 1/5), which fit the fixed grid and nothing between. `vprs` = both. Baselines are PLM rec2
from Phases H, H2, P at the same budgets (60 s for the 1D linear problems, 300 s otherwise).

Pilot (CPU, multiscale k = 16, w = 50, 40 s, seed 0): rec2 3.5e-6 in 140 Jacobians; vp 2.7e-6 in 59
(each Jacobian costs about 2.4× more because every trial point re-solves the output layer).

- **order1, poisson, beam (smooth, linear).** vp reaches the LS floor (≤ 1e-12) on every seed at the
  first Jacobian after warm-up: 4–6 orders below rec2 (1e-7 to 5e-9). rs alone changes nothing (≤ 2×).
- **nlpoisson (smooth, cubic).** Same: vp ≤ 1e-12 vs rec2 4e-10; rs no change.
- **multiscale k = 16.** vp within 2× of rec2 (1e-7): the output layer is not the bottleneck once
  features are learned, and vp's costlier Jacobians roughly cancel its better per-step progress.
- **multiscale k = 32 (2/5 bad seeds in H).** rs removes the bad seeds (all 5 seeds < 1e-4; median
  within 3× of rec2's good seeds, 2e-6). vp alone keeps the bad seeds. The train/validation residual
  gap of the bad seeds (loss 1e-8, error 1.0) disappears under rs.
- **reaction (1/5 bad seed in H2).** rs removes the bad seed; median within 2× of rec2 (8e-5).
- **helmholtz, allen_cahn, burgers (features must be learned, no bad seeds).** vp ties rec2 (within
  3×); rs within 3× of rec2 but not better: with a fixed grid these problems were not overfitting
  (val ≈ train loss in H2/P).
- **Cost.** vp doubles the time per Jacobian at w = 50 (an extra N × 51 Jacobian and SVD per trial
  point); rs adds nothing measurable. vprs is never worse than the better of vp and rs by more than 3×.

Falsifiable headline: with vprs, PLM is never worse than the tuned frozen-feature LS baseline by
more than 10× on any of the ten problems, and no seed ends with error > 1e-2 on k = 32 or reaction.

## Round 5 predictions (written 2026-11-06, before Phase W runs)

**Two-stage method.** Stage 1: PLM rec2 (with lagged Jacobian and collocation resampling on the 2D /
time-dependent problems, as recommended after Phase V) for a fraction `budget_frac` ∈ {0.1, 0.3, 1.0} of the
budget, learning the basis. Stage 2: one output-layer solve on the learned features (least squares; up to 20
Gauss–Newton steps for a PDE nonlinear in u), on the original grid. Baselines at equal budget: rec2 (H, H2,
P), rec2 + rs (V), frozen-feature LS (L, P), interleaved varpro (V). Budgets 60 s (1D linear) / 300 s.

- **Smooth problems (order1, poisson, beam, nlpoisson).** Every switch point reaches the LS floor
  (≤ 1e-12), including fs0.1: the final solve does all the work and stage 1 is irrelevant. Matches LS and
  interleaved varpro, 4–6 orders below rec2.
- **multiscale k = 16.** fs1.0 improves on rec2 (1e-7) by 3–30×: PLM's late iterations are output-layer
  polishing that the solve does exactly. fs0.3 ties rec2; fs0.1 is 10–100× worse than rec2 (features not
  yet learned; the solve cannot make up for them).
- **multiscale k = 32.** The final solve neither fixes nor creates bad seeds: fs1.0 has the same 2 of 5 bad
  seeds as rec2, with the good seeds improved 3–30×. fs0.1 fails on most seeds (features unlearned → LS
  plateau 0.09).
- **helmholtz, burgers, allen_cahn.** fs1.0 improves on the stage-1 baseline (rec2 + rs on helmholtz, 6.5e-6;
  rec2 on burgers 2.8e-7, allen_cahn 4e-5) by 2–10×; fs0.3 within 3× of it; fs0.1 5–50× worse.
- **reaction.** rs keeps the bad seeds away (0 of 5) at every switch point; fs1.0 within 3× of rec2 + rs
  (1.4e-4), the final GN solve on a stiff nonlinear residual gains little.
- **Timing claim.** On every problem fs0.3 reaches an error no worse than 3× rec2's at the full budget,
  i.e. the two-stage method matches full-budget PLM at 30 % of the cost on the feature-learning problems and
  beats it by orders on the smooth ones.

Falsifiable headline: with fs1.0, the two-stage method is never worse than the better of (rec2 + rs, LS) by
more than 3× on any of the ten problems, and is ≥ 10× better than that on at least six of them.
