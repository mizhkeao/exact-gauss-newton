# Findings (running log)

Setup unless stated: width 50, depth 3 tanh MLP (D = 5,251), float64, 60 s training budget per run
(evaluation excluded), 3 seeds, one RTX PRO 5000 per job. Default hyperparameters, **no per-method
tuning yet** (AdamW lr 1e-3, Muon lr 0.02 + AdamW 1e-3 on non-matrix params, L-BFGS history 50
strong-Wolfe, PLM mu0 = 1 identity damping dual solver). Figures in `results/figures/<phase>/`.

## Phase A: six methods on five 1D problems

Final relative L2 error, median (min–max) over 3 seeds:

| problem | AdamW | Muon | L-BFGS | AdamW→L-BFGS | Muon→L-BFGS | PLM |
|---|---|---|---|---|---|---|
| order1 | 1.3e-4 | 9.9e-3 | 7.4e-5 | 5.8e-5 | 1.0e-5 | **1.2e-8** |
| poisson | 6.9e-4 | 3.2e-2 | 2.2e-6 | 2.4e-6 | 6.5e-7 | **5.4e-8** |
| beam (4th order) | 2.8e-2 | 3.3e-2 | 4.5e-6 | 2.3e-5 | **1.9e-6** | 6.3e-6 |
| multiscale k=16 | 8.8e-3 | 1.2e-1 | 2.1e-3 | 1.2e-3 | 1.5e-3 | **3.2e-7** |
| noisy σ=1e-2 | 5.8e-4 | 3.7e-2 | 6.5e-4 | 6.5e-4 | 6.5e-4 | 6.5e-4 |

### Against the predictions
- **E1 (P > A→L ≈ M→L > L > M ≈ A), partly.** P > A→L holds on all three orders. But the lead
  does *not* widen with derivative order: it shrinks. Cause: PLM's per-iteration cost grows with
  the derivative order of the Jacobian (0.04 s/it order 1, 0.08 s/it Poisson, 0.38 s/it beam), so
  on beam PLM gets 157 iterations and is still descending at 60 s. Also L-BFGS alone is *not* the
  worst quasi-Newton variant: it ties for best on beam. M ≈ A: false in the direction M < A.
- **E5 (spectral bias): confirmed strongly.** PLM is 4 orders of magnitude better at k = 16.
- **E2 (noise): σ = 1e-2 sits at the noise floor for every method**, so the knob must go lower
  (Phase E runs σ ∈ {0, 1e-3, 1e-2, 1e-1}).
- **Unpredicted: Muon→L-BFGS beats AdamW→L-BFGS on every problem** (3–10×) although Muon alone is
  the worst method. The original project question ("is Muon a better starting point for L-BFGS?")
  gets a tentative yes at default hyperparameters. Needs the lr sweep (Phase G) to confirm it is
  not an artifact of an over-large Muon lr making the warm-up phase short-and-rough.

### Cost
- First-order methods: 2–3 ms/iteration, ~35 MB. L-BFGS: 5–9 ms per inner iteration.
- PLM: 40–380 ms/iteration, 0.6–5 GB peak (beam: the 4th-order Jacobian's intermediates).
  PLM uses 0 "gradient" evals and 150–1,500 Jacobian evals per run; each Jacobian is roughly N
  reverse passes.

## Phase B: PLM ablations (width 50, 60 s, 3 seeds)

| variant | beam | multiscale k=16 | poisson |
|---|---|---|---|
| base (μ₀=1, identity, dual) | 3.5e-6 | 3.4e-7 | 4.8e-8 |
| base μ₀=1e-2 | 1.4e-6 | 8.1e-7 | 2.1e-8 |
| dense solver | 5.0e-5 | 2.3e-5 | 9.9e-7 |
| Marquardt diag(G) damping | 2.3e-7 | **5.2e+0 (fails)** | **1.7e-9** |
| trust region Δ₀=1 | 3.1e-6 | 6.9e-7 | 4.0e-8 |
| NTK term weighting | 2.4e-7 | 6.4e-7 | 8.9e-8 |
| AdamW warm-up 10% | 1.2e-6 | **6.1e-8** | 2.1e-8 |
| Muon warm-up 10% | 1.9e-7 | 3.5e-7 | 3.8e-8 |
| all (Marquardt + weighting + AdamW warm-up) | **8.8e-8** | 1.4e-1 (fails, via Marquardt) | 4.8e-9 |

- **Beam is fixed by the upgrades.** Marquardt, weighting, and Muon warm-up each give ~15×;
  combined, 40× (8.8e-8), which now beats every other method on beam (best other: 1.9e-6).
  Matches the prediction that term weighting matters most where the operator null space is
  large (beam cubic null space > Poisson linear).
- **Marquardt damping fails catastrophically on multiscale: loss 4e-7, error 5.2.** Diagnosis
  from the per-term logs: PDE and BC residuals both go to ~0 at the 512 collocation points while
  the function oscillates between them. diag(G) scaling divides the step by each parameter's
  own curvature, so parameters with tiny curvature (which control sub-grid-scale content) take
  enormous steps. Fix implemented: floor s_i ≥ 1e-3·max(s) (Phase B2), plus a held-out
  half-grid-shifted residual is now logged (`val_loss`) to catch collocation overfitting.
- **Dense vs dual: iterates are the same, dual is faster.** The per-iteration trajectories
  coincide (beam it 100: 3.3e-5 vs it 101: 3.0e-5); dense simply gets fewer iterations
  (0.58 vs 0.36 s/it at D=5k, 16 s/it at D=20k). Prediction confirmed.
- **Trust region alone does nothing at width 50** (as predicted: redundant with μ₀=1).
- μ₀=1e-2 vs 1: mixed (better on beam and Poisson, worse on multiscale). The "overshoot"
  diagnosis was only partly right; see Phase G for the full μ₀ sweep.

## Phase C: scale sweep (multiscale k=16, widths 25–400, D = 1.3k–322k)

| D | AdamW | Muon | AdamW→L-BFGS | PLM dual | PLM dense |
|---|---|---|---|---|---|
| 1.3k | 1.9 | 1.6e-1 | 4.4e-2 | **8.1e-8** | 1.7e-7 |
| 5.3k | 9.9e-3 | 2.0e-1 | 3.1e-3 | **5.0e-7** | 2.6e-5 |
| 20k | 1.0e-2 | 3.1e-2 | 5.7e-3 | **7.6e-6** | 2.7 (4 iterations) |
| 81k | 8.3e-3 | 5.8e-2 | **5.2e-4** | 1.7 (fails) | OOM/skipped |
| 322k | 1.1e-2 | 1.3e-1 | **4.3e-4** | 2.1 (fails) | — |

Cost (s/iteration, peak MB): first-order 3–6 ms and 35–90 MB at every width; L-BFGS 10–30 ms,
up to 370 MB; PLM dual 0.02 → 2.2 s/it and 0.5 → 14 GB (memory is the N×D Jacobian and the
functorch intermediates, not the N×N system).

- **Prediction E7 confirmed in shape**: dense PLM dies at D≈20k; dual scales ~linearly in D
  per iteration. But **dual PLM fails outright at D ≥ 81k**, which was not predicted. The logs
  show μ being driven to 1e3–1e5 with a third of trial steps rejected: the linearized model is
  untrustworthy for the minimum-norm steps that D ≫ N allows, and 30–90 iterations is too few.
  Phase C2 tests the three obvious rescues (trust region, AdamW warm-up, lagged Jacobian).
- Note AdamW→L-BFGS *improves* with width while AdamW and Muon do not.

## Phase D: 2D and time-dependent (width 50, 60 s, 3 seeds)

| problem | AdamW | Muon | L-BFGS | AdamW→L-BFGS | Muon→L-BFGS | PLM (weighted) |
|---|---|---|---|---|---|---|
| helmholtz a2=4 | 2.1e-1 | 2.0e-1 | 9.9e-3 | 2.9e-2 | 9.5e-3 | **1.7e-4** |
| reaction ρ=5 | 1.0 | 2.4e-2 | 5.7e-2 | 6.9e-2 | **6.0e-3** | 6.7e-2 |
| convection β=30 | 1.3 | 1.5 | 1.1 | 1.3 | 1.3 | 1.4 |
| wave | 4.8e-1 | 5.8e-1 | 4.1e-1 | 4.1e-1 | 4.0e-1 | 5.4e-1 |

- **Helmholtz (E3): PLM wins by 60×**, as predicted.
- **Reaction: Muon is the story.** Muon alone (2.4e-2) beats AdamW (1.0, stuck at the trivial
  solution) by 40×, and Muon→L-BFGS is the best method. PLM is mid-pack: 1 s/iteration on a
  2D problem with N=2,300 residuals leaves ~60 iterations, and the Gauss–Newton approximation
  drops the u(1−u) curvature term.
- **Convection β=30 and wave: nobody learns anything in 60 s** (errors ≈ 1 and ≈ 0.4). Phase D2
  reruns these with 300 s. Convection at β=30 is Rathore et al.'s hard case, so the E8
  scatter will need the β sweep (Phase E).

## Phase B2: second ablation round (width 50, 60 s, 3 seeds)

| variant | beam | multiscale k=16 | poisson |
|---|---|---|---|
| Marquardt with floor 1e-3·max | 3.2e-7 | **2.4e-7 (was 5.2)** | 4.8e-9 |
| lagged Jacobian ×4 | 3.5e-6 | 1.6e-6 (one seed blew up to 4) | 5.1e-8 |
| lagged Jacobian ×8 | 1.9e-6 | **1.3e-7** | 3.9e-8 |
| Marquardt-floor + ×4 lag + weighting | **8.9e-8** | 1.4e-6 | **2.9e-9** |

The floor fixes the collocation-overfitting failure with no loss elsewhere. Lagging the
Jacobian is neutral-to-helpful at width 50 (it is a cost measure, not an accuracy one).

## Phase C2: rescuing PLM at large width (multiscale k=16, 60 s, 3 seeds)

| D | base | trust region | lag ×4 | **AdamW warm-up 10%** | warm + trust + lag | best non-PLM (Phase C) |
|---|---|---|---|---|---|---|
| 20k | 8.2e-6 | 4.9e-5 | 3.5 | **5.7e-7** | 8.6e-7 | 5.7e-3 |
| 81k | 2.1 | 4.9 | 3.6 | **7.0e-6** | 5.9e-5 | 5.2e-4 |
| 322k | 2.1 | 2.9 | 2.9 | **3.1e-4** | 2.4e-4 | 4.3e-4 |

- **The AdamW warm-up alone rescues PLM at every width**: 75× better than the best other
  method at D=81k, and a tie at D=322k (where PLM gets only ~30 Jacobians in the 54 s left).
- **The trust region does not help — prediction wrong.** The failure is not step *size*, it is
  *where* the steps are taken: from the random initialization the Gauss–Newton model is simply
  not trustworthy, and shrinking the step does not change that. Ten percent of AdamW moves to a
  region where the linearization holds, and from there PLM converges quadratically.
- Lagging halves PLM's per-iteration cost at D=322k (0.9 s vs 2.0 s) but hurts from a cold start.
- Recommended PLM for everything that follows: μ₀ = 1, floored Marquardt damping, NTK term
  weighting, 10% AdamW warm-up; add ×4 lag for large N or D.

## Phase D2: 2D and time-dependent at 300 s (width 50, 2 seeds)

| problem | L-BFGS | AdamW→L-BFGS | Muon→L-BFGS | PLM weighted | PLM weighted + warm-up + lag ×4 |
|---|---|---|---|---|---|
| helmholtz a2=4 | 2.5e-3 | 1.7e-3 | 3.9e-3 | **1.2e-5** | 7.8e-5 |
| reaction ρ=5 | 5.1e-1 | 1.0 | 1.1e-2 | 2.1e-3 | **3.4e-4** |
| wave | 3.6e-1 | 3.4e-1 | 3.5e-1 | 2.6e-1 | **8.4e-3** |
| convection β=30 | 1.5 | 1.8 | 1.6 | 2.4 | 1.7 |

- **Reaction: with 5× the budget PLM wins (30× over Muon→L-BFGS).** The Phase D loss was
  budget, not the missing curvature term. AdamW→L-BFGS stays at the trivial solution u ≡ 0
  (error 1.0) in both seeds; Muon→L-BFGS does not. Muon's advantage on this problem is robust.
- **Wave: PLM with warm-up and lag is the only method that learns the solution** (8.4e-3; every
  L-BFGS variant is stuck near 0.35, i.e. it fits the initial condition and little else). The
  wave PDE has second derivatives in both coordinates, so a Jacobian costs ~2 s; the ×4 lag
  triples the iteration count (150 → 469) and that is what makes the difference.
- **Convection β=30: still nobody at 300 s.** Errors 1.5–2.4 across all methods, including PLM
  with the lowest loss. Consistent with the badly-posed-loss prediction (E8); the β sweep (E2)
  will show where it breaks.
- Cost on these 2D problems: first-order/L-BFGS 7–10 ms per inner iteration; PLM 0.35–2 s per
  Jacobian iteration, so 150–900 iterations in 300 s.

## Phase G: hyperparameter robustness (E6; width 50, 60 s, 3 seeds)

Each method's main hyperparameter swept: AdamW lr and Muon lr over 2 decades (5 values),
PLM μ₀ over 8 decades (1e-4 … 1e4). Median final error, best → worst over the sweep:

| problem | AdamW | AdamW→L-BFGS | Muon | Muon→L-BFGS | PLM |
|---|---|---|---|---|---|
| poisson | 1.3e-5 → 3.6e-3 | 9.0e-7 → 3.2e-6 | 2.8e-3 → 1.5e-1 | 4.7e-7 → 2.1e-6 | **2.0e-8 → 5.6e-8** |
| beam | 1.4e-2 → 1.0 (diverges) | 7.6e-7 → 3.2e-5 | 1.6e-2 → 1.0 (diverges) | 3.1e-6 → 1.7e-5 | **1.4e-6 → 8.0e-6** |
| multiscale | 8.0e-4 → 2.9e-1 | 1.1e-3 → 2.8e-3 | 1.6e-2 → 3.0e-1 | 1.5e-4 → 1.0e-2 | **3.0e-7 → 6.2e-7** |

- **PLM is flat across eight decades of μ₀** (worst/best ≤ 6), as predicted. The early
  "overshoot at μ₀ = 1e-2" turned out to cost nothing in the end. Levenberg–Marquardt's
  damping adaptation removes the hyperparameter.
- **Two-stage methods are robust, single-stage ones are not**: AdamW and Muon both span
  2–3 orders of magnitude and both diverge at the top of their range on beam.
- **Muon's "wider working range" did not materialize**: Muon alone is worse than AdamW at
  *every* learning rate on every problem (best Muon 2.8e-3 vs best AdamW 1.3e-5 on Poisson).
  Its best setting is the smallest lr tried (1e-3), so the default 0.02 was far too large, but
  even tuned it does not catch AdamW. Prediction M > A for robustness: wrong.
- **Muon→L-BFGS vs AdamW→L-BFGS after tuning**: Muon better on Poisson (4.7e-7 vs 9.0e-7)
  and multiscale (1.5e-4 vs 1.1e-3, 7×), AdamW better on beam (7.6e-7 vs 3.1e-6). So the
  Phase A result ("Muon→L-BFGS wins everywhere") softens to "comparable, Muon ahead where
  spectral bias matters". Note the opposite lr preferences: Muon→L-BFGS likes a *large* Muon
  lr on Poisson (0.1 best) but a small one on multiscale (1e-3 best).
- Robustness ranking observed: P > A→L ≈ M→L > A ≈ M (predicted P > M > A→L > A). PLM's
  position confirmed; Muon's was wrong.

## Phase E: knob sweeps with the recommended PLM (width 50, 60 s, 3 seeds)

PLM here is the Phase-C2 recommendation (μ₀ = 1, floored Marquardt, NTK weighting, 10 % AdamW
warm-up), tagged `rec`. Figures: `results/figures/E/sweep_multiscale_k.png`, `sweep_noisy_sigma.png`.

**E5 spectral bias (multiscale k = 1, 4, 16, 32):**

| k | AdamW | Muon | AdamW→L-BFGS | Muon→L-BFGS | PLM rec |
|---|---|---|---|---|---|
| 1 | 3.0e-3 | 2.3e-2 | 1.6e-6 | 4.1e-7 | **5.1e-9** |
| 4 | 1.9e-5 | 5.3e-3 | 1.3e-6 | 9.0e-7 | **7.7e-9** |
| 16 | 4.8e-3 | 1.2e-1 | 1.4e-3 | 2.9e-3 | **2.2e-7** |
| 32 | 8.4 | 9.0e-1 | 4.8 | 3.6e-1 | **4.9e-6** (one seed diverges to 2.6) |

- **P > A→L > A at every k, as predicted (τ = 1 for k ≤ 16).** PLM's error grows ~10³ from
  k = 1 to 32; every other method's grows ~10⁶ and all of them are *worse than u = 0* at k = 32.
  The gap widens with k, which is the spectral-bias prediction.
- **Muon finally shows something at k = 32**: Muon (0.9) and Muon→L-BFGS (0.36) beat their
  AdamW counterparts (8.4, 4.8) by 10×, the first problem where Muon alone beats AdamW. At
  k ≤ 16 it is 10–100× worse. So "Muon's first real advantage" (predictions, E5 note) exists
  but only where everyone has already failed.
- **One PLM seed at k = 32 diverged** (final loss 2.6e2 while the other two reached 4e-4–2e-3).
  That seed's best-so-far error was 0.22, so it was never on track: the AdamW warm-up left it
  somewhere the linearization is bad. Under the pre-registered tie rule this makes P ≈ A→L
  at k = 32 (τ = 0.82) although the medians differ by 10⁶.

**E2 noise (σ = 0, 1e-3, 1e-2, 1e-1; A→L, M→L, P):**

| σ | AdamW→L-BFGS | Muon→L-BFGS | PLM rec |
|---|---|---|---|
| 0 | 2.6e-6 | 1.5e-6 | **1.4e-8** |
| 1e-3 | 6.5e-5 | 6.5e-5 | 6.4e-5 |
| 1e-2 | 6.5e-4 | 6.5e-4 | 6.3e-4 |
| 1e-1 | 6.5e-3 | 6.5e-3 | 6.2e-3 |

- **As predicted: at σ = 0 PLM leads by 100×; for σ > 0 everyone sits at the noise floor**
  (error ≈ 0.065 σ, linear in σ). PLM is 3–5 % below the others at every σ, so the
  Gauss–Newton ≠ Hessian argument did not even cost it a tie. The noise experiment measures the
  problem, not the optimizer: the pre-registered "A→L may tie or win at σ = 1e-1" was wrong
  in direction only by a few percent.

**E1 derivative order with the recommended PLM (order1, poisson, beam):**

| problem | AdamW→L-BFGS | Muon→L-BFGS | PLM rec | PLM base (Phase A) |
|---|---|---|---|---|
| order1 | 3.1e-5 | 1.5e-5 | **4.1e-9** | 1.2e-8 |
| poisson | 2.9e-6 | 5.1e-7 | **4.7e-9** | 5.4e-8 |
| beam (4th order) | 2.7e-6 | 2.2e-6 | **1.2e-7** | 6.3e-6 |

- The upgrades turn the Phase-A beam loss into a 20× win (PLM reaches 1e-7 at 45 s while both
  L-BFGS variants plateau at 1–3e-6 from 25 s on; `curves_beam_w50_float64.png`).
- **The lead still shrinks with order** (7,000× → 600× → 20×), so that half of E1 stays
  falsified. The reason is cost, not conditioning: a beam Jacobian costs 0.41 s against
  0.09 s for Poisson, so PLM gets 147 iterations instead of 690.

## Phase F: precision, fp32 vs fp64 (poisson, width 50, 60 s, 3 seeds; E4)

| dtype | AdamW | Muon | L-BFGS | AdamW→L-BFGS | Muon→L-BFGS | PLM rec |
|---|---|---|---|---|---|---|
| float32 | 1.9e-5 | 2.3e-2 | 3.1e-6 | 3.9e-6 | 2.1e-6 | **2.3e-7** |
| float64 | 8.1e-5 | 2.6e-2 | 2.2e-6 | 2.7e-6 | 4.3e-7 | **7.0e-9** |

- **fp64 floors as predicted for PLM (≤ 1e-8) and the L-BFGS variants (1e-6–1e-7).** AdamW and
  Muon were predicted at 1e-3–1e-4; AdamW is better (8e-5) and Muon worse (3e-2).
- **fp32 did not floor everything at 1e-4–1e-5 — wrong.** L-BFGS variants lose only ~1.5×
  (3e-6 vs 2e-6); PLM loses 30× (2.3e-7 vs 7e-9) but still leads by 10×. fp32 hurts the method
  that was going to go deepest.
- **How PLM hits the fp32 wall is instructive.** The dual system J Jᵀ + μI loses positive
  definiteness in fp32 once μ < ε₃₂·‖J Jᵀ‖ ≈ 1e-7 · 1e2; the first campaign run crashed in
  `cholesky`. Treating the failed factorization as a rejected step (μ ×4, now in `plm.py`) makes
  PLM climb μ to 1e16 and stop by its own rule after 8–11 s at 2.3e-7, i.e. it *knows* when the
  precision is exhausted, while L-BFGS keeps taking null steps to the budget. A square-root-free
  solver (QR of J̃ instead of Cholesky of J̃J̃ᵀ) would gain back about half the lost digits;
  not implemented.
- Cost: fp32 halves nothing here (first-order 3 ms/it either way at D = 5k — the GPU is
  latency-bound at this size), so the precision choice is free and fp64 should be the default.

## Phase E2: convection β sweep at 300 s (width 50, 2 seeds; E8)

| β | AdamW→L-BFGS | Muon→L-BFGS | PLM rec + lag ×4 |
|---|---|---|---|
| 1 | 6.6e-4 | 1.8e-2 | **3.0e-6** |
| 10 | 7.7e-4 | 5.5e-2 | **2.5e-5** |
| 30 | 1.8 | 1.6 | 1.8 |
| 40 | 1.5 | 1.3 | 1.4 |

Figure: `results/figures/E2/loss_vs_error.png` (final loss vs final error, every run).

- **E8 confirmed in its strongest form.** PLM reaches a training loss of 1e-11–1e-10 at *every* β
  (3–4 orders below the L-BFGS variants' 2e-7–1e-6). At β ≤ 10 that loss buys an error of
  3e-6–3e-5, 30–200× better than AdamW→L-BFGS. At β ≥ 30 the same 1e-11 loss comes with an
  error of 1.4–2.1, worse than u ≡ 0. The problem has a near-zero-residual solution that is not
  the PDE's solution (the periodic boundary and initial condition are satisfied to 1e-11 by a
  function that is wrong in the interior), and a better optimizer finds *that* faster. The
  cliff between β = 10 and β = 30 is where Rathore et al. report the loss landscape going bad,
  and no optimizer in this study crosses it: the fix must come from the loss (causal/curriculum
  weighting, more collocation points along characteristics), not the optimizer.
- **Muon→L-BFGS is 10–70× worse than AdamW→L-BFGS at β ≤ 10**, the opposite of every 1D
  result. Its loss is the same (5e-7 vs 3e-7) but its error is not — the same loss/error
  decoupling at a milder scale, suggesting the Muon warm start lands in a different basin.

## Phase H: 5-seed confirmation of the 1D results, PLM with the restart rule (width 50, 60 s)

| problem | AdamW→L-BFGS | Muon→L-BFGS | PLM rec2 (restart rule) |
|---|---|---|---|
| order1 | 2.7e-5 (2e-5–9e-5) | 1.5e-5 (8e-6–2e-5) | **4.6e-9** (4e-9–2e-8) |
| poisson | 2.9e-6 (2e-6–4e-6) | 8.5e-7 (4e-7–3e-6) | **5.2e-9** (2e-9–2e-8) |
| beam | 2.0e-5 (1e-6–2e-3) | 5.7e-6 (1e-6–9e-5) | **1.0e-7** (4e-8–1e-6) |
| multiscale k=16 | 1.4e-3 (3e-4–3e-3) | 3.1e-3 (8e-4–1e-2) | **9.9e-8** (6e-8–1e-7) |
| multiscale k=32 | 4.4 (3–6) | 0.36 (4e-2–0.9) | **2.4e-6** (4e-7–2.6) |

Median (min–max) over seeds 0–4.

- **The 3-seed medians hold at 5 seeds** (all within 2×), as predicted. Muon→L-BFGS beats
  AdamW→L-BFGS on 4 of 5 problems by 2–12×, the one 1D result in Muon's favour that survives.
- **The restart rule did nothing.** At k = 32 two of five PLM seeds end at error 2.6 and 0.93
  and the restart counter is 0 in every run. Those seeds never stall and never blow up by the
  rule's definition: the LM steps keep being accepted and the loss keeps falling (to 1e-8) while
  the error climbs from a best of 0.06–0.12. This is the convection failure mode (a low-loss
  wrong function), not a divergence, so a loss-driven restart cannot see it. A first version of
  the rule that fired on the early raw-loss overshoot was worse (it restarted every iteration;
  fixed with a patience of 10 Jacobians before these runs). Prediction "restart removes the k = 32
  divergent seed": **falsified**.

## Phase N: noisy forcing (width 50, 60 s, 3 seeds)

The PDE residual target carries N(0, σ²) noise at every collocation point (fixed per seed);
validation residuals on a shifted grid see clean forcing. Figure: `docs/figures/noise_sweep.png`.

| σ | poisson A→L | poisson M→L | poisson PLM | multiscale A→L | multiscale M→L | multiscale PLM |
|---|---|---|---|---|---|---|
| 1e-3 | 6.5e-6 | 6.0e-6 | 5.4e-6 | 1.4e-3 | 2.4e-3 | **5.4e-6** |
| 1e-2 | 5.5e-5 | 5.4e-5 | 5.3e-5 | 2.9e-3 | 1.4e-3 | **5.4e-5** |
| 1e-1 | 5.4e-4 | 5.4e-4 | 5.4e-4 | 7.1e-4 | 2.5e-3 | **5.4e-4** |

- **The error floor is 5.4e-3 σ, exactly linear in σ, and identical for every method on
  poisson** (the three methods agree to two digits at every σ: they all reach the least-squares
  solution of the noisy problem). The inverse Laplacian damps white noise in the forcing by
  ~200×; the prediction said ≈ 0.1 σ, the observed constant is 0.0054 — right scaling, wrong
  constant.
- **On multiscale only PLM reaches the noise floor.** The L-BFGS variants sit at 1e-3 regardless
  of σ (optimizer-limited, as in Phase E), PLM at 5.4e-3 σ (noise-limited): 250× better at
  σ = 1e-3, 1.3× at σ = 0.1. The multiscale floor is the *same* constant as poisson's, not ≈ σ as
  predicted: the noise is damped by the operator, not measured against the solution's small
  high-frequency part.
- **PLM does not overfit the collocation noise.** Its training loss settles at 0.5 σ² (N−2)/N to
  within 10 % — the irreducible noise energy — and its *validation* loss on the shifted grid is
  10–40× lower on poisson and 2–4× lower on multiscale. A width-50 tanh network cannot interpolate
  white noise at 512 points, so the architecture regularizes and Gauss–Newton cannot overfit what
  the model cannot represent. Prediction "PLM shows the largest train/val gap" **falsified**;
  "P ≈ A→L ≈ M→L at σ ≥ 1e-2 on poisson, P > both at σ = 1e-3" held on poisson, and PLM is better
  than predicted on multiscale.

## Phase H3: causal weighting on convection β ≥ 30 (width 50, 300 s, 3 seeds)

| β | AdamW | AdamW→L-BFGS | Muon→L-BFGS | PLM rec2 + lag ×4 |
|---|---|---|---|---|
| 30 (causal) | 1.3 | 1.2 | 1.5 | 1.2 (0.78–1.7) |
| 40 (causal) | 1.5 | 1.5 | 1.6 | 1.6 |

- **Causal weighting (Wang et al. 2022, ε annealed 1e-2 → 100) does not rescue anybody.** Every
  method still ends at error 1.2–1.6; PLM still reaches loss ≈ 3e-11, and its best iterate along
  the way is 0.8–0.9. Prediction "at least one method gets error < 0.1 at β = 30": **falsified**.
- Likely reasons, untested: the 64 × 32 collocation grid cannot resolve 30 characteristic
  crossings in the time window (Wang et al. use 10⁴–10⁵ points), and the tanh MLP at width 50
  struggles with the 30-period oscillation in t. The loss/error decoupling (E8) is untouched: the
  spurious low-residual function exists on this grid with or without causal weights.

## Phase S: matrix-free PLM at scale (multiscale k=16, widths 100–1600, D = 20k–5.1M, 180 s, 3 seeds)

PLM `solver="cg"`: never forms J; J v by forward-mode `linearize`, Jᵀu by `vjp`, the damped normal
equations solved by Jacobi-PCG (50 iterations, tolerance 1e-2, 4 Hutchinson probes for diag(JᵀJ)).
Figure: `docs/figures/scale_sweep_cg.png`.

| width | D | AdamW | Muon | AdamW→L-BFGS | PLM dual | PLM CG | CG peak MB | LM steps (CG) |
|---|---|---|---|---|---|---|---|---|
| 100 | 20k | (C: 2e-3) | (C: 1e-2) | (C: 1e-3) | **1.7e-8** (14 GB at w=400) | 1.1e-4 | 240 | 292 |
| 400 | 322k | 8.2e-2 | 4.4e-2 | 1.0e-4 | **9.6e-7** (14 GB) | 5.3e-5 | 1 000 | 176 |
| 800 | 1.3M | 3.9e-2 | 8.3e-2 | 8.7e-4 | OOM | **3.1e-4** (4e-5–0.6) | 2 300 | 70 |
| 1600 | 5.1M | 0.39 | 0.15 | 3.5e-3 | OOM | **8.8e-4** (3e-4–9e-4) | 6 000 | 25 |

- **Memory scales as predicted**: CG-PLM is O(D) — 6 GB at 5.1M parameters where dual PLM needed
  14 GB at 322k. Not under 3 GB as predicted (the `linearize`/`vjp` closures keep the forward
  tape; 6 GB vs AdamW→L-BFGS's 4.8 GB at the same D).
- **Accuracy does not scale with PLM.** CG-PLM is 1000× worse than dual PLM at the *same* D
  (1e-4 vs 2e-8 at 20k; 5e-5 vs 1e-6 at 322k) — prediction "within 3×": **badly falsified**. Its
  error is nearly flat in D (1e-4 → 9e-4), which says it is limited by the inexact solve, not by
  the problem: 50 Jacobi-PCG iterations at tolerance 1e-2 recover only a small part of the
  Gauss–Newton direction. Steps per budget fall 292 → 25 as the per-pass cost grows linearly in D.
- **Against first-order methods** CG-PLM ties AdamW→L-BFGS at 322k (5e-5 vs 1e-4), is 3× ahead
  at 1.3M (one of three seeds fails at 0.6) and 4× ahead at 5.1M. Predicted: beats at 1.3M, ties or
  loses at 5.1M — the direction held, the margin is 4× rather than the 10³–10⁴× of width 50.
- Net: PLM's lead over L-BFGS shrinks from 4 orders at D ≤ 100k to under one order at D ≥ 1M at
  equal wall clock. The lever not yet pulled is the solve itself (more CG iterations, tighter
  tolerance, a better preconditioner than Jacobi): the 1000× gap to dual PLM at identical D is all
  in the solver.

## Phase H2: 5-seed confirmation of the 2D / time-dependent results (width 50, 300 s)

| problem | AdamW→L-BFGS | Muon→L-BFGS | PLM rec2 + lag ×4 |
|---|---|---|---|
| helmholtz a2=4 | 1.5e-3 (8e-4–2e-3) | 2.7e-3 (2e-3–5e-3) | **7.8e-5** (5e-5–1e-4) |
| reaction ρ=5 | 1.0 (1.0–1.0) | 1.8e-2 (1e-2–5e-2) | **1.7e-4** (6e-5–1.0) |
| wave | 0.32 (0.3–0.4) | 0.35 (0.3–0.4) | **3.1e-3** (2e-3–3e-2) |

- PLM stays 20–100× ahead at 5 seeds. Helmholtz is 7× worse than the 2-seed D2 number (1.2e-5):
  the rec2 configuration (restart rule, lag ×4) is not the D2 one (lag ×4, no restart) and the
  restart rule costs PLM 5 % of the budget each time it fires on a stall.
- **Reaction has a bad seed too** (1 of 5 at error 1.0 with loss 4e-11, the other four at 6e-5
  to 2e-4). Same signature as k = 32 and convection β ≥ 30: a near-zero-residual function that is
  not the solution (here u ≈ 0 away from the initial Gaussian), reached faster by the better optimizer.
- Muon→L-BFGS vs AdamW→L-BFGS at 5 seeds: Muon wins reaction (1.8e-2 vs 1.0), ties wave, loses
  Helmholtz (2×). The Muon warm start is a basin selector, not a conditioning fix.

## Phase L: "why not just least squares?" — frozen features on the linear problems (60 s, 3 seeds)

`lstsq` freezes every hidden layer at the paired init (first layer scaled by feat_scale ∈ {1, 4})
and solves for the 51–801 output weights by one minimum-norm least-squares solve. Figure:
`docs/figures/lstsq_vs_plm.png`.

| problem | LS w=50 (s4) | LS w=200 (s4) | LS w=800 (s4) | PLM w=50, 60 s (H) | PLM w=200, 60 s |
|---|---|---|---|---|---|
| order1 | 3.9e-13 | 5.4e-14 | 3.1e-14 | 4.6e-9 | — |
| poisson | 1.4e-14 | 7.0e-15 | 5.5e-14 | 5.2e-9 | — |
| beam | 2.9e-14 | 1.2e-14 | 2.2e-14 | 1.0e-7 | — |
| multiscale k=16 | 4.6e-4 (s1: 9e-2) | 2.2e-7 | 3.7e-8 | **9.9e-8** | 7.1e-7 |
| multiscale k=32 | 9.2e-2 | 8.8e-2 | 8.0e-2 | **2.4e-6** (2 of 5 seeds > 0.9) | 1.8 (1 of 3 at 4e-5) |
| helmholtz a2=4 | 3.0 (s1: 1.9) | 1.4e-3 | — | **7.8e-5** (300 s, H2) | 0.17 (10 iterations) |

LS wall time: 0.8–3 s at width 50–200, 3–12 s at 800 (one solve; 2–4 Gauss–Newton steps on the
problems where the first step is not exact to roundoff).

- **On order1, poisson and beam, least squares beats PLM by 5–7 orders of magnitude in 1 s.**
  These are smooth solutions that 50 random tanh features at init already span; the entire PINN
  problem is the linear solve, and PLM's 60 s of Gauss–Newton on the hidden layers buys nothing.
  Every headline number PLM posted on these problems is beaten by a one-line baseline.
- **Least squares fails where the init features do not span the solution**: k = 32 at every width
  and scale (8e-2–1e-1, flat in width — more random features of the wrong bandwidth do not help),
  helmholtz at width 50 (error > 1), k = 16 at feat_scale 1 (9e-2 at every width). feat_scale 4
  (a 4× wider input bandwidth) is what makes k = 16 solvable, and it needs width ≥ 200 to match PLM
  at width 50.
- **PLM at width 200 for 60 s is worse than PLM at width 50**: k = 32 has one good seed (4e-5) and two
  failures; helmholtz got 10 iterations (6 s each) and did not converge. The width-200 PLM rows are
  budget-limited and are not the comparison to quote.

## Phase P: PDEs nonlinear in u (width 50, 300 s, 3 seeds)

| problem | nonlinearity | AdamW→L-BFGS | Muon→L-BFGS | LS (output-layer GN, best scale) | PLM rec2 (+ lag ×4) |
|---|---|---|---|---|---|
| nlpoisson | u³ | 1.7e-6 | 1.1e-6 | **1.5e-14** (4 steps, 1 s) | 4.4e-10 |
| allen_cahn | 5(u³−u) | 1.0e-3 | 3.8e-3 | 4.2e-4 (10 steps, 4 s) | **4.0e-5** |
| burgers ν=0.05 | u·u_x | 7.4e-5 | 3.2e-5 | 5–6e-2 (12 steps, 7 s) | **2.8e-7** |
| reaction ρ=5 | ρu(1−u) | 1.0 | 1.2e-2 | 0.26–0.32 (30–40 steps, 5 s) | **8.0e-5** (1 of 3 at 1.0) |

- **Nonlinearity alone does not defeat least squares.** On the cubic Poisson problem, Gauss–Newton
  on the 51 output weights converges in 4 steps to 1.5e-14, four orders below PLM and eight below
  L-BFGS. The residual is nonlinear in the weights, but the solution is smooth and the features
  span it, so the output-layer problem is nearly linear.
- **PLM wins by 100–10⁵× where the solution has structure the init features lack**: the moving
  front of Burgers (LS 5e-2, L-BFGS 3e-5–7e-5, PLM 3e-7), the sharpening Gaussian of reaction
  (LS 0.3, PLM 8e-5), and even the smooth manufactured Allen–Cahn (LS 4e-4, PLM 4e-5, where the
  stiff 5(u³−u) term makes the output-layer problem genuinely nonlinear and GN with frozen
  features stalls at a residual of 7e-8).
- Ranking predictions held on all four problems (τ = +1, +1, +0.71, +0.52; the reaction ties
  come from PLM's bad seed overlapping Muon→L-BFGS's range).
- Burgers is the cleanest demonstration in the study: PLM is 100× better than tuned L-BFGS and
  10⁵× better than least squares on a problem with a real nonlinear transport term and a real
  (not manufactured) exact solution.

## Phase V: PLM upgrades round 4 — variable projection and collocation resampling (width 50, 5 seeds)

Budgets match the baselines: 60 s on the 1D linear problems (rec2 from Phase H), 300 s otherwise (rec2 from
H2 and P). `vp` = variable projection (output layer solved exactly at every trial point, LM on the hidden layers
through Kaufman's projected Jacobian); `rs` = fresh uniform collocation points at every Jacobian (loss and
validation residual logged on the original grid); `vprs` = both. Full table: `results/figures/V/report.md`;
figure `docs/figures/varpro_resample.png`.

| problem | rec2 | + vp | + rs | + vprs | LS | bad seeds (rec2 / vp / rs / vprs) |
|---|---|---|---|---|---|---|
| order1 | 4.6e-9 | **1.3e-14** | 1.5e-8 | 1.6e-14 | 3.9e-13 | 0/0/0/0 |
| poisson | 5.2e-9 | **1.3e-14** | 7.7e-9 | 1.6e-14 | 1.4e-14 | 0/0/0/0 |
| beam | 1.0e-7 | **2.4e-14** | 1.0e-6 | 3.1e-14 | 2.9e-14 | 0/0/0/0 |
| nlpoisson | 4.4e-10 | **8.7e-15** | 2.2e-9 | 1.0e-14 | 1.5e-14 | 0/0/0/0 |
| multiscale k=16 | **9.9e-8** | 1.5e-6 | 2.0e-5 | 4.4e-6 | 4.6e-4 | 0/0/0/0 |
| multiscale k=32 | **2.4e-6** | 1.4 | 1.9 | 0.51 | 9.2e-2 | 2/3/3/5 |
| helmholtz | 7.8e-5 | 1.1e-4 | **6.5e-6** | 1.8e-4 | 1.9 | 0/0/0/0 |
| allen_cahn | **4.0e-5** | 6.5e-5 | 5.3e-5 | 5.6e-5 | 4.2e-4 | 0/0/0/0 |
| burgers | **2.8e-7** | 1.0e-6 | 3.3e-7 | 6.5e-7 | 4.9e-2 | 0/0/0/0 |
| reaction | 1.4e-4 | 0.22 | **1.4e-4** | 0.26 | 0.26 | 2 of 8 / 5 / **0** / 5 |

**Variable projection does exactly what it was built for, and nothing else.** On the four smooth problems
PLM now reaches 1e-14 at the first Jacobian after the warm-up, equal to (poisson, beam, nlpoisson) or 30× better
than (order1) the frozen-feature least-squares solve, and the roundoff stop rule ends the run after 3–35
Jacobians. Where features must be learned it is neutral to harmful: within 4× on k = 16, helmholtz,
allen_cahn and burgers, but it turns 2 bad seeds into 3 at k = 32 (one output solve blows up to loss 8e13)
and makes every reaction seed end at the frozen-feature solution (error 0.2–1.0). Solving the output layer
exactly from the warm-up iterate pulls the network straight into the random-feature least-squares basin,
and the projected hidden-layer problem cannot climb out of it. It also costs 2–4× per Jacobian on the linear
problems and 25× on the nonlinear ones (up to six output-layer Gauss–Newton solves per trial point).

**Resampling finds a collocation-overfitting problem the fixed grid had hidden.** On helmholtz, rec2's
training loss is 1e-8 while its residual on the half-shifted grid is 1e-3: a 10⁵ train/validation gap on a
32×32 grid. With resampling the gap closes (val/train 1.2) and the error improves 12×, 7.8e-5 → 6.5e-6, the
best helmholtz result of the campaign. On reaction it removes the bad seed (0 of 5 above 1e-2, median
unchanged). The rec2 bad seed on reaction *was* visible to the validation residual (6.8e-8 against a training
loss of 4e-11, 1700× gap; good seeds 1.5–3×), so validation-based seed selection would have caught it.
Costs: 2–10× worse on the smooth problems (the objective changes every Jacobian, so LM's μ never settles
at roundoff), 200× worse on k = 16, and a failure at k = 32: 3 of 5 seeds never leave the warm-up loss
(1e4). 512 i.i.d. uniform points under-resolve sin(32πx) where the uniform grid does not; a jittered grid
would be the right version of this idea.

**Combined (vprs)** inherits the worst of both at k = 32 (5 of 5 bad) and reaction (5 of 5), and the best of
vp on the smooth problems. The falsifiable round-4 headline ("vprs never worse than LS by more than 10×,
no bad seeds on k = 32 or reaction") is **falsified**.

## Phase W: the two-stage method — basis learning, then one output-layer solve (width 50, 5 seeds)

Stage 1: PLM rec2 (lagged Jacobian + collocation resampling on the 2D / time-dependent problems) for 10 %, 30 % or
100 % of the budget (60 s 1D linear, 300 s otherwise). Stage 2: one output-layer solve on the learned features
on the original grid (least squares; up to 20 Gauss–Newton steps for a PDE nonlinear in u). Note that at 10 %
of a 60 s budget the AdamW warm-up (10 % of the full budget) consumes all of stage 1, so `fs0.1` on the 1D
problems is "warm-up + solve" with no LM step. Table: `results/figures/W/report.md`; figure
`docs/figures/two_stage.png`.

| problem | rec2 (+ rs on 2D) | 2-stage 10 % | 2-stage 30 % | 2-stage 100 % | LS | bad seeds 10 / 30 / 100 % |
|---|---|---|---|---|---|---|
| order1 | 4.6e-9 | 1.6e-12 | 5.6e-12 | 1.0e-12 | 3.9e-13 | 0/0/0 |
| poisson | 5.2e-9 | 3.7e-14 | 1.9e-14 | 1.5e-14 | 1.4e-14 | 0/0/0 |
| beam | 1.0e-7 | 2.2e-14 | 2.0e-14 | 1.3e-14 | 2.9e-14 | 0/0/0 |
| nlpoisson | 4.4e-10 | 5.8e-14 | 7.4e-14 | 1.6e-14 | 1.5e-14 | 0/0/0 |
| multiscale k=16 | **9.9e-8** | 0.48 | 5.4e-6 | 4.8e-7 | 4.6e-4 | 5/0/0 |
| multiscale k=32 | **2.4e-6** (2 bad) | 9.0 | 1.6 | 1.2e-3 | 9.2e-2 | 5/5/2 |
| helmholtz | **6.5e-6** (rs) | 0.26 | 3.1e-4 | 3.8e-5 | 1.9 | 5/0/0 |
| allen_cahn | **4.0e-5** | 1.5e-3 | 3.8e-4 | 9.1e-5 | 4.2e-4 | 0/0/0 |
| burgers | **2.8e-7** | 8.6e-5 | 3.1e-6 | 3.1e-7 | 4.9e-2 | 0/0/0 |
| reaction | **1.4e-4** (rs) | 0.29 | 2.4e-3 | 1.4e-4 | 0.26 | 5/0/0 |

**What stage 2 does, measured per run** (full-budget runs, error / loss before → after the solve, median of 5):

| problem | error before → after | per-seed ratio | training loss before → after | solve time |
|---|---|---|---|---|
| order1 | 4.1e-9 → 1.0e-12 | 2e-4 … 3e-4 | 1.2e-14 → 1.1e-21 | 0.4 s |
| poisson | 4.9e-9 → 1.5e-14 | 7e-7 … 8e-5 | 4.1e-13 → 3.9e-22 | 0.5 s |
| beam | 2.2e-7 → 1.3e-14 | 2e-8 … 5e-7 | 4.7e-7 → 6.3e-18 | 1.7 s |
| nlpoisson | 5.3e-10 → 1.6e-14 | 2e-5 … 6e-5 | 2.5e-14 → 1.5e-21 | 0.6 s |
| multiscale k=16 | 7.2e-8 → 4.8e-7 | 0.9, 1.4, 2.3, 38, 43 | 9.4e-7 → 9.3e-7 | 0.2 s |
| multiscale k=32 | 3.0e-4 → 1.2e-3 | 0.7, 0.8, 2.6, 3.9, 17 | 1.4 → 1.4 | 0.2 s |
| helmholtz | 7.1e-6 → 3.8e-5 | 4.1, 4.2, 5.2, 5.3, 12 | 9.8e-7 → 9.1e-7 | 1.1 s |
| allen_cahn | 9.6e-5 → 9.1e-5 | 0.96 … 1.1 | 1.7e-9 → 1.2e-9 | 5.2 s |
| burgers | 3.5e-7 → 3.1e-7 | 0.85 … 0.97 | 1.1e-12 → 9.7e-13 | 4.3 s |
| reaction | 1.4e-4 → 1.4e-4 | 1.0 | 7.7e-11 → 7.2e-11 | 1.7 s |

Three things follow.

1. **The two-stage method gets the least-squares floor wherever the learned basis spans the solution**: the
   four smooth problems drop 5–7 orders in a one-second solve, at every switch point, to 1e-12–1e-14. Nothing
   that least squares alone would not have given, but the method did not have to be told which regime it was in.
2. **Where the basis does not span the solution the exact solve is neutral at best and harmful at worst.** On
   k = 16, k = 32 and helmholtz it lowers the training loss by 1–7 % and raises the error 4–40× on most seeds:
   the minimum-norm solution shifts weight into directions that reduce the residual on the collocation points
   and increase it between them. On helmholtz it gives back most of what resampling had won (6.5e-6 → 3.8e-5).
   On burgers, allen_cahn and reaction it is a no-op (ratios 0.85–1.1). So stage 2 should be *conditional*:
   accept it only if the training loss drops by orders, which is a one-second test of whether the learned
   features span the solution.
3. **Basis learning needs the whole budget.** 30 % of the budget plus the solve is 10–50× worse than full-budget
   PLM on every feature-learning problem (k = 16 5.4e-6 vs 1e-7, helmholtz 3.1e-4 vs 6.5e-6, burgers 3.1e-6 vs
   2.8e-7, reaction 2.4e-3 vs 1.4e-4), and 10 % fails outright (errors 0.1–9). PLM's late iterations are not
   output-layer polishing that a solve could replace; they are still moving the features. The "same accuracy at
   30 % of the cost" claim is falsified.

The round-5 headline ("at full budget never worse than the better of rec2 + rs and LS by more than 3×, and ≥ 10×
better on six problems") is **falsified**: 5–6× worse on k = 16 and helmholtz, 500× worse in the median on k = 32
(two bad seeds plus the solve's damage), and ≥ 10× better only on the four smooth problems — and there only
relative to PLM, not to least squares.

## Scorecard, round 5

Ten rankings, mean **τ = +0.62** ([scorecard.md](scorecard.md)): the smooth problems at +0.75 to +1, burgers
+0.83, helmholtz +0.64, allen_cahn +0.61, k = 16 +0.33, k = 32 and reaction 0. Confirmed: the solve reaches the
LS floor at every switch point on the smooth problems; 10 % fails on the feature-learning problems; resampling
keeps the reaction bad seeds away at every switch point. Falsified: the solve improves full-budget PLM on
k = 16 / helmholtz / burgers / allen_cahn (it hurts or does nothing), and 30 % ties the full budget.

## Scorecard, round 4

Ten rankings, mean **τ = +0.55** ([scorecard.md](scorecard.md)): the four smooth-problem rows at +0.87 to
+1.00 and burgers at +1.00; k = 16, k = 32, allen_cahn and reaction at 0 (the bad seeds and the small
spread tie everything by the range rule). Confirmed: vp reaches the LS floor on the smooth problems; rs
removes the reaction bad seed; neither helps on allen_cahn or burgers. Falsified: rs removes the k = 32 bad
seeds (it adds one and stalls two more), vp is harmless where features are learned (it is not on reaction
and k = 32), rs is neutral on helmholtz (it is the best result of the campaign there: the fixed grid was
being overfitted by 10⁵).

## Scorecard, rounds 2–3

Sixteen rankings, mean **τ = +0.64** ([scorecard.md](scorecard.md)). The least-squares rows are
the strongest (7 of 8 at τ = +1; the misses are ties forced by PLM's bad k = 32 seed and by the
budget-limited width-200 PLM runs). Trend predictions: confirmed — the noise floor ∝ σ, CG-PLM
memory O(D), CG-PLM beats L-BFGS at 1.3M, "nonlinearity alone does not rescue PLM, features do";
falsified — the restart rule removes the bad seed (never fires), causal weighting gets anyone
below 0.1 at β = 30 (nobody), CG-PLM within 3× of dual (1000× worse), CG memory < 3 GB (6 GB),
PLM overfits collocation noise (its validation loss is *below* its training loss), the multiscale
noise floor ≈ σ (it is 5.4e-3 σ, same as poisson).

## Prediction scorecard

[scorecard.md](scorecard.md) scores the eleven pre-registered rankings with Kendall's τ:
mean **τ = +0.71**; 7 of 11 at τ ≥ 0.8; the three E9 rankings are the weak ones (0.24, 0.33,
0 — the last because nobody learns convection at β = 30 and the observed ranking is a four-way
tie). Of the trend predictions: confirmed — spectral bias (E5), term balance (E3), loss/error
decoupling (E8), μ₀ robustness (E6), dense/dual cost scaling (E7), fp64 floors (E4), the
noise floor (E2), the four PLM upgrade predictions that mattered (μ₀, Marquardt, weighting,
dual = dense); falsified — "PLM's lead widens with derivative order" (it shrinks; cost), "Muon's
working lr range is wider than Adam's" (it is worse at every lr), "M ≈ A" (M < A on everything
except k = 32 and reaction), "trust region helps" (it does not; warm-up does), "fp32 floors
everyone at 1e-4" (1e-6 to 1e-7), "dual PLM scales to large D" (it needs the warm-up to).

## Where PLM loses and what to do about it
1. **Large models (D ≥ 81k):** steps too large for the linearization → trust region / warm-up;
   per-iteration cost → lagged Jacobian (`reuse_jac`), which reuses J for several LM steps at
   the cost of one residual evaluation each (≈ N× cheaper than a Jacobian).
2. **2D / time-dependent (N ≈ 2k residuals, 2nd-order derivatives in two coordinates):** the
   Jacobian is ~N reverse passes; same two fixes, plus residual subsampling (not yet implemented).
3. **Nonlinear PDEs (reaction):** Gauss–Newton ignores the residual-weighted curvature term.
   Candidate fix: a Levenberg–Marquardt step on the *full* Hessian-vector product
   (Newton–CG with GN preconditioner), or simply more iterations.
4. **Marquardt damping (fixed):** needs a floor on diag(G) or it overfits the collocation grid.
5. **fp32 (fixed, partly):** Cholesky of J Jᵀ + μI fails once μ < 1e-7‖J Jᵀ‖; now treated as a rejected
   step. PLM stops itself at 2e-7 (still 10× ahead); a QR-based solve would go further.
6. **Badly-posed losses (convection β ≥ 30):** PLM finds the spurious zero-loss solution faster.
   Not an optimizer problem; needs a loss fix.
7. **Occasional bad seed (multiscale k = 32, 2 of 5; reaction 1 of 5):** not a divergence — the loss keeps
   falling while the error rises from 0.06 to > 1. A stall/blow-up restart rule (Phase H) never
   fires on it. Same disease as convection β ≥ 30: a low-residual wrong function. Phase V: the validation
   residual on a shifted grid *does* flag the reaction bad seed (1700× train/validation gap), and
   collocation resampling removes it; at k = 32 i.i.d. resampling under-resolves the solution instead.
8. **Large D (Phase S):** the matrix-free CG solve fixes memory (6 GB at 5.1M parameters) but the
   inexact solve costs 1000× in accuracy at equal D; the lead over L-BFGS is 4× at 5.1M.
9. **Smooth solutions, linear or not (Phases L, P):** a frozen-feature least-squares solve reaches
   1e-13 in 1 s on order1, poisson, beam and the cubic Poisson; PLM's 60–300 s results (5e-9 to
   4e-10) lose that comparison by 4–7 orders. PLM earns its cost only where the init features do
   not span the solution: high frequency (k = 32), 2D (helmholtz), fronts (burgers, reaction).
