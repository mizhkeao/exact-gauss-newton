# Prediction scorecard

Observed ranking: methods ordered by median final error; two methods tie when their
seed ranges overlap. Agreement = Kendall's τ-b. LS = frozen-feature least squares at its better feat_scale.

## Round 1 (predictions of 2026-11-05, Phases A–G)

| experiment | predicted | observed (median error) | τ |
|---|---|---|---|
| E1 order 1 (Phase A) | P > A→L ≈ M→L > L > M ≈ A | P > M→L > A→L ≈ L ≈ A > M  (P 1e-08, A→L 6e-05, M→L 1e-05, L 7e-05, M 1e-02, A 1e-04) | +0.80 |
| E1 order 2 (Phase A) | P > A→L ≈ M→L > L > M ≈ A | P > M→L ≈ L ≈ A→L > A > M  (P 5e-08, A→L 2e-06, M→L 7e-07, L 2e-06, M 3e-02, A 7e-04) | +0.88 |
| E1 order 4 (Phase A) | P > A→L ≈ M→L > L > M ≈ A | M→L ≈ L ≈ P ≈ A→L > A ≈ M  (P 6e-06, A→L 2e-05, M→L 2e-06, L 5e-06, M 3e-02, A 3e-02) | +0.78 |
| E3 Helmholtz (Phase D) | P > A→L > A | P > A→L > A  (P 2e-04, A→L 3e-02, A 2e-01) | +1.00 |
| E5 k = 1 (Phase E) | P > A→L > A | P > A→L > A  (P 5e-09, A→L 2e-06, A 3e-03) | +1.00 |
| E5 k = 4 (Phase E) | P > A→L > A | P > A→L > A  (P 8e-09, A→L 1e-06, A 2e-05) | +1.00 |
| E5 k = 16 (Phase E) | P > A→L > A | P > A→L > A  (P 2e-07, A→L 1e-03, A 5e-03) | +1.00 |
| E5 k = 32 (Phase E) | P > A→L > A | P ≈ A→L > A  (P 5e-06, A→L 5e+00, A 8e+00) | +0.82 |
| E9 reaction (Phase D2) | P ≈ A→L ≈ M→L > L | P > M→L > L > A→L  (P 3e-04, A→L 1e+00, M→L 1e-02, L 5e-01) | +0.24 |
| E9 wave (Phase D2) | P ≈ A→L ≈ M→L > L | P > A→L ≈ M→L ≈ L  (P 8e-03, A→L 3e-01, M→L 4e-01, L 4e-01) | +0.33 |
| E9 convection β=30 (Phase D2) | P ≈ A→L ≈ M→L > L | L ≈ M→L ≈ P ≈ A→L  (P 2e+00, A→L 2e+00, M→L 2e+00, L 1e+00) | +0.00 |
| **mean τ over 11 rankings** | | | **+0.71** |

## Rounds 2–3 (predictions of 2026-11-05/06, Phases H–P)

| experiment | predicted | observed (median error) | τ |
|---|---|---|---|
| L order1 w=50 (H + L) | LS > P | LS > P  (LS 4e-13, P 5e-09) | +1.00 |
| L poisson w=50 (H + L) | LS > P | LS > P  (LS 1e-14, P 5e-09) | +1.00 |
| L beam w=50 (H + L) | LS > P | LS > P  (LS 3e-14, P 1e-07) | +1.00 |
| L multiscale k=16 w=50 (H + L) | P > LS | P > LS  (P 1e-07, LS 5e-04) | +1.00 |
| L multiscale k=32 w=50 (H + L) | P > LS | P ≈ LS  (P 2e-06, LS 9e-02) | +0.00 |
| L multiscale k=16 w=200 (L) | P ≈ LS | LS > P  (P 7e-07, LS 2e-07) | +0.00 |
| L multiscale k=32 w=200 (L) | P > LS | LS ≈ P  (P 2e+00, LS 9e-02) | +0.00 |
| L helmholtz w=50 (H2 300 s + L) | P > LS | P > LS  (P 8e-05, LS 2e+00) | +1.00 |
| P nlpoisson (P) | LS > P > A→L ≈ M→L | LS > P > M→L ≈ A→L  (LS 1e-14, P 4e-10, A→L 2e-06, M→L 1e-06) | +1.00 |
| P burgers (P) | P > A→L ≈ M→L > LS | P > M→L ≈ A→L > LS  (P 3e-07, A→L 7e-05, M→L 3e-05, LS 5e-02) | +1.00 |
| P allen_cahn (P) | P > LS ≈ A→L ≈ M→L | P > LS > A→L > M→L  (P 4e-05, LS 4e-04, A→L 1e-03, M→L 4e-03) | +0.71 |
| P reaction (P) | P > M→L > LS ≈ A→L | P ≈ M→L ≈ LS > A→L  (P 8e-05, M→L 1e-02, LS 3e-01, A→L 1e+00) | +0.52 |
| H2 helmholtz 5 seeds (H2) | P > A→L ≈ M→L | P > A→L ≈ M→L  (P 8e-05, A→L 2e-03, M→L 3e-03) | +1.00 |
| H2 reaction 5 seeds (H2) | P > M→L > A→L | P ≈ M→L ≈ A→L  (P 2e-04, M→L 2e-02, A→L 1e+00) | +0.00 |
| H2 wave 5 seeds (H2) | P > A→L ≈ M→L | P > A→L ≈ M→L  (P 3e-03, A→L 3e-01, M→L 4e-01) | +1.00 |
| H3 convection β=30 causal (H3) | P > A→L ≈ M→L ≈ A | P ≈ A→L ≈ A ≈ M→L  (P 1e+00, A→L 1e+00, M→L 1e+00, A 1e+00) | +0.00 |
| **mean τ over 16 rankings** | | | **+0.64** |

## Round 4 (predictions of 2026-11-06, Phase V: P = PLM rec2, Pv = + varpro, Pr = + resampling, Pvr = both)

| experiment | predicted | observed (median error) | τ |
|---|---|---|---|
| V order1 (H + L + V) | Pv ≈ Pvr ≈ LS > P ≈ Pr | Pv ≈ Pvr > LS > P ≈ Pr  (Pv 1e-14, Pvr 2e-14, LS 4e-13, P 5e-09, Pr 2e-08) | +0.87 |
| V poisson (H + L + V) | Pv ≈ Pvr ≈ LS > P ≈ Pr | Pv ≈ LS ≈ Pvr > P ≈ Pr  (Pv 1e-14, Pvr 2e-14, LS 1e-14, P 5e-09, Pr 8e-09) | +1.00 |
| V beam (H + L + V) | Pv ≈ Pvr ≈ LS > P ≈ Pr | Pv ≈ LS ≈ Pvr > P ≈ Pr  (Pv 2e-14, Pvr 3e-14, LS 3e-14, P 1e-07, Pr 1e-06) | +1.00 |
| V multiscale k=16 (H + L + V) | Pv ≈ Pvr ≈ P ≈ Pr > LS | P ≈ Pv ≈ Pvr ≈ Pr ≈ LS  (Pv 2e-06, Pvr 4e-06, P 1e-07, Pr 2e-05, LS 5e-04) | +0.00 |
| V multiscale k=32 (H + L + V) | Pr ≈ Pvr > P ≈ Pv > LS | P ≈ LS ≈ Pvr ≈ Pv ≈ Pr  (Pr 2e+00, Pvr 5e-01, P 2e-06, Pv 1e+00, LS 9e-02) | +0.00 |
| V nlpoisson (P + V) | Pv ≈ Pvr ≈ LS > P ≈ Pr | Pv ≈ Pvr ≈ LS > P > Pr  (Pv 9e-15, Pvr 1e-14, LS 1e-14, P 4e-10, Pr 2e-09) | +0.93 |
| V helmholtz (H2 + L + V) | Pv ≈ Pvr ≈ P ≈ Pr > LS | Pr > P ≈ Pv ≈ Pvr > LS  (Pv 1e-04, Pvr 2e-04, P 8e-05, Pr 7e-06, LS 2e+00) | +0.76 |
| V allen_cahn (P + V) | Pv ≈ Pvr ≈ P ≈ Pr > LS | P ≈ Pr ≈ Pvr ≈ Pv ≈ LS  (Pv 6e-05, Pvr 6e-05, P 4e-05, Pr 5e-05, LS 4e-04) | +0.00 |
| V burgers (P + V) | Pv ≈ Pvr ≈ P ≈ Pr > LS | P ≈ Pr ≈ Pvr ≈ Pv > LS  (Pv 1e-06, Pvr 7e-07, P 3e-07, Pr 3e-07, LS 5e-02) | +1.00 |
| V reaction (H2 + P + V) | Pr ≈ Pvr > P ≈ Pv > LS | Pr ≈ P ≈ Pv ≈ LS ≈ Pvr  (Pr 1e-04, Pvr 3e-01, P 1e-04, Pv 2e-01, LS 3e-01) | +0.00 |
| **mean τ over 10 rankings** | | | **+0.55** |

## Round 5 (predictions of 2026-11-06, Phase W: F1 / F3 / F10 = two-stage at 10 / 30 / 100 % budget, Pr = rec2 + resampling)

| experiment | predicted | observed (median error) | τ |
|---|---|---|---|
| W order1 | F1 ≈ F3 ≈ F10 ≈ LS ≈ Pv > P | Pv > LS ≈ F10 ≈ F1 ≈ F3 > P  (F1 2e-12, F3 6e-12, F10 1e-12, LS 4e-13, Pv 1e-14, P 5e-09) | +0.75 |
| W poisson | F1 ≈ F3 ≈ F10 ≈ LS ≈ Pv > P | Pv ≈ LS ≈ F10 ≈ F3 ≈ F1 > P  (F1 4e-14, F3 2e-14, F10 2e-14, LS 1e-14, Pv 1e-14, P 5e-09) | +1.00 |
| W beam | F1 ≈ F3 ≈ F10 ≈ LS ≈ Pv > P | F10 ≈ F3 ≈ F1 ≈ Pv ≈ LS > P  (F1 2e-14, F3 2e-14, F10 1e-14, LS 3e-14, Pv 2e-14, P 1e-07) | +1.00 |
| W nlpoisson | F1 ≈ F3 ≈ F10 ≈ LS ≈ Pv > P | Pv ≈ LS ≈ F10 ≈ F1 ≈ F3 > P  (F1 6e-14, F3 7e-14, F10 2e-14, LS 1e-14, Pv 9e-15, P 4e-10) | +1.00 |
| W multiscale k=16 | F10 > P ≈ F3 > F1 > LS | P ≈ F10 ≈ F3 ≈ LS > F1  (F10 5e-07, P 1e-07, F3 5e-06, F1 5e-01, LS 5e-04) | +0.33 |
| W multiscale k=32 | F10 ≈ P > F3 > F1 ≈ LS | P ≈ F10 ≈ LS ≈ F3 ≈ F1  (F10 1e-03, P 2e-06, F3 2e+00, F1 9e+00, LS 9e-02) | +0.00 |
| W helmholtz | F10 > F3 ≈ Pr > P > F1 > LS | Pr > F10 ≈ P > F3 > F1 > LS  (F10 4e-05, F3 3e-04, Pr 7e-06, P 8e-05, F1 3e-01, LS 2e+00) | +0.64 |
| W allen_cahn | F10 > F3 ≈ P ≈ Pr > F1 > LS | P ≈ Pr ≈ F10 > F3 ≈ LS > F1  (F10 9e-05, F3 4e-04, P 4e-05, Pr 5e-05, F1 1e-03, LS 4e-04) | +0.61 |
| W burgers | F10 > F3 ≈ P ≈ Pr > F1 > LS | P ≈ F10 ≈ Pr > F3 > F1 > LS  (F10 3e-07, F3 3e-06, P 3e-07, Pr 3e-07, F1 9e-05, LS 5e-02) | +0.83 |
| W reaction | F10 ≈ F3 ≈ Pr ≈ P > F1 > LS | Pr ≈ F10 ≈ P ≈ F3 ≈ LS ≈ F1  (F10 1e-04, F3 2e-03, Pr 1e-04, P 1e-04, F1 3e-01, LS 3e-01) | +0.00 |
| **mean τ over 10 rankings** | | | **+0.62** |
