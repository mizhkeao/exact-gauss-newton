# P: 60 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW→L-BFGS | Frozen-feature LS [s1] | Frozen-feature LS [s4] | Muon→L-BFGS | PLM [rec2] | PLM [rec2_lag4] |
|---|---|---|---|---|---|---|
| allen_cahn w=50 float64 | 1.0e-03 (1e-03–1e-03) | 4.2e-04 (2e-04–5e-04) | 2.3e-02 (7e-03–2e-02) | 3.8e-03 (2e-03–6e-03) | — | 4.0e-05 (3e-05–9e-05) |
| burgers nu=0.05 w=50 float64 | 7.4e-05 (6e-05–2e-04) | 6.4e-02 (6e-02–7e-02) | 4.9e-02 (5e-02–6e-02) | 3.2e-05 (3e-05–6e-05) | — | 2.8e-07 (2e-07–6e-07) |
| nlpoisson w=50 float64 | 1.7e-06 (2e-06–2e-06) | 4.7e-13 (3e-13–3e-11) | 1.5e-14 (1e-14–2e-14) | 1.1e-06 (6e-07–6e-06) | 4.4e-10 (3e-10–5e-10) | — |
| reaction rho=5 w=50 float64 | 1.0e+00 (1e+00–1e+00) | 3.2e-01 (3e-01–3e-01) | 2.6e-01 (2e-01–3e-01) | 1.2e-02 (1e-02–2e-02) | — | 8.0e-05 (6e-05–1e+00) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| allen_cahn w=50 float64 | AdamW→L-BFGS | 1.24e-02 | 6.9e+01 | 48253 | 0 | 24200 | — | — |
| allen_cahn w=50 float64 | Frozen-feature LS [s1] | 4.43e-01 | 6.9e+03 | 0 | 11 | 10 | 3.2e+00 | — |
| allen_cahn w=50 float64 | Frozen-feature LS [s4] | 4.87e-01 | 6.9e+03 | 0 | 10 | 9 | — | — |
| allen_cahn w=50 float64 | Muon→L-BFGS | 1.28e-02 | 6.9e+01 | 44237 | 0 | 23360 | — | — |
| allen_cahn w=50 float64 | PLM [rec2_lag4] | 5.89e-01 | 1.8e+04 | 5590 | 131 | 510 | 6.1e+01 | — |
| burgers nu=0.05 w=50 float64 | AdamW→L-BFGS | 1.35e-02 | 7.4e+01 | 43540 | 0 | 22160 | 2.4e+01 | — |
| burgers nu=0.05 w=50 float64 | Frozen-feature LS [s1] | 5.59e-01 | 6.9e+03 | 0 | 13 | 12 | — | — |
| burgers nu=0.05 w=50 float64 | Frozen-feature LS [s4] | 5.80e-01 | 6.9e+03 | 0 | 13 | 12 | — | — |
| burgers nu=0.05 w=50 float64 | Muon→L-BFGS | 1.43e-02 | 7.4e+01 | 41198 | 0 | 21040 | 9.0e+01 | — |
| burgers nu=0.05 w=50 float64 | PLM [rec2_lag4] | 6.55e-01 | 1.8e+04 | 5253 | 119 | 461 | 2.5e+01 | 7.0e+01 |
| nlpoisson w=50 float64 | AdamW→L-BFGS | 8.91e-03 | 4.2e+01 | 67943 | 0 | 33660 | 5.5e+00 | 2.9e+01 |
| nlpoisson w=50 float64 | Frozen-feature LS [s1] | 2.23e-01 | 4.4e+02 | 0 | 4 | 4 | 9.3e-01 | 9.3e-01 |
| nlpoisson w=50 float64 | Frozen-feature LS [s4] | 2.96e-01 | 4.4e+02 | 0 | 4 | 4 | 1.1e+00 | 1.1e+00 |
| nlpoisson w=50 float64 | Muon→L-BFGS | 8.71e-03 | 4.2e+01 | 66674 | 0 | 34440 | 9.0e+01 | 9.1e+01 |
| nlpoisson w=50 float64 | PLM [rec2] | 1.03e-01 | 1.1e+03 | 9126 | 2909 | 2909 | 6.1e+00 | 3.0e+01 |
| reaction rho=5 w=50 float64 | AdamW→L-BFGS | 9.69e-03 | 4.9e+01 | 60529 | 0 | 30980 | — | — |
| reaction rho=5 w=50 float64 | Frozen-feature LS [s1] | 1.42e-01 | 3.4e+03 | 0 | 30 | 29 | — | — |
| reaction rho=5 w=50 float64 | Frozen-feature LS [s4] | 1.38e-01 | 3.4e+03 | 0 | 39 | 38 | — | — |
| reaction rho=5 w=50 float64 | Muon→L-BFGS | 9.55e-03 | 4.9e+01 | 61031 | 0 | 31420 | — | — |
| reaction rho=5 w=50 float64 | PLM [rec2_lag4] | 3.28e-01 | 8.9e+03 | 7906 | 235 | 917 | 1.1e+02 | — |
