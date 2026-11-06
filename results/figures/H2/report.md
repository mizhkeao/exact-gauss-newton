# H2: 45 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW→L-BFGS | Muon→L-BFGS | PLM [rec2_lag4] |
|---|---|---|---|
| helmholtz a2=4 w=50 float64 | 1.5e-03 (8e-04–2e-03) | 2.7e-03 (2e-03–5e-03) | 7.8e-05 (5e-05–1e-04) |
| reaction rho=5 w=50 float64 | 1.0e+00 (1e+00–1e+00) | 1.8e-02 (1e-02–5e-02) | 1.7e-04 (6e-05–1e+00) |
| wave w=50 float64 | 3.2e-01 (3e-01–4e-01) | 3.5e-01 (3e-01–4e-01) | 3.1e-03 (2e-03–3e-02) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| helmholtz a2=4 w=50 float64 | AdamW→L-BFGS | 1.15e-02 | 5.8e+01 | 48328 | 0 | 26140 | 2.7e+02 | — |
| helmholtz a2=4 w=50 float64 | Muon→L-BFGS | 1.15e-02 | 5.8e+01 | 46216 | 0 | 26140 | — | — |
| helmholtz a2=4 w=50 float64 | PLM [rec2_lag4] | 1.85e-01 | 4.8e+03 | 6227 | 420 | 1619 | 4.7e+01 | — |
| reaction rho=5 w=50 float64 | AdamW→L-BFGS | 9.12e-03 | 4.9e+01 | 66945 | 0 | 32900 | — | — |
| reaction rho=5 w=50 float64 | Muon→L-BFGS | 9.62e-03 | 4.9e+01 | 60142 | 0 | 31200 | — | — |
| reaction rho=5 w=50 float64 | PLM [rec2_lag4] | 3.28e-01 | 8.9e+03 | 8094 | 236 | 917 | 1.2e+02 | — |
| wave w=50 float64 | AdamW→L-BFGS | 1.38e-02 | 7.7e+01 | 39612 | 0 | 21820 | — | — |
| wave w=50 float64 | Muon→L-BFGS | 1.41e-02 | 7.7e+01 | 38337 | 0 | 21320 | — | — |
| wave w=50 float64 | PLM [rec2_lag4] | 6.56e-01 | 1.8e+04 | 4853 | 124 | 460 | — | — |
