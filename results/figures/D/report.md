# D: 72 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW | AdamW→L-BFGS | L-BFGS | Muon | Muon→L-BFGS | PLM |
|---|---|---|---|---|---|---|
| convection beta=30 w=50 float64 | 1.3e+00 (1e+00–2e+00) | 1.3e+00 (1e+00–2e+00) | 1.1e+00 (1e+00–1e+00) | 1.5e+00 (1e+00–2e+00) | 1.3e+00 (1e+00–1e+00) | 1.4e+00 (1e+00–2e+00) |
| helmholtz a2=4 w=50 float64 | 2.1e-01 (1e-01–3e-01) | 2.9e-02 (1e-02–3e-02) | 9.9e-03 (5e-03–2e-02) | 2.0e-01 (1e-01–2e-01) | 9.5e-03 (9e-03–2e-02) | 1.7e-04 (1e-04–2e-04) |
| reaction rho=5 w=50 float64 | 1.0e+00 (1e+00–1e+00) | 6.9e-02 (2e-02–1e+00) | 5.7e-02 (3e-02–1e+00) | 2.4e-02 (2e-02–2e-02) | 6.0e-03 (6e-03–2e-02) | 6.7e-02 (2e-02–7e-02) |
| wave w=50 float64 | 4.8e-01 (4e-01–5e-01) | 4.1e-01 (4e-01–4e-01) | 4.1e-01 (4e-01–4e-01) | 5.8e-01 (6e-01–6e-01) | 4.0e-01 (4e-01–4e-01) | 5.4e-01 (5e-01–6e-01) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| convection beta=30 w=50 float64 | AdamW | 3.60e-03 | 4.7e+01 | 16680 | 0 | 16680 | — | — |
| convection beta=30 w=50 float64 | AdamW→L-BFGS | 1.02e-02 | 5.2e+01 | 11768 | 0 | 5880 | — | — |
| convection beta=30 w=50 float64 | L-BFGS | 7.14e-03 | 5.2e+01 | 9925 | 0 | 8420 | — | — |
| convection beta=30 w=50 float64 | Muon | 4.05e-03 | 4.7e+01 | 14813 | 0 | 14813 | — | — |
| convection beta=30 w=50 float64 | Muon→L-BFGS | 9.84e-03 | 5.2e+01 | 11237 | 0 | 6100 | — | — |
| convection beta=30 w=50 float64 | PLM | 9.28e-01 | 9.0e+03 | 0 | 65 | 65 | — | — |
| helmholtz a2=4 w=50 float64 | AdamW | 5.39e-03 | 5.4e+01 | 11124 | 0 | 11124 | — | — |
| helmholtz a2=4 w=50 float64 | AdamW→L-BFGS | 1.10e-02 | 5.8e+01 | 10047 | 0 | 5460 | — | — |
| helmholtz a2=4 w=50 float64 | L-BFGS | 7.56e-03 | 5.8e+01 | 8780 | 0 | 7940 | — | — |
| helmholtz a2=4 w=50 float64 | Muon | 5.55e-03 | 5.3e+01 | 10803 | 0 | 10803 | — | — |
| helmholtz a2=4 w=50 float64 | Muon→L-BFGS | 1.13e-02 | 5.8e+01 | 9452 | 0 | 5340 | — | — |
| helmholtz a2=4 w=50 float64 | PLM | 5.56e-01 | 4.8e+03 | 0 | 108 | 108 | 3.4e+01 | — |
| reaction rho=5 w=50 float64 | AdamW | 3.55e-03 | 4.5e+01 | 16908 | 0 | 16908 | — | — |
| reaction rho=5 w=50 float64 | AdamW→L-BFGS | 1.05e-02 | 4.9e+01 | 11677 | 0 | 5740 | — | — |
| reaction rho=5 w=50 float64 | L-BFGS | 6.97e-03 | 4.9e+01 | 10104 | 0 | 8620 | — | — |
| reaction rho=5 w=50 float64 | Muon | 3.95e-03 | 4.5e+01 | 15188 | 0 | 15188 | — | — |
| reaction rho=5 w=50 float64 | Muon→L-BFGS | 9.38e-03 | 4.9e+01 | 12491 | 0 | 6400 | — | — |
| reaction rho=5 w=50 float64 | PLM | 8.88e-01 | 8.9e+03 | 0 | 68 | 68 | — | — |
| wave w=50 float64 | AdamW | 5.71e-03 | 7.2e+01 | 10501 | 0 | 10501 | — | — |
| wave w=50 float64 | AdamW→L-BFGS | 1.30e-02 | 7.7e+01 | 8214 | 0 | 4620 | — | — |
| wave w=50 float64 | L-BFGS | 9.48e-03 | 7.7e+01 | 6913 | 0 | 6340 | — | — |
| wave w=50 float64 | Muon | 6.15e-03 | 7.2e+01 | 9751 | 0 | 9751 | — | — |
| wave w=50 float64 | Muon→L-BFGS | 1.36e-02 | 7.7e+01 | 7672 | 0 | 4420 | — | — |
| wave w=50 float64 | PLM | 1.93e+00 | 1.8e+04 | 0 | 32 | 32 | — | — |
