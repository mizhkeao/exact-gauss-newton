# D2: 40 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW→L-BFGS | L-BFGS | Muon→L-BFGS | PLM [weighted] | PLM [weighted_lag4_warm] |
|---|---|---|---|---|---|
| convection beta=30 w=50 float64 | 1.8e+00 (1e+00–2e+00) | 1.5e+00 (1e+00–2e+00) | 1.6e+00 (1e+00–2e+00) | 2.4e+00 (2e+00–3e+00) | 1.7e+00 (2e+00–2e+00) |
| helmholtz a2=4 w=50 float64 | 1.7e-03 (2e-03–2e-03) | 2.5e-03 (2e-03–3e-03) | 3.9e-03 (3e-03–5e-03) | 1.2e-05 (9e-06–1e-05) | 7.8e-05 (6e-05–9e-05) |
| reaction rho=5 w=50 float64 | 1.0e+00 (1e+00–1e+00) | 5.1e-01 (3e-02–1e+00) | 1.1e-02 (1e-02–1e-02) | 2.1e-03 (2e-03–2e-03) | 3.4e-04 (2e-04–5e-04) |
| wave w=50 float64 | 3.4e-01 (3e-01–3e-01) | 3.6e-01 (4e-01–4e-01) | 3.5e-01 (3e-01–4e-01) | 2.6e-01 (2e-01–3e-01) | 8.4e-03 (5e-03–1e-02) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| convection beta=30 w=50 float64 | AdamW→L-BFGS | 1.03e-02 | 5.2e+01 | 60302 | 0 | 29020 | — | — |
| convection beta=30 w=50 float64 | L-BFGS | 6.58e-03 | 5.2e+01 | 56446 | 0 | 45660 | — | — |
| convection beta=30 w=50 float64 | Muon→L-BFGS | 1.01e-02 | 5.2e+01 | 55651 | 0 | 29700 | — | — |
| convection beta=30 w=50 float64 | PLM [weighted] | 9.06e-01 | 9.0e+03 | 0 | 332 | 332 | — | — |
| convection beta=30 w=50 float64 | PLM [weighted_lag4_warm] | 3.46e-01 | 9.0e+03 | 8434 | 227 | 871 | — | — |
| helmholtz a2=4 w=50 float64 | AdamW→L-BFGS | 1.10e-02 | 5.8e+01 | 50918 | 0 | 27250 | — | — |
| helmholtz a2=4 w=50 float64 | L-BFGS | 8.32e-03 | 5.8e+01 | 40456 | 0 | 36080 | — | — |
| helmholtz a2=4 w=50 float64 | Muon→L-BFGS | 1.09e-02 | 5.8e+01 | 49464 | 0 | 27620 | — | — |
| helmholtz a2=4 w=50 float64 | PLM [weighted] | 5.42e-01 | 4.8e+03 | 0 | 554 | 554 | 3.3e+01 | 2.6e+02 |
| helmholtz a2=4 w=50 float64 | PLM [weighted_lag4_warm] | 1.85e-01 | 4.8e+03 | 6105 | 417 | 1621 | 3.6e+01 | — |
| reaction rho=5 w=50 float64 | AdamW→L-BFGS | 9.75e-03 | 4.9e+01 | 62270 | 0 | 30780 | — | — |
| reaction rho=5 w=50 float64 | L-BFGS | 6.58e-03 | 4.9e+01 | 56482 | 0 | 45660 | — | — |
| reaction rho=5 w=50 float64 | Muon→L-BFGS | 9.83e-03 | 4.9e+01 | 59790 | 0 | 30540 | — | — |
| reaction rho=5 w=50 float64 | PLM [weighted] | 8.76e-01 | 8.9e+03 | 0 | 343 | 343 | — | — |
| reaction rho=5 w=50 float64 | PLM [weighted_lag4_warm] | 3.36e-01 | 8.9e+03 | 7586 | 234 | 894 | 1.8e+02 | — |
| wave w=50 float64 | AdamW→L-BFGS | 1.37e-02 | 7.7e+01 | 38834 | 0 | 21950 | — | — |
| wave w=50 float64 | L-BFGS | 1.04e-02 | 7.7e+01 | 32610 | 0 | 29150 | — | — |
| wave w=50 float64 | Muon→L-BFGS | 1.41e-02 | 7.7e+01 | 38213 | 0 | 21370 | — | — |
| wave w=50 float64 | PLM [weighted] | 2.01e+00 | 1.8e+04 | 0 | 150 | 150 | — | — |
| wave w=50 float64 | PLM [weighted_lag4_warm] | 6.42e-01 | 1.8e+04 | 5266 | 122 | 469 | — | — |
