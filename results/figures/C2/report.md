# C2: 45 runs

## Final relative L2 error, median (min–max) over seeds

| setting | PLM [base] | PLM [lag4] | PLM [trust] | PLM [warm_adamw] | PLM [warm_trust_lag4] |
|---|---|---|---|---|---|
| multiscale k=16 w=100 float64 | 8.2e-06 (3e-06–1e-05) | 3.5e+00 (3e-06–6e+00) | 4.9e-05 (3e-05–9e-05) | 5.7e-07 (5e-07–7e-07) | 8.6e-07 (2e-07–1e-06) |
| multiscale k=16 w=200 float64 | 2.1e+00 (2e+00–2e+00) | 3.6e+00 (3e+00–5e+00) | 4.9e+00 (4e+00–7e+00) | 7.0e-06 (6e-06–3e-05) | 5.9e-05 (1e-05–7e-05) |
| multiscale k=16 w=400 float64 | 2.1e+00 (1e+00–5e+00) | 2.9e+00 (1e+00–3e+00) | 4.9e+00 (4e+00–6e+00) | 3.1e-04 (6e-06–6e-04) | 2.4e-04 (2e-04–6e-04) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| multiscale k=16 w=100 float64 | PLM [base] | 1.89e-01 | 2.4e+03 | 0 | 318 | 318 | 3.3e+01 | 5.0e+01 |
| multiscale k=16 w=100 float64 | PLM [lag4] | 7.99e-02 | 2.4e+03 | 0 | 124 | 383 | 3.9e+01 | 5.3e+01 |
| multiscale k=16 w=100 float64 | PLM [trust] | 1.91e-01 | 2.4e+03 | 0 | 315 | 315 | 5.0e+01 | — |
| multiscale k=16 w=100 float64 | PLM [warm_adamw] | 2.09e-01 | 2.4e+03 | 2143 | 288 | 288 | 1.2e+01 | 2.4e+01 |
| multiscale k=16 w=100 float64 | PLM [warm_trust_lag4] | 7.92e-02 | 2.4e+03 | 2161 | 193 | 759 | 1.2e+01 | 2.9e+01 |
| multiscale k=16 w=200 float64 | PLM [base] | 6.21e-01 | 5.6e+03 | 0 | 97 | 97 | — | — |
| multiscale k=16 w=200 float64 | PLM [lag4] | 2.68e-01 | 5.6e+03 | 0 | 27 | 83 | — | — |
| multiscale k=16 w=200 float64 | PLM [trust] | 6.35e-01 | 5.6e+03 | 0 | 95 | 95 | — | — |
| multiscale k=16 w=200 float64 | PLM [warm_adamw] | 6.87e-01 | 5.6e+03 | 1650 | 88 | 88 | 1.9e+01 | 5.6e+01 |
| multiscale k=16 w=200 float64 | PLM [warm_trust_lag4] | 2.59e-01 | 5.6e+03 | 1729 | 60 | 233 | 2.4e+01 | — |
| multiscale k=16 w=400 float64 | PLM [base] | 2.03e+00 | 1.4e+04 | 0 | 30 | 30 | — | — |
| multiscale k=16 w=400 float64 | PLM [lag4] | 9.05e-01 | 1.4e+04 | 0 | 21 | 68 | — | — |
| multiscale k=16 w=400 float64 | PLM [trust] | 2.08e+00 | 1.4e+04 | 0 | 29 | 29 | — | — |
| multiscale k=16 w=400 float64 | PLM [warm_adamw] | 2.28e+00 | 1.4e+04 | 1015 | 27 | 27 | 4.0e+01 | 3.3e+01 |
| multiscale k=16 w=400 float64 | PLM [warm_trust_lag4] | 8.98e-01 | 1.4e+04 | 1023 | 18 | 68 | 4.8e+01 | — |
