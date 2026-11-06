# H: 75 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW→L-BFGS | Muon→L-BFGS | PLM [rec2] |
|---|---|---|---|
| beam w=50 float64 | 2.0e-05 (1e-06–2e-03) | 5.7e-06 (1e-06–9e-05) | 1.0e-07 (4e-08–1e-06) |
| multiscale k=16 w=50 float64 | 1.4e-03 (3e-04–3e-03) | 3.1e-03 (8e-04–1e-02) | 9.9e-08 (6e-08–1e-07) |
| multiscale k=32 w=50 float64 | 4.4e+00 (3e+00–6e+00) | 3.6e-01 (4e-02–9e-01) | 2.4e-06 (4e-07–3e+00) |
| order1 w=50 float64 | 2.7e-05 (2e-05–9e-05) | 1.5e-05 (8e-06–2e-05) | 4.6e-09 (4e-09–2e-08) |
| poisson w=50 float64 | 2.9e-06 (2e-06–4e-06) | 8.5e-07 (4e-07–3e-06) | 5.2e-09 (2e-09–2e-08) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| beam w=50 float64 | AdamW→L-BFGS | 2.26e-02 | 6.5e+01 | 4887 | 0 | 2660 | 2.2e+01 | 2.9e+01 |
| beam w=50 float64 | Muon→L-BFGS | 2.12e-02 | 6.5e+01 | 5248 | 0 | 2840 | 1.9e+01 | 2.2e+01 |
| beam w=50 float64 | PLM [rec2] | 4.06e-01 | 5.0e+03 | 536 | 148 | 148 | 7.4e+00 | 1.1e+01 |
| multiscale k=16 w=50 float64 | AdamW→L-BFGS | 8.39e-03 | 4.2e+01 | 14454 | 0 | 7160 | 3.5e+01 | — |
| multiscale k=16 w=50 float64 | Muon→L-BFGS | 8.42e-03 | 4.2e+01 | 13469 | 0 | 7140 | 4.0e+01 | — |
| multiscale k=16 w=50 float64 | PLM [rec2] | 8.86e-02 | 1.1e+03 | 1763 | 677 | 677 | 9.0e+00 | 1.1e+01 |
| multiscale k=32 w=50 float64 | AdamW→L-BFGS | 8.12e-03 | 4.2e+01 | 14569 | 0 | 7400 | — | — |
| multiscale k=32 w=50 float64 | Muon→L-BFGS | 8.60e-03 | 4.2e+01 | 12471 | 0 | 6980 | — | — |
| multiscale k=32 w=50 float64 | PLM [rec2] | 8.88e-02 | 1.1e+03 | 1952 | 676 | 676 | 2.6e+01 | 3.6e+01 |
| order1 w=50 float64 | AdamW→L-BFGS | 6.24e-03 | 3.5e+01 | 21405 | 0 | 9620 | 8.2e-01 | — |
| order1 w=50 float64 | Muon→L-BFGS | 6.69e-03 | 3.9e+01 | 17596 | 0 | 8980 | 1.8e+01 | 1.8e+01 |
| order1 w=50 float64 | PLM [rec2] | 4.64e-02 | 6.5e+02 | 2569 | 1295 | 1295 | 8.5e-01 | 7.5e+00 |
| poisson w=50 float64 | AdamW→L-BFGS | 8.19e-03 | 4.2e+01 | 15122 | 0 | 7340 | 8.0e-01 | 1.8e+01 |
| poisson w=50 float64 | Muon→L-BFGS | 8.05e-03 | 4.2e+01 | 14012 | 0 | 7460 | 1.8e+01 | 1.9e+01 |
| poisson w=50 float64 | PLM [rec2] | 8.78e-02 | 1.1e+03 | 1941 | 684 | 684 | 9.5e-01 | 7.4e+00 |
