# S: 45 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW | AdamW→L-BFGS | Muon | PLM [cg_rec2] | PLM [dual_rec2] |
|---|---|---|---|---|---|
| multiscale k=16 w=100 float64 | — | — | — | 1.1e-04 (2e-05–3e-04) | 1.7e-08 (1e-08–2e-08) |
| multiscale k=16 w=400 float64 | 8.2e-02 (2e-02–1e-01) | 1.0e-04 (5e-05–4e-04) | 4.4e-02 (2e-02–5e-02) | 5.3e-05 (2e-05–8e-05) | 9.6e-07 (4e-07–1e-06) |
| multiscale k=16 w=800 float64 | 3.9e-02 (3e-02–3e-01) | 8.7e-04 (6e-04–1e-03) | 8.3e-02 (2e-02–9e-02) | 3.1e-04 (4e-05–6e-01) | — |
| multiscale k=16 w=1600 float64 | 3.9e-01 (2e-01–6e-01) | 3.5e-03 (2e-03–7e-03) | 1.5e-01 (1e-01–2e-01) | 8.8e-04 (3e-04–9e-04) | — |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| multiscale k=16 w=100 float64 | PLM [cg_rec2] | 6.17e-01 | 2.4e+02 | 41594 | 292 | 292 | 2.1e+01 | 8.5e+01 |
| multiscale k=16 w=100 float64 | PLM [dual_rec2] | 2.02e-01 | 2.4e+03 | 6378 | 891 | 891 | 1.9e+01 | 2.7e+01 |
| multiscale k=16 w=400 float64 | AdamW | 5.86e-03 | 9.2e+01 | 30718 | 0 | 30718 | — | — |
| multiscale k=16 w=400 float64 | AdamW→L-BFGS | 1.32e-02 | 3.6e+02 | 24229 | 0 | 13680 | 7.0e+01 | 1.4e+02 |
| multiscale k=16 w=400 float64 | Muon | 5.72e-03 | 8.9e+01 | 31450 | 0 | 31450 | — | — |
| multiscale k=16 w=400 float64 | PLM [cg_rec2] | 1.03e+00 | 1.0e+03 | 28718 | 176 | 176 | 2.3e+01 | 8.1e+01 |
| multiscale k=16 w=400 float64 | PLM [dual_rec2] | 2.21e+00 | 1.4e+04 | 3108 | 82 | 82 | 2.2e+01 | 7.7e+01 |
| multiscale k=16 w=800 float64 | AdamW | 2.04e-02 | 1.8e+02 | 8807 | 0 | 8807 | — | — |
| multiscale k=16 w=800 float64 | AdamW→L-BFGS | 3.81e-02 | 1.2e+03 | 7839 | 0 | 4740 | 1.1e+02 | — |
| multiscale k=16 w=800 float64 | Muon | 2.00e-02 | 1.7e+02 | 9011 | 0 | 9011 | — | — |
| multiscale k=16 w=800 float64 | PLM [cg_rec2] | 2.57e+00 | 2.3e+03 | 10984 | 70 | 70 | 2.4e+01 | 1.7e+02 |
| multiscale k=16 w=1600 float64 | AdamW | 7.13e-02 | 4.8e+02 | 2531 | 0 | 2531 | — | — |
| multiscale k=16 w=1600 float64 | AdamW→L-BFGS | 1.27e-01 | 4.8e+03 | 2287 | 0 | 1420 | 1.4e+02 | — |
| multiscale k=16 w=1600 float64 | Muon | 7.13e-02 | 4.4e+02 | 2530 | 0 | 2530 | — | — |
| multiscale k=16 w=1600 float64 | PLM [cg_rec2] | 7.26e+00 | 6.0e+03 | 3368 | 25 | 25 | 1.5e+02 | — |
