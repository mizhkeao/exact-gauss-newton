# C: 69 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW | AdamW→L-BFGS | Muon | PLM | PLM [dense] |
|---|---|---|---|---|---|
| multiscale k=16 w=25 float64 | 1.9e+00 (8e-03–2e+00) | 4.4e-02 (6e-04–2e+00) | 1.6e-01 (2e-01–2e-01) | 8.1e-08 (7e-09–9e-08) | 1.7e-07 (9e-08–5e-07) |
| multiscale k=16 w=50 float64 | 9.9e-03 (5e-03–1e-02) | 3.1e-03 (2e-03–4e-03) | 2.0e-01 (1e-01–3e-01) | 5.0e-07 (3e-07–8e-07) | 2.6e-05 (1e-05–7e-05) |
| multiscale k=16 w=100 float64 | 1.0e-02 (6e-03–3e-02) | 5.7e-03 (6e-04–9e-03) | 3.1e-02 (3e-02–1e-01) | 7.6e-06 (2e-06–1e-05) | 2.7e+00 (2e+00–4e+00) |
| multiscale k=16 w=200 float64 | 8.3e-03 (4e-03–4e-02) | 5.2e-04 (4e-04–3e-03) | 5.8e-02 (5e-02–9e-02) | 1.7e+00 (2e+00–2e+00) | — |
| multiscale k=16 w=400 float64 | 1.1e-02 (1e-02–1e-01) | 4.3e-04 (1e-04–6e-04) | 1.3e-01 (7e-02–1e-01) | 2.1e+00 (1e+00–5e+00) | — |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| multiscale k=16 w=25 float64 | AdamW | 2.76e-03 | 3.5e+01 | 21728 | 0 | 21728 | — | — |
| multiscale k=16 w=25 float64 | AdamW→L-BFGS | 2.98e-02 | 3.6e+01 | 8764 | 0 | 2020 | 4.0e+01 | — |
| multiscale k=16 w=25 float64 | Muon | 3.49e-03 | 3.5e+01 | 17172 | 0 | 17172 | — | — |
| multiscale k=16 w=25 float64 | PLM | 2.25e-02 | 5.3e+02 | 0 | 2671 | 2671 | 4.7e+00 | 1.2e+01 |
| multiscale k=16 w=25 float64 | PLM [dense] | 3.58e-02 | 5.3e+02 | 0 | 1675 | 1675 | 7.3e+00 | 1.8e+01 |
| multiscale k=16 w=50 float64 | AdamW | 3.21e-03 | 3.7e+01 | 18680 | 0 | 18680 | 3.0e+01 | — |
| multiscale k=16 w=50 float64 | AdamW→L-BFGS | 9.46e-03 | 4.2e+01 | 12994 | 0 | 6360 | 3.0e+01 | — |
| multiscale k=16 w=50 float64 | Muon | 3.25e-03 | 3.7e+01 | 18440 | 0 | 18440 | — | — |
| multiscale k=16 w=50 float64 | PLM | 8.95e-02 | 1.1e+03 | 0 | 671 | 671 | 1.4e+01 | 2.3e+01 |
| multiscale k=16 w=50 float64 | PLM [dense] | 2.89e-01 | 1.1e+03 | 0 | 208 | 208 | 4.5e+01 | 5.7e+01 |
| multiscale k=16 w=100 float64 | AdamW | 2.79e-03 | 4.3e+01 | 21535 | 0 | 21535 | 1.6e+01 | — |
| multiscale k=16 w=100 float64 | AdamW→L-BFGS | 3.17e-02 | 6.0e+01 | 8469 | 0 | 1900 | 2.0e+01 | — |
| multiscale k=16 w=100 float64 | Muon | 3.28e-03 | 4.3e+01 | 18280 | 0 | 18280 | — | — |
| multiscale k=16 w=100 float64 | PLM | 1.87e-01 | 2.4e+03 | 0 | 322 | 322 | 3.2e+01 | 5.0e+01 |
| multiscale k=16 w=100 float64 | PLM [dense] | 1.64e+01 | 6.6e+03 | 0 | 4 | 4 | — | — |
| multiscale k=16 w=200 float64 | AdamW | 3.56e-03 | 5.7e+01 | 16867 | 0 | 16867 | 4.4e+01 | — |
| multiscale k=16 w=200 float64 | AdamW→L-BFGS | 1.07e-02 | 1.2e+02 | 11454 | 0 | 5620 | 3.2e+01 | — |
| multiscale k=16 w=200 float64 | Muon | 3.65e-03 | 5.6e+01 | 16436 | 0 | 16436 | — | — |
| multiscale k=16 w=200 float64 | PLM | 6.89e-01 | 5.6e+03 | 0 | 88 | 88 | — | — |
| multiscale k=16 w=400 float64 | AdamW | 5.86e-03 | 9.2e+01 | 10241 | 0 | 10241 | — | — |
| multiscale k=16 w=400 float64 | AdamW→L-BFGS | 1.29e-02 | 3.6e+02 | 8105 | 0 | 4640 | 2.9e+01 | — |
| multiscale k=16 w=400 float64 | Muon | 6.34e-03 | 8.9e+01 | 9470 | 0 | 9470 | — | — |
| multiscale k=16 w=400 float64 | PLM | 2.04e+00 | 1.4e+04 | 0 | 30 | 30 | — | — |
