# F: 36 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW | AdamW→L-BFGS | L-BFGS | Muon | Muon→L-BFGS | PLM [rec] |
|---|---|---|---|---|---|---|
| poisson w=50 float32 | 1.9e-05 (4e-06–1e-02) | 3.9e-06 (3e-06–5e-06) | 3.1e-06 (3e-06–3e-06) | 2.3e-02 (2e-02–3e-02) | 2.1e-06 (2e-06–3e-06) | 2.3e-07 (2e-07–2e-07) |
| poisson w=50 float64 | 8.1e-05 (7e-05–9e-05) | 2.7e-06 (3e-06–4e-06) | 2.2e-06 (2e-06–3e-06) | 2.6e-02 (2e-02–4e-02) | 4.3e-07 (2e-07–1e-06) | 7.0e-09 (4e-09–1e-08) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| poisson w=50 float32 | AdamW | 2.87e-03 | 3.5e+01 | 20942 | 0 | 20942 | 1.4e+00 | 2.8e+01 |
| poisson w=50 float32 | AdamW→L-BFGS | 1.09e-03 | 3.7e+01 | 17619 | 0 | 55240 | 1.2e+00 | 1.8e+01 |
| poisson w=50 float32 | L-BFGS | 9.08e-04 | 3.7e+01 | 16726 | 0 | 66100 | 5.5e-01 | 1.5e+00 |
| poisson w=50 float32 | Muon | 3.28e-03 | 3.5e+01 | 18277 | 0 | 18277 | — | — |
| poisson w=50 float32 | Muon→L-BFGS | 1.23e-03 | 3.7e+01 | 16374 | 0 | 48980 | 1.8e+01 | 1.8e+01 |
| poisson w=50 float32 | PLM [rec] | 6.36e-02 | 5.6e+02 | 2073 | 126 | 126 | 1.5e+00 | 6.6e+00 |
| poisson w=50 float64 | AdamW | 2.77e-03 | 3.7e+01 | 21700 | 0 | 21700 | 1.2e+00 | 3.2e+01 |
| poisson w=50 float64 | AdamW→L-BFGS | 8.00e-03 | 4.2e+01 | 15127 | 0 | 7500 | 1.3e+00 | 1.8e+01 |
| poisson w=50 float64 | L-BFGS | 5.48e-03 | 4.2e+01 | 12474 | 0 | 10960 | 4.5e-01 | 1.8e+00 |
| poisson w=50 float64 | Muon | 3.21e-03 | 3.7e+01 | 18689 | 0 | 18689 | — | — |
| poisson w=50 float64 | Muon→L-BFGS | 8.46e-03 | 4.2e+01 | 14786 | 0 | 7100 | 1.8e+01 | 1.8e+01 |
| poisson w=50 float64 | PLM [rec] | 8.77e-02 | 1.1e+03 | 2094 | 685 | 685 | 1.4e+00 | 7.4e+00 |
