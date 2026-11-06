# B2: 36 runs

## Final relative L2 error, median (min–max) over seeds

| setting | PLM [lag4] | PLM [lag8] | PLM [marq_lag4_weighted] | PLM [marquardt_floor] |
|---|---|---|---|---|
| beam w=50 float64 | 3.5e-06 (1e-06–4e-06) | 1.9e-06 (1e-06–3e-06) | 8.9e-08 (9e-09–2e-07) | 3.2e-07 (1e-07–4e-06) |
| multiscale k=16 w=50 float64 | 1.6e-06 (1e-06–4e+00) | 1.3e-07 (9e-08–3e-07) | 1.4e-06 (1e-07–2e-06) | 2.4e-07 (2e-07–3e-07) |
| poisson w=50 float64 | 5.1e-08 (3e-09–7e-08) | 3.9e-08 (7e-09–6e-08) | 2.9e-09 (9e-10–2e-08) | 4.8e-09 (3e-09–8e-09) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| beam w=50 float64 | PLM [lag4] | 1.11e-01 | 5.0e+03 | 0 | 142 | 542 | 1.6e+01 | 4.3e+01 |
| beam w=50 float64 | PLM [lag8] | 8.01e-02 | 5.0e+03 | 0 | 130 | 753 | 1.9e+01 | 4.1e+01 |
| beam w=50 float64 | PLM [marq_lag4_weighted] | 1.12e-01 | 5.0e+03 | 0 | 140 | 537 | 5.9e+00 | 8.4e+00 |
| beam w=50 float64 | PLM [marquardt_floor] | 3.65e-01 | 5.0e+03 | 0 | 165 | 165 | 1.3e+01 | 2.0e+01 |
| multiscale k=16 w=50 float64 | PLM [lag4] | 3.31e-02 | 1.1e+03 | 0 | 484 | 1815 | 1.7e+01 | 2.8e+01 |
| multiscale k=16 w=50 float64 | PLM [lag8] | 2.44e-02 | 1.1e+03 | 0 | 426 | 2458 | 1.2e+01 | 2.0e+01 |
| multiscale k=16 w=50 float64 | PLM [marq_lag4_weighted] | 3.22e-02 | 1.1e+03 | 0 | 507 | 1866 | 2.5e+01 | 3.1e+01 |
| multiscale k=16 w=50 float64 | PLM [marquardt_floor] | 7.93e-02 | 1.1e+03 | 0 | 758 | 758 | 1.4e+01 | 2.5e+01 |
| poisson w=50 float64 | PLM [lag4] | 3.13e-02 | 1.1e+03 | 0 | 493 | 1918 | 8.3e-01 | 3.1e+00 |
| poisson w=50 float64 | PLM [lag8] | 2.40e-02 | 1.1e+03 | 0 | 412 | 2506 | 7.0e-01 | 3.1e+00 |
| poisson w=50 float64 | PLM [marq_lag4_weighted] | 3.11e-02 | 1.1e+03 | 0 | 498 | 1932 | 1.4e+00 | 1.6e+00 |
| poisson w=50 float64 | PLM [marquardt_floor] | 7.78e-02 | 1.1e+03 | 0 | 772 | 772 | 9.6e-01 | 1.9e+00 |
