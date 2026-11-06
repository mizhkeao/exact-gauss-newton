# L: 111 runs

## Final relative L2 error, median (min–max) over seeds

| setting | Frozen-feature LS [s1] | Frozen-feature LS [s4] | PLM [rec2] |
|---|---|---|---|
| beam w=50 float64 | 1.2e-12 (5e-13–1e-12) | 2.9e-14 (9e-15–3e-14) | — |
| beam w=200 float64 | 5.0e-13 (3e-13–1e-12) | 1.2e-14 (9e-15–5e-14) | — |
| beam w=800 float64 | 4.7e-13 (2e-13–5e-13) | 2.2e-14 (1e-14–4e-14) | — |
| helmholtz a2=4 w=50 float64 | 1.9e+00 (1e+00–6e+00) | 3.0e+00 (2e+00–6e+00) | — |
| helmholtz a2=4 w=200 float64 | 9.1e-03 (7e-03–1e-02) | 1.4e-03 (9e-04–3e-03) | 1.7e-01 (1e-01–5e-01) |
| multiscale k=16 w=50 float64 | 9.2e-02 (9e-02–9e-02) | 4.6e-04 (1e-05–3e-03) | — |
| multiscale k=16 w=200 float64 | 9.2e-02 (9e-02–9e-02) | 2.2e-07 (2e-07–3e-07) | 7.1e-07 (6e-07–1e-06) |
| multiscale k=16 w=800 float64 | 9.2e-02 (9e-02–9e-02) | 3.7e-08 (3e-08–4e-08) | — |
| multiscale k=32 w=50 float64 | 1.0e-01 (1e-01–1e-01) | 9.2e-02 (9e-02–9e-02) | — |
| multiscale k=32 w=200 float64 | 1.0e-01 (1e-01–1e-01) | 8.8e-02 (9e-02–9e-02) | 1.8e+00 (4e-05–2e+01) |
| multiscale k=32 w=800 float64 | 1.0e-01 (1e-01–1e-01) | 8.0e-02 (8e-02–8e-02) | — |
| order1 w=50 float64 | 1.9e-11 (3e-12–2e-11) | 3.9e-13 (3e-13–5e-13) | — |
| order1 w=200 float64 | 6.8e-12 (2e-12–7e-12) | 5.4e-14 (3e-14–8e-14) | — |
| order1 w=800 float64 | 6.5e-12 (6e-12–8e-12) | 3.1e-14 (3e-14–5e-14) | — |
| poisson w=50 float64 | 4.3e-13 (1e-13–5e-13) | 1.4e-14 (1e-14–1e-14) | — |
| poisson w=200 float64 | 4.2e-13 (2e-13–9e-13) | 7.0e-15 (4e-15–1e-14) | — |
| poisson w=800 float64 | 3.0e-13 (3e-13–6e-13) | 5.5e-14 (3e-14–6e-14) | — |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| beam w=50 float64 | Frozen-feature LS [s1] | 1.29e+00 | 2.2e+03 | 0 | 1 | 1 | 1.3e+00 | 1.3e+00 |
| beam w=50 float64 | Frozen-feature LS [s4] | 1.60e+00 | 2.2e+03 | 0 | 1 | 1 | 1.6e+00 | 1.6e+00 |
| beam w=200 float64 | Frozen-feature LS [s1] | 1.99e+00 | 9.0e+03 | 0 | 1 | 1 | 2.0e+00 | 2.0e+00 |
| beam w=200 float64 | Frozen-feature LS [s4] | 1.87e+00 | 9.0e+03 | 0 | 1 | 1 | 1.9e+00 | 1.9e+00 |
| beam w=800 float64 | Frozen-feature LS [s1] | 8.88e+00 | 3.6e+04 | 0 | 1 | 1 | 8.9e+00 | 8.9e+00 |
| beam w=800 float64 | Frozen-feature LS [s4] | 8.67e+00 | 3.6e+04 | 0 | 1 | 1 | 8.7e+00 | 8.7e+00 |
| helmholtz a2=4 w=50 float64 | Frozen-feature LS [s1] | 8.45e-01 | 1.9e+03 | 0 | 2 | 1 | — | — |
| helmholtz a2=4 w=50 float64 | Frozen-feature LS [s4] | 7.95e-01 | 1.9e+03 | 0 | 2 | 1 | — | — |
| helmholtz a2=4 w=200 float64 | Frozen-feature LS [s1] | 1.90e+00 | 7.3e+03 | 0 | 4 | 3 | — | — |
| helmholtz a2=4 w=200 float64 | Frozen-feature LS [s4] | 1.47e+00 | 7.3e+03 | 0 | 2 | 2 | 1.6e+00 | — |
| helmholtz a2=4 w=200 float64 | PLM [rec2] | 6.23e+00 | 2.2e+04 | 451 | 10 | 10 | — | — |
| multiscale k=16 w=50 float64 | Frozen-feature LS [s1] | 2.86e-01 | 4.4e+02 | 0 | 4 | 4 | — | — |
| multiscale k=16 w=50 float64 | Frozen-feature LS [s4] | 2.51e-01 | 4.4e+02 | 0 | 5 | 4 | 7.6e-01 | 7.7e-01 |
| multiscale k=16 w=200 float64 | Frozen-feature LS [s1] | 4.14e-01 | 1.7e+03 | 0 | 7 | 7 | — | — |
| multiscale k=16 w=200 float64 | Frozen-feature LS [s4] | 4.87e-01 | 1.7e+03 | 0 | 5 | 5 | 1.1e+00 | 1.1e+00 |
| multiscale k=16 w=200 float64 | PLM [rec2] | 6.86e-01 | 5.6e+03 | 1691 | 88 | 88 | 1.8e+01 | 2.8e+01 |
| multiscale k=16 w=800 float64 | Frozen-feature LS [s1] | 3.49e+00 | 6.5e+03 | 0 | 4 | 3 | — | — |
| multiscale k=16 w=800 float64 | Frozen-feature LS [s4] | 3.76e+00 | 6.5e+03 | 0 | 4 | 3 | 3.3e+00 | 3.3e+00 |
| multiscale k=32 w=50 float64 | Frozen-feature LS [s1] | 2.52e-01 | 4.4e+02 | 0 | 5 | 4 | — | — |
| multiscale k=32 w=50 float64 | Frozen-feature LS [s4] | 2.31e-01 | 4.4e+02 | 0 | 5 | 5 | — | — |
| multiscale k=32 w=200 float64 | Frozen-feature LS [s1] | 5.18e-01 | 1.7e+03 | 0 | 4 | 4 | — | — |
| multiscale k=32 w=200 float64 | Frozen-feature LS [s4] | 5.60e-01 | 1.7e+03 | 0 | 4 | 3 | — | — |
| multiscale k=32 w=200 float64 | PLM [rec2] | 6.68e-01 | 5.6e+03 | 2569 | 84 | 90 | 5.4e+01 | — |
| multiscale k=32 w=800 float64 | Frozen-feature LS [s1] | 3.04e+00 | 6.5e+03 | 0 | 5 | 5 | — | — |
| multiscale k=32 w=800 float64 | Frozen-feature LS [s4] | 3.45e+00 | 6.5e+03 | 0 | 4 | 4 | — | — |
| order1 w=50 float64 | Frozen-feature LS [s1] | 5.16e-01 | 2.4e+02 | 0 | 4 | 3 | 1.3e+00 | 1.3e+00 |
| order1 w=50 float64 | Frozen-feature LS [s4] | 7.65e-01 | 2.4e+02 | 0 | 1 | 1 | 7.7e-01 | 7.7e-01 |
| order1 w=200 float64 | Frozen-feature LS [s1] | 1.13e+00 | 8.4e+02 | 0 | 1 | 1 | 1.1e+00 | 1.1e+00 |
| order1 w=200 float64 | Frozen-feature LS [s4] | 9.95e-01 | 8.4e+02 | 0 | 1 | 1 | 1.0e+00 | 1.0e+00 |
| order1 w=800 float64 | Frozen-feature LS [s1] | 2.56e+00 | 3.3e+03 | 0 | 1 | 1 | 2.6e+00 | 2.6e+00 |
| order1 w=800 float64 | Frozen-feature LS [s4] | 2.74e+00 | 3.3e+03 | 0 | 1 | 1 | 2.7e+00 | 2.7e+00 |
| poisson w=50 float64 | Frozen-feature LS [s1] | 9.10e-01 | 4.4e+02 | 0 | 1 | 1 | 9.1e-01 | 9.1e-01 |
| poisson w=50 float64 | Frozen-feature LS [s4] | 1.12e+00 | 4.4e+02 | 0 | 1 | 1 | 1.1e+00 | 1.1e+00 |
| poisson w=200 float64 | Frozen-feature LS [s1] | 1.16e+00 | 1.7e+03 | 0 | 1 | 1 | 1.2e+00 | 1.2e+00 |
| poisson w=200 float64 | Frozen-feature LS [s4] | 1.11e+00 | 1.7e+03 | 0 | 1 | 1 | 1.1e+00 | 1.1e+00 |
| poisson w=800 float64 | Frozen-feature LS [s1] | 3.46e+00 | 6.5e+03 | 0 | 1 | 1 | 3.5e+00 | 3.5e+00 |
| poisson w=800 float64 | Frozen-feature LS [s4] | 3.26e+00 | 6.5e+03 | 0 | 1 | 1 | 3.3e+00 | 3.3e+00 |
