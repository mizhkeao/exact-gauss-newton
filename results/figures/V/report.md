# V: PLM upgrades, round 4 (width 50)

Final relative L2, median (min–max) over seeds; `bad` = seeds with error > 1e-2; s/Jac = train time per Jacobian; val/train = validation residual / training loss at the end (fixed grid).

| problem | variant | n | median (min–max) | bad | s/Jac | val/train |
|---|---|---|---|---|---|---|
| order1 | PLM rec2 | 5 | 4.6e-09 (4e-09–2e-08) | 0 | 0.05 | 0.9 |
| order1 | + varpro | 5 | 1.3e-14 (1e-14–2e-14) | 0 | 1.43 | 0.9 |
| order1 | + resample | 5 | 1.5e-08 (7e-09–3e-08) | 0 | 0.05 | 0.9 |
| order1 | + both | 5 | 1.6e-14 (1e-14–3e-14) | 0 | 1.00 | 0.8 |
| order1 | frozen-feature LS | 3 | 3.9e-13 (3e-13–5e-13) | 0 | 0.77 | nan |
| poisson | PLM rec2 | 5 | 5.2e-09 (2e-09–2e-08) | 0 | 0.09 | 0.9 |
| poisson | + varpro | 5 | 1.3e-14 (1e-14–2e-14) | 0 | 1.20 | 0.9 |
| poisson | + resample | 5 | 7.7e-09 (2e-09–1e-08) | 0 | 0.09 | 1.0 |
| poisson | + both | 5 | 1.6e-14 (1e-14–3e-14) | 0 | 0.84 | 0.9 |
| poisson | frozen-feature LS | 3 | 1.4e-14 (1e-14–1e-14) | 0 | 1.12 | nan |
| beam | PLM rec2 | 5 | 1.0e-07 (4e-08–1e-06) | 0 | 0.41 | 0.9 |
| beam | + varpro | 5 | 2.4e-14 (1e-14–4e-14) | 0 | 3.19 | 0.9 |
| beam | + resample | 5 | 1.0e-06 (3e-07–2e-06) | 0 | 0.46 | 1.0 |
| beam | + both | 5 | 3.1e-14 (1e-14–3e-14) | 0 | 1.69 | 0.9 |
| beam | frozen-feature LS | 3 | 2.9e-14 (9e-15–3e-14) | 0 | 1.60 | nan |
| multiscale k=16 | PLM rec2 | 5 | 9.9e-08 (6e-08–1e-07) | 0 | 0.09 | 1.0 |
| multiscale k=16 | + varpro | 5 | 1.5e-06 (5e-08–4e-06) | 0 | 0.22 | 1.0 |
| multiscale k=16 | + resample | 5 | 2.0e-05 (2e-06–3e-05) | 0 | 0.10 | 1.0 |
| multiscale k=16 | + both | 5 | 4.4e-06 (8e-07–5e-05) | 0 | 0.35 | 1.0 |
| multiscale k=16 | frozen-feature LS | 3 | 4.6e-04 (1e-05–3e-03) | 0 | 0.25 | nan |
| multiscale k=32 | PLM rec2 | 5 | 2.4e-06 (4e-07–3e+00) | 2 | 0.09 | 1.0 |
| multiscale k=32 | + varpro | 5 | 1.4e+00 (4e-04–3e+03) | 3 | 0.22 | 1.0 |
| multiscale k=32 | + resample | 5 | 1.9e+00 (8e-05–2e+00) | 3 | 0.09 | 1.0 |
| multiscale k=32 | + both | 5 | 5.1e-01 (6e-02–4e+00) | 5 | 0.35 | 1.0 |
| multiscale k=32 | frozen-feature LS | 3 | 9.2e-02 (9e-02–9e-02) | 3 | 0.23 | nan |
| nlpoisson | PLM rec2 | 3 | 4.4e-10 (3e-10–5e-10) | 0 | 0.10 | 0.9 |
| nlpoisson | + varpro | 5 | 8.7e-15 (5e-15–2e-14) | 0 | 4.68 | 1.0 |
| nlpoisson | + resample | 5 | 2.2e-09 (9e-10–4e-09) | 0 | 0.11 | 1.0 |
| nlpoisson | + both | 5 | 1.0e-14 (5e-15–2e-14) | 0 | 3.04 | 1.0 |
| nlpoisson | frozen-feature LS | 3 | 1.5e-14 (1e-14–2e-14) | 0 | 0.30 | nan |
| helmholtz | PLM rec2 | 5 | 7.8e-05 (5e-05–1e-04) | 0 | 0.19 | 147839.4 |
| helmholtz | + varpro | 5 | 1.1e-04 (7e-05–2e-04) | 0 | 1.08 | 272.8 |
| helmholtz | + resample | 5 | 6.5e-06 (4e-06–2e-05) | 0 | 0.19 | 1.2 |
| helmholtz | + both | 5 | 1.8e-04 (1e-04–2e-04) | 0 | 1.12 | 1.1 |
| helmholtz | frozen-feature LS | 3 | 1.9e+00 (1e+00–6e+00) | 3 | 0.84 | nan |
| allen_cahn | PLM rec2 | 3 | 4.0e-05 (3e-05–9e-05) | 0 | 0.59 | 2.2 |
| allen_cahn | + varpro | 5 | 6.5e-05 (2e-05–3e-04) | 0 | 16.21 | 1.0 |
| allen_cahn | + resample | 5 | 5.3e-05 (3e-05–1e-04) | 0 | 0.61 | 1.0 |
| allen_cahn | + both | 5 | 5.6e-05 (4e-05–2e-04) | 0 | 18.10 | 0.9 |
| allen_cahn | frozen-feature LS | 3 | 4.2e-04 (2e-04–5e-04) | 0 | 0.44 | nan |
| burgers | PLM rec2 | 3 | 2.8e-07 (2e-07–6e-07) | 0 | 0.66 | 0.9 |
| burgers | + varpro | 5 | 1.0e-06 (9e-08–5e-06) | 0 | 14.65 | 1.2 |
| burgers | + resample | 5 | 3.3e-07 (2e-07–6e-07) | 0 | 0.68 | 0.8 |
| burgers | + both | 5 | 6.5e-07 (1e-07–6e-06) | 0 | 15.79 | 1.1 |
| burgers | frozen-feature LS | 3 | 4.9e-02 (5e-02–6e-02) | 3 | 0.58 | nan |
| reaction | PLM rec2 | 8 | 1.4e-04 (6e-05–1e+00) | 2 | 0.33 | 2.4 |
| reaction | + varpro | 5 | 2.2e-01 (2e-01–1e+00) | 5 | 10.00 | 0.8 |
| reaction | + resample | 5 | 1.4e-04 (7e-05–3e-04) | 0 | 0.34 | 1.2 |
| reaction | + both | 5 | 2.6e-01 (4e-02–3e-01) | 5 | 10.77 | 0.8 |
| reaction | frozen-feature LS | 3 | 2.6e-01 (2e-01–3e-01) | 3 | 0.14 | nan |
