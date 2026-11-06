# V: PLM upgrades, round 4 (width 50)

Final relative L2, median (min–max) over seeds; `bad` = seeds with error > 1e-2; s/Jac = train time per Jacobian; val/train = validation residual / training loss at the end (fixed grid).

| problem | variant | n | median (min–max) | bad | s/Jac | val/train |
|---|---|---|---|---|---|---|
| order1 | PLM rec2 | 5 | 4.6e-09 (4e-09–2e-08) | 0 | 0.05 | 0.9 |
| order1 | + resample | 5 | 1.5e-08 (7e-09–3e-08) | 0 | 0.05 | 0.9 |
| order1 | 2-stage 10 % | 5 | 1.6e-12 (2e-13–6e-12) | 0 | 6.55 | 0.9 |
| order1 | 2-stage 30 % | 5 | 5.6e-12 (1e-12–1e-11) | 0 | 0.06 | 0.9 |
| order1 | 2-stage 100 % | 5 | 1.0e-12 (7e-13–3e-12) | 0 | 0.05 | 0.9 |
| order1 | frozen-feature LS | 3 | 3.9e-13 (3e-13–5e-13) | 0 | 0.77 | nan |
| poisson | PLM rec2 | 5 | 5.2e-09 (2e-09–2e-08) | 0 | 0.09 | 0.9 |
| poisson | + resample | 5 | 7.7e-09 (2e-09–1e-08) | 0 | 0.09 | 1.0 |
| poisson | 2-stage 10 % | 5 | 3.7e-14 (2e-14–4e-14) | 0 | 6.69 | 0.9 |
| poisson | 2-stage 30 % | 5 | 1.9e-14 (1e-14–3e-14) | 0 | 0.12 | 1.0 |
| poisson | 2-stage 100 % | 5 | 1.5e-14 (8e-15–1e-13) | 0 | 0.09 | 0.9 |
| poisson | frozen-feature LS | 3 | 1.4e-14 (1e-14–1e-14) | 0 | 1.12 | nan |
| beam | PLM rec2 | 5 | 1.0e-07 (4e-08–1e-06) | 0 | 0.41 | 0.9 |
| beam | + resample | 5 | 1.0e-06 (3e-07–2e-06) | 0 | 0.46 | 1.0 |
| beam | 2-stage 10 % | 5 | 2.2e-14 (6e-15–9e-14) | 0 | 8.11 | 1.0 |
| beam | 2-stage 30 % | 5 | 2.0e-14 (6e-15–3e-13) | 0 | 0.60 | 1.0 |
| beam | 2-stage 100 % | 5 | 1.3e-14 (6e-15–1e-13) | 0 | 0.42 | 0.9 |
| beam | frozen-feature LS | 3 | 2.9e-14 (9e-15–3e-14) | 0 | 1.60 | nan |
| multiscale k=16 | PLM rec2 | 5 | 9.9e-08 (6e-08–1e-07) | 0 | 0.09 | 1.0 |
| multiscale k=16 | + resample | 5 | 2.0e-05 (2e-06–3e-05) | 0 | 0.10 | 1.0 |
| multiscale k=16 | 2-stage 10 % | 5 | 4.8e-01 (2e-02–5e+00) | 5 | 6.40 | 0.9 |
| multiscale k=16 | 2-stage 30 % | 5 | 5.4e-06 (3e-06–1e-03) | 0 | 0.12 | 1.0 |
| multiscale k=16 | 2-stage 100 % | 5 | 4.8e-07 (6e-08–2e-05) | 0 | 0.09 | 1.0 |
| multiscale k=16 | frozen-feature LS | 3 | 4.6e-04 (1e-05–3e-03) | 0 | 0.25 | nan |
| multiscale k=32 | PLM rec2 | 5 | 2.4e-06 (4e-07–3e+00) | 2 | 0.09 | 1.0 |
| multiscale k=32 | + resample | 5 | 1.9e+00 (8e-05–2e+00) | 3 | 0.09 | 1.0 |
| multiscale k=32 | 2-stage 10 % | 5 | 9.0e+00 (3e+00–1e+01) | 5 | 6.68 | 1.0 |
| multiscale k=32 | 2-stage 30 % | 5 | 1.6e+00 (1e+00–5e+00) | 5 | 0.13 | 1.0 |
| multiscale k=32 | 2-stage 100 % | 5 | 1.2e-03 (9e-06–2e+00) | 2 | 0.09 | 1.0 |
| multiscale k=32 | frozen-feature LS | 3 | 9.2e-02 (9e-02–9e-02) | 3 | 0.23 | nan |
| nlpoisson | PLM rec2 | 3 | 4.4e-10 (3e-10–5e-10) | 0 | 0.10 | 0.9 |
| nlpoisson | + resample | 5 | 2.2e-09 (9e-10–4e-09) | 0 | 0.11 | 1.0 |
| nlpoisson | 2-stage 10 % | 5 | 5.8e-14 (1e-14–2e-13) | 0 | 31.08 | 0.9 |
| nlpoisson | 2-stage 30 % | 5 | 7.4e-14 (3e-14–9e-14) | 0 | 0.17 | 0.9 |
| nlpoisson | 2-stage 100 % | 5 | 1.6e-14 (1e-14–4e-14) | 0 | 0.10 | 1.0 |
| nlpoisson | frozen-feature LS | 3 | 1.5e-14 (1e-14–2e-14) | 0 | 0.30 | nan |
| helmholtz | PLM rec2 | 5 | 7.8e-05 (5e-05–1e-04) | 0 | 0.19 | 147839.4 |
| helmholtz | + resample | 5 | 6.5e-06 (4e-06–2e-05) | 0 | 0.19 | 1.2 |
| helmholtz | 2-stage 10 % | 5 | 2.6e-01 (2e-01–6e-01) | 5 | 31.31 | 1.0 |
| helmholtz | 2-stage 30 % | 5 | 3.1e-04 (3e-04–5e-04) | 0 | 0.27 | 1.4 |
| helmholtz | 2-stage 100 % | 5 | 3.8e-05 (2e-05–1e-04) | 0 | 0.19 | 1.5 |
| helmholtz | frozen-feature LS | 3 | 1.9e+00 (1e+00–6e+00) | 3 | 0.84 | nan |
| allen_cahn | PLM rec2 | 3 | 4.0e-05 (3e-05–9e-05) | 0 | 0.59 | 2.2 |
| allen_cahn | + resample | 5 | 5.3e-05 (3e-05–1e-04) | 0 | 0.61 | 1.0 |
| allen_cahn | 2-stage 10 % | 5 | 1.5e-03 (1e-03–2e-03) | 0 | 38.06 | 0.9 |
| allen_cahn | 2-stage 30 % | 5 | 3.8e-04 (2e-04–5e-04) | 0 | 0.86 | 1.1 |
| allen_cahn | 2-stage 100 % | 5 | 9.1e-05 (4e-05–1e-04) | 0 | 0.68 | 1.1 |
| allen_cahn | frozen-feature LS | 3 | 4.2e-04 (2e-04–5e-04) | 0 | 0.44 | nan |
| burgers | PLM rec2 | 3 | 2.8e-07 (2e-07–6e-07) | 0 | 0.66 | 0.9 |
| burgers | + resample | 5 | 3.3e-07 (2e-07–6e-07) | 0 | 0.68 | 0.8 |
| burgers | 2-stage 10 % | 5 | 8.6e-05 (4e-05–1e-04) | 0 | 35.94 | 0.9 |
| burgers | 2-stage 30 % | 5 | 3.1e-06 (2e-06–2e-05) | 0 | 0.98 | 0.9 |
| burgers | 2-stage 100 % | 5 | 3.1e-07 (2e-07–5e-07) | 0 | 0.71 | 0.9 |
| burgers | frozen-feature LS | 3 | 4.9e-02 (5e-02–6e-02) | 3 | 0.58 | nan |
| reaction | PLM rec2 | 8 | 1.4e-04 (6e-05–1e+00) | 2 | 0.33 | 2.4 |
| reaction | + resample | 5 | 1.4e-04 (7e-05–3e-04) | 0 | 0.34 | 1.2 |
| reaction | 2-stage 10 % | 5 | 2.9e-01 (3e-01–1e+00) | 5 | 46.22 | 0.7 |
| reaction | 2-stage 30 % | 5 | 2.4e-03 (2e-03–5e-03) | 0 | 0.48 | 1.1 |
| reaction | 2-stage 100 % | 5 | 1.4e-04 (6e-05–2e-04) | 0 | 0.34 | 1.3 |
| reaction | frozen-feature LS | 3 | 2.6e-01 (2e-01–3e-01) | 3 | 0.14 | nan |
