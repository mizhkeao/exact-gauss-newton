# A: 90 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW | AdamW→L-BFGS | L-BFGS | Muon | Muon→L-BFGS | PLM |
|---|---|---|---|---|---|---|
| beam w=50 float64 | 2.8e-02 (2e-02–4e-02) | 2.3e-05 (9e-06–4e-05) | 4.5e-06 (2e-06–5e-06) | 3.3e-02 (3e-02–1e-01) | 1.9e-06 (1e-06–1e-05) | 6.3e-06 (2e-06–6e-05) |
| multiscale k=16 w=50 float64 | 8.8e-03 (5e-03–1e-02) | 1.2e-03 (1e-04–2e-02) | 2.1e-03 (2e-03–8e-03) | 1.2e-01 (3e-02–2e-01) | 1.5e-03 (5e-04–2e-03) | 3.2e-07 (2e-07–8e-07) |
| noisy sigma=0.01 w=50 float64 | 5.8e-04 (4e-04–2e-02) | 6.5e-04 (7e-04–7e-04) | 6.5e-04 (7e-04–7e-04) | 3.7e-02 (1e-02–4e-02) | 6.5e-04 (7e-04–7e-04) | 6.5e-04 (7e-04–7e-04) |
| order1 w=50 float64 | 1.3e-04 (2e-05–3e-03) | 5.8e-05 (3e-05–9e-05) | 7.4e-05 (5e-05–8e-05) | 9.9e-03 (7e-03–2e-02) | 1.0e-05 (7e-06–2e-05) | 1.2e-08 (1e-08–1e-08) |
| poisson w=50 float64 | 6.9e-04 (8e-06–2e-03) | 2.4e-06 (1e-06–3e-06) | 2.2e-06 (2e-06–3e-06) | 3.2e-02 (1e-02–3e-02) | 6.5e-07 (6e-07–2e-06) | 5.4e-08 (4e-08–6e-08) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| beam w=50 float64 | AdamW | 1.16e-02 | 6.0e+01 | 5172 | 0 | 5172 | — | — |
| beam w=50 float64 | AdamW→L-BFGS | 2.27e-02 | 6.5e+01 | 4992 | 0 | 2660 | 2.2e+01 | 2.7e+01 |
| beam w=50 float64 | L-BFGS | 1.63e-02 | 6.5e+01 | 4538 | 0 | 3700 | 6.1e+00 | 1.7e+01 |
| beam w=50 float64 | Muon | 1.07e-02 | 6.0e+01 | 5625 | 0 | 5625 | — | — |
| beam w=50 float64 | Muon→L-BFGS | 2.23e-02 | 6.5e+01 | 4895 | 0 | 2700 | 1.9e+01 | 2.2e+01 |
| beam w=50 float64 | PLM | 3.82e-01 | 5.0e+03 | 0 | 157 | 157 | 2.3e+01 | 3.7e+01 |
| multiscale k=16 w=50 float64 | AdamW | 2.77e-03 | 3.7e+01 | 21639 | 0 | 21639 | 2.5e+01 | — |
| multiscale k=16 w=50 float64 | AdamW→L-BFGS | 7.90e-03 | 4.2e+01 | 15270 | 0 | 7600 | 3.7e+01 | — |
| multiscale k=16 w=50 float64 | L-BFGS | 6.96e-03 | 4.2e+01 | 9507 | 0 | 8640 | 4.7e+01 | — |
| multiscale k=16 w=50 float64 | Muon | 3.24e-03 | 3.7e+01 | 18514 | 0 | 18514 | — | — |
| multiscale k=16 w=50 float64 | Muon→L-BFGS | 8.56e-03 | 4.2e+01 | 12446 | 0 | 7020 | 3.7e+01 | — |
| multiscale k=16 w=50 float64 | PLM | 7.84e-02 | 1.1e+03 | 0 | 766 | 766 | 1.2e+01 | 2.0e+01 |
| noisy sigma=0.01 w=50 float64 | AdamW | 3.15e-03 | 3.7e+01 | 19021 | 0 | 19021 | 8.4e-01 | — |
| noisy sigma=0.01 w=50 float64 | AdamW→L-BFGS | 8.79e-03 | 4.2e+01 | 13993 | 0 | 6840 | 8.5e-01 | — |
| noisy sigma=0.01 w=50 float64 | L-BFGS | 6.40e-03 | 4.2e+01 | 11829 | 0 | 9380 | 3.6e-01 | — |
| noisy sigma=0.01 w=50 float64 | Muon | 3.95e-03 | 3.7e+01 | 15176 | 0 | 15176 | — | — |
| noisy sigma=0.01 w=50 float64 | Muon→L-BFGS | 8.93e-03 | 4.2e+01 | 12409 | 0 | 6720 | 1.8e+01 | — |
| noisy sigma=0.01 w=50 float64 | PLM | 1.20e-01 | 1.3e+03 | 0 | 499 | 499 | 1.3e+00 | — |
| order1 w=50 float64 | AdamW | 2.00e-03 | 3.4e+01 | 29998 | 0 | 29998 | 1.4e+00 | — |
| order1 w=50 float64 | AdamW→L-BFGS | 4.40e-03 | 3.5e+01 | 26220 | 0 | 13640 | 6.5e-01 | — |
| order1 w=50 float64 | L-BFGS | 5.41e-03 | 3.9e+01 | 13872 | 0 | 11100 | 4.0e-01 | — |
| order1 w=50 float64 | Muon | 2.60e-03 | 3.4e+01 | 23069 | 0 | 23069 | — | — |
| order1 w=50 float64 | Muon→L-BFGS | 6.60e-03 | 3.9e+01 | 18542 | 0 | 9100 | 1.8e+01 | 1.8e+01 |
| order1 w=50 float64 | PLM | 4.08e-02 | 6.5e+02 | 0 | 1470 | 1470 | 5.0e-01 | 3.4e+00 |
| poisson w=50 float64 | AdamW | 2.72e-03 | 3.7e+01 | 22075 | 0 | 22075 | 1.3e+00 | 3.2e+01 |
| poisson w=50 float64 | AdamW→L-BFGS | 8.16e-03 | 4.2e+01 | 15793 | 0 | 7360 | 1.4e+00 | 1.8e+01 |
| poisson w=50 float64 | L-BFGS | 5.85e-03 | 4.2e+01 | 11870 | 0 | 10260 | 5.0e-01 | 1.9e+00 |
| poisson w=50 float64 | Muon | 3.30e-03 | 3.7e+01 | 18171 | 0 | 18171 | — | — |
| poisson w=50 float64 | Muon→L-BFGS | 8.02e-03 | 4.2e+01 | 14188 | 0 | 7480 | 1.8e+01 | 1.9e+01 |
| poisson w=50 float64 | PLM | 7.77e-02 | 1.1e+03 | 0 | 773 | 773 | 1.0e+00 | 3.7e+00 |
