# E: 123 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW | AdamW→L-BFGS | Muon | Muon→L-BFGS | PLM [rec] |
|---|---|---|---|---|---|
| beam w=50 float64 | — | 2.7e-06 (2e-06–2e-05) | — | 2.2e-06 (8e-07–7e-06) | 1.2e-07 (5e-08–4e-07) |
| multiscale k=1 w=50 float64 | 3.0e-03 (1e-03–8e-03) | 1.6e-06 (1e-06–3e-06) | 2.3e-02 (1e-02–3e-02) | 4.1e-07 (3e-07–2e-06) | 5.1e-09 (1e-09–1e-08) |
| multiscale k=4 w=50 float64 | 1.9e-05 (9e-06–7e-03) | 1.3e-06 (5e-07–1e-06) | 5.3e-03 (5e-03–8e-03) | 9.0e-07 (5e-07–2e-06) | 7.7e-09 (4e-09–1e-08) |
| multiscale k=16 w=50 float64 | 4.8e-03 (5e-03–8e-03) | 1.4e-03 (7e-04–3e-03) | 1.2e-01 (1e-01–2e-01) | 2.9e-03 (7e-04–4e-03) | 2.2e-07 (1e-07–3e-07) |
| multiscale k=32 w=50 float64 | 8.4e+00 (7e+00–1e+01) | 4.8e+00 (1e+00–5e+00) | 9.0e-01 (3e-01–1e+00) | 3.6e-01 (5e-02–9e-01) | 4.9e-06 (3e-06–3e+00) |
| noisy sigma=0 w=50 float64 | — | 2.6e-06 (2e-06–3e-06) | — | 1.5e-06 (1e-06–3e-06) | 1.4e-08 (1e-08–2e-08) |
| noisy sigma=0.001 w=50 float64 | — | 6.5e-05 (7e-05–7e-05) | — | 6.5e-05 (7e-05–7e-05) | 6.4e-05 (6e-05–7e-05) |
| noisy sigma=0.01 w=50 float64 | — | 6.5e-04 (7e-04–7e-04) | — | 6.5e-04 (7e-04–7e-04) | 6.3e-04 (6e-04–7e-04) |
| noisy sigma=0.1 w=50 float64 | — | 6.5e-03 (7e-03–7e-03) | — | 6.5e-03 (7e-03–7e-03) | 6.2e-03 (6e-03–6e-03) |
| order1 w=50 float64 | — | 3.1e-05 (3e-05–6e-05) | — | 1.5e-05 (1e-05–2e-05) | 4.1e-09 (4e-09–2e-08) |
| poisson w=50 float64 | — | 2.9e-06 (3e-06–4e-06) | — | 5.1e-07 (2e-07–8e-07) | 4.7e-09 (4e-09–1e-08) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| beam w=50 float64 | AdamW→L-BFGS | 2.06e-02 | 6.5e+01 | 5340 | 0 | 2920 | 2.1e+01 | 3.0e+01 |
| beam w=50 float64 | Muon→L-BFGS | 2.11e-02 | 6.5e+01 | 5211 | 0 | 2840 | 1.9e+01 | 2.1e+01 |
| beam w=50 float64 | PLM [rec] | 4.09e-01 | 5.0e+03 | 548 | 147 | 147 | 8.0e+00 | 1.2e+01 |
| multiscale k=1 w=50 float64 | AdamW | 2.68e-03 | 3.7e+01 | 22415 | 0 | 22415 | 1.0e+00 | 2.0e+01 |
| multiscale k=1 w=50 float64 | AdamW→L-BFGS | 8.50e-03 | 4.2e+01 | 14880 | 0 | 7060 | 1.2e+00 | 1.8e+01 |
| multiscale k=1 w=50 float64 | Muon | 3.23e-03 | 3.7e+01 | 18549 | 0 | 18549 | — | — |
| multiscale k=1 w=50 float64 | Muon→L-BFGS | 8.11e-03 | 4.2e+01 | 14867 | 0 | 7400 | 1.8e+01 | 1.9e+01 |
| multiscale k=1 w=50 float64 | PLM [rec] | 8.74e-02 | 1.1e+03 | 2108 | 687 | 687 | 1.0e+00 | 7.4e+00 |
| multiscale k=4 w=50 float64 | AdamW | 2.81e-03 | 3.7e+01 | 21345 | 0 | 21345 | 4.7e+00 | 4.4e+01 |
| multiscale k=4 w=50 float64 | AdamW→L-BFGS | 8.44e-03 | 4.2e+01 | 14858 | 0 | 7120 | 5.0e+00 | 1.9e+01 |
| multiscale k=4 w=50 float64 | Muon | 3.28e-03 | 3.7e+01 | 18306 | 0 | 18306 | — | — |
| multiscale k=4 w=50 float64 | Muon→L-BFGS | 7.88e-03 | 4.2e+01 | 14649 | 0 | 7620 | 1.8e+01 | 2.0e+01 |
| multiscale k=4 w=50 float64 | PLM [rec] | 8.86e-02 | 1.1e+03 | 1716 | 677 | 677 | 5.8e+00 | 7.0e+00 |
| multiscale k=16 w=50 float64 | AdamW | 2.83e-03 | 3.7e+01 | 21235 | 0 | 21235 | 2.6e+01 | — |
| multiscale k=16 w=50 float64 | AdamW→L-BFGS | 8.31e-03 | 4.2e+01 | 14441 | 0 | 7220 | 2.9e+01 | — |
| multiscale k=16 w=50 float64 | Muon | 3.33e-03 | 3.7e+01 | 18027 | 0 | 18027 | — | — |
| multiscale k=16 w=50 float64 | Muon→L-BFGS | 7.92e-03 | 4.2e+01 | 13566 | 0 | 7580 | 4.0e+01 | — |
| multiscale k=16 w=50 float64 | PLM [rec] | 8.98e-02 | 1.1e+03 | 1771 | 669 | 669 | 8.2e+00 | 1.1e+01 |
| multiscale k=32 w=50 float64 | AdamW | 2.78e-03 | 3.7e+01 | 21595 | 0 | 21595 | — | — |
| multiscale k=32 w=50 float64 | AdamW→L-BFGS | 8.67e-03 | 4.2e+01 | 13076 | 0 | 6920 | — | — |
| multiscale k=32 w=50 float64 | Muon | 3.27e-03 | 3.7e+01 | 18335 | 0 | 18335 | — | — |
| multiscale k=32 w=50 float64 | Muon→L-BFGS | 8.71e-03 | 4.2e+01 | 12652 | 0 | 6900 | — | — |
| multiscale k=32 w=50 float64 | PLM [rec] | 8.85e-02 | 1.1e+03 | 1999 | 678 | 678 | 4.1e+01 | 4.6e+01 |
| noisy sigma=0 w=50 float64 | AdamW→L-BFGS | 8.80e-03 | 4.2e+01 | 13381 | 0 | 6820 | 1.1e+00 | 1.9e+01 |
| noisy sigma=0 w=50 float64 | Muon→L-BFGS | 1.13e-02 | 4.2e+01 | 10838 | 0 | 5340 | 1.8e+01 | 1.8e+01 |
| noisy sigma=0 w=50 float64 | PLM [rec] | 1.02e-01 | 1.3e+03 | 1634 | 590 | 590 | 1.2e+00 | 7.6e+00 |
| noisy sigma=0.001 w=50 float64 | AdamW→L-BFGS | 8.84e-03 | 4.2e+01 | 13683 | 0 | 6800 | 1.0e+00 | — |
| noisy sigma=0.001 w=50 float64 | Muon→L-BFGS | 8.70e-03 | 4.2e+01 | 13249 | 0 | 6900 | 1.8e+01 | 1.9e+01 |
| noisy sigma=0.001 w=50 float64 | PLM [rec] | 1.02e-01 | 1.3e+03 | 1626 | 588 | 588 | 1.2e+00 | — |
| noisy sigma=0.01 w=50 float64 | AdamW→L-BFGS | 9.63e-03 | 4.2e+01 | 13235 | 0 | 6240 | 8.3e-01 | — |
| noisy sigma=0.01 w=50 float64 | Muon→L-BFGS | 9.07e-03 | 4.2e+01 | 12544 | 0 | 6620 | 1.8e+01 | — |
| noisy sigma=0.01 w=50 float64 | PLM [rec] | 1.02e-01 | 1.3e+03 | 1742 | 591 | 591 | 7.4e-01 | — |
| noisy sigma=0.1 w=50 float64 | AdamW→L-BFGS | 9.32e-03 | 4.2e+01 | 13670 | 0 | 6440 | — | — |
| noisy sigma=0.1 w=50 float64 | Muon→L-BFGS | 8.60e-03 | 4.2e+01 | 13646 | 0 | 6980 | — | — |
| noisy sigma=0.1 w=50 float64 | PLM [rec] | 1.03e-01 | 1.3e+03 | 1803 | 585 | 585 | — | — |
| order1 w=50 float64 | AdamW→L-BFGS | 4.20e-03 | 3.5e+01 | 26951 | 0 | 14300 | 5.6e-01 | — |
| order1 w=50 float64 | Muon→L-BFGS | 6.66e-03 | 3.9e+01 | 17927 | 0 | 9020 | 1.8e+01 | — |
| order1 w=50 float64 | PLM [rec] | 4.61e-02 | 6.5e+02 | 3032 | 1301 | 1301 | 6.5e-01 | 7.8e+00 |
| poisson w=50 float64 | AdamW→L-BFGS | 8.34e-03 | 4.2e+01 | 15529 | 0 | 7200 | 1.3e+00 | 1.8e+01 |
| poisson w=50 float64 | Muon→L-BFGS | 8.24e-03 | 4.2e+01 | 14433 | 0 | 7280 | 1.8e+01 | 1.9e+01 |
| poisson w=50 float64 | PLM [rec] | 8.68e-02 | 1.1e+03 | 2127 | 692 | 692 | 1.2e+00 | 7.3e+00 |
