# B: 81 runs

## Final relative L2 error, median (min–max) over seeds

| setting | PLM [all] | PLM [base] | PLM [base_mu1e-2] | PLM [dense] | PLM [marquardt] | PLM [trust] | PLM [warm_adamw] | PLM [warm_muon] | PLM [weighted] |
|---|---|---|---|---|---|---|---|---|---|
| beam w=50 float64 | 8.8e-08 (3e-08–4e-07) | 3.5e-06 (2e-06–3e-05) | 1.4e-06 (1e-06–2e-06) | 5.0e-05 (6e-06–5e-04) | 2.3e-07 (1e-07–3e-07) | 3.1e-06 (1e-06–8e-06) | 1.2e-06 (1e-06–3e-06) | 1.9e-07 (4e-08–1e-06) | 2.4e-07 (2e-07–2e-06) |
| multiscale k=16 w=50 float64 | 1.4e-01 (2e-02–2e+00) | 3.4e-07 (2e-07–7e-07) | 8.1e-07 (3e-07–8e-07) | 2.3e-05 (1e-05–6e-05) | 5.2e+00 (4e+00–6e+00) | 6.9e-07 (4e-07–9e-07) | 6.1e-08 (2e-08–3e-07) | 3.5e-07 (3e-07–2e-06) | 6.4e-07 (3e-07–1e-06) |
| poisson w=50 float64 | 4.8e-09 (4e-09–1e-08) | 4.8e-08 (4e-08–6e-08) | 2.1e-08 (9e-09–3e-08) | 9.9e-07 (8e-07–4e-05) | 1.7e-09 (1e-10–3e-09) | 4.0e-08 (2e-08–2e-07) | 2.1e-08 (9e-09–2e-07) | 3.8e-08 (4e-08–7e-08) | 8.9e-08 (2e-08–1e-07) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| beam w=50 float64 | PLM [all] | 3.99e-01 | 5.0e+03 | 564 | 151 | 151 | 7.6e+00 | 1.2e+01 |
| beam w=50 float64 | PLM [base] | 3.58e-01 | 5.0e+03 | 0 | 168 | 168 | 2.0e+01 | 3.3e+01 |
| beam w=50 float64 | PLM [base_mu1e-2] | 3.58e-01 | 5.0e+03 | 0 | 168 | 168 | 1.5e+01 | 2.2e+01 |
| beam w=50 float64 | PLM [dense] | 5.78e-01 | 5.0e+03 | 0 | 104 | 104 | 3.3e+01 | 3.5e+01 |
| beam w=50 float64 | PLM [marquardt] | 3.59e-01 | 5.0e+03 | 0 | 168 | 168 | 1.4e+01 | 2.5e+01 |
| beam w=50 float64 | PLM [trust] | 3.60e-01 | 5.0e+03 | 0 | 167 | 167 | 2.3e+01 | 5.2e+01 |
| beam w=50 float64 | PLM [warm_adamw] | 4.04e-01 | 5.0e+03 | 502 | 149 | 149 | 1.5e+01 | 4.0e+01 |
| beam w=50 float64 | PLM [warm_muon] | 3.96e-01 | 5.0e+03 | 486 | 152 | 152 | 7.3e+00 | 2.0e+01 |
| beam w=50 float64 | PLM [weighted] | 3.60e-01 | 5.0e+03 | 0 | 167 | 167 | 6.9e+00 | 1.7e+01 |
| multiscale k=16 w=50 float64 | PLM [all] | 8.77e-02 | 1.1e+03 | 1967 | 685 | 685 | — | — |
| multiscale k=16 w=50 float64 | PLM [base] | 7.73e-02 | 1.1e+03 | 0 | 776 | 776 | 1.2e+01 | 2.0e+01 |
| multiscale k=16 w=50 float64 | PLM [base_mu1e-2] | 7.70e-02 | 1.1e+03 | 0 | 780 | 780 | 1.1e+01 | 2.3e+01 |
| multiscale k=16 w=50 float64 | PLM [dense] | 2.85e-01 | 1.1e+03 | 0 | 211 | 211 | 4.5e+01 | 5.7e+01 |
| multiscale k=16 w=50 float64 | PLM [marquardt] | 7.99e-02 | 1.1e+03 | 0 | 751 | 751 | — | — |
| multiscale k=16 w=50 float64 | PLM [trust] | 7.80e-02 | 1.1e+03 | 0 | 770 | 770 | 1.5e+01 | 2.7e+01 |
| multiscale k=16 w=50 float64 | PLM [warm_adamw] | 8.54e-02 | 1.1e+03 | 2060 | 703 | 703 | 1.0e+01 | 1.7e+01 |
| multiscale k=16 w=50 float64 | PLM [warm_muon] | 8.60e-02 | 1.1e+03 | 1698 | 698 | 698 | 1.0e+01 | 2.3e+01 |
| multiscale k=16 w=50 float64 | PLM [weighted] | 7.74e-02 | 1.1e+03 | 0 | 775 | 775 | 9.9e+00 | 1.3e+01 |
| poisson w=50 float64 | PLM [all] | 1.97e-01 | 1.1e+03 | 2127 | 304 | 304 | 1.3e+00 | 8.2e+00 |
| poisson w=50 float64 | PLM [base] | 7.74e-02 | 1.1e+03 | 0 | 775 | 775 | 1.5e+00 | 4.2e+00 |
| poisson w=50 float64 | PLM [base_mu1e-2] | 7.73e-02 | 1.1e+03 | 0 | 776 | 776 | 1.1e+00 | 3.0e+00 |
| poisson w=50 float64 | PLM [dense] | 2.67e-01 | 1.1e+03 | 0 | 225 | 225 | 3.0e+00 | 1.4e+01 |
| poisson w=50 float64 | PLM [marquardt] | 7.72e-02 | 1.1e+03 | 0 | 778 | 778 | 1.6e+00 | 2.2e+00 |
| poisson w=50 float64 | PLM [trust] | 7.76e-02 | 1.1e+03 | 0 | 773 | 773 | 1.8e+00 | 4.6e+00 |
| poisson w=50 float64 | PLM [warm_adamw] | 8.62e-02 | 1.1e+03 | 1778 | 696 | 696 | 1.3e+00 | 8.1e+00 |
| poisson w=50 float64 | PLM [warm_muon] | 1.93e-01 | 1.1e+03 | 1717 | 312 | 312 | 6.6e+00 | 9.7e+00 |
| poisson w=50 float64 | PLM [weighted] | 8.98e-02 | 1.1e+03 | 0 | 669 | 669 | 1.6e+00 | 4.2e+00 |
