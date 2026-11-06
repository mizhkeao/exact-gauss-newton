# N: 54 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW→L-BFGS | Muon→L-BFGS | PLM [rec2] |
|---|---|---|---|
| multiscale k=16 pde_noise=0.001 w=50 float64 | 1.4e-03 (6e-04–4e-03) | 2.4e-03 (7e-04–1e-02) | 5.4e-06 (2e-06–7e-06) |
| multiscale k=16 pde_noise=0.01 w=50 float64 | 2.9e-03 (1e-03–4e-03) | 1.4e-03 (5e-04–2e-03) | 5.4e-05 (2e-05–7e-05) |
| multiscale k=16 pde_noise=0.1 w=50 float64 | 7.1e-04 (5e-04–1e-03) | 2.5e-03 (1e-03–6e-03) | 5.4e-04 (2e-04–7e-04) |
| poisson pde_noise=0.001 w=50 float64 | 6.5e-06 (4e-06–7e-06) | 6.0e-06 (4e-06–7e-06) | 5.4e-06 (2e-06–7e-06) |
| poisson pde_noise=0.01 w=50 float64 | 5.5e-05 (2e-05–7e-05) | 5.4e-05 (2e-05–7e-05) | 5.3e-05 (2e-05–7e-05) |
| poisson pde_noise=0.1 w=50 float64 | 5.4e-04 (2e-04–7e-04) | 5.4e-04 (2e-04–7e-04) | 5.4e-04 (2e-04–7e-04) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| multiscale k=16 pde_noise=0.001 w=50 float64 | AdamW→L-BFGS | 8.44e-03 | 4.2e+01 | 14445 | 0 | 7120 | 2.8e+01 | — |
| multiscale k=16 pde_noise=0.001 w=50 float64 | Muon→L-BFGS | 8.44e-03 | 4.2e+01 | 13180 | 0 | 7120 | 3.8e+01 | — |
| multiscale k=16 pde_noise=0.001 w=50 float64 | PLM [rec2] | 8.81e-02 | 1.1e+03 | 2053 | 682 | 682 | 8.3e+00 | 1.4e+01 |
| multiscale k=16 pde_noise=0.01 w=50 float64 | AdamW→L-BFGS | 8.47e-03 | 4.2e+01 | 14267 | 0 | 7100 | 3.0e+01 | — |
| multiscale k=16 pde_noise=0.01 w=50 float64 | Muon→L-BFGS | 8.33e-03 | 4.2e+01 | 13376 | 0 | 7200 | 2.9e+01 | — |
| multiscale k=16 pde_noise=0.01 w=50 float64 | PLM [rec2] | 8.74e-02 | 1.1e+03 | 2071 | 687 | 687 | 1.1e+01 | — |
| multiscale k=16 pde_noise=0.1 w=50 float64 | AdamW→L-BFGS | 8.21e-03 | 4.2e+01 | 14590 | 0 | 7320 | 3.1e+01 | — |
| multiscale k=16 pde_noise=0.1 w=50 float64 | Muon→L-BFGS | 8.41e-03 | 4.2e+01 | 13259 | 0 | 7140 | 3.1e+01 | — |
| multiscale k=16 pde_noise=0.1 w=50 float64 | PLM [rec2] | 8.79e-02 | 1.1e+03 | 1964 | 683 | 683 | 8.8e+00 | — |
| poisson pde_noise=0.001 w=50 float64 | AdamW→L-BFGS | 8.47e-03 | 4.2e+01 | 15498 | 0 | 7100 | 1.4e+00 | 1.8e+01 |
| poisson pde_noise=0.001 w=50 float64 | Muon→L-BFGS | 7.42e-03 | 4.2e+01 | 14403 | 0 | 8100 | 1.8e+01 | 1.9e+01 |
| poisson pde_noise=0.001 w=50 float64 | PLM [rec2] | 8.74e-02 | 1.1e+03 | 1837 | 687 | 687 | 1.5e+00 | 7.7e+00 |
| poisson pde_noise=0.01 w=50 float64 | AdamW→L-BFGS | 8.37e-03 | 4.2e+01 | 15437 | 0 | 7180 | 1.4e+00 | — |
| poisson pde_noise=0.01 w=50 float64 | Muon→L-BFGS | 8.44e-03 | 4.2e+01 | 13844 | 0 | 7120 | 1.8e+01 | — |
| poisson pde_noise=0.01 w=50 float64 | PLM [rec2] | 8.67e-02 | 1.1e+03 | 1918 | 692 | 692 | 1.5e+00 | — |
| poisson pde_noise=0.1 w=50 float64 | AdamW→L-BFGS | 8.65e-03 | 4.2e+01 | 14678 | 0 | 6940 | 1.2e+00 | — |
| poisson pde_noise=0.1 w=50 float64 | Muon→L-BFGS | 9.85e-03 | 4.2e+01 | 11632 | 0 | 6100 | 1.8e+01 | — |
| poisson pde_noise=0.1 w=50 float64 | PLM [rec2] | 8.80e-02 | 1.1e+03 | 1915 | 682 | 682 | 1.3e+00 | — |
