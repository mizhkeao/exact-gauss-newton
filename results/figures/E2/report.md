# E2: 24 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW→L-BFGS | Muon→L-BFGS | PLM [rec_lag4] |
|---|---|---|---|
| convection beta=1 w=50 float64 | 6.6e-04 (4e-04–9e-04) | 1.8e-02 (1e-02–2e-02) | 3.0e-06 (2e-06–4e-06) |
| convection beta=10 w=50 float64 | 7.7e-04 (6e-04–1e-03) | 5.5e-02 (3e-03–1e-01) | 2.5e-05 (2e-05–3e-05) |
| convection beta=30 w=50 float64 | 1.8e+00 (1e+00–2e+00) | 1.6e+00 (1e+00–2e+00) | 1.8e+00 (2e+00–2e+00) |
| convection beta=40 w=50 float64 | 1.5e+00 (1e+00–1e+00) | 1.3e+00 (1e+00–1e+00) | 1.4e+00 (1e+00–1e+00) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| convection beta=1 w=50 float64 | AdamW→L-BFGS | 1.06e-02 | 5.2e+01 | 59984 | 0 | 28200 | 9.3e+01 | — |
| convection beta=1 w=50 float64 | Muon→L-BFGS | 9.47e-03 | 5.2e+01 | 62102 | 0 | 31700 | — | — |
| convection beta=1 w=50 float64 | PLM [rec_lag4] | 3.41e-01 | 9.0e+03 | 7786 | 227 | 882 | 3.4e+01 | 2.0e+02 |
| convection beta=10 w=50 float64 | AdamW→L-BFGS | 9.40e-03 | 5.2e+01 | 65084 | 0 | 31940 | 9.8e+01 | — |
| convection beta=10 w=50 float64 | Muon→L-BFGS | 9.50e-03 | 5.2e+01 | 60736 | 0 | 31630 | — | — |
| convection beta=10 w=50 float64 | PLM [rec_lag4] | 3.44e-01 | 9.0e+03 | 8092 | 226 | 874 | 4.5e+01 | — |
| convection beta=30 w=50 float64 | AdamW→L-BFGS | 9.28e-03 | 5.2e+01 | 65916 | 0 | 32330 | — | — |
| convection beta=30 w=50 float64 | Muon→L-BFGS | 1.01e-02 | 5.2e+01 | 58451 | 0 | 29950 | — | — |
| convection beta=30 w=50 float64 | PLM [rec_lag4] | 3.45e-01 | 9.0e+03 | 8408 | 228 | 872 | — | — |
| convection beta=40 w=50 float64 | AdamW→L-BFGS | 9.52e-03 | 5.2e+01 | 64250 | 0 | 31530 | — | — |
| convection beta=40 w=50 float64 | Muon→L-BFGS | 9.07e-03 | 5.2e+01 | 63307 | 0 | 33090 | — | — |
| convection beta=40 w=50 float64 | PLM [rec_lag4] | 3.46e-01 | 9.0e+03 | 7554 | 224 | 868 | — | — |
