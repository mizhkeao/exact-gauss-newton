# H3: 24 runs

## Final relative L2 error, median (min–max) over seeds

| setting | AdamW | AdamW→L-BFGS | Muon→L-BFGS | PLM [rec2_lag4] |
|---|---|---|---|---|
| convection beta=30 causal=1 w=50 float64 | 1.3e+00 (1e+00–1e+00) | 1.2e+00 (1e+00–2e+00) | 1.5e+00 (1e+00–2e+00) | 1.2e+00 (8e-01–2e+00) |
| convection beta=40 causal=1 w=50 float64 | 1.5e+00 (1e+00–1e+00) | 1.5e+00 (1e+00–2e+00) | 1.6e+00 (1e+00–2e+00) | 1.6e+00 (1e+00–2e+00) |

## Cost

| setting | series | s/iter | peak MB | grad evals | jac evals | iters | t→1e-3 | t→1e-5 |
|---|---|---|---|---|---|---|---|---|
| convection beta=30 causal=1 w=50 float64 | AdamW | 3.84e-03 | 4.7e+01 | 78199 | 0 | 78199 | — | — |
| convection beta=30 causal=1 w=50 float64 | AdamW→L-BFGS | 1.03e-02 | 5.2e+01 | 58015 | 0 | 29120 | — | — |
| convection beta=30 causal=1 w=50 float64 | Muon→L-BFGS | 1.05e-02 | 5.2e+01 | 55798 | 0 | 28600 | — | — |
| convection beta=30 causal=1 w=50 float64 | PLM [rec2_lag4] | 3.65e-01 | 9.0e+03 | 11233 | 216 | 823 | — | — |
| convection beta=40 causal=1 w=50 float64 | AdamW | 3.76e-03 | 4.7e+01 | 79760 | 0 | 79760 | — | — |
| convection beta=40 causal=1 w=50 float64 | AdamW→L-BFGS | 1.10e-02 | 5.2e+01 | 57796 | 0 | 27400 | — | — |
| convection beta=40 causal=1 w=50 float64 | Muon→L-BFGS | 1.02e-02 | 5.2e+01 | 58978 | 0 | 29460 | — | — |
| convection beta=40 causal=1 w=50 float64 | PLM [rec2_lag4] | 3.41e-01 | 9.0e+03 | 7756 | 226 | 881 | — | — |
