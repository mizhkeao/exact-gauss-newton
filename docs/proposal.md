# Investigation of the Muon Optimizer for Training PINNs

**Paper:** Rathore et al., "Challenges in Training PINNs: A Loss Landscape Perspective" (ICML 2024).

**Contribution:** Extend their optimizer study to test Muon against Adam and Adam followed by L-BFGS.

**Question:** Does Muon reach accurate solutions faster, and provide a better starting point for L-BFGS?

## Implementation
Train PINNs with four strategies: Adam, Muon, Adam → L-BFGS, Muon → L-BFGS.
Muon updates hidden-layer weight matrices; Adam updates biases and the input/output layers.
Keep the same network architecture, loss definitions, training points, and paired
initializations across methods, with comparable hyperparameter-tuning budgets.

## Dataset and benchmark
Convection, reaction, and wave problems from Rathore et al.'s public implementation.
Training uses equation residuals and initial/boundary conditions; reference solutions
provide independent evaluation.

## Metrics
- Relative solution error vs. elapsed training time
- Time to a preset accuracy
- Variability across five random seeds

Goal: determine whether Muon improves early training, final accuracy, or both, and whether
its benefits persist when followed by L-BFGS.
