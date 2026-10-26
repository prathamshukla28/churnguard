# ADR-0003: Model selection by cross-validation, then tuning

**Status:** Accepted

## Context
Picking a model by training a few and eyeballing a single train/test split is noisy
and invites overfitting to that split. "Use the fanciest model" is a common reflex
that is often wrong on medium-sized tabular data, where a well-regularised linear
model can match or beat tree ensembles while being faster and more interpretable.

## Decision
Compare the candidate estimators (logistic regression, random forest, gradient
boosting) by **stratified k-fold cross-validated ROC-AUC** and let the data choose
the winner. Then run a **randomized hyper-parameter search** on the winner only
(cheaper than grid search, good coverage). All candidates and CV settings live in
`config.yaml`, so the comparison is declarative and reproducible under one seed.

## Consequences
- The selected model is justified by a stable estimate, not one lucky split.
- On this dataset logistic regression wins — and brings interpretability and clean
  calibration (see ADR-0004, ADR-0006) as a bonus.
- Tuning only the winner keeps training fast enough for CI.
- The comparison is re-runnable and logged (MLflow, best-effort).
