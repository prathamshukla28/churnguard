# ADR-0003: Model selection by cross-validation, then tuning

**Status:** Accepted

## Context
Picking a model by training a few and eyeballing a single train/test split is noisy
and invites overfitting to that split. "Use the fanciest model" is a common reflex
that is often wrong on medium-sized tabular data, where a well-regularised linear
model can match or beat tree ensembles while being faster and more interpretable.

## Decision
Compare the candidate estimators (logistic regression, random forest, gradient
boosting) by **stratified 5-fold cross-validated ROC-AUC** on the training split,
then apply the **one-standard-error rule** (`model.selection_rule:
one_standard_error`): take the best mean score, subtract its standard error across
folds, and pick the *simplest* candidate (config order: logistic regression, random
forest, gradient boosting) whose mean clears that floor. Setting `selection_rule:
best` instead picks the highest mean outright. Then run a **randomized
hyper-parameter search** on the selected model only (cheaper than grid search, good
coverage). All candidates and CV settings live in `config.yaml`, so the comparison
is declarative and reproducible under one seed.

## Consequences
- The selected model is justified by a stable estimate, not one lucky split.
- On the real Telco data logistic regression does **not** have the best raw CV
  score. Untuned 5-fold ROC-AUC on the training split (seed 42):

  | Candidate | Mean | SD across folds | Standard error |
  | --- | --- | --- | --- |
  | Random forest | **0.8480** | 0.0106 | 0.0047 |
  | Gradient boosting | 0.8479 | 0.0126 | 0.0056 |
  | Logistic regression | 0.8456 | 0.0138 | 0.0062 |

  The forest is best by 0.0024. The 1-SE floor is 0.8480 − 0.0047 = 0.8433, and
  logistic regression (0.8456) clears it, so the rule selects it as the simplest
  candidate. Under `selection_rule: best` the forest would be chosen. After tuning,
  logistic regression's CV ROC-AUC is 0.8457 (`cv_roc_auc` in
  `reports/metrics.json` records this tuned score for the selected model and the
  untuned scores for the others).
- The trade is a CV difference well inside fold-to-fold noise in exchange for
  exact SHAP values, clean calibration (see ADR-0004, ADR-0006) and much cheaper
  serving.
- Tuning only the selected model keeps training fast enough for CI.
- The comparison is re-runnable and logged (MLflow, best-effort).
