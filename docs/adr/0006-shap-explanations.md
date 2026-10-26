# ADR-0006: Per-prediction SHAP explanations

**Status:** Accepted

## Context
A retention agent acting on a score needs to know *why* a customer is flagged, and
stakeholders will not trust a black box. Global feature importance is not enough —
the question is per-customer ("why is *this* account high risk?"). Explanations must
speak in business terms (`Contract`), not transformed columns (`cat__Contract_Month-to-month`).

## Decision
Return the top-k **SHAP** risk drivers with each prediction. For the linear serving
model, SHAP values have a closed form — `coef_j · (x_j − E[x_j])` — computed directly
with no approximation; for tree models, fall back to `shap.TreeExplainer`. A SHAP
**background sample** (transformed) is stored in the bundle. Contributions in the
transformed space are summed back to their source columns so the output is
business-readable, with a direction ("increases/decreases risk").

## Consequences
- Every `/predict` with `explain=true` (and the demo page) shows actionable reasons.
- The bundle keeps both the calibrated serving model *and* the uncalibrated base
  pipeline, because the explainer needs the raw estimator and preprocessor.
- Linear closed-form is exact and fast; tree explanations cost more compute.
- Explanation values are offsets from a background mean, not probabilities — the UI
  presents them as relative drivers, not standalone percentages.
