# ADR-0004: Isotonic probability calibration

**Status:** Accepted

## Context
Downstream use of ChurnGuard is **probabilistic**: ranking customers by risk, making
expected-value retention decisions, and choosing a cost-based threshold (ADR-0005).
All of that assumes a predicted score of 0.7 means roughly a 70% chance of churn.
Many classifiers — especially with `class_weight="balanced"` — produce well-ranked
but poorly *calibrated* scores, so the raw number is misleading.

## Decision
Wrap the tuned model in `CalibratedClassifierCV` with **isotonic** regression,
fitted by cross-validation on the training data. Verify calibration explicitly with
the **Brier score** and a reliability curve (`reports/figures/calibration_curve.png`),
both produced during evaluation.

## Consequences
- Predicted probabilities are meaningful, so cost-based decisions are sound.
- Brier score ≈ 0.13 on the test split; the reliability curve tracks the diagonal.
- The calibrated model is the serving model; the *uncalibrated* base pipeline is
  kept in the bundle for SHAP (the explainer needs the raw estimator — ADR-0006).
- Isotonic is flexible but needs enough data; sigmoid is available in config for
  smaller datasets.
