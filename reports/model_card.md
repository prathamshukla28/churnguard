# Model card — ChurnGuard

**Date:** 2026-11-09
**Model:** logistic_regression (tuned=True, calibrated=True)
**Task:** Binary classification — will a customer churn?

## Intended use
Rank a telecom customer base by churn risk so a retention team can prioritise
outreach. Not for automated, consequential decisions without a human in the loop.

## Data
Public IBM *Telco Customer Churn* sample dataset (7,043 customers, 21 columns; IBM describes it as a fictional telco). Downloaded from `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`, SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.

Split: stratified 80% train / 20% test (seed 42). Model selection, tuning,
calibration and the decision threshold use the training split only; the test split
below is scored exactly once.

## Metrics (held-out test split, n = 1409, churn rate 26.5%)
| Metric | Value |
| --- | --- |
| ROC-AUC | 0.8402 (95% bootstrap CI 0.8168–0.8619) |
| PR-AUC | 0.6253 (95% CI 0.5714–0.6761; base rate 0.2654) |
| Brier score (lower = better) | 0.1389 (base rate alone: 0.195) |
| Precision @ 0.16 | 0.4462 |
| Recall @ 0.16 | 0.9091 |
| F1 @ 0.16 | 0.5986 |
| Share of customers flagged | 54.1% |

## Decision threshold
**0.16**, chosen to minimise expected cost with cost_fn=5,
cost_fp=1 on out-of-fold predictions on the training split. On the untouched test split it
gives a cost of 592 vs. 1014 at
the default 0.5 (42% lower). The saving is only as real as the 5:1 cost
ratio, which is an assumption, not a measured business number.

## Where it is weaker (test split, by segment)
| Segment | n | Churn rate | ROC-AUC | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| Contract = Month-to-month | 773 | 0.43 | 0.739 | 0.47 | 0.97 |
| Contract = Two year | 336 | 0.03 | 0.768 | 0.00 | 0.00 |
| Contract = One year | 300 | 0.12 | 0.743 | 0.25 | 0.56 |

| Segment | n | Churn rate | ROC-AUC | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| tenure_band = 49+ mo | 450 | 0.08 | 0.830 | 0.27 | 0.60 |
| tenure_band = 0-12 mo | 449 | 0.48 | 0.744 | 0.53 | 0.97 |
| tenure_band = 25-48 mo | 304 | 0.21 | 0.795 | 0.36 | 0.80 |
| tenure_band = 13-24 mo | 206 | 0.29 | 0.813 | 0.40 | 0.98 |

| Segment | n | Churn rate | ROC-AUC | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| InternetService = Fiber optic | 613 | 0.41 | 0.778 | 0.49 | 0.95 |
| InternetService = DSL | 484 | 0.20 | 0.813 | 0.38 | 0.85 |
| InternetService = No | 312 | 0.08 | 0.891 | 0.30 | 0.76 |

Within-segment ROC-AUC is usually below the overall figure because contract type
and tenure, the strongest signals, partly define the segments. Watch the low-churn
segments: with few churners there, the threshold flags few of them (low recall).

## Top drivers (permutation importance on the test split)
- `tenure` (drop in test ROC-AUC when shuffled: 0.1615)
- `MonthlyCharges` (drop in test ROC-AUC when shuffled: 0.1422)
- `InternetService` (drop in test ROC-AUC when shuffled: 0.137)
- `Contract` (drop in test ROC-AUC when shuffled: 0.0394)
- `num_addon_services` (drop in test ROC-AUC when shuffled: 0.0293)

These are associations the model uses, not causal effects.

## Explanations
Per-prediction SHAP reasons explain the **uncalibrated base estimator** in its raw
output space (log-odds for linear and gradient-boosted models). The probability
returned by the API comes from the **calibrated** wrapper. Calibration is
monotone per fold, so the direction and ranking of reasons is informative, but
SHAP impacts do not sum to the calibrated probability.

## Monitoring
`churnguard/monitor.py` (CLI and `POST /monitor`) computes PSI data drift,
prediction drift and, when labels are supplied, performance drift against the
reference profile captured at evaluation time. It runs on demand: nothing
schedules it or pages anyone yet.

## Limitations
- One static public snapshot; no timestamps, so no out-of-time validation. Real
  churn data shifts with pricing and competition.
- The threshold depends on an assumed 5:1 cost ratio; retune when real costs are known.
- Drift checks are on-demand only; wire them to a schedule and an alert before relying on them.
- No fairness audit beyond the per-segment table (e.g. `SeniorCitizen`, `gender` are inputs).
