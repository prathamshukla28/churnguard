# Model card — ChurnGuard

**Date:** 2026-10-26
**Model:** logistic_regression (tuned=True, calibrated=True)
**Task:** Binary classification — will a customer churn?

## Intended use
Rank a telecom customer base by churn risk so a retention team can prioritise
outreach. Not for automated, consequential decisions without a human in the loop.

## Data
Telco-style customer dataset (7043 rows). See
`churnguard/data.py`; run `make data-real` to train on the genuine Kaggle Telco CSV.

## Metrics (held-out test split, n = 1409)
| Metric | Value |
| --- | --- |
| ROC-AUC | 0.8144 |
| PR-AUC | 0.5376 |
| Brier score (calibration, lower=better) | 0.1291 |
| Precision @ threshold | 0.4104 |
| Recall @ threshold | 0.7893 |
| F1 @ threshold | 0.54 |

## Decision threshold
Chosen at **0.2** (not the default 0.5) to minimise expected
business cost with cost_fn=5, cost_fp=1. This cut expected cost
from 1047 (at 0.5) to 654.

## Top churn drivers
- `tenure` (importance 0.1814)
- `Contract` (importance 0.0565)
- `PaymentMethod` (importance 0.014)
- `MonthlyCharges` (importance 0.0109)
- `InternetService` (importance 0.0077)

## Limitations
- Trained on a single snapshot; no concept-drift monitoring.
- Threshold reflects the configured cost ratio; retune if the business changes.
