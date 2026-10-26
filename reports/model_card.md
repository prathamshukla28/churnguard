# Model card — ChurnGuard

**Date:** 2026-10-26
**Model:** logistic_regression
**Task:** Binary classification — will a customer churn?

## Intended use
Rank a telecom customer base by churn risk so a retention team can prioritise
outreach. Not intended for automated, consequential decisions without a
human in the loop.

## Data
Telco-style customer dataset (7043 rows). See
`churnguard/data.py` for the schema and the (synthetic) generating process;
swap in the real Kaggle Telco CSV at `data/churn.csv` to retrain on genuine data.

## Metrics (held-out test split, n = 1409)
| Metric | Value |
| --- | --- |
| ROC-AUC | 0.8144 |
| PR-AUC | 0.541 |
| Precision | 0.413 |
| Recall | 0.7625 |
| F1 | 0.5358 |

Baseline churn rate in the test split: 0.2122.

## Top churn drivers
- `tenure` (importance 0.1994)
- `Contract` (importance 0.059)
- `PaymentMethod` (importance 0.0151)
- `InternetService` (importance 0.0114)
- `TechSupport` (importance 0.006)

## Limitations
- Trained on a single snapshot; no concept-drift monitoring.
- Decision threshold fixed at 0.5; tune it to the business
  cost of false positives vs. false negatives before production use.
