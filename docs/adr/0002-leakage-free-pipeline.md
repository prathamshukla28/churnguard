# ADR-0002: Leak-free scikit-learn Pipeline

**Status:** Accepted

## Context
The most common and most damaging bug in tabular ML is **data leakage**: fitting a
preprocessing step (scaler means, encoder categories, imputation statistics) on the
whole dataset — including the rows later used for validation or test — inflates
offline metrics and silently fails in production. It is also easy to serve inputs
shaped differently from training ("train/serve skew").

## Decision
Put *all* preprocessing inside a single scikit-learn `Pipeline`
(`ColumnTransformer` → estimator) and cross-validate the Pipeline **as one unit**,
so every transform is fitted only on each training fold. Do feature engineering in
one pure function, `engineer_features()`, called identically at training and serving
time. The serialized `ModelBundle` stores the exact input feature schema; serving
reindexes incoming records to it with `OneHotEncoder(handle_unknown="ignore")`.

## Consequences
- Offline metrics are trustworthy — no optimistic leakage.
- Train/serve parity is structural, not a convention someone must remember.
- Slightly less flexibility than ad-hoc preprocessing, which is the point.
- Unseen categories at serving time degrade gracefully instead of crashing.
