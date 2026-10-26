# ADR-0008: PSI-based drift monitoring

**Status:** Accepted

## Context
A model decays as the world shifts; a model that is never watched is a liability.
We need to detect **input drift** (the live population diverging from training),
**prediction drift** (the score distribution shifting), and **performance drift**
(accuracy dropping, when labels are available). A monitoring framework (e.g.
Evidently) does this but adds a heavy dependency and still needs a reference captured
somewhere.

## Decision
Measure drift with the **Population Stability Index (PSI)** — the standard in churn
and credit scoring — computed per feature against a **reference profile** captured at
evaluation time and committed as a small JSON (`reports/reference_profile.json`).
Bands come from config: `PSI < 0.10` stable, `0.10–0.25` moderate, `> 0.25`
significant. Prediction drift is PSI over the score distribution; performance drift
compares live ROC-AUC to the stored baseline when labels are supplied. It is a few
lines of NumPy, exposed via `POST /monitor` and `python -m churnguard.monitor`.

## Consequences
- Monitoring works the instant someone clones the repo — no server to run.
- PSI is industry-standard, explainable, and threshold-driven.
- The reference is a point-in-time snapshot; regenerating it (via `make evaluate`)
  deliberately moves the goalposts and should be a conscious act.
- PSI flags *that* a distribution moved, not *why* — triage still needs a human
  (see the runbook). KS/JS tests could be added later for extra rigor.
