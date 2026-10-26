# ADR-0007: File-based model registry

**Status:** Accepted

## Context
Operating a model needs provenance and a lifecycle: what versions exist, which is in
production, what metrics each achieved, and a one-command rollback. A hosted MLflow
Model Registry provides this but requires a tracking server and backing store to
stand up, run, and secure — heavy for a single-service project and awkward to test.

## Decision
Implement a **file-based registry** (`churnguard/registry.py`): each registered model
is an immutable, versioned directory (`models/registry/v<N>/`) with its bundle and
metrics, tracked in a `registry.json` manifest carrying the lifecycle **stage**
(`staging` → `production` → `archived`). Promotion to production atomically archives
the incumbent. The champion/challenger decision (`evaluate_promotion`) is shared by
the retraining job and documented for a CI promotion gate.

## Consequences
- Full provenance and one-call rollback with zero infrastructure.
- Trivially unit-testable (point it at a temp dir) and diff-able.
- The registry is local runtime state (git-ignored), not a shared service — fine for
  one service; a team with many models would graduate to a hosted registry.
- MLflow is still used for *experiment* tracking (best-effort), a complementary concern.
