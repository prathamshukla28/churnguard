# ADR-0001: Record architecture decisions

**Status:** Accepted

## Context
ChurnGuard makes several non-obvious engineering choices (how probabilities are
calibrated, how the decision threshold is picked, why the registry is a flat file).
Six months later — or to a new teammate or an interviewer — the code shows *what*
was done but not *why*, and the reasoning is the expensive part to reconstruct.

## Decision
Keep lightweight Architecture Decision Records in `docs/adr/`, one Markdown file
per significant decision, using Nygard's Context / Decision / Consequences format.
ADRs are immutable once accepted; to change course, add a new ADR that supersedes
the old one rather than editing history.

## Consequences
- The rationale behind each choice is versioned alongside the code it explains.
- A small, ongoing discipline: a meaningful decision means writing an ADR.
- Reviewers and interviewers can see judgement, not just implementation.
