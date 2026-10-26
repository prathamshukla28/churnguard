---
description: "Independently verify a phase against its Definition of Done (read-only, fresh context)."
argument-hint: "N  (e.g. /verify-phase 3)"
---

Independently verify **phase $1** of ChurnGuard against its Definition of Done
(`CLAUDE.md` §6). Delegate to the **verifier** agent (read-only, fresh context — it
must not have built the code).

The verifier must, itself, run and report the **real output** of:

- the full test suite (`pytest -q`, with `CHURNGUARD_TRACKING=0`),
- `ruff check churnguard tests scripts`,
- `mypy churnguard`,
- the regression gate (`python scripts/check_regression.py`),
- any phase-specific behaviour check from the API/ADR contract.

Then produce a **DoD checklist**: each item pass/fail with the exact command and its
real output as evidence. Finish with a verdict: **VERIFIED** or **NOT VERIFIED**
(list blockers). A box that cannot be proven by a command actually run is a fail,
never an assumption.

Do not edit anything. If VERIFIED, report that phase $1 is ready for the human gate
and commit/tag step; if NOT VERIFIED, hand the blocking items back to the builder.
