---
description: "Build phase N end-to-end: plan → tests-first → build → parallel review → fix → verify → human gate → commit."
argument-hint: "N  (e.g. /build-phase 3)"
---

Build **phase $1** of ChurnGuard, strictly following the phase protocol in
`CLAUDE.md` §5 and the phase map in §5. Use the right owner agent for this phase
(data-engineer / backend-ml-engineer / devops-mlops / architect).

Execute the loop in order — do not skip or reorder:

1. **Plan.** Restate phase $1's scope and Definition of Done (`CLAUDE.md` §6). List
   the files you expect to touch. Surface any SPEC-GAPs now with conservative
   defaults. Output the plan and stop if the scope is unclear.
2. **Tests first.** Have **qa-engineer** write failing tests for phase $1's behaviour
   from the DoD and the API/ADR contracts. Confirm they fail for the right reason.
3. **Build.** The owning builder implements until the phase tests pass, with
   `ruff check` and `mypy churnguard` clean. During iteration, run **only the
   affected tests** (`CLAUDE.md` §9).
4. **Review (parallel).** Launch **code-reviewer** and **security-reviewer**
   together in one batch (and **ux-reviewer** if phase $1 touches the demo/API).
   Each returns severity-ranked findings; they are read-only.
5. **Fix.** The builder resolves findings; re-run only the affected tests.
6. **Verify.** Launch the **verifier** (fresh, read-only) to re-run the full suite +
   lint + types + regression gate and confirm the DoD, reporting only what it ran.
7. **Human gate.** Present: plan result, diff summary, verifier report, and all
   collected SPEC-GAPs. **Wait for explicit approval.**
8. **Commit.** On approval only: `git add -A` then
   `git commit -m "<type>(p$1): <summary>"` (choose the right `<type>`), then
   `git tag p$1-verified`. **Never** push, rebase, reset --hard or amend; no
   co-author/trailer lines.
