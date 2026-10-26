---
description: "Fix a bug test-first: reproduce with a failing test, then make it pass, then verify."
argument-hint: "<description of the bug, with repro if known>"
---

Fix this bug **test-first**, following `CLAUDE.md`. The bug:

$ARGUMENTS

Steps — do not skip the failing test:

1. **Reproduce.** Have **qa-engineer** write a **failing** test that captures the bug
   (the minimal input → wrong/crash behaviour). Run it and show it fails for the
   right reason. If you cannot reproduce it, say so and ask for more detail — do not
   guess a fix.
2. **Diagnose.** The owning builder finds the root cause and states it in one line
   (`file:line` + why). No shotgun changes.
3. **Fix.** Make the smallest change that turns the new test green. Run **only the
   affected tests** during iteration (`CLAUDE.md` §9).
4. **Guard against regressions.** Keep the new test; run the full suite once, plus
   `ruff check` and `mypy churnguard`.
5. **Review.** If the fix touches security-sensitive or contract surface, launch
   **code-reviewer** (and **security-reviewer** if relevant), read-only, in parallel.
6. **Commit.** On approval: `git add -A` then
   `git commit -m "fix(pNN): <summary>"` (use the current phase's `NN`). Never push,
   rebase, reset --hard or amend; no co-author/trailer lines.

If the "bug" is actually undefined behaviour (not covered by the spec), log a
`SPEC-GAP` and raise it rather than inventing the intended behaviour.
