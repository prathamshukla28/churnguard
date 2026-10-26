---
description: "Report project status: phase progress, test/lint/type/gate state, open SPEC-GAPs, and the next action."
argument-hint: "(no args)"
---

Report ChurnGuard's current status concisely. Gather and show **only real,
just-run** results — do not estimate.

1. **Phase progress.** Which phases (P00–P10, `CLAUDE.md` §5) are done? Read the
   `pNN-verified` tags (`git tag --list 'p*-verified'`) and the latest commits
   (`git log --oneline -10`). Name the current phase and what remains in it.
2. **Health check.** Run and report the real output of:
   - `pytest -q` (pass/fail counts) with `CHURNGUARD_TRACKING=0`,
   - `ruff check churnguard tests scripts`,
   - `mypy churnguard`,
   - `python scripts/check_regression.py` (if a baseline exists).
   If any can't run, say why.
3. **Working tree.** `git status --short` — uncommitted changes, if any.
4. **Open SPEC-GAPs.** List every unresolved SPEC-GAP (from `CLAUDE.md` §10 and the
   current phase) with its proposed default.
5. **Next action.** One clear recommendation: the next command to run (e.g.
   `/build-phase N`, `/verify-phase N`) and any human gate that is pending.

Keep it to a short scannable summary. Do not change any files.
