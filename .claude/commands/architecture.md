---
description: "Phase 0 — produce ADRs, data model, API contracts, repo scaffold and CI skeleton (no application code)."
argument-hint: "(no args)"
---

Run **Phase 0 (P00): Architecture & scaffold** for ChurnGuard, following the phase
protocol in `CLAUDE.md` §5. Do **not** write application logic — scaffolding and
specs only.

Delegate to the **architect** agent to produce:

1. **ADRs** in `docs/adr/NNNN-*.md` (Nygard format) for every foundational decision:
   leak-free pipeline, CV model selection + tuning, probability calibration,
   cost-based threshold, SHAP explanations, file-based registry, PSI monitoring,
   dependency-free observability. Add an ADR-0001 "record architecture decisions".
2. **Data model** — the full column schema (types, dropped, engineered, target
   encoding) and the initial `config/config.yaml`.
3. **API contracts** — every endpoint: method, path, request + response schema,
   status codes, auth. Capture as an OpenAPI sketch or a precise markdown contract.
4. **Repo scaffold** — package layout and empty modules (docstrings + signatures,
   `NotImplementedError` bodies), `tests/` layout, `pyproject.toml`, `Makefile`,
   `Dockerfile`, `.env.example`, `.gitignore`.
5. **CI skeleton** — `.github/workflows/` running ruff + mypy + pytest (will go green
   as phases land) and the regression-gate + scheduled-retrain stubs.

Then:
- Collect all **SPEC-GAPs** (seed from `CLAUDE.md` §10; add any new ones).
- Present the plan result, the file list, and the SPEC-GAPs at a **human gate**.
- On approval, commit: `git add -A` then
  `git commit -m "chore(p00): architecture, contracts and scaffold"`, and tag
  `git tag p00-verified`. Never push.
