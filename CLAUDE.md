# ChurnGuard — Build Constitution

This file is the **operating constitution** for building ChurnGuard from scratch to
an industry-level product with Claude Code, phase by phase. Every agent and command
in `.claude/` obeys it. When anything here conflicts with a model's instinct, this
file wins.

---

## 1. Project summary

ChurnGuard is an **end-to-end machine-learning service** that predicts which telecom
customers are about to churn, so a retention team can act before they leave. It
covers the full lifecycle — **data → features → training → evaluation → explainability
→ a served REST API → monitoring → model lifecycle → observability** — with
production engineering practices (config-driven, leak-free pipeline, tests, Docker,
CI, drift monitoring, a model registry, a runbook, and ADRs).

It is a **scoring service, not an actuator**: it emits calibrated probabilities,
explanations and a cost-based decision; a human or downstream tool takes the action.

---

## 2. Sources of truth (priority order)

When two sources disagree, the higher one wins. Lower sources are updated to match,
never the reverse.

1. **The spec** — this `CLAUDE.md`, the phase map (§6) and `docs/adr/` (Architecture
   Decision Records). ADRs explain *why*; the spec says *what*.
2. **The design & ops docs** — `docs/DESIGN.md`, `docs/RUNBOOK.md`, `config/config.yaml`.
3. **The code & tests** — the implementation. If code contradicts a higher source,
   the code is wrong (fix the code or raise a SPEC-GAP to change the spec first).

A decision is only "real" once it is in source #1 or #2. Code alone is never a spec.

---

## 3. Tech stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.10–3.12 |
| ML / data | scikit-learn, pandas, NumPy, SciPy, SHAP |
| Serving | FastAPI, Uvicorn, Pydantic v2 |
| Packaging | Docker, pip-tools (`requirements.lock`) |
| Quality | pytest + coverage, ruff, mypy, pre-commit |
| MLOps | file-based model registry, champion/challenger, CI regression gate |
| Monitoring | PSI drift (data/prediction/performance), MLflow tracking (best-effort) |
| Observability | structured JSON logs, Prometheus text metrics, rate limiting, API-key auth |
| CI/CD | GitHub Actions (test matrix + smoke + regression gate + scheduled retrain) |
| Deploy | Render blueprint / Hugging Face Spaces (production is a human gate) |

Add a dependency only when a phase needs it; pin it and update `requirements.lock`.

---

## 4. Team (agents)

Defined in `.claude/agents/`. Each agent has a single scope and an output contract,
and **reports only what it actually ran** — never predicted or assumed results.

| Agent | Role | Model | Access |
| --- | --- | --- | --- |
| `architect` | Phase 0: ADRs, data model, API contracts, scaffold, CI | opus | read/write |
| `data-engineer` | Data generation, loading, schema validation | sonnet | read/write |
| `backend-ml-engineer` | Features, training, evaluation, explainability, API | sonnet | read/write |
| `devops-mlops` | Docker, CI, registry, retraining, observability wiring | sonnet | read/write |
| `qa-engineer` | Writes tests first; owns the test suite | sonnet | read/write |
| `security-reviewer` | Security review (deps, inputs, secrets, auth) | opus | **read-only** |
| `code-reviewer` | Correctness, clarity, leakage, contract adherence | opus | **read-only** |
| `ux-reviewer` | The demo page / API ergonomics | sonnet | **read-only** |
| `verifier` | Independent final check of a phase's DoD | sonnet | **read-only** |

Reviewers and the verifier never edit code; they only read, run read-only checks,
and report. They are a different context from the builder on purpose.

---

## 5. Phase protocol

Every phase follows the same loop. Do not skip steps; do not reorder them.

1. **Plan** — restate the phase's scope and DoD; list files to touch; surface
   SPEC-GAPs *before* building. Output a short written plan.
2. **Tests first** — `qa-engineer` writes failing tests for the phase's behaviour
   (red). No implementation yet.
3. **Build** — the owning builder implements until the phase tests pass (green),
   plus lint + types clean.
4. **Review (parallel)** — `code-reviewer` and `security-reviewer` run independently
   and in parallel; `ux-reviewer` too when the phase touches the demo/API surface.
   Each returns findings ranked by severity.
5. **Fix** — the builder addresses findings. During fix loops, run **only the
   affected tests** (see §10), not the whole suite.
6. **Independent verify** — `verifier` (fresh context, read-only) re-runs the full
   phase suite + lint + types + the regression gate and confirms the DoD, reporting
   only what it ran.
7. **Human gate** — present the plan outcome, the diff summary, verifier report and
   all collected SPEC-GAPs. **Wait for explicit human approval** (§8).
8. **Commit** — only after approval. Commit (§7), then tag `pNN-verified`.

Anti-hallucination rule: **if it is not in the spec, log a `SPEC-GAP`, never guess.**
A SPEC-GAP is a one-line entry: `SPEC-GAP[pNN]: <question> → proposed default: <x>`.
Collect them and resolve them at the human gate — do not block mid-build on each.

### Phase map

| Phase | Name | Owner | Produces |
| --- | --- | --- | --- |
| **P00** | Architecture & scaffold (`/architecture`) | architect | ADRs, data model, API contracts (OpenAPI), repo scaffold, CI skeleton |
| **P01** | Data layer | data-engineer | generator, loader, real-data download, schema validation + tests |
| **P02** | Features & preprocessing | backend-ml-engineer | `engineer_features`, leak-free `ColumnTransformer` + tests |
| **P03** | Training | backend-ml-engineer | CV selection → tuning → calibration → `ModelBundle` + tests |
| **P04** | Evaluation | backend-ml-engineer | metrics, figures, cost threshold, model card, reference profile, baseline |
| **P05** | Explainability | backend-ml-engineer | per-prediction SHAP mapped to business features + tests |
| **P06** | Serving API & batch CLI | backend-ml-engineer | `/predict`, `/health`, `/metrics`, demo page, batch CLI + tests |
| **P07** | Drift monitoring | devops-mlops | PSI monitor, `/monitor`, reference-profile capture + tests |
| **P08** | Model registry & retraining | devops-mlops | registry, champion/challenger, retrain + regression gate + tests |
| **P09** | Observability & hardening | devops-mlops | JSON logs, Prometheus, rate limit, API-key auth + tests |
| **P10** | Docs & release | architect | DESIGN, RUNBOOK, ADRs, dossier PDF, scheduled retrain, release |

(This mirrors the shipped architecture; a rebuild walks the same path in order.)

---

## 6. Definition of Done (per phase)

A phase is Done only when **all** of these hold and the verifier confirms them:

- [ ] Phase tests exist, were written before the code, and pass.
- [ ] `ruff check` clean and `mypy churnguard` clean.
- [ ] Full suite passes at least once this phase; coverage does not drop.
- [ ] `code-reviewer` and `security-reviewer` findings are resolved or logged as
      an accepted SPEC-GAP with a human decision.
- [ ] CI regression gate passes (no model-quality regression beyond tolerance).
- [ ] Public behaviour matches the API contract / ADRs; no undocumented behaviour.
- [ ] No secret, credential or real `.env` content in code, logs or history.
- [ ] All SPEC-GAPs for the phase are resolved at the human gate.
- [ ] `verifier` report attached, reporting only commands it actually ran.

---

## 7. Git rules

- Stage with plain **`git add -A`**.
- Commit with **`git commit -m "<type>(pNN): <summary>"`** where `<type>` ∈
  {feat, fix, test, docs, chore, refactor, ci, perf} and `pNN` is the phase
  (e.g. `feat(p03): CV model selection + tuning`).
- After the human gate, tag the phase: **`git tag pNN-verified`**.
- **No** co-author lines, trailers, `Signed-off-by`, or any generated-by signature
  in commit messages. Commits are authored by the human only.
- **Never** run `git push`, `git remote`, `git rebase`, `git reset --hard`,
  `git commit --amend`, or `git filter-branch`. History is append-only and local.
- If a mistake needs undoing, make a new commit — do not rewrite history.

---

## 8. Human gates (stop and ask — never self-approve)

Pause and require explicit human approval before:

- Changing settings/config that affect tooling, permissions or CI behaviour.
- Using any **paid or external service** (cloud deploy, hosted DB, paid API, a
  dataset download that incurs cost or has license terms).
- Anything touching **production** (deploying, promoting a model to `production`,
  changing a live endpoint).
- **Deleting or overwriting data** (datasets, models, the registry, reports) or any
  irreversible action.
- Adding a new runtime dependency or raising the dependency's major version.
- Each **phase commit** (the gate at step 7 of §5).

At a gate, present: the plan result, a diff summary, the verifier report, and the
full SPEC-GAP list. Wait for a clear yes.

---

## 9. Speed rules

- **Fix loops run only the affected tests** (`pytest tests/test_<area>.py -q` or
  `-k <pattern>`). Run the **full suite once per phase**, at the verify step.
- Run **independent agents in parallel** (e.g. code-reviewer + security-reviewer, or
  several read-only reviews) in a single batch, never one-at-a-time when they don't
  depend on each other.
- **Collect all SPEC-GAPs** and resolve them together at the human gate; do not stop
  the build to ask about each one as it appears.
- Prefer targeted reads/searches over re-reading whole files; cite `file:line`.
- Don't regenerate heavy artifacts (full training, figures, the PDF) during a fix
  loop — only at the verify step or when the change requires it.

---

## 10. Known SPEC-GAPs (seed log)

Resolved defaults from the initial doc review. Revisit any of these at a human gate
if the product gives a firmer answer.

1. **Data of record** — synthetic for dev/CI; any *published* metric must come from
   `make data-real`. Never present synthetic numbers as real.
2. **SLA/latency** — default p99 < 50 ms single-row, 99.5% monthly availability.
3. **Cost ratio** — `cost_fn:cost_fp = 5:1`, config-driven, flagged as an assumption.
4. **Deploy target** — Render blueprint as reference; production is a human gate.
5. **Auth** — single `CHURNGUARD_API_KEY` + gateway; multi-tenant out of scope.
6. **PII/retention** — drop IDs before modelling; no PII in logs; demo data non-PII.
7. **Retrain cadence** — monthly + on-alert; promotion to prod human-gated.
8. **Reference refresh** — only on an intentional, versioned baseline change.
9. **Label latency** — performance drift computed only when labels are attached.
10. **Throughput** — stateless horizontal scale; batch cap 1000/request.
11. **Frontend** — keep the demo page; a dashboard is future work.
12. **Python matrix** — support 3.10–3.12; CI on 3.10 & 3.12.
13. **Secrets** — env vars + platform secrets; only `.env.example` in repo.
14. **Dataset license** — real Telco CSV for eval only; cite, don't redistribute.
15. **Release policy** — semver; `pNN-verified` tag per phase; release tag at a gate.
