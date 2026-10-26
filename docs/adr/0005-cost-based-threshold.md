# ADR-0005: Cost-based decision threshold

**Status:** Accepted

## Context
A classifier outputs a probability, but the business needs a yes/no: who gets a
retention offer? The default 0.5 cutoff is arbitrary and optimises nothing the
business cares about. The two mistakes have very different costs — missing a real
churner (false negative) is far more expensive than a wasted retention offer to
someone who would have stayed (false positive).

## Decision
Choose the decision threshold that **minimises expected business cost**. Assign
`cost_fn` and `cost_fp` (config; default 5:1), sweep every threshold in
`[0.01, 0.99]` on the held-out set, and pick the cost-minimising point. Persist it
into the `ModelBundle` so serving uses it automatically, and plot cost vs. threshold
(`reports/figures/cost_curve.png`).

## Consequences
- The operating point reflects the real cost trade-off, not a default.
- On the test split the chosen threshold (~0.20) cuts expected cost ~37% vs. 0.5.
- The threshold is only as good as the cost ratio; it must be re-tuned if the
  business economics change (noted in the model card and the runbook).
- Precision/recall are reported *at this threshold*, not at 0.5, so headline numbers
  match what production actually does.
