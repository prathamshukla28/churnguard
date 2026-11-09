# ADR-0005: Cost-based decision threshold

**Status:** Accepted (amended: selection moved from the test split to out-of-fold training predictions)

## Context
A classifier outputs a probability, but the business needs a yes/no: who gets a
retention offer? The default 0.5 cutoff is arbitrary and optimises nothing the
business cares about. The two mistakes have very different costs — missing a real
churner (false negative) is far more expensive than a wasted retention offer to
someone who would have stayed (false positive).

## Decision
Choose the decision threshold that **minimises expected business cost**. Assign
`cost_fn` and `cost_fp` (config; default 5:1), sweep every threshold in
`[0.01, 0.99]` on **out-of-fold predictions over the training split** (each row
scored by a calibrated model that never saw it), and pick the cost-minimising
point. Persist it into the `ModelBundle` at training time so serving uses it
automatically. The test split never influences the choice; `evaluate` reports the
cost at the frozen threshold once. An earlier version chose the threshold on the
test split and reported the saving on that same split, which overstated it.

## Consequences
- The operating point reflects the real cost trade-off, not a default.
- On the real Telco data the OOF-chosen threshold is 0.19; on the untouched test
  split it cuts expected cost ~35% vs. 0.5 (629 vs. 967) under the assumed 5:1 ratio.
- The threshold is only as good as the cost ratio; it must be re-tuned if the
  business economics change (noted in the model card and the runbook).
- Precision/recall are reported *at this threshold*, not at 0.5, so headline numbers
  match what production actually does.
- Risk bands are derived from the same threshold (`low` < threshold ≤ `medium` <
  `high`), so `will_churn` and the band never disagree.
