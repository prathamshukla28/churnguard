"""The business decision layer: cost-based threshold and consistent risk bands.

Kept separate from training and evaluation so both share one definition:

* :func:`choose_threshold` picks the cut that minimises expected business cost.
  Training calls it on **out-of-fold** predictions over the training split; the
  held-out test split never reaches it (see ``tests/test_train.py``).
* :func:`risk_band` derives the band *from the same threshold* that decides
  ``will_churn``, so a response can never say ``will_churn: true`` with
  ``risk_band: "low"``.
"""

from __future__ import annotations

import numpy as np

THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
RISK_BANDS = ("low", "medium", "high")


def expected_cost(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float, cost_fn: float, cost_fp: float
) -> float:
    """Total cost of acting on ``y_proba >= threshold``."""
    y_true = np.asarray(y_true)
    pred = np.asarray(y_proba) >= threshold
    fn = int(((~pred) & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    return float(cost_fn * fn + cost_fp * fp)


def choose_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, cost_fn: float, cost_fp: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the threshold minimising expected business cost, and the cost curve.

    Ties resolve to the lowest threshold on the grid (``np.argmin``).
    """
    costs = np.asarray(
        [expected_cost(y_true, y_proba, t, cost_fn, cost_fp) for t in THRESHOLD_GRID],
        dtype=float,
    )
    best = float(THRESHOLD_GRID[int(np.argmin(costs))])
    return best, THRESHOLD_GRID.copy(), costs


def high_risk_cutoff(threshold: float, high_risk_probability: float = 0.5) -> float:
    """Probability at which ``medium`` becomes ``high``; never below the threshold."""
    return max(float(threshold), float(high_risk_probability))


def risk_band(p: float, threshold: float, high_risk_probability: float = 0.5) -> str:
    """Band a calibrated probability consistently with the decision threshold.

    * ``low``    — ``p < threshold``: below the action threshold (``will_churn`` false).
    * ``medium`` — at/above the threshold but below the high-risk cutoff.
    * ``high``   — ``p >= max(threshold, high_risk_probability)``: with calibrated
      probabilities and the default 0.5, "more likely than not to churn".

    Invariant: ``risk_band(p, t) != "low"``  <=>  ``p >= t``.
    """
    if p < threshold:
        return "low"
    if p >= high_risk_cutoff(threshold, high_risk_probability):
        return "high"
    return "medium"
