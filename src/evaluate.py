"""Metrics for a replacement shortlist: recall at a length budget, and calibration of predicted counts."""
import numpy as np
import pandas as pd

BUDGETS = [0.0025, 0.005, 0.01, 0.02, 0.05]


def caught_at_budget(frame, score, budget):
    """Breaks on the top-scored pipes that together make up `budget` of total length."""
    order = np.argsort(-np.asarray(score), kind="stable")
    length = frame.length.to_numpy()[order]
    target = frame.target.to_numpy()[order]
    within = np.cumsum(length) <= budget * length.sum()
    return target[within].sum()


def recall_at_budget(frame, score, budget):
    """Share of future breaks on the top-scored pipes that together make up `budget` of total length."""
    return caught_at_budget(frame, score, budget) / frame.target.sum()


def recall_table(frame, score):
    """Recall and lift over random at every budget, overall and for pipes with no break history.

    Lift = recall / budget: picking pipes at random catches ~budget share of breaks, so lift 1 = no skill.
    """
    out = {f"R@{b * 100:g}%": recall_at_budget(frame, score, b) for b in BUDGETS}
    out.update({f"lift@{b * 100:g}%": out[f"R@{b * 100:g}%"] / b for b in BUDGETS})
    clean = (frame.breaks_all == 0).to_numpy()
    out.update({f"R@{b * 100:g}%_no_history": recall_at_budget(frame[clean], np.asarray(score)[clean], b) for b in BUDGETS})
    return out


def calibration_by_decile(frame, expected_breaks):
    """Predicted vs actual break counts per decile of predicted risk."""
    d = pd.DataFrame({"expected": np.asarray(expected_breaks), "actual": frame.target.to_numpy()})
    d["decile"] = pd.qcut(d.expected.rank(method="first"), 10, labels=False) + 1
    return d.groupby("decile")[["expected", "actual"]].sum()
