"""Train vs validation vs test scores for the final model, with the same metric everywhere.

For each graded year Y the model is trained on 1996..Y-1, then scored on
  - year Y-1  (inside its training data -> "train", in-sample)
  - year Y    (not seen               -> validation for 2016-2020, test for 2021-2025)
Scores are pooled over the five years of each split.

Usage:
    python src/score_table.py      # results/scores_train_val_test.csv
"""
import numpy as np
import pandas as pd

from evaluate import caught_at_budget
from rolling import load_annual
from train import FEATURE_SETS, ROOT, exposure, train_lgbm

SPLITS = {"validation": range(2016, 2021), "test": range(2021, 2026)}
BUDGETS = [0.0025, 0.01]


def pooled(rows):
    t = pd.DataFrame(rows)
    out = {f"R@{b * 100:g}%": t[f"caught@{b}"].sum() / t.breaks.sum() for b in BUDGETS}
    out.update({f"lift@{b * 100:g}%": out[f"R@{b * 100:g}%"] / b for b in BUDGETS})
    out["predicted_over_actual"] = t.predicted.sum() / t.breaks.sum()
    out["breaks"] = int(t.breaks.sum())
    return out


def score(frame, rate, expected):
    row = {"breaks": frame.target.sum(), "predicted": expected.sum()}
    row.update({f"caught@{b}": caught_at_budget(frame, rate, b) for b in BUDGETS})
    return row


def main():
    numeric = FEATURE_SETS["nearby"]
    results = []
    for split, years in SPLITS.items():
        in_sample, held_out = [], []
        for year in years:
            model, prep = train_lgbm(load_annual(range(1996, year)).reset_index(drop=True), numeric, seed=0)
            for frame_year, bucket in ((year - 1, in_sample), (year, held_out)):
                frame = load_annual([frame_year]).reset_index(drop=True)
                rate = np.exp(model.predict(prep(frame), raw_score=True))
                bucket.append(score(frame, rate, rate * exposure(frame)))
        results.append({"split": f"train (in-sample, years {years[0] - 1}-{years[-1] - 1})", **pooled(in_sample)})
        results.append({"split": f"{split} (held out, {years[0]}-{years[-1]})", **pooled(held_out)})
    table = pd.DataFrame(results)
    table.round(4).to_csv(ROOT / "results" / "scores_train_val_test.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
