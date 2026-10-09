"""Tune LightGBM regularisation on the validation years only, logging each setting to MLflow.

For each setting and each validation year Y: train on 1996..Y-1, score year Y (held out) and year Y-1
(in-sample). A large in-sample vs held-out gap means the model memorises training breaks.

Usage:
    python src/tune.py
"""
import itertools

import mlflow
import numpy as np
import pandas as pd

from evaluate import caught_at_budget
from rolling import load_annual
from train import FEATURE_SETS, LGBM_PARAMS, ROOT, exposure, train_lgbm
import train as T

GRID = {"min_child_samples": [50, 500, 2000], "n_estimators": [150, 500], "reg_lambda": [0.0, 10.0]}
YEARS = range(2016, 2021)


def evaluate_setting(params):
    T.LGBM_PARAMS = {**LGBM_PARAMS, **params}
    held, inside, short_share = [], [], []
    for year in YEARS:
        model, prep = train_lgbm(load_annual(range(1996, year)).reset_index(drop=True), FEATURE_SETS["nearby"], seed=0)
        for frame_year, bucket in ((year, held), (year - 1, inside)):
            f = load_annual([frame_year]).reset_index(drop=True)
            rate = np.exp(model.predict(prep(f), raw_score=True))
            bucket.append({"breaks": f.target.sum(), "predicted": (rate * exposure(f)).sum(),
                           "c25": caught_at_budget(f, rate, 0.0025), "c100": caught_at_budget(f, rate, 0.01)})
            if frame_year == year:
                top = f.assign(r=rate).sort_values("r", ascending=False)
                top = top[top.length.cumsum() <= 0.0025 * top.length.sum()]
                short_share.append((top.length < 20).mean())
    h, i = pd.DataFrame(held).sum(), pd.DataFrame(inside).sum()
    return {"val_R_at_0_25pct": h.c25 / h.breaks, "val_R_at_1pct": h.c100 / h.breaks,
            "train_R_at_0_25pct": i.c25 / i.breaks, "train_R_at_1pct": i.c100 / i.breaks,
            "val_predicted_over_actual": h.predicted / h.breaks,
            "share_of_shortlist_under_20m": float(np.mean(short_share))}


def main():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("tuning")
    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        with mlflow.start_run(run_name="lgbm-" + "-".join(f"{k}={v}" for k, v in params.items())):
            m = evaluate_setting(params)
            m["gap_R_at_1pct"] = m["train_R_at_1pct"] - m["val_R_at_1pct"]
            mlflow.log_params(params); mlflow.log_metrics(m)
        rows.append({**params, **m})
        print({**params, **{k: round(v, 3) for k, v in m.items()}}, flush=True)
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "tuning_validation.csv", index=False)


if __name__ == "__main__":
    main()
