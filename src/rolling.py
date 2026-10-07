"""Compare evaluation designs year by year: how well does the ranking catch each year's breaks?

Designs:
  five_year  one ranking made from 5-year snapshots at the start of the window, reused every year
  annual     a fresh model every 1 January, trained on 1-year snapshots from the previous `window` years

Usage:
    python src/rolling.py --design annual --window 4 --model lgbm --features nearby
    python src/rolling.py --design five_year --model lgbm --features nearby
"""
import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from evaluate import recall_table
from train import FEATURE_SETS, MODELS, ROOT, STOCHASTIC, load

ANNUAL_DIR = ROOT / "data" / "features_annual"
YEARS = {"validation": range(2016, 2021), "test": range(2021, 2026)}


def load_annual(years):
    return pd.concat([pd.read_parquet(ANNUAL_DIR / f"frame_{y}.parquet") for y in years])


def annual_scores(year, window, model, numeric, seed):
    first = 1996 if window == "all" else year - int(window)
    train = load_annual(range(first, year)).reset_index(drop=True)
    test = load_annual([year])
    score, _, _ = MODELS[model](train, test.reset_index(drop=True), numeric, seed)
    return test, score


def five_year_scores(year, start, model, numeric, seed):
    """Score with the ranking made at `start` from 5-year snapshots; pipes laid since then rank last."""
    train_years = [y for y in (1996, 2001, 2006, 2011, 2016) if y + 5 <= start]
    frame_start = pd.read_parquet(ROOT / "data" / "features" / f"frame_{start}.parquet")
    score, _, _ = MODELS[model](load(train_years), frame_start.reset_index(drop=True), numeric, seed)
    ranking = pd.Series(score, index=frame_start.index)
    test = load_annual([year])
    return test, ranking.reindex(test.index).fillna(-np.inf).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["annual", "five_year"], required=True)
    ap.add_argument("--window", default="4", help="annual design: years of training data, or 'all'")
    ap.add_argument("--model", choices=MODELS, default="lgbm")
    ap.add_argument("--features", choices=FEATURE_SETS, default="nearby")
    ap.add_argument("--split", choices=YEARS, default="validation")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    numeric = FEATURE_SETS[args.features]
    years = YEARS[args.split]
    seeds = range(args.seeds if args.model in STOCHASTIC else 1)

    rows = []
    for year in years:
        for seed in seeds:
            if args.design == "annual":
                test, score = annual_scores(year, args.window, args.model, numeric, seed)
            else:
                test, score = five_year_scores(year, years[0], args.model, numeric, seed)
            rows.append({"year": year, "seed": seed, **recall_table(test, score)})
    per_year = pd.DataFrame(rows).groupby("year").mean().drop(columns="seed")

    name = f"{args.design}{'' if args.design == 'five_year' else '-w' + args.window}-{args.model}-{args.features}-{args.split}"
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("design-comparison")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"design": args.design, "window": args.window if args.design == "annual" else "n/a",
                           "model": args.model, "features": args.features, "split": args.split,
                           "years": list(years), "n_seeds": len(seeds)})
        means = per_year.mean().to_dict()
        mlflow.log_metrics({k.replace("@", "_at_").replace("%", "pct").replace(".", "_"): v for k, v in means.items()})
        mlflow.log_text(per_year.round(4).to_csv(), "per_year.csv")
    print(f"{name}: " + "  ".join(f"{k}={means[k]:.3f}" for k in ["R@0.25%", "R@1%", "R@0.25%_no_history", "R@1%_no_history"]))
    print((per_year[["R@0.25%", "R@1%"]] * 100).round(1).T.to_string())


if __name__ == "__main__":
    main()
