"""Compare evaluation designs year by year: how well does the ranking catch each year's breaks?

Designs:
  five_year  one ranking made from 5-year snapshots at the start of the window, reused every year
  annual     a fresh model every 1 January, trained on 1-year snapshots from the previous `window` years

Usage:
    python -m src.experiments.rolling --design annual --window 4 --model lgbm --features nearby
    python -m src.experiments.rolling --design five_year --model lgbm --features nearby
"""
import argparse
import mlflow
import numpy as np
import pandas as pd

from src.evaluate import BUDGETS, caught_at_budget, recall_table
from src.experiments.compare_models import MODELS, STOCHASTIC
from src.model import FEATURE_SETS, ROOT, load, load_annual

YEARS = {"validation": range(2016, 2021), "test": range(2021, 2026)}


def annual_scores(year, window, model, numeric, seed):
    first = 1996 if window == "all" else year - int(window)
    train = load_annual(range(first, year)).reset_index(drop=True)
    test = load_annual([year])
    score, expected, _ = MODELS[model](train, test.reset_index(drop=True), numeric, seed)
    return test, score, expected


def five_year_scores(year, start, model, numeric, seed):
    """Score with the ranking made at `start` from 5-year snapshots; pipes laid since then rank last."""
    train_years = [y for y in (1996, 2001, 2006, 2011, 2016) if y + 5 <= start]
    frame_start = pd.read_parquet(ROOT / "data" / "features" / f"frame_{start}.parquet")
    score, _, _ = MODELS[model](load(train_years), frame_start.reset_index(drop=True), numeric, seed)
    ranking = pd.Series(score, index=frame_start.index)
    test = load_annual([year])
    return test, ranking.reindex(test.index).fillna(-np.inf).to_numpy(), None


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
    # Recalibration: scale this year's expected count by last year's actual/predicted ratio (past data only).
    need_prior = args.design == "annual" and args.model in ("glm", "lgbm", "xgb")
    prior_ratio = {}
    if need_prior:
        for seed in seeds:
            t, _, e = annual_scores(years[0] - 1, args.window, args.model, numeric, seed)
            prior_ratio[seed] = t.target.sum() / e.sum()
    for year in years:
        for seed in seeds:
            if args.design == "annual":
                test, score, expected = annual_scores(year, args.window, args.model, numeric, seed)
            else:
                test, score, expected = five_year_scores(year, years[0], args.model, numeric, seed)
            row = {"year": year, "seed": seed, "breaks": test.target.sum(), **recall_table(test, score)}
            row.update({f"caught@{b * 100:g}%": caught_at_budget(test, score, b) for b in BUDGETS})
            if expected is not None:
                row["predicted"] = expected.sum()
                row["predicted_recalibrated"] = expected.sum() * prior_ratio[seed]
                prior_ratio[seed] = test.target.sum() / expected.sum()
            rows.append(row)
    per_year = pd.DataFrame(rows).groupby("year").mean().drop(columns="seed")

    name = f"{args.design}{'' if args.design == 'five_year' else '-w' + args.window}-{args.model}-{args.features}-{args.split}"
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("design-comparison")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"design": args.design, "window": args.window if args.design == "annual" else "n/a",
                           "model": args.model, "features": args.features, "split": args.split,
                           "years": list(years), "n_seeds": len(seeds)})
        means = per_year.drop(columns=[c for c in per_year if c.startswith("caught") or c in ("breaks", "predicted", "predicted_recalibrated")]).mean().to_dict()
        for b in BUDGETS:
            pooled = per_year[f"caught@{b * 100:g}%"].sum() / per_year.breaks.sum()
            means[f"pooled_R@{b * 100:g}%"] = pooled
            means[f"pooled_lift@{b * 100:g}%"] = pooled / b
        if "predicted" in per_year:
            means["predicted_over_actual_raw"] = per_year.predicted.sum() / per_year.breaks.sum()
            means["predicted_over_actual_recalibrated"] = per_year.predicted_recalibrated.sum() / per_year.breaks.sum()
        mlflow.log_metrics({k.replace("@", "_at_").replace("%", "pct").replace(".", "_"): v for k, v in means.items()})
        mlflow.log_text(per_year.round(4).to_csv(), "per_year.csv")
    keys = ["pooled_R@0.25%", "pooled_lift@0.25%", "pooled_R@1%", "R@0.25%_no_history", "R@1%_no_history",
            "predicted_over_actual_raw", "predicted_over_actual_recalibrated"]
    print(f"{name}: " + "  ".join(f"{k}={means[k]:.3f}" for k in keys if k in means))
    cols = [c for c in ["breaks", "caught@0.25%", "predicted", "predicted_recalibrated"] if c in per_year]
    print(per_year[cols].round(1).T.to_string())


if __name__ == "__main__":
    main()
