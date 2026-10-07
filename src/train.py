"""Train one ranking method, evaluate it, and log the run to MLflow.

Usage:
    python src/train.py --model rule                            # evaluated on the validation window
    python src/train.py --model rule --split test
"""
import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from evaluate import calibration_by_decile, recall_table
from features import HORIZON

ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "data" / "features"
SEED = 0

# Planning years whose 5-year labels end before the evaluation window starts.
SPLITS = {"validation": ([1996, 2001, 2006, 2011], 2016), "test": ([1996, 2001, 2006, 2011, 2016], 2021)}

BASE = ["age", "diameter", "log_length", "breaks_all", "breaks_10y", "breaks_per_km",
        "breaks_per_km_10y", "years_since_break"]
CATEGORICAL = ["material", "p_zone"]
FEATURE_SETS = {"base": BASE}


def load(years):
    return pd.concat([pd.read_parquet(FEATURE_DIR / f"frame_{y}.parquet") for y in years], ignore_index=True)


def exposure(frame):
    return frame.km.to_numpy() * HORIZON


def fit_rule(train, test, numeric):
    """Past breaks per km — the baseline any model must beat. No expected counts."""
    return test.breaks_per_km.to_numpy(), None, {}


def fit_isolation_forest(train, test, numeric):
    """Unsupervised: ranks pipes by how unusual their features are. Never sees the target."""
    pre = ColumnTransformer([("num", StandardScaler(), numeric),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), ["material"])])
    model = make_pipeline(pre, IsolationForest(n_estimators=300, random_state=SEED))
    model.fit(train)
    return -model.score_samples(test), None, {"n_estimators": 300}


def fit_glm(train, test, numeric):
    """Poisson regression on break rate per km-year, weighted by exposure."""
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    pre = ColumnTransformer([("num", make_pipeline(log1p, StandardScaler()), numeric),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), CATEGORICAL)])
    model = make_pipeline(pre, PoissonRegressor(alpha=1e-3, max_iter=3000))
    model.fit(train, train.target / exposure(train), poissonregressor__sample_weight=exposure(train))
    rate = model.predict(test)
    return rate, rate * exposure(test), {"alpha": 1e-3}


MODELS = {"rule": fit_rule, "iforest": fit_isolation_forest, "glm": fit_glm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--features", choices=FEATURE_SETS, default="base")
    ap.add_argument("--split", choices=SPLITS, default="validation")
    args = ap.parse_args()

    train_years, eval_year = SPLITS[args.split]
    train, test = load(train_years), load([eval_year])
    numeric = FEATURE_SETS[args.features]

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("replacement-ranking") is None:
        mlflow.create_experiment("replacement-ranking", artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment("replacement-ranking")
    with mlflow.start_run(run_name=f"{args.model}-{args.features}-{args.split}"):
        score, expected, params = MODELS[args.model](train, test, numeric)
        importance = params.pop("importance", None)
        mlflow.log_params({"model": args.model, "features": args.features, "split": args.split,
                           "train_years": train_years, "eval_year": eval_year, "n_features": len(numeric), **params})
        metrics = recall_table(test, score)
        if expected is not None:
            cal = calibration_by_decile(test, expected)
            metrics["expected_over_actual"] = cal.expected.sum() / cal.actual.sum()
            metrics["top_decile_expected_over_actual"] = cal.expected.iloc[-1] / cal.actual.iloc[-1]
            mlflow.log_text(cal.round(1).to_csv(), "calibration_by_decile.csv")
        if importance is not None:
            mlflow.log_text(importance.sort_values(ascending=False).round(4).to_csv(), "feature_importance.csv")
        mlflow.log_metrics({k.replace("@", "_at_").replace("%", "pct"): v for k, v in metrics.items()})
        print(f"{args.model}-{args.features} on {eval_year}: " +
              "  ".join(f"{k}={v:.3f}" for k, v in metrics.items() if "no_history" not in k))


if __name__ == "__main__":
    main()
