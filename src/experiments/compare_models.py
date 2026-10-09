"""Train one ranking method, evaluate it, and log the run to MLflow.

Usage:
    python -m src.experiments.compare_models --model rule        # evaluated on the validation window
    python -m src.experiments.compare_models --model lgbm --features nearby --split test
"""
import argparse

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.evaluate import calibration_by_decile, recall_table
from src.model import CATEGORICAL, FEATURE_SETS, ROOT, SEED, exposure, fit_lgbm, load

# Planning years whose 5-year labels end before the evaluation window starts.
SPLITS = {"validation": ([1996, 2001, 2006, 2011], 2016), "test": ([1996, 2001, 2006, 2011, 2016], 2021)}

def fit_rule(train, test, numeric, seed):
    """Past breaks per km — the baseline any model must beat. No expected counts."""
    return test.breaks_per_km.to_numpy(), None, {}


def fit_isolation_forest(train, test, numeric, seed):
    """Unsupervised: ranks pipes by how unusual their features are. Never sees the target."""
    pre = ColumnTransformer([("num", StandardScaler(), numeric),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), ["material"])])
    model = make_pipeline(pre, IsolationForest(n_estimators=300, random_state=seed))
    model.fit(train)
    return -model.score_samples(test), None, {"n_estimators": 300}


def fit_glm(train, test, numeric, seed):
    """Poisson regression on break rate per km-year, weighted by exposure."""
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    pre = ColumnTransformer([("num", make_pipeline(log1p, StandardScaler()), numeric),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), CATEGORICAL)])
    model = make_pipeline(pre, PoissonRegressor(alpha=1e-3, max_iter=3000))
    model.fit(train, train.target / exposure(train), poissonregressor__sample_weight=exposure(train))
    rate = model.predict(test)
    return rate, rate * exposure(test), {"alpha": 1e-3}


XGB_PARAMS = {"objective": "count:poisson", "n_estimators": 500, "learning_rate": 0.03, "max_depth": 4,
              "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
              "enable_categorical": True, "random_state": SEED}


def fit_xgb(train, test, numeric, seed):
    """Same tree-boosting idea as LightGBM, but trees grow level by level; offset via base_margin."""
    cols = numeric + CATEGORICAL
    def prep(f):
        x = f[cols].copy()
        for c in CATEGORICAL:
            x[c] = pd.Categorical(x[c], categories=sorted(train[c].dropna().unique()))
        return x
    params = {**XGB_PARAMS, "random_state": seed}
    model = xgb.XGBRegressor(**params).fit(prep(train), train.target, base_margin=np.log(exposure(train)))
    rate = np.exp(model.predict(prep(test), output_margin=True, base_margin=np.zeros(len(test))))
    importance = pd.Series(model.feature_importances_, index=cols)
    return rate, rate * exposure(test), {**params, "importance": importance / importance.sum()}


MODELS = {"rule": fit_rule, "iforest": fit_isolation_forest, "glm": fit_glm, "lgbm": fit_lgbm, "xgb": fit_xgb}
STOCHASTIC = {"iforest", "lgbm", "xgb"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--features", choices=FEATURE_SETS, default="base")
    ap.add_argument("--split", choices=SPLITS, default="validation")
    ap.add_argument("--seeds", type=int, default=5, help="seeds to average for stochastic models")
    args = ap.parse_args()

    train_years, eval_year = SPLITS[args.split]
    train, test = load(train_years), load([eval_year])
    numeric = FEATURE_SETS[args.features]

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("replacement-ranking") is None:
        mlflow.create_experiment("replacement-ranking", artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment("replacement-ranking")
    with mlflow.start_run(run_name=f"{args.model}-{args.features}-{args.split}"):
        seeds = range(args.seeds if args.model in STOCHASTIC else 1)
        per_seed = []
        for seed in seeds:
            score, expected, params = MODELS[args.model](train, test, numeric, seed)
            importance = params.pop("importance", None)
            m = recall_table(test, score)
            if expected is not None:
                cal = calibration_by_decile(test, expected)
                m["expected_over_actual"] = cal.expected.sum() / cal.actual.sum()
                m["top_decile_expected_over_actual"] = cal.expected.iloc[-1] / cal.actual.iloc[-1]
            per_seed.append(m)
        params.pop("random_state", None)
        mlflow.log_params({"model": args.model, "features": args.features, "split": args.split, "n_seeds": len(seeds),
                           "train_years": train_years, "eval_year": eval_year, "n_features": len(numeric), **params})
        metrics = pd.DataFrame(per_seed).mean().to_dict()
        metrics["R@1%_seed_std"] = pd.DataFrame(per_seed)["R@1%"].std(ddof=0)
        if expected is not None:
            mlflow.log_text(cal.round(1).to_csv(), "calibration_by_decile_last_seed.csv")
        if importance is not None:
            mlflow.log_text(importance.sort_values(ascending=False).round(4).to_csv(), "feature_importance.csv")
        mlflow.log_metrics({k.replace("@", "_at_").replace("%", "pct"): v for k, v in metrics.items()})
        print(f"{args.model}-{args.features} on {eval_year}: " +
              "  ".join(f"{k}={v:.3f}" for k, v in metrics.items() if "no_history" not in k and "lift" not in k))


if __name__ == "__main__":
    main()
