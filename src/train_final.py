"""Train the production model on every labelled year and register it in MLflow.

Usage:
    python src/train_final.py      # registers model "pipe-break-ranker" (new version each run)
"""
import mlflow
import mlflow.lightgbm

from rolling import load_annual
from train import CATEGORICAL, FEATURE_SETS, LGBM_PARAMS, ROOT, train_lgbm

MODEL_NAME = "pipe-break-ranker"
FEATURES = "nearby"
LAST_LABELLED_YEAR = 2025


def main():
    train = load_annual(range(1996, LAST_LABELLED_YEAR + 1)).reset_index(drop=True)
    numeric = FEATURE_SETS[FEATURES]
    model, _ = train_lgbm(train, numeric, seed=0)

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("production")
    with mlflow.start_run(run_name=f"final-model-1996-{LAST_LABELLED_YEAR}"):
        mlflow.log_params({**LGBM_PARAMS, "features": FEATURES, "train_years": f"1996-{LAST_LABELLED_YEAR}",
                           "rows": len(train)})
        mlflow.log_dict({"numeric": numeric, "categorical": CATEGORICAL,
                         "categories": {c: sorted(train[c].dropna().unique().tolist()) for c in CATEGORICAL}},
                        "inputs.json")
        mlflow.lightgbm.log_model(model, name="model", registered_model_name=MODEL_NAME)
    print(f"Registered {MODEL_NAME}: trained on {len(train):,} pipe-years, 1996-{LAST_LABELLED_YEAR}")


if __name__ == "__main__":
    main()
