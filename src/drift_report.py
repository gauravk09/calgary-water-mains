"""Data drift between the validation years (2016-2020) and the test years (2021-2025), with Evidently.

Usage:
    python src/drift_report.py      # writes results/drift_report.html and logs it to MLflow
"""
import mlflow
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from rolling import load_annual
from train import CATEGORICAL, FEATURE_SETS, ROOT

COLUMNS = FEATURE_SETS["nearby"] + CATEGORICAL + ["target"]
SAMPLE = 50_000
OUT = ROOT / "results" / "drift_report.html"


def main():
    reference = load_annual(range(2016, 2021))[COLUMNS].sample(SAMPLE, random_state=0)
    current = load_annual(range(2021, 2026))[COLUMNS].sample(SAMPLE, random_state=0)
    snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    snapshot.save_html(str(OUT))

    drifted = [m for m in snapshot.dict()["metrics"] if "ValueDrift" in m["metric_name"]]
    table = pd.DataFrame({"column": [m["config"]["column"] for m in drifted],
                          "drift_score": [m["value"] for m in drifted]}).sort_values("drift_score")
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("monitoring")
    with mlflow.start_run(run_name="feature-drift-2016-20-vs-2021-25"):
        mlflow.log_artifact(str(OUT))
        mlflow.log_text(table.to_csv(index=False), "drift_scores.csv")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
