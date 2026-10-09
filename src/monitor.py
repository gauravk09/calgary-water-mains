"""Yearly calibration monitor with Evidently: are actual breaks within ±15% of the model's predicted total?

Replays deployment for 2016-2025: each year the model is retrained on all earlier years, predicts every pipe,
and an Evidently test checks the year's actual break count against the predicted total. An alarm fires when
the check fails two years running.

Usage:
    python src/monitor.py      # results/calibration_monitor.{csv,png}, results/monitor_latest_year.html
"""
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from evidently import Report
from evidently.metrics import SumValue
from evidently.tests import gte, lte

from rolling import annual_scores
from train import FEATURE_SETS, ROOT

YEARS = range(2016, 2026)
TOLERANCE = 0.15
RESULTS = ROOT / "results"


def check_year(year):
    pipes, _, expected = annual_scores(year, "all", "lgbm", FEATURE_SETS["nearby"], seed=0)
    data = pd.DataFrame({"prediction": expected, "target": pipes.target.to_numpy()})
    predicted = data.prediction.sum()
    report = Report([SumValue(column="prediction"),
                     SumValue(column="target", tests=[gte(predicted * (1 - TOLERANCE)), lte(predicted * (1 + TOLERANCE))])])
    snapshot = report.run(current_data=data)
    passed = all(t["status"].name == "SUCCESS" for t in snapshot.dict()["tests"])
    return {"year": year, "predicted": predicted, "actual": data.target.sum(), "within_15pct": passed}, snapshot


def main():
    rows = []
    for year in YEARS:
        row, snapshot = check_year(year)
        rows.append(row)
    snapshot.save_html(str(RESULTS / "monitor_latest_year.html"))

    t = pd.DataFrame(rows)
    t["actual_over_predicted"] = t.actual / t.predicted
    t["alarm"] = ~t.within_15pct & ~t.within_15pct.shift(fill_value=True)
    t.round(3).to_csv(RESULTS / "calibration_monitor.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.fill_between(t.year, t.predicted * (1 - TOLERANCE), t.predicted * (1 + TOLERANCE), color="#2a78d6", alpha=0.15,
                    label="predicted ±15%")
    ax.plot(t.year, t.predicted, color="#2a78d6", linewidth=2, label="predicted breaks")
    ax.plot(t.year, t.actual, color="#eb6834", linewidth=2, marker="o", label="actual breaks")
    alarms = t[t.alarm]
    ax.scatter(alarms.year, alarms.actual, s=160, facecolors="none", edgecolors="#e34948", linewidths=2, label="alarm")
    ax.axvline(2020.5, color="#52514e", linestyle="--", linewidth=1)
    ax.text(2020.6, ax.get_ylim()[1] * 0.97, "test years →", fontsize=9, color="#52514e", va="top")
    ax.set_ylabel("breaks per year"); ax.set_xticks(list(YEARS))
    ax.set_title("Calibration monitor: actual vs predicted breaks (alarm = outside ±15% two years running)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); fig.savefig(RESULTS / "calibration_monitor.png", dpi=150, bbox_inches="tight")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("monitoring")
    with mlflow.start_run(run_name="calibration-monitor-2016-2025"):
        mlflow.log_params({"tolerance": TOLERANCE, "alarm_rule": "outside tolerance two years running"})
        for name in ("calibration_monitor.csv", "calibration_monitor.png", "monitor_latest_year.html"):
            mlflow.log_artifact(str(RESULTS / name))
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
