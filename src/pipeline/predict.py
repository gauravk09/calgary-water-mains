"""Load the registered model and produce the replacement shortlist for a planning year.

Usage:
    python -m src.pipeline.predict --year 2026 --budget 0.0025      # results/shortlist_2026.csv
"""
import argparse

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import shap

from src.data import load_pipes
from src.features import RADII_M, all_breaks, build_frame, neighbour_pairs, usable_breaks
from src.model import ROOT, exposure
from src.pipeline.train import MODEL_NAME

MIN_REPLACEABLE_M = 10


def load_model():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    versions = mlflow.MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    version = max(versions, key=lambda v: int(v.version))
    model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/{version.version}")
    inputs = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/inputs.json")
    return model, inputs, version.version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--budget", type=float, default=0.0025, help="share of network length to replace")
    args = ap.parse_args()

    model, inputs, version = load_model()
    pipes, geoms = load_pipes()
    breaks = usable_breaks(pipes, geoms)
    everything = all_breaks(geoms)
    neighbours = {r: neighbour_pairs(everything, geoms, r) for r in RADII_M}
    frame = build_frame(args.year, pipes, breaks, neighbours, horizon=1)

    cols = inputs["numeric"] + inputs["categorical"]
    X = frame[cols].copy()
    for c in inputs["categorical"]:
        X[c] = pd.Categorical(X[c], categories=inputs["categories"][c])
    frame["rate_per_km_yr"] = np.exp(model.predict(X, raw_score=True))
    frame["expected_breaks"] = frame.rate_per_km_yr * exposure(frame)

    # Replaceable units only: stubs under 10 m are fittings/junction pieces replaced with their main,
    # and INACTIVE pipes carry no water. They stay in the network-wide expected count.
    eligible = frame[(frame.length >= MIN_REPLACEABLE_M) & (pipes.loc[frame.index, "status_ind"] == "ACTIVE")]
    ranked = eligible.sort_values("rate_per_km_yr", ascending=False)
    shortlist = ranked[ranked.length.cumsum() <= args.budget * frame.length.sum()].copy()

    contrib = pd.DataFrame(shap.TreeExplainer(model).shap_values(X.loc[shortlist.index]),
                           columns=cols, index=shortlist.index)
    def reasons(row):
        top = row.abs().sort_values(ascending=False).index[:3]
        return "; ".join(f"{c} x{np.exp(row[c]):.2f}" for c in top)
    shortlist["top_reasons"] = contrib.apply(reasons, axis=1)

    shortlist["globalid"] = pipes.loc[shortlist.index, "globalid"]
    shortlist.insert(0, "rank", range(1, len(shortlist) + 1))
    out = shortlist[["rank", "globalid", "material", "install_year", "diameter", "length", "breaks_all",
                     "nearby_150m_10y", "rate_per_km_yr", "expected_breaks", "top_reasons"]]
    path = ROOT / "results" / f"shortlist_{args.year}.csv"
    out.round(4).to_csv(path, index=False)
    print(f"Model {MODEL_NAME} v{version} -> {len(out)} pipes, {out.length.sum() / 1000:.1f} km "
          f"({args.budget:.2%} of {frame.length.sum() / 1000:,.0f} km); "
          f"expected breaks on the list in {args.year}: {out.expected_breaks.sum():.1f}; "
          f"expected network-wide: {frame.expected_breaks.sum():.0f}")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
