# Calgary Water Main Replacement Prioritisation — Classical ML

Avathon AI/ML Hiring Challenge · **Track C — Classical ML**

## Problem statement

Calgary can replace only about 1% of its ~5,400 km of water mains over a five-year planning horizon
(~8 km in 2025, ~15 km/yr planned for 2027–2030). Every break means an emergency dig, homes without
water and a damaged street. The question each planning cycle: **which pipes should be replaced first so
the most future breaks are avoided?**

This project ranks every pipe by its predicted number of breaks per km over the next five years. The
output is a replacement shortlist for an asset engineer, who makes the final decision.

## Data source

City of Calgary Open Data, [Open Government Licence – City of Calgary](https://www.calgary.ca/cs/iis/licensing-city-data/licensing-city-data.html):

- [Water Main Breaks](https://data.calgary.ca/Environment/Water-Main-Breaks/dpcu-jr23) — 37,538 breaks, 1956–2026, with failure mode and location
- [Public Water Main](https://data.calgary.ca/Environment/Public-Water-Main/w6h9-w33i) — 60,740 pipe segments with material, diameter, install year and length

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing results

1. `dvc repro` — downloads the data and builds the feature tables (`data/features/`); `dvc.lock` records the exact file versions used
2. `notebooks/01_eda.ipynb` — data exploration
3. `python src/train.py --model rule` — train and evaluate a method on the validation window; every run is logged to MLflow
4. `mlflow ui --backend-store-uri sqlite:///mlflow.db` — compare runs

## Repository structure

```
src/        data download and loading code
notebooks/  analysis notebooks
data/       raw data (downloaded, not committed)
results/    figures and metrics
write-up/   technical write-up
```
