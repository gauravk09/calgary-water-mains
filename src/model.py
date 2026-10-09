"""Shared model code: feature sets, frozen settings, data loading and the LightGBM Poisson model."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import MODES, RADII_M

ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "data" / "features"
ANNUAL_DIR = ROOT / "data" / "features_annual"
SEED = 0

BASE = ["age", "diameter", "log_length", "breaks_all", "breaks_10y", "breaks_per_km",
        "breaks_per_km_10y", "years_since_break"]
TYPES = [f"past_{m}" for m in MODES.values()]
NEARBY = [f"nearby_{r}m_{k}" for r in RADII_M for k in ("10y", "decay")]
CATEGORICAL = ["material", "p_zone"]
FEATURE_SETS = {"base": BASE, "types": BASE + TYPES, "nearby": BASE + NEARBY, "all": BASE + TYPES + NEARBY}

MIN_EXPOSURE_KM = 0.01   # segments shorter than 10 m count as 10 m; tiny offsets made boosting unstable

# Frozen production settings (regularisation chosen on validation, see experiments/tune.py).
LGBM_PARAMS = {"objective": "poisson", "n_estimators": 500, "learning_rate": 0.03, "num_leaves": 15,
               "min_child_samples": 2000, "reg_lambda": 10.0, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
               "random_state": SEED, "verbose": -1}


def load(years):
    """5-year-horizon feature tables."""
    return pd.concat([pd.read_parquet(FEATURE_DIR / f"frame_{y}.parquet") for y in years], ignore_index=True)


def load_annual(years):
    """1-year-horizon feature tables (index = pipe)."""
    return pd.concat([pd.read_parquet(ANNUAL_DIR / f"frame_{y}.parquet") for y in years])


def exposure(frame):
    return frame.km.clip(lower=MIN_EXPOSURE_KM).to_numpy() * frame.horizon.to_numpy()


def train_lgbm(train, numeric, seed):
    """Fit LightGBM on break counts with log(exposure) as offset; returns the model and its input builder."""
    cols = numeric + CATEGORICAL
    def prep(f):
        x = f[cols].copy()
        for c in CATEGORICAL:
            x[c] = pd.Categorical(x[c], categories=sorted(train[c].dropna().unique()))
        return x
    params = {**LGBM_PARAMS, "random_state": seed}
    model = lgb.LGBMRegressor(**params).fit(prep(train), train.target, init_score=np.log(exposure(train)))
    return model, prep


def fit_lgbm(train, test, numeric, seed):
    """Gradient boosting on break counts with log(exposure) as offset, so it predicts a rate per km-year."""
    model, prep = train_lgbm(train, numeric, seed)
    cols = numeric + CATEGORICAL
    params = {**LGBM_PARAMS, "random_state": seed}
    rate = np.exp(model.predict(prep(test), raw_score=True))
    importance = pd.Series(model.booster_.feature_importance("gain"), index=cols)
    return rate, rate * exposure(test), {**params, "importance": importance / importance.sum()}
