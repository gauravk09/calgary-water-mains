"""Leakage test: adding breaks in the future must not change the features for a planning date."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import MODES, RADII_M, build_frame  # noqa: E402

PLANNING_YEAR = 2016


def tiny_network():
    pipes = pd.DataFrame({"year": [1950, 1975, 2001], "length": [120.0, 80.0, 60.0],
                          "material": ["CI", "PDI", "PVC"], "p_zone": ["A", "A", "B"], "diam": [150, 200, 200],
                          "globalid": ["g0", "g1", "g2"], "status_ind": ["ACTIVE"] * 3})
    breaks = pd.DataFrame({"pipe_idx": [0, 0, 1], "break_year": [1990, 2012, 2014]})
    for name in MODES.values():
        breaks[f"mode_{name}"] = name == "corrosion"
    neighbours = {r: pd.DataFrame({"pipe_idx": [1, 2], "break_year": [2012, 2014]}) for r in RADII_M}
    return pipes, breaks, neighbours


def add_future_breaks(breaks, neighbours):
    """Breaks on every pipe in and after the planning year — information a planner can't have on 1 January."""
    future = pd.DataFrame({"pipe_idx": [0, 1, 2, 2], "break_year": [PLANNING_YEAR, 2017, 2018, 2020]})
    for name in MODES.values():
        future[f"mode_{name}"] = True
    near = {r: pd.concat([n, future[["pipe_idx", "break_year"]]], ignore_index=True) for r, n in neighbours.items()}
    return pd.concat([breaks, future], ignore_index=True), near


def test_future_breaks_do_not_change_features():
    pipes, breaks, neighbours = tiny_network()
    before = build_frame(PLANNING_YEAR, pipes, breaks, neighbours, horizon=1)
    after = build_frame(PLANNING_YEAR, pipes, *add_future_breaks(breaks, neighbours), horizon=1)

    features = [c for c in before.columns if c != "target"]
    pd.testing.assert_frame_equal(before[features], after[features])
    # Sanity check that the test can fail: the future breaks do reach the label.
    assert after.target.sum() > before.target.sum()
