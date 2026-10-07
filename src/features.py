"""Build one row per (pipe, planning date) with features known at that date and the 5-year break count.

Usage:
    python src/features.py      # writes data/features/frame_<year>.parquet for every planning date
"""
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

from data import link_breaks_to_pipes, load_breaks, load_pipes

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "features"
PLANNING_YEARS = [1996, 2001, 2006, 2011, 2016, 2021]
HORIZON = 5
RADII_M = [50, 150, 300]
MODES = {"A": "circular", "B": "split", "C": "corrosion", "D": "fitting",
         "E": "joint", "F": "diagonal", "G": "hole", "S": "saddle"}


def usable_breaks(pipes, geoms):
    """ACTIVE breaks linked to a pipe that already existed when it broke (see EDA Part 5)."""
    b = link_breaks_to_pipes(load_breaks(), geoms)
    b["break_year"] = b.break_date.dt.year
    b = b.join(pipes.year.rename("install_year"), on="pipe_idx")
    b = b[(b.STATUS == "ACTIVE") & (b.install_year <= b.break_year)].reset_index(drop=True)
    letters = b.BREAK_TYPE.str.replace(r"\d", "", regex=True)
    for code, name in MODES.items():
        b[f"mode_{name}"] = letters.str.contains(code)
    return b


def neighbour_pairs(breaks, geoms, radius):
    """(pipe, break year) pairs for breaks within `radius` m of a pipe, excluding the pipe's own breaks."""
    tree = STRtree([Point(x, y) for x, y in zip(breaks.x, breaks.y)])
    pipe_rows, break_rows = tree.query([g.buffer(radius) for g in geoms], predicate="intersects")
    pairs = pd.DataFrame({"pipe_idx": pipe_rows, "break_year": breaks.break_year.values[break_rows]})
    return pairs[pairs.pipe_idx != breaks.pipe_idx.values[break_rows]]


def build_frame(year, pipes, breaks, neighbours):
    """Pipes in the ground before 1 Jan `year`, described with information available on that date."""
    f = pipes[(pipes.year < year) & (pipes.length > 0)].copy()
    f["planning_year"] = year
    f["km"] = f.length / 1000
    f["age"] = year - f.year
    f["log_length"] = np.log(f.length.clip(lower=1))

    past = breaks[breaks.break_year < year]
    by_pipe = past.groupby("pipe_idx")
    f["breaks_all"] = f.index.map(by_pipe.size()).fillna(0)
    f["breaks_10y"] = f.index.map(past[past.break_year >= year - 10].groupby("pipe_idx").size()).fillna(0)
    f["breaks_per_km"] = f.breaks_all / f.km
    f["breaks_per_km_10y"] = f.breaks_10y / f.km
    f["years_since_break"] = (year - f.index.map(by_pipe.break_year.max())).fillna(99)

    for name in MODES.values():
        f[f"past_{name}"] = f.index.map(past[past[f"mode_{name}"]].groupby("pipe_idx").size()).fillna(0)

    for radius, pairs in neighbours.items():
        p = pairs[pairs.break_year < year]
        f[f"nearby_{radius}m_10y"] = f.index.map(p[p.break_year >= year - 10].groupby("pipe_idx").size()).fillna(0)
        f[f"nearby_{radius}m_decay"] = f.index.map(np.exp(-(year - p.break_year) / 5).groupby(p.pipe_idx).sum()).fillna(0)

    future = breaks[breaks.break_year.between(year, year + HORIZON - 1)]
    f["target"] = f.index.map(future.groupby("pipe_idx").size()).fillna(0)
    return f.drop(columns=["globalid", "status_ind"]).rename(columns={"year": "install_year", "diam": "diameter"})


def main():
    pipes, geoms = load_pipes()
    breaks = usable_breaks(pipes, geoms)
    neighbours = {r: neighbour_pairs(breaks, geoms, r) for r in RADII_M}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in PLANNING_YEARS:
        frame = build_frame(year, pipes, breaks, neighbours)
        frame.to_parquet(OUT_DIR / f"frame_{year}.parquet")
        print(f"{year}: {len(frame):,} pipes, {int(frame.target.sum()):,} breaks in the next {HORIZON} years")


if __name__ == "__main__":
    main()
