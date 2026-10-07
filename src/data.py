"""Load the Calgary break and pipe tables and link each break to its pipe."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Degrees -> metres around Calgary (latitude ~51 N). An equirectangular approximation is
# accurate to well under 1% over the city's ~40 km extent, which is plenty for nearest-pipe matching.
M_PER_DEG_LNG = np.cos(np.radians(51.04)) * 111_320
M_PER_DEG_LAT = 110_540


def _to_metres(geom):
    return shapely.transform(geom, lambda c: c * [M_PER_DEG_LNG, M_PER_DEG_LAT])


def load_pipes():
    """One row per pipe segment in the ground today, plus its geometry in metres."""
    features = json.loads((RAW_DIR / "pipes.geojson").read_text())["features"]
    pipes = pd.DataFrame([f["properties"] for f in features])
    for col in ["year", "length", "diam"]:
        pipes[col] = pd.to_numeric(pipes[col], errors="coerce")
    geoms = [_to_metres(shape(f["geometry"])) for f in features]
    return pipes, geoms


def load_breaks():
    """One row per recorded break, with its location in metres."""
    breaks = pd.read_csv(RAW_DIR / "breaks.csv")
    breaks["break_date"] = pd.to_datetime(breaks.BREAK_DATE, format="%Y/%m/%d")
    lng_lat = breaks.point.str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    breaks["x"] = lng_lat[0] * M_PER_DEG_LNG
    breaks["y"] = lng_lat[1] * M_PER_DEG_LAT
    return breaks


def link_breaks_to_pipes(breaks, geoms):
    """Attach each break to its nearest pipe; returns a copy with `pipe_idx` and `link_dist_m`."""
    tree = STRtree(geoms)
    points = [Point(x, y) for x, y in zip(breaks.x, breaks.y)]
    (break_rows, pipe_rows), dist = tree.query_nearest(points, return_distance=True, all_matches=False)
    out = breaks.copy()
    out.loc[break_rows, "pipe_idx"] = pipe_rows
    out.loc[break_rows, "link_dist_m"] = dist
    return out
