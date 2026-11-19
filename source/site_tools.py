"""
Site constraints for the EA, built on the outputs of wolfe_island_site.py:
    wolfe_island_mask.npz   feasibility raster (setbacks from houses, roads, shore)
    wolfe_island_site.gpkg  map layers + the real 86-turbine layout

Run `python source/wolfe_island_site.py` once before using this module.
"""

from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from pyproj import Transformer

from wind_model import ROTOR_D

DATA_DIR = Path(__file__).resolve().parent.parent
N_TURBINES = 86
MIN_SPACING = 4 * ROTOR_D          # ~372 m; tune between 3D and 5D
# Approximate island collector from FITNESS_PENALTIES.md, projected to metres.
# A fixed root makes every turbine's cable path reach the same destination.
COLLECTOR_XY = np.array(Transformer.from_crs(
    "EPSG:4326", "EPSG:32618", always_xy=True).transform(-76.4725, 44.1790))

try:
    _m = np.load(DATA_DIR / "wolfe_island_mask.npz")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Site mask has not been generated: {DATA_DIR / 'wolfe_island_mask.npz'}\n"
        "From the project root, run `python source/wolfe_island_site.py` once, "
        "then rerun the EA. The site builder needs internet access to download map data."
    ) from None
MASK = _m["mask"]
X0, Y0, RES = float(_m["x0"]), float(_m["y0"]), float(_m["res"])
_rows, _cols = np.nonzero(MASK)
FEASIBLE_XY = np.column_stack([X0 + (_cols + 0.5) * RES, Y0 + (_rows + 0.5) * RES])
ROAD_DIST = _m["road_dist"]
HABITAT_DIST = _m["habitat_dist"] if "habitat_dist" in _m else None


def in_mask(xy):
    """True for each turbine inside an allowed cell."""
    j = np.floor((xy[:, 0] - X0) / RES).astype(int)
    i = np.floor((xy[:, 1] - Y0) / RES).astype(int)
    ok = (i >= 0) & (i < MASK.shape[0]) & (j >= 0) & (j < MASK.shape[1])
    out = np.zeros(len(xy), dtype=bool)
    out[ok] = MASK[i[ok], j[ok]]
    return out


def pair_distances(xy):
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d


def spacing_violations(xy, min_d=MIN_SPACING):
    """Number of turbine pairs closer than min_d."""
    return int((pair_distances(xy) < min_d).sum() // 2)


def random_feasible_points(rng, m):
    """m uniform random points inside the feasible area."""
    idx = rng.integers(len(FEASIBLE_XY), size=m)
    return FEASIBLE_XY[idx] + rng.uniform(-RES / 2, RES / 2, size=(m, 2))

def road_distance(xy):
    """Distance (m) from each turbine to the nearest main road (grid lookup)"""
    j = np.clip(np.floor((xy[:, 0] - X0) / RES).astype(int), 0, ROAD_DIST.shape[1] - 1)
    i = np.clip(np.floor((xy[:, 1] - Y0) / RES).astype(int), 0, ROAD_DIST.shape[0] - 1)
    return ROAD_DIST[i, j]


def habitat_distance(xy):
    """Nearest wetland distance (m), from the site builder's 25 m raster.

    Infinity means the supplied site contains no mapped wetlands. Missing data
    is an error rather than silently treating habitat penalties as zero.
    """
    if HABITAT_DIST is None:
        raise RuntimeError("Habitat distances are missing. Rebuild the site with "
                           "`python source/wolfe_island_site.py`, or set --habitat-weight 0.")
    if not in_mask(xy).all():
        raise ValueError("Habitat lookup requires turbines inside the feasible mask.")
    j = np.floor((xy[:, 0] - X0) / RES).astype(int)
    i = np.floor((xy[:, 1] - Y0) / RES).astype(int)
    return HABITAT_DIST[i, j]


def cable_tree(xy):
    """
    Straight-line minimum spanning tree over turbines and the fixed collector.
    Each shared segment is counted once. This is a routing estimate; it does
    not enforce terrain restrictions or cable capacity on the routes.
    Returns (total length in m, edges); node n is the collector.
    """
    n = len(xy)
    points = np.vstack([xy, COLLECTOR_XY])
    G = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    G = np.maximum(G, 1e-6)
    np.fill_diagonal(G, 0)
    T = minimum_spanning_tree(G).tocoo()
    edges = [(min(a, b), max(a, b)) for a, b in zip(T.row, T.col)]
    return float(T.data.sum()), edges

def cable_length(xy):
    """Total cable length (m) of the turbine + collector spanning tree."""
    return cable_tree(xy)[0]

def snap_to_feasible(xy):
    """Move any turbine outside the mask to the nearest allowed cell centre."""
    xy = xy.copy()
    for t in np.nonzero(~in_mask(xy))[0]:
        xy[t] = FEASIBLE_XY[np.argmin(((FEASIBLE_XY - xy[t]) ** 2).sum(axis=1))]
    return xy


def relocate(xy, t, rng, min_d=MIN_SPACING, tries=200):
    """Move turbine t to a random feasible spot that respects spacing (if one is found)."""
    others = np.delete(xy, t, axis=0)
    cand = random_feasible_points(rng, tries)
    dmin = np.min(np.linalg.norm(cand[:, None, :] - others[None, :, :], axis=-1), axis=1)
    good = np.nonzero(dmin >= min_d)[0]
    xy[t] = cand[good[0]] if len(good) else cand[np.argmax(dmin)]
    return xy


def repair(xy, rng, min_d=MIN_SPACING, passes=3):
    """
    Make a layout feasible:
      1. snap turbines outside the mask to the nearest allowed cell
      2. for pairs closer than min_d, nudge one turbine away; if the nudge
         leaves the mask, relocate it to a random feasible spot
    Leftover violations are handled by the fitness penalty.
    """
    xy = snap_to_feasible(np.asarray(xy, dtype=float))
    for _ in range(passes):
        d = pair_distances(xy)
        bad_i, bad_j = np.nonzero(np.triu(d < min_d))
        if len(bad_i) == 0:
            break
        for i, j in zip(bad_i, bad_j):
            dij = np.linalg.norm(xy[j] - xy[i])
            if dij >= min_d:                       # already fixed earlier this pass
                continue
            direction = (xy[j] - xy[i]) / dij if dij > 0 else rng.normal(size=2)
            direction /= np.linalg.norm(direction)
            moved = xy[j] + direction * (min_d - dij + 5.0)
            if in_mask(moved[None, :])[0]:
                xy[j] = moved
            else:
                xy = relocate(xy, j, rng, min_d)
    return xy


def random_layout(rng, n=N_TURBINES, min_d=MIN_SPACING, batch=500, max_rounds=200):
    """Uniform random feasible layout that respects minimum spacing."""
    pts = np.empty((0, 2))
    for _ in range(max_rounds):
        for p in random_feasible_points(rng, batch):
            if len(pts) == 0 or np.min(np.hypot(*(pts - p).T)) >= min_d:
                pts = np.vstack([pts, p])
                if len(pts) == n:
                    return pts
    raise RuntimeError("Couldn't place all turbines: lower MIN_SPACING or check the mask.")


def load_real_layout():
    """The existing 86-turbine Wolfe Island layout from OSM (UTM metres), or None."""
    try:
        import geopandas as gpd
        real = gpd.read_file(DATA_DIR / "wolfe_island_site.gpkg", layer="existing_turbines")
        return np.column_stack([real.geometry.x, real.geometry.y])
    except Exception:
        return None
