"""
Wolfe Island wind farm: apply the wind model to turbine layouts.

Everything in one file:
  1. wind climate (GWA wind rose) -> joint probability
  2. power curve + Jensen/Park wake model -> AEP
  3. site mask: feasibility checks, repair, random layouts
  4. fitness function for the EA
  5. evaluate the real layout vs a random layout and plot both

Run wolfe_island_site.py ONCE first so these files exist in this folder:
    wolfe_island_mask.npz, wolfe_island_site.gpkg
Then:  python wolfe_wind_eval.py
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from math import gamma

# =====================================================================
# 1. WIND CLIMATE (Global Wind Atlas, measured from rose images)
#    Replace with exact GWA tooltip values when you have them.
# =====================================================================

sector_dirs_from = np.arange(0, 360, 30)        # wind comes FROM, clockwise from north

freq = np.array([4.1, 7.0, 4.3, 1.1, 0.5, 4.7, 23.4, 12.7, 16.6, 9.7, 7.8, 4.8])        # % time
speed_share = np.array([6.0, 8.1, 7.1, 2.0, 1.1, 4.8, 15.8, 12.8, 14.7, 9.8, 8.9, 6.0])  # % speed
MEAN_SPEED = 7.5                                 # site mean speed at 80 m hub height (m/s)
K_WEIBULL = 2.0

f = freq / freq.sum()
rel = speed_share / freq
u_sector = np.clip(rel / (f * rel).sum() * MEAN_SPEED, 4.0, 11.0)   # clip noisy tiny sectors

speeds = np.arange(0.5, 30, 1.0)                 # 1 m/s speed bins


def weibull_pdf(v, A, k):
    return (k / A) * (v / A) ** (k - 1) * np.exp(-(v / A) ** k)


A_sector = u_sector / gamma(1 + 1 / K_WEIBULL)
PROB = np.array([f[s] * weibull_pdf(speeds, A_sector[s], K_WEIBULL)
                 for s in range(12)])            # (12 sectors, n_speeds)
PROB /= PROB.sum()

SUB_OFFSETS = np.array([-12.5, -7.5, -2.5, 2.5, 7.5, 12.5])   # 72 directions total

# =====================================================================
# 2. TURBINE + WAKE MODEL
# =====================================================================

N_TURBINES = 86
D = 93.0                       # SWT-2.3-93 rotor diameter (m)
R = D / 2


def power_curve(v, P_r=2300.0, v_ci=4.0, v_r=14.0, v_co=25.0):
    """Approximate SWT-2.3-93 power (kW) at hub-height speed v (m/s)."""
    v = np.asarray(v, dtype=float)
    P = P_r * (v**3 - v_ci**3) / (v_r**3 - v_ci**3)
    P = np.where(v < v_ci, 0.0, P)
    P = np.where((v >= v_r) & (v <= v_co), P_r, P)
    P = np.where(v > v_co, 0.0, P)
    return P


def turbine_aep(xy, k=0.075, Ct=0.8):
    """Annual energy (kWh) of EACH turbine. xy: (n, 2) UTM metres, x=east, y=north."""
    a = 1 - np.sqrt(1 - Ct)
    dxy = xy[:, None, :] - xy[None, :, :]        # position of i relative to j
    per_turbine = np.zeros(len(xy))

    for s in range(12):
        for off in SUB_OFFSETS:
            phi = np.radians(sector_dirs_from[s] + off)
            w = np.array([-np.sin(phi), -np.cos(phi)])   # direction wind blows TOWARD
            n = np.array([-w[1], w[0]])                  # crosswind

            x = dxy @ w                                  # downwind distance
            r = np.abs(dxy @ n)                          # crosswind offset
            in_wake = (x > 0) & (r < R + k * x)
            delta = np.where(in_wake, a / (1 + k * np.maximum(x, 0) / R) ** 2, 0.0)
            d_eff = np.sqrt((delta ** 2).sum(axis=1))    # sum-of-squares combination

            U = speeds[:, None] * (1 - d_eff)[None, :]
            per_turbine += (PROB[s][:, None] * power_curve(U)).sum(axis=0) / len(SUB_OFFSETS)

    return per_turbine * 8760


def aep(xy):
    """Total annual energy production (kWh)."""
    return turbine_aep(xy).sum()


def wake_loss(xy):
    """Fraction of energy lost to wakes vs. the same turbines with no interaction."""
    ideal = len(xy) * (PROB * power_curve(speeds)[None, :]).sum() * 8760
    return 1 - aep(xy) / ideal

# =====================================================================
# 3. SITE MASK: feasibility, repair, random layouts
# =====================================================================

MIN_SPACING = 4 * D            # tune: 3-5 D
PENALTY_GWH = 10.0             # per spacing violation

_m = np.load("wolfe_island_mask.npz")
MASK = _m["mask"]
X0, Y0, RES = float(_m["x0"]), float(_m["y0"]), float(_m["res"])
_rows, _cols = np.nonzero(MASK)
FEASIBLE_XY = np.column_stack([X0 + (_cols + 0.5) * RES, Y0 + (_rows + 0.5) * RES])


def in_mask(xy):
    """True for each turbine sitting in an allowed cell."""
    j = np.floor((xy[:, 0] - X0) / RES).astype(int)
    i = np.floor((xy[:, 1] - Y0) / RES).astype(int)
    ok = (i >= 0) & (i < MASK.shape[0]) & (j >= 0) & (j < MASK.shape[1])
    out = np.zeros(len(xy), dtype=bool)
    out[ok] = MASK[i[ok], j[ok]]
    return out


def spacing_violations(xy, min_d=MIN_SPACING):
    """Number of turbine pairs closer than min_d."""
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return int((d < min_d).sum() // 2)


def snap_to_feasible(xy):
    """Move any turbine outside the mask to the nearest allowed cell centre."""
    xy = xy.copy()
    for t in np.nonzero(~in_mask(xy))[0]:
        xy[t] = FEASIBLE_XY[np.argmin(((FEASIBLE_XY - xy[t]) ** 2).sum(axis=1))]
    return xy


def random_layout(rng, n=N_TURBINES, min_d=MIN_SPACING, max_tries=50_000):
    """Uniform random feasible layout respecting minimum spacing."""
    pts = np.empty((0, 2))
    for _ in range(max_tries):
        p = FEASIBLE_XY[rng.integers(len(FEASIBLE_XY))] + rng.uniform(-RES / 2, RES / 2, 2)
        if len(pts) == 0 or np.min(np.hypot(*(pts - p).T)) >= min_d:
            pts = np.vstack([pts, p])
            if len(pts) == n:
                return pts
    raise RuntimeError("Couldn't place all turbines: lower MIN_SPACING or check the mask.")

# =====================================================================
# 4. FITNESS (what the EA calls)
# =====================================================================

def fitness(xy):
    """AEP in GWh/yr minus spacing penalty. Higher is better. Returns (fitness, repaired xy)."""
    xy = snap_to_feasible(xy)
    return aep(xy) / 1e6 - PENALTY_GWH * spacing_violations(xy), xy

# =====================================================================
# 5. EVALUATE + PLOT
# =====================================================================

def load_real_layout(gpkg="wolfe_island_site.gpkg"):
    real = gpd.read_file(gpkg, layer="existing_turbines")
    return np.column_stack([real.geometry.x, real.geometry.y])


def report(name, xy):
    e = turbine_aep(xy)
    print(f"{name:>8}: {len(xy)} turbines | AEP {e.sum()/1e6:6.1f} GWh/yr | "
          f"wake loss {wake_loss(xy):5.1%} | in mask {in_mask(xy).sum()}/{len(xy)} | "
          f"spacing violations {spacing_violations(xy)}")
    return e


def plot_layout(ax, xy, energy, title, gpkg="wolfe_island_site.gpkg"):
    gpd.read_file(gpkg, layer="site").plot(ax=ax, color="#ddd")
    gpd.read_file(gpkg, layer="feasible").plot(ax=ax, color="#9c9", alpha=0.6)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=energy / 1e6, cmap="viridis",
                    s=30, edgecolor="k", linewidth=0.4, zorder=3)
    ax.set_title(title)
    ax.set_axis_off()
    return sc


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    layouts = {}
    try:
        layouts["real"] = load_real_layout()
    except Exception as err:
        print(f"no real layout found ({err}); skipping baseline")
    layouts["random"] = random_layout(rng)

    fig, axes = plt.subplots(1, len(layouts), figsize=(7 * len(layouts), 7))
    axes = np.atleast_1d(axes)
    for ax, (name, xy) in zip(axes, layouts.items()):
        energy = report(name, xy)
        sc = plot_layout(ax, xy, energy, f"{name}: {energy.sum()/1e6:.0f} GWh/yr")
    fig.colorbar(sc, ax=axes, shrink=0.7, label="turbine AEP (GWh/yr)")
    plt.savefig("layout_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()
