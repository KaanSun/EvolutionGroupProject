"""
Wind model for Wolfe Island.

Data (Global Wind Atlas exports in this repo):
  windFrequencyRose.csv        share of time the wind comes FROM each 30-degree sector
  windSpeedRose.csv            each sector's share of total wind speed
  area_1_wind-speed_100m.tif   mean wind speed map at 100 m (spatial variation over the site)

Model:
  - local mean speed at each turbine read from the GeoTIFF, scaled from 100 m to hub height
  - per-sector Weibull distributions (shape k) built from the two roses
  - SWT-2.3-93 power curve
  - Jensen/Park wake model, sum-of-squares wake combination
  - 72 wind directions (each 30-degree sector split into 6 sub-directions)

Layouts are numpy arrays of shape (n, 2) in UTM 18N metres: x = east, y = north.
"""

from math import gamma
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

DATA_DIR = Path(__file__).parent

# ------------------------------------------------------------------ turbine
ROTOR_D = 93.0                 # Siemens SWT-2.3-93
ROTOR_R = ROTOR_D / 2
HUB_HEIGHT = 80.0
RATED_KW = 2300.0

# ------------------------------------------------------------------ assumptions
SHEAR_ALPHA = 0.14             # power-law shear exponent, 100 m -> hub height (assumed)
K_WEIBULL = 2.0                # Weibull shape (assumed; GWA typical ~2)
WAKE_K = 0.075                 # Jensen wake decay constant (onshore)
CT = 0.8                       # thrust coefficient (constant simplification)
SUB_OFFSETS = np.array([-12.5, -7.5, -2.5, 2.5, 7.5, 12.5])   # degrees within each sector

# ------------------------------------------------------------------ wind roses
_freq = pd.read_csv(DATA_DIR / "windFrequencyRose.csv").sort_values("sector")
_speed = pd.read_csv(DATA_DIR / "windSpeedRose.csv").sort_values("sector")

SECTOR_DIRS = _freq["center_degrees"].to_numpy(dtype=float)     # direction wind comes FROM
FREQ = _freq["value"].to_numpy(dtype=float)
FREQ = FREQ / FREQ.sum()                                        # P(sector)

_rel = _speed["value"].to_numpy(dtype=float) / _freq["value"].to_numpy(dtype=float)
REL_SECTOR_SPEED = _rel / (FREQ * _rel).sum()                   # sector mean / overall mean (avg = 1)

# Speeds are handled in NORMALIZED units (multiples of the local mean speed),
# so one probability table works for every turbine even though the mean
# speed varies across the site.
VNORM = np.arange(0.025, 4.0, 0.05)                             # bin centres, width 0.05


def _weibull_pdf(v, A, k):
    return (k / A) * (v / A) ** (k - 1) * np.exp(-(v / A) ** k)


_A = REL_SECTOR_SPEED / gamma(1 + 1 / K_WEIBULL)
PROB = np.array([FREQ[s] * _weibull_pdf(VNORM, _A[s], K_WEIBULL) * 0.05
                 for s in range(len(FREQ))])                    # (12, n_bins)
PROB /= PROB.sum()

# ------------------------------------------------------------------ speed map
with rasterio.open(DATA_DIR / "area_1_wind-speed_100m.tif") as _src:
    _SPEED_100 = _src.read(1).astype(float)
    _TIF_TRANSFORM = _src.transform
    _TIF_CRS = _src.crs

_to_tif = Transformer.from_crs("EPSG:32618", _TIF_CRS, always_xy=True)
_SHEAR = (HUB_HEIGHT / 100.0) ** SHEAR_ALPHA
_FALLBACK = float(np.nanmean(_SPEED_100))


def local_mean_speed(xy):
    """Mean hub-height wind speed (m/s) at each turbine position (nearest GeoTIFF cell)."""
    lon, lat = _to_tif.transform(xy[:, 0], xy[:, 1])
    col, row = ~_TIF_TRANSFORM * (np.asarray(lon), np.asarray(lat))
    row = np.clip(np.floor(row).astype(int), 0, _SPEED_100.shape[0] - 1)
    col = np.clip(np.floor(col).astype(int), 0, _SPEED_100.shape[1] - 1)
    v = _SPEED_100[row, col]
    v = np.where(np.isfinite(v), v, _FALLBACK)
    return v * _SHEAR

# ------------------------------------------------------------------ power curve
def power_curve(v, v_ci=4.0, v_co=25.0, cp=0.44, rho=1.225):
    """
    Approximate SWT-2.3-93 power (kW): aerodynamic power 0.5*rho*A*Cp*v^3,
    capped at rated power, zero outside cut-in/cut-out. Cp ~0.44 gives roughly
    the published curve shape (~0.9 MW at 8 m/s, rated near 11-13 m/s).
    """
    v = np.asarray(v, dtype=float)
    area = np.pi * ROTOR_R ** 2
    P = 0.5 * rho * area * cp * v ** 3 / 1000.0
    P = np.minimum(P, RATED_KW)
    P = np.where((v < v_ci) | (v > v_co), 0.0, P)
    return P

# ------------------------------------------------------------------ wake + AEP
def turbine_aep(xy, wakes=True):
    """Annual energy (kWh) of EACH turbine in layout xy -> array of length n."""
    xy = np.asarray(xy, dtype=float)
    u_local = local_mean_speed(xy)                         # (n,)
    a = 1 - np.sqrt(1 - CT)
    dxy = xy[:, None, :] - xy[None, :, :]                  # position of i relative to j
    per_turbine = np.zeros(len(xy))

    for s in range(len(SECTOR_DIRS)):
        for off in SUB_OFFSETS:
            if wakes:
                phi = np.radians(SECTOR_DIRS[s] + off)
                w = np.array([-np.sin(phi), -np.cos(phi)])     # direction wind blows TOWARD
                n = np.array([-w[1], w[0]])                    # crosswind
                x = dxy @ w                                    # downwind distance of i behind j
                r = np.abs(dxy @ n)                            # crosswind offset
                in_wake = (x > 0) & (r < ROTOR_R + WAKE_K * x)
                delta = np.where(in_wake,
                                 a / (1 + WAKE_K * np.maximum(x, 0) / ROTOR_R) ** 2, 0.0)
                d_eff = np.sqrt((delta ** 2).sum(axis=1))      # sum of squares
            else:
                d_eff = np.zeros(len(xy))

            U = VNORM[:, None] * (u_local * (1 - d_eff))[None, :]   # (bins, n)
            per_turbine += (PROB[s][:, None] * power_curve(U)).sum(axis=0) / len(SUB_OFFSETS)

    return per_turbine * 8760


def aep(xy):
    """Total annual energy production (kWh)."""
    return turbine_aep(xy).sum()


def wake_loss(xy):
    """Fraction of energy lost to wakes vs. the same turbines with no interaction."""
    return 1 - aep(xy) / turbine_aep(xy, wakes=False).sum()


if __name__ == "__main__":
    print("sector  P(from)  rel. speed")
    for d, p, r in zip(SECTOR_DIRS, FREQ, REL_SECTOR_SPEED):
        print(f"{d:6.0f}  {p:7.3f}  {r:9.2f}")
    print(f"speed map at 100 m: {np.nanmin(_SPEED_100):.2f}-{np.nanmax(_SPEED_100):.2f} m/s, "
          f"hub-height factor {_SHEAR:.3f}")
