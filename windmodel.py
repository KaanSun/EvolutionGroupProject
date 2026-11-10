import numpy as np
from math import gamma

# ---------- 1. Wind climate (from Global Wind Atlas, hub height) ----------

sector_dirs_from = np.arange(0, 360, 30)          # 12 sectors: 0, 30, ..., 330 (direction wind comes FROM)

freq = np.array([4.1, 7.0, 4.3, 1.1, 0.5, 4.7, 23.4, 12.7, 16.6, 9.7, 7.8, 4.8])
#               N    30   60   E    120  150  S     210   240   W    300  330                          # frequency rose, % per sector (12 values)
speed_share = np.array([6.0, 8.1, 7.1, 2.0, 1.1, 4.8, 15.8, 12.8, 14.7, 9.8, 8.9, 6.0])
#                       N    30   60   E    120  150  S     210   240   W    300  330                 # speed rose, % per sector (12 values)
mean_speed = 7.5                                   # site mean wind speed at hub height (m/s), from GWA
k_weibull = 2.0                                    # Weibull shape, from GWA (≈2 is typical)

f = freq / freq.sum()                              # sector probabilities, sum to 1
rel = (speed_share / freq)                         # relative mean speed per sector
u_sector = rel / (f * rel).sum() * mean_speed      # scale so the overall mean equals mean_speed

# ---------- 2. Joint probability p(direction, speed) ----------

speeds = np.arange(0.5, 30, 1.0)                   # speed bin centres (m/s), 1 m/s wide

def weibull_pdf(v, A, k):
    return (k / A) * (v / A) ** (k - 1) * np.exp(-(v / A) ** k)

A_sector = u_sector / gamma(1 + 1 / k_weibull)     # Weibull scale from each sector's mean speed
prob = np.array([f[s] * weibull_pdf(speeds, A_sector[s], k_weibull) * 1.0   # × bin width
                 for s in range(len(f))])          # shape (12, n_speeds)
prob /= prob.sum()                                 # normalize to exactly 1

# ---------- 3. Power curve ----------

def power_curve(v, P_r=2300.0, v_ci=4.0, v_r=14.0, v_co=25.0):
    """Approximate SWT-2.3-93 power (kW) at hub-height wind speed v (m/s)."""
    v = np.asarray(v, dtype=float)
    P = P_r * (v**3 - v_ci**3) / (v_r**3 - v_ci**3)
    P = np.where(v < v_ci, 0.0, P)
    P = np.where((v >= v_r) & (v <= v_co), P_r, P)
    P = np.where(v > v_co, 0.0, P)
    return P

# ---------- 4. Jensen/Park wake model + AEP ----------

def aep(xy, k=0.075, R=46.5, Ct=0.8):
    """Annual energy production (kWh) of a layout xy: (n_turbines, 2) in metres, x = east, y = north."""
    a = 1 - np.sqrt(1 - Ct)
    dxy = xy[:, None, :] - xy[None, :, :]          # position of turbine i relative to turbine j
    total = 0.0

    for s, phi in enumerate(np.radians(sector_dirs_from)):
        w = np.array([-np.sin(phi), -np.cos(phi)]) # direction the wind blows TOWARD
        n = np.array([-w[1], w[0]])                # crosswind direction

        x = dxy @ w                                # downwind distance of i behind j
        r = np.abs(dxy @ n)                        # crosswind offset
        in_wake = (x > 0) & (r < R + k * x)
        delta = np.where(in_wake, a / (1 + k * np.maximum(x, 0) / R) ** 2, 0.0)
        d_eff = np.sqrt((delta ** 2).sum(axis=1))  # combine wakes: sum of squares

        U = speeds[:, None] * (1 - d_eff)[None, :] # every speed bin × every turbine
        total += (prob[s][:, None] * power_curve(U)).sum()

    return total * 8760                            # kWh per year 