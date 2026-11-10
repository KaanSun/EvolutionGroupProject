"""
Evolutionary algorithm for Wolfe Island wind turbine layout.

Design (from the team's outline):
  Genotype        86 turbines as continuous (x, y) UTM coordinates -> array (86, 2)
  Fitness         AEP (GWh/yr) from the Jensen wake model, minus a spacing penalty
  Initialization  uniform random feasible layouts respecting minimum spacing
  Parent select.  tournament, size k
  Recombination   sorted one-point spatial crossover: pick x or y (50/50), sort both
                  parents along it, take the first k turbines from parent 1 and the
                  last 86-k from parent 2, with k/86 ~ N(0.5, sd) -> always 86 turbines
  Mutation        "scheduled": P(explore) decays with progress
                      explore -> random relocation of a few turbines
                      exploit -> small Gaussian moves, sigma shrinking over the run
                  "cauchy": single heavy-tailed operator, no schedule (comparison)
  Repair          snap into the feasible mask + fix spacing at the crossover seam
  Replacement     (mu + lambda): parents and offspring compete, best mu survive
  Termination     fixed budget of fitness evaluations

Usage:
  python ea.py                         # default run
  python ea.py --mutation cauchy --seed 3 --evals 5000
"""

import argparse
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

import site_tools as st
from wind_model import aep, turbine_aep, wake_loss

PENALTY_GWH = 10.0          # per spacing violation (~1.5x one turbine's annual output)


@dataclass
class Config:
    mu: int = 50                    # population size
    lam: int = 100                  # offspring per generation
    evals: int = 20_000             # fitness evaluation budget
    tournament_k: int = 3
    p_crossover: float = 0.9
    split_sd: float = 0.12          # sd of the crossover split fraction around 0.5
    mutation: str = "scheduled"     # "scheduled" or "cauchy"
    p_explore0: float = 0.4         # initial probability of an exploration mutation
    relocate_max: int = 4           # turbines relocated per exploration (early in run)
    sigma0: float = 200.0           # Gaussian step size at start (m)
    sigma_min: float = 20.0         # Gaussian step size at end (m)
    p_turbine: float = 2 / 86       # per-turbine mutation probability (exploitation)
    cauchy_scale: float = 60.0      # Cauchy scale (m) for the "cauchy" mode
    seed: int = 0


# ------------------------------------------------------------------ fitness
def fitness(xy):
    """AEP in GWh/yr minus spacing penalty. Assumes xy is already repaired."""
    return aep(xy) / 1e6 - PENALTY_GWH * st.spacing_violations(xy)


# ------------------------------------------------------------------ operators
def tournament(fits, k, rng):
    """Index of the fittest of k randomly drawn individuals."""
    contenders = rng.integers(len(fits), size=k)
    return contenders[np.argmax(fits[contenders])]


def sorted_crossover(p1, p2, rng, sd):
    """Spatial one-point crossover that always yields exactly 86 turbines."""
    axis = rng.integers(2)                              # 0 = cut along x, 1 = along y
    a = p1[np.argsort(p1[:, axis])]
    b = p2[np.argsort(p2[:, axis])]
    n = len(a)
    frac = np.clip(rng.normal(0.5, sd), 0.1, 0.9)
    k = int(round(frac * n))
    return np.vstack([a[:k], b[k:]])


def mutate(xy, progress, cfg, rng):
    """progress in [0, 1]: fraction of the evaluation budget used so far."""
    xy = xy.copy()
    n = len(xy)

    if cfg.mutation == "cauchy":
        hit = rng.random(n) < cfg.p_turbine
        if not hit.any():
            hit[rng.integers(n)] = True
        xy[hit] += cfg.cauchy_scale * rng.standard_cauchy(size=(hit.sum(), 2))
        return xy

    # scheduled exploration vs exploitation
    p_explore = cfg.p_explore0 * (1 - progress)
    if rng.random() < p_explore:
        m = max(1, int(round(cfg.relocate_max * (1 - progress))))
        for t in rng.choice(n, size=m, replace=False):
            xy[t] = st.random_feasible_points(rng, 1)[0]
    else:
        sigma = cfg.sigma_min + (cfg.sigma0 - cfg.sigma_min) * (1 - progress)
        hit = rng.random(n) < cfg.p_turbine
        if not hit.any():
            hit[rng.integers(n)] = True
        xy[hit] += rng.normal(0, sigma, size=(hit.sum(), 2))
    return xy


# ------------------------------------------------------------------ main loop
def run(cfg: Config, verbose=True):
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()

    pop = [st.random_layout(rng) for _ in range(cfg.mu)]
    fits = np.array([fitness(x) for x in pop])
    evals = cfg.mu
    history = [(0, evals, fits.max(), fits.mean())]
    gen = 0

    while evals + cfg.lam <= cfg.evals:
        gen += 1
        progress = evals / cfg.evals
        children, child_fits = [], []

        for _ in range(cfg.lam):
            p1 = pop[tournament(fits, cfg.tournament_k, rng)]
            if rng.random() < cfg.p_crossover:
                p2 = pop[tournament(fits, cfg.tournament_k, rng)]
                child = sorted_crossover(p1, p2, rng, cfg.split_sd)
            else:
                child = p1.copy()
            child = mutate(child, progress, cfg, rng)
            child = st.repair(child, rng)
            children.append(child)
            child_fits.append(fitness(child))
        evals += cfg.lam

        # (mu + lambda) survivor selection
        all_pop = pop + children
        all_fits = np.concatenate([fits, child_fits])
        keep = np.argsort(all_fits)[::-1][: cfg.mu]
        pop = [all_pop[i] for i in keep]
        fits = all_fits[keep]

        history.append((gen, evals, fits.max(), fits.mean()))
        if verbose and gen % 10 == 0:
            print(f"gen {gen:4d}  evals {evals:6d}  best {fits.max():7.2f}  "
                  f"mean {fits.mean():7.2f} GWh/yr  ({time.time() - t0:.0f}s)")

    best = pop[int(np.argmax(fits))]
    return best, float(fits.max()), history


def save_run(cfg, best, best_fit, history, out_root="results"):
    out = Path(out_root) / f"{cfg.mutation}_seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "best_layout.npy", best)
    with open(out / "history.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["generation", "evals", "best_fitness", "mean_fitness"])
        w.writerows(history)
    with open(out / "config.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerows(list(asdict(cfg).items()) + [("best_fitness", best_fit)])
    return out


def plot_run(best, history, out):
    import matplotlib.pyplot as plt
    import geopandas as gpd

    real = st.load_real_layout()
    layouts = {"optimized": best} if real is None else {"real (2009)": real, "optimized": best}

    fig, axes = plt.subplots(1, len(layouts) + 1, figsize=(6 * (len(layouts) + 1), 6))
    gpkg = st.DATA_DIR / "wolfe_island_site.gpkg"
    for ax, (name, xy) in zip(axes, layouts.items()):
        try:
            gpd.read_file(gpkg, layer="site").plot(ax=ax, color="#ddd")
            gpd.read_file(gpkg, layer="feasible").plot(ax=ax, color="#9c9", alpha=0.6)
        except Exception:
            pass
        e = turbine_aep(xy) / 1e6
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=e, cmap="viridis", s=28,
                        edgecolor="k", linewidth=0.4, zorder=3)
        ax.set_title(f"{name}: {e.sum():.0f} GWh/yr, wake loss {wake_loss(xy):.1%}")
        ax.set_axis_off()
    fig.colorbar(sc, ax=axes[:-1], shrink=0.7, label="turbine AEP (GWh/yr)")

    h = np.array(history)
    axes[-1].plot(h[:, 1], h[:, 2], label="best")
    axes[-1].plot(h[:, 1], h[:, 3], label="population mean")
    if real is not None:
        axes[-1].axhline(fitness(real), color="grey", ls="--", label="real layout")
    axes[-1].set_xlabel("fitness evaluations")
    axes[-1].set_ylabel("fitness (GWh/yr)")
    axes[-1].legend()
    plt.savefig(out / "summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for field, default in asdict(Config()).items():
        ap.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    cfg = Config(**vars(ap.parse_args()))

    real = st.load_real_layout()
    if real is not None:
        print(f"real layout: {len(real)} turbines, {aep(real) / 1e6:.1f} GWh/yr, "
              f"wake loss {wake_loss(real):.1%}, in mask {st.in_mask(real).sum()}/{len(real)}, "
              f"spacing violations {st.spacing_violations(real)}")

    best, best_fit, history = run(cfg)
    out = save_run(cfg, best, best_fit, history)
    plot_run(best, history, out)
    print(f"best fitness {best_fit:.2f} GWh/yr -> saved to {out}/")
