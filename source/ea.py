"""
Evolutionary algorithm for Wolfe Island wind turbine layout.

Design (from the team's outline):
  Genotype        86 turbines as continuous (x, y) UTM coordinates -> array (86, 2)
  Fitness         delivered AEP minus installation, spacing and habitat penalties
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
  python source/ea.py                         # default run
  python source/ea.py --mutation cauchy --seed 3 --evals 5000
"""

import argparse
import csv
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np

try:
    import site_tools as st
except FileNotFoundError as exc:
    if __name__ == "__main__":
        raise SystemExit(str(exc)) from None
    raise
from wind_model import aep, turbine_aep, wake_loss
from fitness_penalties import PenaltyConfig, evaluate_layout


@dataclass
class Config(PenaltyConfig):
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
def fitness(xy, cable_weight=None, *, cfg=None):
    """Penalized fitness (GWh/year); reject invalid or out-of-mask layouts.

    The optional positional cable_weight preserves the old GWh/km override.
    Pass cfg to use the same penalty settings as an EA run.
    """
    cfg = Config() if cfg is None else cfg
    if cable_weight is not None:
        cfg = replace(cfg, cable_weight=cable_weight)
    try:
        xy = np.asarray(xy, dtype=float)
    except (TypeError, ValueError):
        return -np.inf
    if (xy.shape != (st.N_TURBINES, 2) or not np.isfinite(xy).all()
            or not st.in_mask(xy).all()):
        return -np.inf
    return evaluate_layout(xy, cfg).fitness_gwh


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

    if verbose:
        print(f"Initializing {cfg.mu} layouts...", flush=True)
    pop = [st.random_layout(rng) for _ in range(cfg.mu)]
    fits = np.array([fitness(x, cfg=cfg) for x in pop])
    evals = cfg.mu
    b = pop[int(np.argmax(fits))]
    history = [(0, evals, fits.max(), fits.mean(), aep(b) / 1e6, st.cable_length(b) / 1000)]
    gen = 0
    if verbose:
        print(f"Starting optimization | evaluations {evals}/{cfg.evals} | "
              f"best fitness {fits.max():.2f}", flush=True)

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
            child_fits.append(fitness(child, cfg=cfg))
        evals += cfg.lam

        # (mu + lambda) survivor selection
        all_pop = pop + children
        all_fits = np.concatenate([fits, child_fits])
        keep = np.argsort(all_fits)[::-1][: cfg.mu]
        pop = [all_pop[i] for i in keep]
        fits = all_fits[keep]

        b = pop[0]
        best_aep, best_km = aep(b) / 1e6, st.cable_length(b) / 1000
        history.append((gen, evals, fits.max(), fits.mean(), best_aep, best_km))
        if verbose:
            print(f"Generation {gen} | evaluations {evals}/{cfg.evals} | "
                  f"best fitness {fits.max():.2f} | {time.time() - t0:.0f}s", flush=True)

    best = pop[int(np.argmax(fits))]
    return best, float(fits.max()), history


def save_run(cfg, best, best_fit, history, out_root=None):
    if out_root is None:
        out_root = st.DATA_DIR / "results"
    out = Path(out_root) / f"{cfg.mutation}_seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "best_layout.npy", best)
    with open(out / "history.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["generation", "evals", "best_fitness", "mean_fitness", "best_aep_gwh", "best_cable_km"])
        w.writerows(history)
    with open(out / "config.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerows(list(asdict(cfg).items()) + [("best_fitness", best_fit)])
    with open(out / "fitness_breakdown.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["metric", "value"])
        w.writerows(asdict(evaluate_layout(best, cfg)).items())
    return out


def plot_run(best, history, out, cfg=None):
    import matplotlib.pyplot as plt
    import geopandas as gpd

    real = st.load_real_layout()
    layouts = {"optimized": best} if real is None else {"real (2009)": real, "optimized": best}

    fig, axes = plt.subplots(1, len(layouts) + 1,
                             figsize=(6 * (len(layouts) + 1), 6),
                             layout="constrained")
    energies = {name: turbine_aep(xy) / 1e6 for name, xy in layouts.items()}
    norm = plt.Normalize(vmin=min(e.min() for e in energies.values()),
                         vmax=max(e.max() for e in energies.values()))
    gpkg = st.DATA_DIR / "wolfe_island_site.gpkg"
    for ax, (name, xy) in zip(axes, layouts.items()):
        try:
            gpd.read_file(gpkg, layer="site").plot(ax=ax, color="#ddd")
            gpd.read_file(gpkg, layer="feasible").plot(ax=ax, color="#9c9", alpha=0.6)
            gpd.read_file(gpkg, layer="roads").plot(ax=ax, color="#555", linewidth=0.5)

        except Exception:
            pass
        cable_m, edges = st.cable_tree(xy)
        n = len(xy)
        for a,b in edges:
            if b < n:
                ax.plot(*xy[[a, b]].T, color="orange", lw=0.8, zorder=2)
            else:
                ax.plot(*np.vstack([xy[a], st.COLLECTOR_XY]).T,
                        color="orange", lw=0.8, zorder=2)
        ax.scatter(*st.COLLECTOR_XY, marker="s", c="orange", s=60, zorder=4)
        e = energies[name]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=e, cmap="viridis", norm=norm, s=28,
                        edgecolor="k", linewidth=0.4, zorder=3)
        ax.set_title(f"{name}\n{e.sum():.0f} GWh/yr | wake loss {wake_loss(xy):.1%}\n"
                     f"cable {cable_m / 1000:.1f} km", fontsize=11)
        ax.set_axis_off()
    fig.colorbar(sc, ax=axes[:-1], shrink=0.7, label="turbine AEP (GWh/yr)")

    h = np.array(history)
    axes[-1].plot(h[:, 1], h[:, 2], label="best")
    axes[-1].plot(h[:, 1], h[:, 3], label="population mean")
    if real is not None:
        real_fitness = fitness(real, cfg=cfg)
        if np.isfinite(real_fitness):
            axes[-1].axhline(real_fitness, color="grey", ls="--", label="real layout")
    axes[-1].set_xlabel("fitness evaluations")
    axes[-1].set_ylabel("fitness (GWh/yr equivalent)")
    axes[-1].set_title("Fitness improvement")
    axes[-1].legend()
    plt.savefig(out / "summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for field, default in asdict(Config()).items():
        ap.add_argument(f"--{field.replace('_', '-')}",
                        type=float if default is None else type(default), default=default)
    cfg = Config(**vars(ap.parse_args()))

    real = st.load_real_layout()
    if real is not None:
        print(f"real layout: {len(real)} turbines, {aep(real) / 1e6:.1f} GWh/yr, "
              f"wake loss {wake_loss(real):.1%}, in mask {st.in_mask(real).sum()}/{len(real)}, "
              f"spacing violations {st.spacing_violations(real)}, "
              f"cable {st.cable_length(real) / 1000:.1f} km")

    best, best_fit, history = run(cfg)
    out = save_run(cfg, best, best_fit, history)
    plot_run(best, history, out, cfg)
    print(f"best fitness {best_fit:.2f} GWh/yr -> saved to {out}/")
