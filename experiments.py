"""
Run each EA configuration over several random seeds and compare them.

  python experiments.py --seeds 20 --evals 20000 --workers 4

Outputs in results/:
  summary.csv         final best fitness for every (config, seed)
  convergence.png     mean +/- sd best-fitness curves per config
  printed             mean/sd per config + Mann-Whitney U test between configs
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from ea import Config, run, save_run

# Add or edit configurations to compare here.
CONFIGS = {
    "scheduled": Config(mutation="scheduled"),
    "cauchy": Config(mutation="cauchy"),
    "no_crossover": Config(mutation="scheduled", p_crossover=0.0),
}


def _one(args):
    name, cfg = args
    best, best_fit, history = run(cfg, verbose=False)
    save_run(cfg, best, best_fit, history, out_root=f"results/{name}")
    print(f"{name:>14} seed {cfg.seed:3d}: {best_fit:.2f} GWh/yr")
    return name, cfg.seed, best_fit, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--evals", type=int, default=20_000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = [(name, replace(cfg, seed=s, evals=args.evals))
            for name, cfg in CONFIGS.items() for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_one, jobs))

    Path("results").mkdir(exist_ok=True)
    with open("results/summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "seed", "best_fitness"])
        w.writerows((n, s, f) for n, s, f, _ in results)

    finals = {n: np.array([f for m, _, f, _ in results if m == n]) for n in CONFIGS}
    print("\nconfig           mean      sd     best")
    for n, v in finals.items():
        print(f"{n:<14} {v.mean():7.2f} {v.std(ddof=1):7.2f} {v.max():8.2f}")

    print("\nMann-Whitney U (two-sided):")
    for a, b in combinations(CONFIGS, 2):
        stat, p = mannwhitneyu(finals[a], finals[b], alternative="two-sided")
        print(f"  {a} vs {b}: U = {stat:.0f}, p = {p:.4f}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in CONFIGS:
        curves = [np.array(h) for m, _, _, h in results if m == n]
        length = min(len(c) for c in curves)
        evals = curves[0][:length, 1]
        best = np.stack([c[:length, 2] for c in curves])
        mean, sd = best.mean(0), best.std(0)
        ax.plot(evals, mean, label=n)
        ax.fill_between(evals, mean - sd, mean + sd, alpha=0.2)
    ax.set_xlabel("fitness evaluations")
    ax.set_ylabel("best fitness (GWh/yr)")
    ax.legend()
    plt.savefig("results/convergence.png", dpi=150, bbox_inches="tight")
    print("\nsaved results/summary.csv and results/convergence.png")


if __name__ == "__main__":
    main()
