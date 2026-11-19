# Evolution Group Project

Course project for CISC 455: Evolutionary Algorithms.

## Setup

Requires conda. If `conda --version` doesn't work, install [Miniforge](https://github.com/conda-forge/miniforge) first.

```bash
conda env create -f environment.yml
conda activate evolutionaryproject
```

If `environment.yml` changes, update with `conda env update -f environment.yml`.

## Run

```bash
python source/wolfe_island_site.py
python source/ea.py --seed 0 --evals 5000
```

The first command downloads map data and generates `wolfe_island_mask.npz`
and `wolfe_island_site.gpkg` in the project root. These generated files are
excluded from Git, so each new checkout needs this setup step once. Both
scripts resolve data paths from their file locations, independently of the
terminal's current directory.

The builder uses the Dresel North America Overpass mirror for map downloads
([public server list](https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances)).
To select a different endpoint, set `OSMNX_OVERPASS_URL` before running the site builder.

Fitness combines wake-adjusted energy with cable losses, installation cost,
spacing, and habitat proximity. See [FITNESS_PENALTIES.md](FITNESS_PENALTIES.md)
for assumptions, weights, output metrics, and tests. Rebuild site data if your
mask predates the habitat-distance raster.

All Python code is in `source/`, including tests in `source/tests/`. Input data
and generated site files stay in the project root; runs are saved in `results/`.

```bash
python -m unittest discover -s source/tests -v
```
