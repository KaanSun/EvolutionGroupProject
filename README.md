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
python wolfe_island_site.py
```