# Fitness penalties

`fitness_penalties.evaluate_layout(xy, cfg)` evaluates a layout in projected UTM
metres and returns a `FitnessBreakdown`. `ea.fitness` uses its `fitness_gwh` field
for selection. All terms in the final score are GWh/year equivalents:

```text
fitness = waked AEP
          - electrical_weight * cable energy loss
          - annualized installation cost / energy value
          - spacing_weight * (violating pairs + normalized spacing shortfall)
          - habitat_weight * summed habitat proximity
```

Turbine count/type stay fixed. `ea.fitness` rejects malformed, nonfinite,
wrong-count and out-of-mask layouts with negative infinity. The site mask
excludes building/road/shore setbacks and mapped wetlands. Spacing repair is
best effort; any remaining violations receive a penalty. These constraints and
the numerical defaults are project assumptions, not verified siting rules.

## Wake loss and spacing

`wind_model.power_scenarios` supplies simultaneous turbine power for the same
72 directions and Weibull speed bins used by `turbine_aep`. Scenario probabilities
sum to one. AEP already includes Jensen wake losses, so **no second wake penalty
is subtracted**. `wake_loss(gross_energy_mwh, waked_energy_mwh)` remains available
for reporting the nonnegative lost energy. `wind_model.wake_loss(xy)` reports the
fraction instead.

Minimum spacing is `site_tools.MIN_SPACING = 4 * ROTOR_D = 372 m`. Each unique
pair below this threshold incurs `spacing_weight` plus a severity term:
`spacing_weight * max(0, 1 - distance / MIN_SPACING)`. This enforces the minimum
spacing assumption separately from physical wake effects. The default weight
is 10 GWh per violating pair, increasing to 20 GWh for coincident turbines.

## Cable routing and installation

`site_tools.cable_tree` builds a straight-line minimum spanning tree containing
all turbines and a fixed collector, counting each segment once. The collector
uses the approximate location already selected for this project:
44.1790 latitude, -76.4725 longitude, projected to EPSG:32618. Its exact position
still needs verification. The previous virtual road node omitted the connection
along roads to the collector and could not describe shared electrical flows.

Installation cost is total cable metres times `cable_cost_per_m`. Convert it to
an annual GWh-equivalent penalty using:

```text
installation_penalty_gwh = cost / project_lifetime_years
                               / energy_value_per_mwh / 1000
```

Defaults are illustrative: 300 currency units/m, 25 years, and 100 of the same
currency units/MWh. That gives 0.12 GWh/year per km. This is straight-line
annualization without discounting. `--cable-weight VALUE` overrides this
conversion with the previous direct GWh/year-per-km weight (e.g. `0.3`). It
replaces the installation penalty rather than adding a duplicate cable term.

## Electrical energy loss

The cable tree is rooted at the collector. A segment carries the sum of all
downstream turbines' simultaneous power in each wind scenario. For each segment:

```text
R = resistivity_ohm_m * length_m / (area_mm2 / 1_000_000)
I = power_w / (sqrt(3) * voltage_ll_v * power_factor)
annual_loss_mwh = sum(probability * 3 * I**2 * R) * 8760 / 1_000_000
```

This integrates squared current, rather than squaring annual mean current.
Default assumptions: aluminium resistivity 2.82e-8 ohm-m, 630 mm² conductors,
34.5 kV line-to-line, and power factor 0.95. `electrical_weight=1` subtracts the
estimated physical energy loss once. Zero skips loss calculation; other weights
alter the optimization tradeoff. `net_aep_gwh` reports AEP minus the calculated
unweighted loss; when disabled, that loss is reported as zero.

Installation and electrical losses use the same tree and segment lengths.
This model assumes constant conductor properties and does not model voltage
drop, transformer losses, capacity limits, feedback of losses into upstream
flows, or terrain-aware cable routing. A single conductor size is used for all
segments, including shared trunks; this is a comparison model, not cable design.

## Habitat proximity

The site builder saves a `habitat_dist` raster in `wolfe_island_mask.npz` and the
mapped wetlands in a GeoPackage layer. Distance is estimated on the site's
25 m grid. Wetlands remain hard exclusions. Outside them, each turbine scores:

```text
proximity = max(0, 1 - nearest_wetland_distance_m / habitat_buffer_m)
```

Sum across turbines and multiply by `habitat_weight` (default 1 GWh per unit).
The default soft buffer is 200 m. This is a proximity proxy for the mapped
wetlands only. It does not estimate species-specific impacts or include
unmapped habitat. No mapped wetlands produce infinite distances and zero scores.
Missing habitat data raises an explicit error; it is not silently scored zero.

## Running and inspecting results

Rebuild site data once to include the habitat raster:

```bash
conda activate evolutionaryproject
python source/wolfe_island_site.py
python source/ea.py --seed 0 --evals 5000
```

For an older mask, explicitly disable the unavailable term with
`--habitat-weight 0`, or rebuild it. Each penalty setting is an EA command-line
option; for example:

```bash
python source/ea.py --habitat-buffer-m 300 --habitat-weight 2 --cable-cost-per-m 400
```

Runs save `fitness_breakdown.csv` alongside the existing layout, configuration,
history, and plot. The breakdown reports waked AEP, cable loss, net AEP, cable
length, raw installation cost, spacing counts/severity, habitat scores, weighted
penalties, and final fitness. Historical results produced with the old objective
are not directly comparable to new scores.

Run regression tests without downloading OSM data:

```bash
python -m unittest discover -s source/tests -v
```

The tests supply a synthetic feasibility raster and explicit wind scenarios
where exact electrical losses are known; they also check the real wind model
against its AEP calculation and run a small EA smoke test.

## Background references

- [NREL/NLR wake models](https://samrepo.nlr.gov/help/wind_farm.html)
- [Turbine placement and cable routing](https://arxiv.org/abs/2501.07203)
- [US Fish & Wildlife Service wind siting context](https://www.fws.gov/story/avoidance-and-minimization-measures-land-based-wind)
- [Previously selected approximate collector location](https://mapcarta.com/W166793711)
- [ABB electrical installation handbook](https://library.e.abb.com/public/a5861200ea7c59fec1256dc50039e0ed/1SDC010001D0201.pdf)
