# Fitness penalty functions

**Hard constraints** reject unacceptable layouts, such as turbines inside building
setbacks or protected no-build areas. Handle these separately.

**Soft penalties** discourage allowed but less desirable layouts. The short
formulas in `fitness_penalties.py` show the intended calculations. They do not
load data, decode solutions, or implement wind/cable models. Adapt their inputs
when the team settles the architecture.

## 1. Wake loss

`wake_loss(gross_energy_mwh, waked_energy_mwh)`

Upstream turbines leave slower wind behind them, reducing downstream output. [1]
Return the difference between energy without wakes and energy with wakes,
clipped at zero. The future model must estimate both using turbine positions,
wind speed/direction, and turbine characteristics. Available wind power follows
`P = 0.5 * air_density * rotor_area * wind_speed^3`; actual turbine output needs
a power curve and rated-power limit.

`spacing_penalty(distance_m, preferred_spacing_m)` is an optional alternative:
`max(0, preferred_spacing - distance) / preferred_spacing`, with positive preferred
spacing. It returns 0..1 per unique pair. This is a proximity proxy, not a physical
wake estimate; use it instead of `wake_loss` initially, not in addition to it.

## 2. Cable installation cost

`cable_installation_cost(distance_to_collector_m, cost_per_m)`

Return cable distance times installed cost per metre. Initially estimate a direct
route from each turbine to the fixed island collector substation. Shared routes
can be considered later. [2] Start with a mild weight; spread one-time cost over
the project lifetime if comparing it with annual energy value.

`cable_length(turbine_xy, collector_xy)` estimates `sqrt(dx^2 + dy^2)` in projected
metres. The installed cost/m should depend on cable size, material, and trenching.

## 3. Electrical energy loss

`electrical_loss(current_a, resistance_per_phase_ohm, hours)`

Return `3 * current^2 * resistance * hours / 1_000_000` MWh lost on a balanced
three-phase cable segment. Two helpers show where the inputs come from:

- `cable_resistance(length_m, cross_section_mm2, resistivity_ohm_m)` uses
  `R = rho * L / A` for one phase conductor, converting mm² to m².
- `three_phase_current(power_w, voltage_ll_v, power_factor)` uses
  `I = P / (sqrt(3) * V_LL * power_factor)`; voltage/current are RMS. [5]

Longer cable raises resistance. Larger conductor area lowers resistance but can
raise installation cost. For the same power and power factor, doubling voltage
halves current and quarters resistive loss if resistance is unchanged.

Use the same route length and cable choice for installation and loss estimates.
These are rough conductor formulas: real AC cable resistance also depends on
temperature and cable construction. Sum losses over operating periods, not from
an average current. Shared segments carry combined output. Equipment ratings and
voltage limits remain separate hard checks.

The intended flow is simply:

```python
length_m = cable_length(turbine_xy, collector_xy)
cost = cable_installation_cost(length_m, cost_per_m)
resistance = cable_resistance(length_m, area_mm2, resistivity)
current = three_phase_current(power_w, voltage_ll_v, power_factor)
lost_mwh = electrical_loss(current, resistance, hours)
```

## 4. Habitat proximity

`habitat_proximity(distance_to_habitat_m, buffer_m)`

Return `max(0, 1 - distance / buffer)`, with a positive buffer. One turbine scores
1 inside selected habitat and 0 at or beyond the buffer. Future map code supplies
the nearest-habitat distance. Select relevant woodland/wetland/habitat classes;
this is a disturbance proxy, not a wildlife-impact prediction. [3]

## Integration later

- `PENALTY_FUNCTIONS` lists the callables; each needs its own inputs. Sum per-turbine
  or per-segment results where appropriate, counting each loss/cost once.
- Inputs use the units in their names. Keep missing data explicit, not zero.
- Convert raw results to points using chosen weights/scales; do not directly add
  currency, MWh, metres, and proximity scores. Keep turbine count/type fixed.
- Do not subtract wake/electrical losses already included in the energy reward.
- Use the island collector, not Marysville, as the collection destination. Its
  approximate mapped location is **44.1790, -76.4725**; verify and project before
  measuring distances. [4] A separate grid-distance term would duplicate that
  cost when the collector and export connection are fixed.

## References

1. [NREL/NLR: wake models](https://samrepo.nlr.gov/help/wind_farm.html).
2. [Turbine placement and cable routing](https://arxiv.org/abs/2501.07203).
3. [US Fish & Wildlife Service: wind siting](https://www.fws.gov/story/avoidance-and-minimization-measures-land-based-wind) (context, not Ontario rules).
4. [Mapped island substation](https://mapcarta.com/W166793711).
5. [ABB: electrical installation handbook, conductor losses](https://library.e.abb.com/public/a5861200ea7c59fec1256dc50039e0ed/1SDC010001D0201.pdf).
