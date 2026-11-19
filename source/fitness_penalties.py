"""Layout fitness in GWh/year, with shared cable and wind scenario losses.

The scalar/array helpers return raw metrics in the units in their names.
``evaluate_layout`` converts annualized installation cost and habitat scores
to GWh equivalents before combining them with net delivered energy.
"""

from dataclasses import dataclass
from math import hypot, sqrt

import numpy as np


@dataclass
class PenaltyConfig:
    # Illustrative modelling assumptions, not surveyed engineering/cost data.
    spacing_weight: float = 10.0       # GWh per violating pair + relative shortfall
    cable_weight: float | None = None  # optional legacy GWh/km installation override
    cable_cost_per_m: float = 300.0    # currency units / m, installed
    project_lifetime_years: float = 25.0
    energy_value_per_mwh: float = 100.0  # same currency units / MWh
    cable_area_mm2: float = 630.0
    cable_resistivity_ohm_m: float = 2.82e-8  # aluminium reference assumption
    collection_voltage_v: float = 34_500.0   # RMS line-to-line
    power_factor: float = 0.95
    electrical_weight: float = 1.0     # physical energy loss multiplier
    habitat_buffer_m: float = 200.0    # soft buffer outside excluded wetlands
    habitat_weight: float = 1.0        # GWh per summed turbine proximity score

    def __post_init__(self):
        positive = ("project_lifetime_years", "energy_value_per_mwh", "cable_area_mm2",
                    "cable_resistivity_ohm_m", "collection_voltage_v", "habitat_buffer_m")
        nonnegative = ("spacing_weight", "cable_cost_per_m", "electrical_weight", "habitat_weight")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"Invalid penalty parameter {name}: {value}")
        if not 0 < self.power_factor <= 1:
            raise ValueError("power_factor must be in (0, 1].")
        if self.cable_weight is not None and (not np.isfinite(self.cable_weight)
                                              or self.cable_weight < 0):
            raise ValueError("cable_weight must be nonnegative or None.")


@dataclass(frozen=True)
class FitnessBreakdown:
    aep_gwh: float
    electrical_loss_gwh: float
    net_aep_gwh: float
    cable_km: float
    installation_cost: float
    installation_penalty_gwh: float
    spacing_violations: int
    spacing_shortfall: float
    spacing_penalty_gwh: float
    habitat_score: float
    habitat_penalty_gwh: float
    fitness_gwh: float


def cable_length(turbine_xy, collector_xy):
    """Straight-line route estimate; both coordinates must be in metres."""
    dx = turbine_xy[0] - collector_xy[0]
    dy = turbine_xy[1] - collector_xy[1]
    return hypot(dx, dy)


def cable_resistance(length_m, cross_section_mm2, resistivity_ohm_m):
    """Approximate resistance of ONE phase conductor: R = rho * L / A."""
    area_m2 = cross_section_mm2 / 1_000_000
    resistance_ohm = resistivity_ohm_m * length_m / area_m2
    return resistance_ohm


def three_phase_current(power_w, voltage_ll_v, power_factor):
    """Balanced three-phase current; voltage is RMS line-to-line voltage."""
    current_a = power_w / (sqrt(3) * voltage_ll_v * power_factor)
    return current_a


def wake_loss(gross_energy_mwh, waked_energy_mwh):
    """Energy lost because upstream turbines reduce downstream wind."""
    energy_lost_mwh = np.maximum(0, gross_energy_mwh - waked_energy_mwh)
    return energy_lost_mwh


def spacing_penalty(distance_m, preferred_spacing_m):
    """0..1 normalized minimum-spacing shortfall per unique turbine pair."""
    spacing_shortfall_m = np.maximum(0, preferred_spacing_m - distance_m)
    return spacing_shortfall_m / preferred_spacing_m


def cable_installation_cost(distance_to_collector_m, cost_per_m):
    """One-time installation cost for one turbine's direct cable route."""
    installation_cost = distance_to_collector_m * cost_per_m
    return installation_cost


def electrical_loss(current_a, resistance_per_phase_ohm, hours):
    """Three-phase cable energy loss for one segment and operating period."""
    # Use the helpers above; sum operating periods instead of averaging current.
    power_lost_w = 3 * current_a ** 2 * resistance_per_phase_ohm
    energy_lost_mwh = power_lost_w * hours / 1_000_000
    return energy_lost_mwh


def habitat_proximity(distance_to_habitat_m, buffer_m):
    """Proximity score from 0 to 1 for one turbine; buffer_m must be positive."""
    proximity_score = np.clip(1 - distance_to_habitat_m / buffer_m, 0, 1)
    return proximity_score


def cable_segment_loads(xy, edges, collector_xy):
    """Return segment lengths and a (segment, turbine) downstream indicator.

    Root the tree at node n (the collector). Shared segments carry the sum of
    their downstream turbines' simultaneous power, not a per-turbine average.
    """
    n = len(xy)
    points = np.vstack([xy, collector_xy])
    neighbours = [[] for _ in range(n + 1)]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    parent = np.full(n + 1, -1, dtype=int)
    parent[n] = n
    order = [n]
    for node in order:
        for other in neighbours[node]:
            if parent[other] == -1:
                parent[other] = node
                order.append(other)
    if len(edges) != n or len(order) != n + 1:
        raise ValueError("Cable network must be a connected tree rooted at collector node n.")
    downstream = np.zeros((n + 1, n))
    downstream[:n] = np.eye(n)
    for node in reversed(order[1:]):
        downstream[parent[node]] += downstream[node]
    lengths = np.linalg.norm(points[:n] - points[parent[:n]], axis=1)
    return lengths, downstream[:n]


def evaluate_layout(xy, cfg=None):
    """Evaluate a feasible UTM layout and return all fitness components.

    Hard mask violations are rejected here; ea.fitness maps them to -infinity.
    Minimum-spacing violations left by repair receive a count and severity
    penalty. Wake loss is already included in the wind-model power scenarios.
    """
    import site_tools as st
    from wind_model import power_scenarios

    cfg = PenaltyConfig() if cfg is None else cfg
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or not np.isfinite(xy).all():
        raise ValueError("Layout must be a nonempty finite array of shape (n, 2).")
    if not st.in_mask(xy).all():
        raise ValueError("Layout contains turbines outside the feasible mask.")

    # Check habitat availability before the expensive wind calculation.
    habitat_score = (float(habitat_proximity(st.habitat_distance(xy), cfg.habitat_buffer_m).sum())
                     if cfg.habitat_weight else 0.0)
    _, edges = st.cable_tree(xy)
    lengths, downstream = cable_segment_loads(xy, edges, st.COLLECTOR_XY)
    resistance = cable_resistance(lengths, cfg.cable_area_mm2, cfg.cable_resistivity_ohm_m)
    energy_kwh, loss_mwh = 0.0, 0.0
    for power_kw, probability in power_scenarios(xy):
        energy_kwh += float(probability @ power_kw.sum(axis=1)) * 8760
        if cfg.electrical_weight:
            segment_power_w = (power_kw @ downstream.T) * 1000
            current = three_phase_current(segment_power_w, cfg.collection_voltage_v, cfg.power_factor)
            scenario_loss = electrical_loss(current, resistance, 8760).sum(axis=1)
            loss_mwh += float(probability @ scenario_loss)

    distances = st.pair_distances(xy)[np.triu_indices(len(xy), k=1)]
    violations = int((distances < st.MIN_SPACING).sum())
    shortfall = float(spacing_penalty(distances, st.MIN_SPACING).sum())
    spacing_gwh = cfg.spacing_weight * (violations + shortfall)
    cable_m = float(lengths.sum())
    cost = cable_installation_cost(cable_m, cfg.cable_cost_per_m)
    install_gwh = (cost / cfg.project_lifetime_years / cfg.energy_value_per_mwh / 1000
                   if cfg.cable_weight is None else cfg.cable_weight * cable_m / 1000)
    habitat_gwh = cfg.habitat_weight * habitat_score
    aep_gwh = energy_kwh / 1e6
    loss_gwh = loss_mwh / 1000
    net_gwh = aep_gwh - loss_gwh
    fit = aep_gwh - cfg.electrical_weight * loss_gwh - install_gwh - spacing_gwh - habitat_gwh
    return FitnessBreakdown(aep_gwh, loss_gwh, net_gwh, cable_m / 1000, cost,
                            install_gwh, violations, shortfall, spacing_gwh,
                            habitat_score, habitat_gwh, fit)
