"""Starter formulas for FITNESS_PENALTIES.md; adapt inputs to the future design.

Inputs are assumed valid: distances/power >= 0, voltage/area/buffer > 0,
and 0 < power_factor <= 1. Penalties return raw metrics; the evaluator applies
weights. Coordinate pairs below are provisional inputs, not a solution format.
"""

from math import hypot, sqrt


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
    # TODO: obtain these energy estimates from a wind/wake model.
    energy_lost_mwh = max(0, gross_energy_mwh - waked_energy_mwh)
    return energy_lost_mwh


def spacing_penalty(distance_m, preferred_spacing_m):
    """Optional wake proxy: 0..1 shortfall score for one unique turbine pair."""
    spacing_shortfall_m = max(0, preferred_spacing_m - distance_m)
    return spacing_shortfall_m / preferred_spacing_m


def cable_installation_cost(distance_to_collector_m, cost_per_m):
    """One-time installation cost for one turbine's direct cable route."""
    # TODO: choose an installed cost/m matching the cable material and size.
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
    # TODO: find the nearest relevant habitat using future map data.
    proximity_score = max(0, 1 - distance_to_habitat_m / buffer_m)
    return proximity_score


# spacing_penalty can replace wake_loss initially; do not enable both by default.
PENALTY_FUNCTIONS = [
    wake_loss,
    cable_installation_cost,
    electrical_loss,
    habitat_proximity,
]
