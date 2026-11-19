"""Regression tests with a synthetic site; never fetch OSM or overwrite site data."""

import csv
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitness_penalties as fp
import wind_model as wm


def setUpModule():
    global st, ea
    collector = np.array(Transformer.from_crs(
        "EPSG:4326", "EPSG:32618", always_xy=True).transform(-76.4725, 44.1790))
    # Explicit stand-in for the generated mask, which is not checked into git.
    fixture = {"mask": np.ones((160, 160), dtype=bool),
               "x0": collector[0] - 8000, "y0": collector[1] - 8000,
               "res": 100.0, "road_dist": np.full((160, 160), 100.0),
               "habitat_dist": np.full((160, 160), 100.0)}
    real_load = np.load

    def load(path, *args, **kwargs):
        if Path(path).name == "wolfe_island_mask.npz":
            return fixture
        return real_load(path, *args, **kwargs)

    with patch("numpy.load", side_effect=load):
        st = importlib.import_module("site_tools")
        ea = importlib.import_module("ea")


class PenaltyTests(unittest.TestCase):
    def setUp(self):
        self.xy = st.COLLECTOR_XY + [[2000, 0], [1000, 0]]

    @staticmethod
    def scenarios(xy):
        # Half the year idle, half at 2 MW per turbine. Mean is 1 MW each.
        yield np.array([[0., 0.], [2000., 2000.]]), np.array([0.5, 0.5])

    def evaluate(self, **kwargs):
        cfg = fp.PenaltyConfig(**kwargs)
        with patch("wind_model.power_scenarios", self.scenarios):
            return fp.evaluate_layout(self.xy, cfg)

    def test_scalar_units(self):
        self.assertEqual(fp.cable_length((0, 0), (3, 4)), 5)
        self.assertAlmostEqual(fp.cable_resistance(1000, 1000, 1e-8), 0.01)
        self.assertAlmostEqual(fp.electrical_loss(100, 0.5, 2), 0.03)
        self.assertEqual(fp.wake_loss(100, 80), 20)
        self.assertEqual(fp.wake_loss(80, 100), 0)
        self.assertEqual(fp.cable_installation_cost(1000, 300), 300_000)

    def test_rooted_shared_segments_and_real_collector(self):
        length, edges = st.cable_tree(self.xy)
        self.assertAlmostEqual(length, 2000)
        lengths, loads = fp.cable_segment_loads(self.xy, edges, st.COLLECTOR_XY)
        np.testing.assert_allclose(lengths, [1000, 1000])
        np.testing.assert_array_equal(loads, [[1, 0], [1, 1]])
        with self.assertRaises(ValueError):
            fp.cable_segment_loads(self.xy, [], st.COLLECTOR_XY)

    def test_exact_scenario_losses_and_fitness_units(self):
        result = self.evaluate(cable_area_mm2=1000, cable_resistivity_ohm_m=1e-8,
                               collection_voltage_v=10_000, power_factor=1,
                               habitat_weight=2)
        self.assertAlmostEqual(result.aep_gwh, 17.52)
        # 0.01 ohm/segment: 400 W branch + 1600 W trunk, for 4380 hours.
        self.assertAlmostEqual(result.electrical_loss_gwh, 0.00876)
        self.assertAlmostEqual(result.net_aep_gwh, 17.51124)
        self.assertAlmostEqual(result.installation_cost, 600_000)
        self.assertAlmostEqual(result.installation_penalty_gwh, 0.24)
        self.assertAlmostEqual(result.habitat_score, 1)
        self.assertAlmostEqual(result.fitness_gwh, 15.27124)

    def test_voltage_scaling_and_legacy_installation_override(self):
        low = self.evaluate(collection_voltage_v=10_000)
        high = self.evaluate(collection_voltage_v=20_000, cable_weight=0.3)
        self.assertAlmostEqual(low.electrical_loss_gwh, 4 * high.electrical_loss_gwh)
        self.assertAlmostEqual(high.installation_penalty_gwh, 0.6)

    def test_no_double_counting_of_wakes(self):
        result = self.evaluate(cable_cost_per_m=0, electrical_weight=0, habitat_weight=0)
        self.assertAlmostEqual(result.fitness_gwh, 17.52)

    def test_spacing_counts_pairs_once_and_includes_severity(self):
        self.xy[0] = self.xy[1] + [st.MIN_SPACING / 2, 0]
        result = self.evaluate()
        self.assertEqual(result.spacing_violations, 1)
        self.assertAlmostEqual(result.spacing_shortfall, 0.5)
        self.assertAlmostEqual(result.spacing_penalty_gwh, 15)

    def test_habitat_buffer_and_missing_data(self):
        np.testing.assert_allclose(fp.habitat_proximity(np.array([0, 100, 200, np.inf]), 200),
                                   [1, 0.5, 0, 0])
        with patch.object(st, "HABITAT_DIST", None):
            with self.assertRaisesRegex(RuntimeError, "Habitat distances are missing"):
                self.evaluate()
            self.assertEqual(self.evaluate(habitat_weight=0).habitat_penalty_gwh, 0)

    def test_ea_rejects_invalid_layouts(self):
        with patch.object(st, "N_TURBINES", 2):
            self.assertEqual(ea.fitness([[1, 2], [3]]), -np.inf)
            self.assertEqual(ea.fitness([["bad", 0], [0, 0]]), -np.inf)
            self.assertEqual(ea.fitness([[np.nan, 0], [0, 0]]), -np.inf)
            self.assertEqual(ea.fitness([[0, 0], [0, 0]]), -np.inf)
            self.assertEqual(ea.fitness(self.xy[:1]), -np.inf)
            with patch("wind_model.power_scenarios", self.scenarios):
                cfg = ea.Config(cable_cost_per_m=0, electrical_weight=0, habitat_weight=0)
                self.assertAlmostEqual(ea.fitness(self.xy, cfg=cfg), 17.52)

    def test_invalid_penalty_settings(self):
        for kwargs in ({"collection_voltage_v": 0}, {"cable_area_mm2": -1},
                       {"habitat_buffer_m": 0}, {"power_factor": 1.1},
                       {"cable_weight": -1}, {"spacing_weight": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ea.Config(**kwargs)

    def test_actual_wind_energy_matches_evaluator(self):
        result = fp.evaluate_layout(self.xy)
        self.assertAlmostEqual(result.aep_gwh, wm.aep(self.xy) / 1e6, places=10)
        self.assertGreater(result.electrical_loss_gwh, 0)
        self.assertLessEqual(result.aep_gwh, wm.turbine_aep(self.xy, wakes=False).sum() / 1e6)
        probability = sum(float(p.sum()) for _, p in wm.power_scenarios(self.xy))
        self.assertAlmostEqual(probability, 1)

    def test_small_ea_run_and_saved_breakdown(self):
        cfg = ea.Config(mu=2, lam=2, evals=4, seed=7)
        best, fit, history = ea.run(cfg, verbose=False)
        self.assertEqual(best.shape, (86, 2))
        self.assertTrue(np.isfinite(fit))
        self.assertTrue(st.in_mask(best).all())
        self.assertAlmostEqual(fit, ea.fitness(best, cfg=cfg))
        self.assertEqual(history[-1][1], 4)
        with tempfile.TemporaryDirectory() as tmp:
            out = ea.save_run(cfg, best, fit, history, out_root=tmp)
            with open(out / "fitness_breakdown.csv", newline="") as fh:
                metrics = dict(list(csv.reader(fh))[1:])
            self.assertAlmostEqual(float(metrics["fitness_gwh"]), fit)


if __name__ == "__main__":
    unittest.main()
