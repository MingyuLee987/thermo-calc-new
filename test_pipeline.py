"""Offline regression checks; test data are synthetic and not corrosion evidence."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from core import Checkpoint, diverse, feasible, grid, load_settings, scheil_metrics, strict_csv
from learning import expected_improvement, fit_gp, predict, recommend, measured
from pipeline import write_template, manufacturing_pool
from tc_backend import equilibrium, scheil

CFG = load_settings(Path(__file__).with_name("settings.json"))


class PipelineTests(unittest.TestCase):
    def test_grid(self):
        df = grid(CFG)
        self.assertEqual(len(df), 60000)
        self.assertAlmostEqual(df.Ce_wt.max(), .099)
        self.assertAlmostEqual(df.Cr_wt.max(), 1.399)

    def test_scheil_cooling_order(self):
        result = scheil_metrics([1100, 1200, 1300, 1400], [1., .99, .90, 0.])
        self.assertAlmostEqual(result["dT90_99_C"], 100.)
        self.assertAlmostEqual(result["T_90solid_C"], 1026.85)

    def test_incomplete_scheil_is_not_solidus(self):
        result = scheil_metrics([1400, 1300, 1200], [0., .8, .95])
        self.assertTrue(np.isnan(result["T_99solid_C"]))
        with self.assertRaises(ValueError):
            scheil_metrics([1400, 1300, 1200], [0., .8, .7])

    def test_failed_query_not_feasible(self):
        df = pd.DataFrame(dict(calculation_status=["ok", "failed", "ok"], matrix_phase_fraction=[.99]*3, harmful_phase_fraction=[0, 0, np.nan], matrix_Cr_wt=[1.]*3))
        self.assertEqual(feasible(df, CFG).tolist(), [True, False, False])

    def test_checkpoint_resume_and_config_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Checkpoint(tmp, CFG)
            cp.put("equilibrium", {"Ce_wt": .01, "Cr_wt": 1., "calculation_status": "ok"})
            cp.close()
            cp = Checkpoint(tmp, CFG)
            self.assertEqual(len(cp.records("equilibrium")), 1)
            cp.close()
            with self.assertRaises(ValueError):
                Checkpoint(tmp, {**CFG, "temperature_C": 950})

    def test_bad_csv_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bad.csv"
            path.write_text("a,b\n1,2,3\n")
            with self.assertRaises(ValueError):
                strict_csv(path)

    def test_ei_minimization(self):
        scores = expected_improvement(np.array([-1., 1., 0.]), np.array([0., 0., 1.]), 0.)
        np.testing.assert_allclose(scores, [1., 0., 1/np.sqrt(2*np.pi)])

    def test_noise_scaling_invariant(self):
        x = np.array([[0,0],[1,0],[0,1],[1,1],[.3,.7],[.7,.3]], float)
        y = np.array([1, 3, 2, 5, 2, 4.], float)
        cfg = {**CFG, "optimizer_restarts": 0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            a = predict(fit_gp(x, y, np.ones(6)*.2, "RBF", cfg), x)
            b = predict(fit_gp(x, y*100, np.ones(6)*20, "RBF", cfg), x)
        np.testing.assert_allclose(a[0]*100, b[0], rtol=1e-5)
        np.testing.assert_allclose(a[1]*100, b[1], rtol=1e-5)

    def test_exact_phase_sets_and_units(self):
        class Quantity:
            def __getattr__(self, name):
                return lambda *args: (name, *args)
        q = Quantity()
        fractions = {"FCC_A1#1": .6, "FCC_A1#2": .35, "SIGMA#1": .05}
        masses = {"FCC_A1#1": .5, "FCC_A1#2": .4}
        crs = {"FCC_A1#1": .01, "FCC_A1#2": .02}
        class Result:
            def get_stable_phases(self): return list(fractions)
            def get_value_of(self, query):
                name, p, *other = query
                return {"mole_fraction_of_a_phase": fractions, "mass_fraction_of_a_phase": masses, "composition_of_phase_as_weight_fraction": crs}[name][p]
            def get_values_of(self, *args): return [1400, 1300, 1200], [0., .90, .99]
        class Calc:
            def __init__(self): self.conditions = {}; self.unit = None
            def set_condition(self, q, value): self.conditions[q] = value; return self
            def set_composition_unit(self, unit): self.unit = unit; return self
            def set_start_temperature(self, t): return self
            def set_composition(self, element, wt): self.conditions[element] = wt; return self
            def calculate(self): return Result()
        calc = Calc()
        system = SimpleNamespace(with_single_equilibrium_calculation=lambda: calc, with_scheil_calculation=lambda: calc)
        fake = SimpleNamespace(ThermodynamicQuantity=q, ScheilQuantity=q, CompositionUnit=SimpleNamespace(MASS_PERCENT="mass%"))
        with patch.dict("sys.modules", {"tc_python": fake}):
            r = equilibrium(system, CFG, .01, 1.)
            self.assertAlmostEqual(r["matrix_phase_fraction"], .95)
            self.assertAlmostEqual(r["matrix_Cr_wt"], (.5*.01+.4*.02)/.9*100)
            self.assertEqual(calc.conditions[("mass_fraction_of_a_component", "CR")], .01)
            scheil(system, CFG, .01, 1.)
            self.assertEqual(calc.unit, "mass%")
            self.assertEqual(calc.conditions["CR"], 1.)

    def test_offline_full_learning_and_guards(self):
        cfg = {**CFG, "ce_grid": [0, .09, .01], "cr_grid": [.8, 1.3, .1], "optimizer_restarts": 0}
        pool = grid(cfg)
        pool["matrix_Cr_wt"] = pool.Cr_wt
        initial = diverse(pool, 8, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            path = out/"synthetic_ONLY.csv"
            write_template(initial, cfg, path)
            df = strict_csv(path)
            df["icorr_mean_A_cm2"] = 1e-5*(1+df.Ce_wt+df.Cr_wt)
            df["icorr_sd_A_cm2"] = 1e-6
            df["Epit_mean_V_AgAgCl"] = -.3+.1*df.Cr_wt-df.Ce_wt
            df["Epit_sd_V"] = .01
            df["n_replicates"] = 3
            df.to_csv(path, index=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                selected = recommend(pool, path, cfg, out)
            self.assertEqual(len(selected.drop_duplicates(["Ce_wt", "Cr_wt"])), 3)
            measured_pairs = set(map(tuple, initial[["Ce_wt", "Cr_wt"]].to_numpy()))
            self.assertFalse(any(tuple(r) in measured_pairs for r in selected[["Ce_wt", "Cr_wt"]].to_numpy()))
            self.assertTrue((out/"prediction_maps.png").exists())
            self.assertEqual(len(pd.read_csv(out/"kernel_comparison.csv")), 6)
            df.loc[0, "icorr_mean_A_cm2"] = 0
            from learning import target_data
            with self.assertRaises(ValueError): target_data(df, "icorr")
            df.loc[0, "experiment_context"] = "different electrolyte"
            df.to_csv(path, index=False)
            with self.assertRaises(ValueError): measured(path, cfg)
            df = pd.concat([df.iloc[1:], df.iloc[1:2]])
            df.to_csv(path, index=False)
            with self.assertRaises(ValueError): measured(path, cfg)


if __name__ == "__main__":
    unittest.main()
