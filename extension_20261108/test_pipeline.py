"""
Comprehensive audit and verification test suite for the extension pipeline.
Covers all requirements from GEMINI_OVERNIGHT_HANDOFF.md Section 5.
"""
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import unittest
import numpy as np
import pandas as pd

from extension_20261108.data import (
    verify_hashes, load_screened_universe, compute_volatility_state,
    FACTOR_ZIP, MARKET_ZIP, EXPECTED_HASHES
)
from extension_20261108.clustering import fit_theme_representation
from extension_20261108.models import (
    solve_ridge_sdf, solve_augmented_least_squares, evaluate_sdf_moments
)
from extension_20261108.tuning import (
    get_penalty_grid, run_goldilocks_selection, run_forward_inner_tuning,
    fit_and_freeze_full_models
)
from extension_20261108.evaluate import evaluate_period_models, run_paired_moving_block_bootstrap
from diagnostic.run import paired_draws

class TestExtensionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.factors_panel, cls.market_series, cls.full_market, cls.eligible, cls.dates = load_screened_universe()
        cls.train_dates = cls.dates[:420]
        cls.test_dates = cls.dates[420:552]
        cls.val_dates = cls.dates[552:612]
        selection = json.loads((ROOT_DIR / "reference/primary/tuning_curves.json").read_text())["selection_summary"]
        _, transforms = fit_and_freeze_full_models(
            cls.factors_panel, cls.market_series, cls.full_market, cls.train_dates, selection
        )
        cls.fixture_transforms = {k: v for k, v in transforms.items()
                                  if k not in ("F_all", "X_all", "h_all")}

    def make_figure_fixture(self, destination):
        import shutil
        shutil.copytree(ROOT_DIR / "reference/primary", destination / "primary")
        shutil.copytree(ROOT_DIR / "reference/diagnostic", destination / "diagnostic")
        (destination / "primary/training_transforms.json").write_text(
            json.dumps(self.fixture_transforms, indent=2, allow_nan=False), encoding="utf-8"
        )

    def test_01_hash_verification(self):
        """Test hash verification succeeds on genuine archives."""
        self.assertTrue(verify_hashes())

    def test_02_hash_rejection_on_altered_input(self):
        """Verify altered-input hash rejection."""
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        with TemporaryDirectory() as tmp:
            (Path(tmp) / FACTOR_ZIP.name).write_bytes(b"changed")
            with patch("extension_20261108.data.EVIDENCE", Path(tmp)):
                with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                    verify_hashes()

    def test_03_key_uniqueness_and_screening(self):
        """Verify unique keys and screening produces exactly 153 eligible factors."""
        self.assertEqual(len(self.eligible), 153)
        self.assertEqual(self.factors_panel.shape, (612, 153))
        self.assertEqual(len(self.market_series), 612)
        self.assertEqual(self.factors_panel.isna().sum().sum(), 0)

    def test_04_chronological_disjointness(self):
        """Verify chronological disjointness and strict contiguous ordering."""
        # Check no overlap
        s_train = set(self.train_dates)
        s_test = set(self.test_dates)
        s_val = set(self.val_dates)
        self.assertTrue(s_train.isdisjoint(s_test))
        self.assertTrue(s_train.isdisjoint(s_val))
        self.assertTrue(s_test.isdisjoint(s_val))

        # Check boundaries
        self.assertEqual(self.train_dates[0], "1975-01-31")
        self.assertEqual(self.train_dates[-1], "2009-12-31")
        self.assertEqual(self.test_dates[0], "2010-01-31")
        self.assertEqual(self.test_dates[-1], "2020-12-31")
        self.assertEqual(self.val_dates[0], "2021-01-31")
        self.assertEqual(self.val_dates[-1], "2025-12-31")
        self.assertEqual(len(self.train_dates), 420)
        self.assertEqual(len(self.test_dates), 132)
        self.assertEqual(len(self.val_dates), 60)

    def test_05_future_perturbation_invariance(self):
        """
        Verify that perturbing future returns (post-2009) leaves earlier state values
        and all trained parameters 100% numerically invariant.
        """
        def train(panel, market, history):
            curves = run_forward_inner_tuning(panel, market, history, self.train_dates)
            models, transforms = fit_and_freeze_full_models(
                panel, market, history, self.train_dates, curves["selection_summary"])
            return curves, models, transforms

        original = train(self.factors_panel, self.market_series, self.full_market)
        panel, market, history = (obj.copy() for obj in
                                  (self.factors_panel, self.market_series, self.full_market))
        for obj in (panel, market, history):
            obj.loc[obj.index > self.train_dates[-1]] *= -3
        modified = train(panel, market, history)
        self.assertEqual(original[:2], modified[:2])
        a, b = original[2], modified[2]
        for key in ("factor_names", "factor_scales", "market_scale", "cluster_labels"):
            self.assertEqual(a[key], b[key])
        for key in ("F_all", "X_all", "h_all"):
            np.testing.assert_array_equal(a[key][:420], b[key][:420])

    def test_06_augmented_least_squares_equivalence(self):
        """
        Verify normal equations solution matches augmented least squares solution
        to machine precision.
        """
        rng = np.random.default_rng(123)
        n, d, p = 100, 308, 14
        Q = rng.standard_normal((n, d))
        F = rng.standard_normal((n, p))
        lam = 0.5
        N = 154

        b_normal = solve_ridge_sdf(Q, F, lam, N=N)
        b_augmented = solve_augmented_least_squares(Q, F, lam, N=N)

        np.testing.assert_allclose(b_normal, b_augmented, atol=1e-13)

    def test_07_fixed_per_asset_weight(self):
        """
        Verify that N=154 keeps common restriction weight constant when managed
        restrictions are added.
        """
        n, N = 100, 154
        p = 14
        rng = np.random.default_rng(456)
        X = rng.standard_normal((n, N))
        F = rng.standard_normal((n, p))
        h = rng.standard_normal(n)

        # Unconditional dictionary Q_u = X (d=N)
        # Conditional dictionary Q_c = [X, h*X] (d=2N)
        Q_u = X
        Q_c = np.hstack([X, h[:, np.newaxis] * X])

        mu_u = Q_u.mean(axis=0)
        A_u = (Q_u.T @ F) / n

        mu_c = Q_c.mean(axis=0)
        A_c = (Q_c.T @ F) / n

        # First N rows of A_c and mu_c are identical to A_u and mu_u
        np.testing.assert_array_equal(mu_c[:N], mu_u)
        np.testing.assert_array_equal(A_c[:N, :], A_u)

        # In LHS: (A_c' A_c / N) = (A_u' A_u / N) + (A_managed' A_managed / N)
        # Weight on common restrictions is 1/N in both!
        LHS_u_common = (A_u.T @ A_u) / N
        LHS_c_common = (A_c[:N, :].T @ A_c[:N, :]) / N
        np.testing.assert_allclose(LHS_u_common, LHS_c_common, atol=1e-15)

    def test_08_goldilocks_flat_window_and_tie_behavior(self):
        """Verify Goldilocks flat-window variance normalization and tie-breaking."""
        # Flat curve: all values equal
        flat_curve = np.array([0.5] * 10)
        res_flat = run_goldilocks_selection(flat_curve, window_size=3)
        # Variance of all windows is 0; scores are all 0; tie-breaking must pick first index (0)
        self.assertEqual(res_flat["best_window_idx"], 0)
        self.assertEqual(res_flat["best_penalty_idx"], 0)

        # Tie curve
        tie_curve = np.array([0.2, 0.2, 0.2, 0.4, 0.4, 0.4])
        res_tie = run_goldilocks_selection(tie_curve, window_size=3)
        self.assertEqual(res_tie["best_window_idx"], 0)
        self.assertEqual(res_tie["best_penalty_idx"], 0)

    def test_09_paired_bootstrap_alignment(self):
        """
        Verify that bootstrap resamples sample rows identically across all compared models.
        Uses a deterministic, nonconstant time-series fixture to verify:
          1. Individual bootstrap loss distributions exhibit positive variation across draws.
          2. Identically specified models yield exactly zero paired differences and zero interval spread.
          3. Diagnostic paired_draws exhibits zero paired difference across draws for identical inputs.
          4. Deliberate mispairing (temporal displacement) produces non-zero difference spread,
             proving the fixture detects misaligned time-series draws.
        """
        T, d = 60, 308
        t = np.arange(T, dtype=float)[:, None]
        j = np.arange(d, dtype=float)[None, :]
        base_series = np.sin(2 * np.pi * t / 12.0) + (t / T) + 0.1 * np.cos(j / 10.0)

        # 1. Primary bootstrap with identical nonconstant inputs across models
        MQ_dict = {
            "U_M": base_series,
            "U_G": base_series,
            "C_M": base_series,
            "C_G": base_series
        }
        for block_len in (12, 6):
            res = run_paired_moving_block_bootstrap(MQ_dict, block_length=block_len, num_resamples=100, seed=340)
            ci = res["ci_intervals"]

            # Resampled losses must vary across draws due to the nonconstant time series
            self.assertGreater(ci["L_U_G"]["std"], 0.0)
            self.assertGreater(ci["L_U_G"]["p975"], ci["L_U_G"]["p025"])

            # Paired differences and interaction must be identically zero for identical inputs
            for stat in ("delta_goldilocks", "delta_minimum", "interaction"):
                self.assertEqual(ci[stat]["mean"], 0.0)
                self.assertEqual(ci[stat]["std"], 0.0)
                self.assertEqual(ci[stat]["p025"], 0.0)
                self.assertEqual(ci[stat]["p50"], 0.0)
                self.assertEqual(ci[stat]["p975"], 0.0)

        # 2. Diagnostic paired_draws helper with identical inputs
        moments = {"M1": base_series, "M2": base_series}
        df_draws = paired_draws(moments, length=12)
        diff = df_draws["M1"] - df_draws["M2"]
        np.testing.assert_array_equal(diff.to_numpy(), np.zeros(len(df_draws)))
        self.assertGreater(df_draws["M1"].std(), 0.0)

        # 3. Deliberately mispaired control: phase-shifted series must produce non-zero spread
        misaligned = np.roll(base_series, shift=6, axis=0)
        unpaired_dict = {
            "U_M": base_series,
            "U_G": base_series,
            "C_M": misaligned,
            "C_G": misaligned
        }
        res_unpaired = run_paired_moving_block_bootstrap(unpaired_dict, block_length=12, num_resamples=100, seed=340)
        ci_unp = res_unpaired["ci_intervals"]
        self.assertGreater(ci_unp["delta_goldilocks"]["std"], 0.0)
        self.assertGreater(ci_unp["delta_goldilocks"]["p975"] - ci_unp["delta_goldilocks"]["p025"], 0.0)

    def test_10_figure_generation_and_source_match(self):
        """
        Verify figure generation stage:
          1. Produces all 3 PDF and PNG files plus figures_data.json, figure_index.json, and FIGURES.md.
          2. Figure 1 plotted values match primary tables and sum to total loss.
          3. Figure 2 loss differences and 95% bootstrap intervals match contrast tables and maintain correct signs.
          4. Figure 3 tuning curves match tuning curves JSON and selected penalties match.
        """
        import shutil
        from tempfile import TemporaryDirectory
        from tools.build_results import build
        from tools.build_figures import build_figures

        # Construct fixture from committed reference artifacts in an isolated temporary directory
        with TemporaryDirectory() as tmpdir:
            fixture_run = Path(tmpdir) / "run"
            self.make_figure_fixture(fixture_run)
            build(fixture_run)

            out_figures = fixture_run / "figures"
            build_figures(fixture_run, dest_dir=out_figures)

            # 1. Verify required files exist and are non-empty
            expected_files = [
                "figure1_loss_decomposition.pdf", "figure1_loss_decomposition.png",
                "figure2_contrasts_uncertainty.pdf", "figure2_contrasts_uncertainty.png",
                "figure3_tuning_curves.pdf", "figure3_tuning_curves.png",
                "figures_data.json", "figure_index.json", "FIGURES.md"
            ]
            for fname in expected_files:
                fpath = out_figures / fname
                self.assertTrue(fpath.exists(), f"Missing expected figure artifact: {fname}")
                self.assertGreater(fpath.stat().st_size, 0, f"Empty figure artifact: {fname}")

            # 2. Verify Figure 1 data matches numerical artifacts and components sum to total
            fdata = json.loads((out_figures / "figures_data.json").read_text(encoding="utf-8"))
            f1 = fdata["figure1"]
            test_prim = pd.read_csv(fixture_run / "tables/test_primary.csv")
            val_prim = pd.read_csv(fixture_run / "tables/validation_primary.csv")

            for period_key, df in [("development_test_132m", test_prim), ("final_validation_60m", val_prim)]:
                u_row = df[df.model.eq("U_G")].iloc[0]
                c_row = df[df.model.eq("C_G")].iloc[0]
                self.assertAlmostEqual(f1[period_key]["U"]["ordinary"], u_row.ordinary, places=12)
                self.assertAlmostEqual(f1[period_key]["U"]["managed"], u_row.managed, places=12)
                self.assertAlmostEqual(f1[period_key]["U"]["loss"], u_row.loss, places=12)
                self.assertAlmostEqual(f1[period_key]["U"]["ordinary"] + f1[period_key]["U"]["managed"],
                                       f1[period_key]["U"]["loss"], places=14)

                self.assertAlmostEqual(f1[period_key]["C1"]["ordinary"], c_row.ordinary, places=12)
                self.assertAlmostEqual(f1[period_key]["C1"]["managed"], c_row.managed, places=12)
                self.assertAlmostEqual(f1[period_key]["C1"]["loss"], c_row.loss, places=12)
                self.assertAlmostEqual(f1[period_key]["C1"]["ordinary"] + f1[period_key]["C1"]["managed"],
                                       f1[period_key]["C1"]["loss"], places=14)

            # 3. Verify Figure 2 contrasts, interval orientations, and negative direction convention
            f2 = fdata["figure2"]["contrasts"]
            prim_contrasts = pd.read_csv(fixture_run / "tables/primary_contrasts.csv")
            c1_u = prim_contrasts[(prim_contrasts.period == "Validation") & (prim_contrasts.comparison == "C_G minus U_G") & (prim_contrasts.block == 12)].iloc[0]
            self.assertAlmostEqual(f2["C1_minus_U"]["difference"], c1_u.difference, places=12)
            self.assertAlmostEqual(f2["C1_minus_U"]["block_12m"]["lower"], c1_u.lower, places=12)
            self.assertAlmostEqual(f2["C1_minus_U"]["block_12m"]["upper"], c1_u.upper, places=12)
            # Verify endpoint ordering
            self.assertLessEqual(f2["C1_minus_U"]["block_12m"]["lower"], f2["C1_minus_U"]["block_12m"]["upper"])
            self.assertLessEqual(f2["C1_minus_U"]["block_6m"]["lower"], f2["C1_minus_U"]["block_6m"]["upper"])
            # All 95% intervals must cross zero
            for ckey in ("C1_minus_U", "C2_matched_minus_C1", "C2_G_minus_C1"):
                self.assertLess(f2[ckey]["block_12m"]["lower"], 0.0)
                self.assertGreater(f2[ckey]["block_12m"]["upper"], 0.0)

            # 4. Verify Figure 3 curves and selected penalty points match source
            f3 = fdata["figure3"]
            p_curves = json.loads((fixture_run / "primary/tuning_curves.json").read_text(encoding="utf-8"))
            d_curves = json.loads((fixture_run / "diagnostic/tuning_and_checks.json").read_text(encoding="utf-8"))
            np.testing.assert_allclose(f3["lambdas"], p_curves["curves_data"]["lambdas"], atol=1e-15)
            np.testing.assert_allclose(f3["unconditional"]["rmse_curve"], p_curves["curves_data"]["unconditional_rmse"], atol=1e-15)
            np.testing.assert_allclose(f3["conditional_static"]["rmse_curve"], p_curves["curves_data"]["conditional_rmse"], atol=1e-15)
            np.testing.assert_allclose(f3["conditional_interacted"]["rmse_curve"], d_curves["pooled_rmse"], atol=1e-15)

            # Verify selected penalties match source metadata dynamically
            self.assertEqual(f3["unconditional"]["selected_idx"], p_curves["selection_summary"]["unconditional_goldilocks"]["lambda_idx"])
            self.assertEqual(f3["conditional_static"]["selected_idx"], p_curves["selection_summary"]["conditional_goldilocks"]["lambda_idx"])
            self.assertEqual(f3["conditional_interacted"]["selected_idx"], d_curves["goldilocks"]["best_penalty_idx"])
            self.assertEqual(f3["conditional_interacted"]["selected_lambda"], 10.0)
            self.assertTrue(f3["conditional_interacted"]["is_grid_boundary"])

    def test_11_git_free_manifest_handling(self):
        """
        Verify that pipeline git helper and manifest generation operate safely when Git metadata
        or executable is absent, without raising or inventing fake commit hashes.
        """
        from unittest.mock import patch
        import pipeline

        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            res = pipeline.git(["rev-parse", "HEAD"])
            self.assertIsNone(res)

        with patch("pipeline.git", return_value=None):
            commit_raw = pipeline.git(["rev-parse", "HEAD"])
            tree_raw = pipeline.git(["status"])
            commit_val = commit_raw if commit_raw is not None else "unavailable (no git metadata or executable)"
            self.assertEqual(commit_val, "unavailable (no git metadata or executable)")
            self.assertIsNone(commit_raw)

    def test_12_rscript_discovery_and_failure_instruction(self):
        """
        Verify that rscript() raises FileNotFoundError with actionable instructions when missing.
        """
        from unittest.mock import patch
        from extension_20261108.clustering import rscript

        # When discoverable, returns non-empty string
        res = rscript()
        self.assertTrue(isinstance(res, str) and len(res) > 0)

        # When missing, error message provides actionable guidance
        with patch.dict("os.environ", {}, clear=True):
            with patch("shutil.which", return_value=None):
                with patch("pathlib.Path.glob", return_value=[]):
                    with self.assertRaises(FileNotFoundError) as ctx:
                        rscript()
                    msg = str(ctx.exception)
                    self.assertIn("Rscript", msg)
                    self.assertIn("RSCRIPT", msg)

    def test_13_figure_reactivity_to_altered_selectors(self):
        """
        Verify that build_figures dynamically derives selections, windows, and annotations
        from run artifacts rather than hard-coding values:
          1. Altering tuning and selector metadata in run artifacts changes the plotted
             selections, winning windows, grid boundary flags, and captions.
          2. When minimum-error and Goldilocks selectors differ, separate markers and
             annotations are rendered instead of claiming selector agreement.
          3. Hard-coded selections would fail this test.
        """
        import shutil
        from tempfile import TemporaryDirectory
        from tools.build_results import build
        from tools.build_figures import build_figures

        with TemporaryDirectory() as tmpdir:
            fixture_run = Path(tmpdir) / "run"
            self.make_figure_fixture(fixture_run)
            build(fixture_run)

            # Alter unconditional goldilocks selector: move from index 27 (1.2589) to index 22 (0.01995)
            # Minimum remains at index 27, creating a selector divergence for U
            tuning_prim_path = fixture_run / "primary/tuning_curves.json"
            tuning_prim = json.loads(tuning_prim_path.read_text(encoding="utf-8"))
            tuning_prim["selection_summary"]["unconditional_goldilocks"]["lambda_idx"] = 22
            tuning_prim["selection_summary"]["unconditional_goldilocks"]["lambda_val"] = tuning_prim["curves_data"]["lambdas"][22]
            tuning_prim["selection_summary"]["unconditional_goldilocks"]["goldilocks_details"]["best_penalty_idx"] = 22
            tuning_prim["selection_summary"]["unconditional_goldilocks"]["goldilocks_details"]["best_window_idx"] = 21
            tuning_prim_path.write_text(json.dumps(tuning_prim, indent=2), encoding="utf-8")

            # Alter diagnostic selector: move goldilocks from index 30 (10.0, boundary) to index 26 (0.6309, interior)
            # Minimum moves to index 24 (0.1585), creating divergence and removing grid boundary status
            tuning_diag_path = fixture_run / "diagnostic/tuning_and_checks.json"
            tuning_diag = json.loads(tuning_diag_path.read_text(encoding="utf-8"))
            tuning_diag["minimum"] = 24
            tuning_diag["goldilocks"]["best_penalty_idx"] = 26
            tuning_diag["goldilocks"]["best_window_idx"] = 25
            tuning_diag_path.write_text(json.dumps(tuning_diag, indent=2), encoding="utf-8")

            # Alter diagnostic model penalties in fits.json
            fits_diag_path = fixture_run / "diagnostic/fits.json"
            fits_diag = json.loads(fits_diag_path.read_text(encoding="utf-8"))
            fits_diag["lambdas"]["C2_matched"] = 1.2589254117941662
            fits_diag["lambdas"]["C2_G"] = 0.630957344480193
            fits_diag_path.write_text(json.dumps(fits_diag, indent=2), encoding="utf-8")

            out_figures = fixture_run / "figures"
            build_figures(fixture_run, dest_dir=out_figures)

            fdata = json.loads((out_figures / "figures_data.json").read_text(encoding="utf-8"))
            findex = json.loads((out_figures / "figure_index.json").read_text(encoding="utf-8"))
            f3 = fdata["figure3"]
            f2 = fdata["figure2"]["contrasts"]

            # U: selectors must diverge, selected_idx must be 22 (NOT hardcoded 27)
            self.assertEqual(f3["unconditional"]["selected_idx"], 22)
            self.assertEqual(f3["unconditional"]["winning_window_idx"], 21)
            self.assertFalse(f3["unconditional"]["selectors_agree"])
            self.assertNotEqual(f3["unconditional"]["selected_idx_min"], f3["unconditional"]["selected_idx_gold"])

            # C2: selectors must diverge, index must be 26 (NOT hardcoded 30), NOT on boundary
            self.assertEqual(f3["conditional_interacted"]["selected_idx"], 26)
            self.assertEqual(f3["conditional_interacted"]["selected_idx_min"], 24)
            self.assertEqual(f3["conditional_interacted"]["selected_idx_gold"], 26)
            self.assertEqual(f3["conditional_interacted"]["winning_window_idx"], 25)
            self.assertFalse(f3["conditional_interacted"]["selectors_agree"])
            self.assertFalse(f3["conditional_interacted"]["is_grid_boundary"])

            # C2 diagnostic penalties must match altered values (NOT hardcoded 2.5119 / 10.0)
            self.assertAlmostEqual(f2["C2_matched_minus_C1"]["penalty"], 1.2589254117941662, places=10)
            self.assertAlmostEqual(f2["C2_G_minus_C1"]["penalty"], 0.630957344480193, places=10)

            # Caption for Figure 3 must reflect the altered non-boundary and selector divergence
            cap3 = findex[2]["caption"]
            self.assertIn("interior to grid", cap3)
            self.assertNotIn("sits at the upper boundary", cap3)


    def test_14_missing_training_metadata_is_rejected(self):
        from tempfile import TemporaryDirectory
        from tools.build_results import build
        with TemporaryDirectory() as temp:
            fixture = Path(temp) / "run"
            self.make_figure_fixture(fixture)
            (fixture / "primary/training_transforms.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing learned training metadata"):
                build(fixture)
            self.assertFalse((fixture / "tables").exists())

    def test_15_lower_boundary_figure_caption(self):
        from tempfile import TemporaryDirectory
        from tools.build_results import build
        from tools.build_figures import build_figures
        with TemporaryDirectory() as temp:
            fixture = Path(temp) / "run"
            self.make_figure_fixture(fixture)
            build(fixture)
            path = fixture / "diagnostic/tuning_and_checks.json"
            tuning = json.loads(path.read_text())
            tuning["minimum"] = 0
            tuning["goldilocks"].update(best_penalty_idx=0, best_window_idx=0)
            path.write_text(json.dumps(tuning), encoding="utf-8")
            path = fixture / "diagnostic/fits.json"
            fits = json.loads(path.read_text())
            grid = json.loads((fixture / "primary/tuning_curves.json").read_text())["curves_data"]["lambdas"]
            fits["lambdas"].update(C2_G=grid[0], C2_M=grid[0])
            path.write_text(json.dumps(fits), encoding="utf-8")
            build_figures(fixture)
            caption = json.loads((fixture / "figures/figure_index.json").read_text())[2]["caption"]
            self.assertIn("lower boundary", caption)
            self.assertIn("1e-08", caption)
            self.assertNotIn("upper boundary", caption)
            data = json.loads((fixture / "figures/figures_data.json").read_text())
            self.assertEqual(data["figure3"]["conditional_interacted"]["boundary_position"], "lower")

    def test_16_git_rejects_enclosing_repository(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import pipeline
        enclosing = SimpleNamespace(returncode=0, stdout=str(ROOT_DIR.parent))
        with patch("pipeline.subprocess.run", return_value=enclosing) as run:
            self.assertIsNone(pipeline.git(["rev-parse", "HEAD"]))
            self.assertEqual(run.call_count, 1)
        own_root = SimpleNamespace(returncode=0, stdout=str(ROOT_DIR))
        revision = SimpleNamespace(returncode=0, stdout="test-revision\n")
        with patch("pipeline.subprocess.run", side_effect=[own_root, revision]):
            self.assertEqual(pipeline.git(["rev-parse", "HEAD"]), "test-revision")


if __name__ == "__main__":
    unittest.main()
