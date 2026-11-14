"""
Comprehensive audit and verification test suite for the extension pipeline.
Covers all requirements from GEMINI_OVERNIGHT_HANDOFF.md Section 5.
"""
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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

if __name__ == "__main__":
    unittest.main()
