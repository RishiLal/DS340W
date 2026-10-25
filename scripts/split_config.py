"""
split_config.py — Centralized Split Configuration and Chronological Boundaries
DS340W Capstone Project: Three-Paper Synthesis

Defines explicit date boundaries and observation counts across all pipeline stages:
  1. Representation Learning Split (T_rep = 180 return months):
     - Returns: August 1976 to July 1991 (180 months)
     - Preceding States: July 1976 to June 1991 (180 months)
     - Used exclusively for state imputation, normalization, GAN training, and PCA basis.
  2. Econometric Inference Split (T_inf = 167 return months):
     - Returns: August 1991 to June 2005 (167 months)
     - Preceding States: July 1991 to May 2005 (167 months)
     - Used exclusively for factor pricing evaluation (Reverse Cross-Fitting PLR).
  3. Outer Out-of-Sample Test Split (T_test = 100 return months):
     - Returns: July 2005 to October 2013 (100 months)
     - Preceding State for first return: June 2005 (last month of training split)
  4. Outer Validation Split (T_val = 50 return months):
     - Returns: November 2013 to December 2017 (50 months)
     - Preceding State for first return: October 2013 (last month of test split)
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data_splits"

# Explicit date constants
REP_RETURN_START = 19760831
REP_RETURN_END   = 19910731
INF_RETURN_START = 19910830
INF_RETURN_END   = 20050630

TEST_RETURN_START = 20050729
TEST_RETURN_END   = 20131031
VAL_RETURN_START  = 20131129
VAL_RETURN_END    = 20171229

T_REP_MONTHS = 180
T_INF_MONTHS = 167
T_TRAIN_LAGGED_TOTAL = 347
T_TEST_MONTHS = 100
T_VAL_MONTHS = 50


def load_raw_splits(data_dir=DATA_DIR):
    """Load raw factor and portfolio return splits and verify integrity."""
    f_tr = pd.read_csv(data_dir / "factors_train_70.csv")
    f_te = pd.read_csv(data_dir / "factors_test_20.csv")
    f_va = pd.read_csv(data_dir / "factors_val_10.csv")

    p_tr = pd.read_csv(data_dir / "port_3x2_excess_train_70.csv")
    p_te = pd.read_csv(data_dir / "port_3x2_excess_test_20.csv")
    p_va = pd.read_csv(data_dir / "port_3x2_excess_val_10.csv")

    # Assert date equality between factors and portfolios
    assert (f_tr["Date"].values == p_tr["Date"].values).all(), "Train factor/portfolio date mismatch"
    assert (f_te["Date"].values == p_te["Date"].values).all(), "Test factor/portfolio date mismatch"
    assert (f_va["Date"].values == p_va["Date"].values).all(), "Val factor/portfolio date mismatch"

    return (f_tr, f_te, f_va), (p_tr, p_te, p_va)


def get_representation_and_inference_splits(f_tr, p_tr):
    """
    Partition the 348 raw training months into strictly lagged representation (180)
    and inference (167) pairs.
    """
    assert len(f_tr) == 348, f"Expected 348 raw training observations, found {len(f_tr)}"

    factor_cols = [c for c in f_tr.columns if c not in ["Date", "RF"]]
    port_cols = [c for c in p_tr.columns if c != "Date"]

    dates = f_tr["Date"].values

    # Preceding states (rows 0 to 346) and return targets (rows 1 to 347)
    Z_all_raw = f_tr[factor_cols].copy()
    R_all_raw = p_tr[port_cols].copy()

    # Representation Split:
    # States: rows 0 to 179 (dates[0] to dates[179])
    # Returns: rows 1 to 180 (dates[1] to dates[180])
    rep_state_df = Z_all_raw.iloc[:T_REP_MONTHS].copy()
    rep_ret_df   = R_all_raw.iloc[1:T_REP_MONTHS + 1].copy()
    rep_return_dates = dates[1:T_REP_MONTHS + 1]
    rep_state_dates  = dates[:T_REP_MONTHS]

    assert len(rep_return_dates) == T_REP_MONTHS
    assert rep_return_dates[0] == REP_RETURN_START
    assert rep_return_dates[-1] == REP_RETURN_END

    # Inference Split:
    # States: rows 180 to 346 (dates[180] to dates[346])
    # Returns: rows 181 to 347 (dates[181] to dates[347])
    inf_state_df = Z_all_raw.iloc[T_REP_MONTHS:-1].copy()
    inf_ret_df   = R_all_raw.iloc[T_REP_MONTHS + 1:].copy()
    inf_return_dates = dates[T_REP_MONTHS + 1:]
    inf_state_dates  = dates[T_REP_MONTHS:-1]

    assert len(inf_return_dates) == T_INF_MONTHS
    assert inf_return_dates[0] == INF_RETURN_START
    assert inf_return_dates[-1] == INF_RETURN_END

    return {
        "factor_cols": factor_cols,
        "port_cols": port_cols,
        "rep": {
            "states_raw": rep_state_df,
            "returns_raw": rep_ret_df,
            "return_dates": rep_return_dates,
            "state_dates": rep_state_dates
        },
        "inf": {
            "states_raw": inf_state_df,
            "returns_raw": inf_ret_df,
            "return_dates": inf_return_dates,
            "state_dates": inf_state_dates
        }
    }
