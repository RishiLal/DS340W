"""
Data loading, verification, screening, and volatility state preparation
for the extension methodology.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"

FACTOR_ZIP = EVIDENCE / "JKP_current_US_factors.zip"
MARKET_ZIP = EVIDENCE / "JKP_current_US_market.zip"

EXPECTED_HASHES = {
    "JKP_current_US_factors.zip": "f766aa2c7db1f97594f79f9063275161b7038e936db9ea960eac6eb7ec27ca80",
    "JKP_current_US_market.zip": "a90c1f2c8e89bf5cc45219d8f678696458412f19380769448bbaa2c9cc567752",
}

def verify_hashes():
    """Verify SHA-256 hashes of input zip files."""
    for filename, expected in EXPECTED_HASHES.items():
        path = EVIDENCE / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        with path.open("rb") as f:
            computed = hashlib.file_digest(f, "sha256").hexdigest()
        if computed != expected:
            raise ValueError(f"Hash mismatch for {filename}: expected {expected}, got {computed}")
    return True

def load_screened_universe():
    """
    Load current US factors and market excess return, screen on first 210 Training months,
    and assert key uniqueness and completeness.
    Returns:
        factor_panel: DataFrame (612 months x 153 factors) from 1975-01 to 2025-12
        market_series: Series (612 months) from 1975-01 to 2025-12
        full_market: Series (all available dates) for rolling lag calculation
        eligible_factors: list of 153 factor names
        dates: list of 612 date strings
    """
    verify_hashes()
    
    # Load factors
    factors_df = pd.read_csv(FACTOR_ZIP)
    assert factors_df.location.eq("usa").all(), "Non-USA factors found"
    assert factors_df.weighting.eq("vw_cap").all(), "Non-vw_cap factors found"
    assert factors_df.freq.eq("monthly").all(), "Non-monthly factors found"
    assert not factors_df.duplicated(["name", "date"]).any(), "Duplicate factor keys found"
    
    # Load market
    market_df = pd.read_csv(MARKET_ZIP).sort_values("date").reset_index(drop=True)
    assert market_df.location.eq("usa").all(), "Non-USA market found"
    assert market_df.name.eq("mkt").all(), "Non-mkt series found"
    assert not market_df.duplicated(["date"]).any(), "Duplicate market dates found"
    
    # Screen on first 210 Training months (1975-01-01 to 1992-06-30)
    init_mask = (factors_df.date >= "1975-01-01") & (factors_df.date <= "1992-06-30") & (factors_df.n_stocks_min >= 5)
    init_df = factors_df.loc[init_mask]
    stats = init_df.groupby("name").ret.agg(["count", "std"])
    eligible = stats[(stats["count"] == 210) & (stats["std"] > 0)].index.tolist()
    eligible = sorted(eligible)
    assert len(eligible) == 153, f"Expected 153 eligible factors, got {len(eligible)}"
    
    # Target period: 1975-01-01 to 2025-12-31 (612 months)
    target_mask = (factors_df.date >= "1975-01-01") & (factors_df.date <= "2025-12-31") & factors_df.name.isin(eligible)
    panel = factors_df.loc[target_mask].pivot(index="date", columns="name", values="ret")[eligible]
    assert panel.shape == (612, 153), f"Expected shape (612, 153), got {panel.shape}"
    assert panel.isna().sum().sum() == 0, "Missing cells found in factor panel"
    
    # Target market
    mkt_target = market_df[(market_df.date >= "1975-01-01") & (market_df.date <= "2025-12-31")].set_index("date")["ret"]
    assert len(mkt_target) == 612, f"Expected 612 market months, got {len(mkt_target)}"
    assert not mkt_target.isna().any(), "Missing cells found in market series"
    assert (panel.index == mkt_target.index).all(), "Date mismatch between factor panel and market series"
    
    # Full market for lag calculation
    full_market = market_df.set_index("date")["ret"]
    
    return panel, mkt_target, full_market, eligible, panel.index.tolist()

def compute_volatility_state(full_market, target_dates, train_dates):
    """
    Compute predetermined trailing 12-month market volatility state h_{t-1}.
    For month t, v_{t-1} is the sample standard deviation of market returns over [t-12, t-1] (denominator 11).
    c is the 2/3 quantile of v_{t-1} over train_dates (linear interpolation).
    p_H is the training frequency of v_{t-1} > c (strict greater-than).
    h_{t-1} = (1{v_{t-1} > c} - p_H) / sqrt(p_H * (1 - p_H)).
    """
    # Build a series of trailing 12-month standard deviation ending at t-1
    all_dates = full_market.index.tolist()
    v_dict = {}
    for d in target_dates:
        idx = all_dates.index(d)
        assert idx >= 12, f"Insufficient lag history for date {d}"
        lag_window = full_market.iloc[idx - 12 : idx].values
        assert len(lag_window) == 12, f"Lag window length {len(lag_window)} != 12"
        v_dict[d] = np.std(lag_window, ddof=1)
        
    v_series = pd.Series(v_dict)
    
    # Compute threshold on train_dates
    v_train = v_series.loc[train_dates]
    c = float(np.quantile(v_train.values, 2.0 / 3.0))
    ind_train = (v_train > c).astype(float)
    p_H = float(ind_train.mean())
    assert 0 < p_H < 1, f"Invalid p_H: {p_H}"
    
    # Compute standardized state for all target_dates
    ind_all = (v_series > c).astype(float)
    scale = np.sqrt(p_H * (1.0 - p_H))
    h_series = (ind_all - p_H) / scale
    
    metadata = {
        "c_threshold": c,
        "p_H_train_frequency": p_H,
        "train_high_count": int(ind_train.sum()),
        "train_total_count": len(train_dates),
        "target_high_count": int(ind_all.sum()),
        "target_total_count": len(target_dates)
    }
    
    return h_series, v_series, metadata
