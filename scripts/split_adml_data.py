"""
Split ADML Data into 70% Training, 20% Testing, and 10% Validation Sets
========================================================================
Paper: "Identifying Factors via Automatic Debiased Machine Learning"
Authors: Esfandiar Maasoumi, Jianqiu Wang, Zhuo Wang, Ke Wu (2023)
Data: 150 equity factors, 750 bivariate portfolios (port_3x2), 1825 fine portfolios (port_5x5), 202 anomaly portfolios (port202).

Chronological Partition:
- Total Sample: T = 498 months (1976-07 to 2017-12)
- Training (70%): Months 1 to 348 (1976-07-30 to 2005-06-30, 348 months)
- Testing (20%): Months 349 to 448 (2005-07-29 to 2013-10-31, 100 months)
- Validation (10%): Months 449 to 498 (2013-11-29 to 2017-12-29, 50 months)
"""

import os
import json
import numpy as np
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rep_dir = os.path.join(base_dir, 'resources', 'replication_files_all', 'Identifying factors via automatic debiased machine learning (replication data)')
    out_dir = os.path.join(base_dir, 'data_splits')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("ADML DATA SPLIT PIPELINE (70% Train / 20% Test / 10% Validation)")
    print("=" * 80)

    # 1. Load factors.csv
    factors_path = os.path.join(rep_dir, 'factors.csv')
    factors_df = pd.read_csv(factors_path)
    T_total = len(factors_df)
    dates = factors_df['Date'].values
    rf = factors_df['RF'].values

    print(f"Factors loaded: {factors_df.shape[0]} months x {factors_df.shape[1]} columns (Date, RF, + 150 factors)")
    print(f"Time span: {dates[0]} to {dates[-1]}")

    # Determine split indices (chronological, no lookahead)
    n_train = int(np.floor(0.70 * T_total))  # 348
    n_test = int(np.floor(0.20 * T_total))   # 99.6 -> let's make test exactly 100 so val is 50
    # 348 + 100 + 50 = 498 (70%, 20.08%, 10.04%)
    n_test = 100
    n_val = T_total - n_train - n_test       # 50

    idx_train = slice(0, n_train)
    idx_test = slice(n_train, n_train + n_test)
    idx_val = slice(n_train + n_test, T_total)

    train_dates = (dates[0], dates[n_train - 1])
    test_dates = (dates[n_train], dates[n_train + n_test - 1])
    val_dates = (dates[n_train + n_test], dates[-1])

    print(f"\nPartitioning Protocol:")
    print(f"  Training Set (70%):   {n_train} months ({n_train/T_total*100:.2f}%) -> {train_dates[0]} to {train_dates[1]}")
    print(f"  Testing Set  (20%):   {n_test} months ({n_test/T_total*100:.2f}%)  -> {test_dates[0]} to {test_dates[1]}")
    print(f"  Validation   (10%):   {n_val} months ({n_val/T_total*100:.2f}%)   -> {val_dates[0]} to {val_dates[1]}")

    # Save factor splits
    factors_train = factors_df.iloc[idx_train]
    factors_test = factors_df.iloc[idx_test]
    factors_val = factors_df.iloc[idx_val]

    factors_train.to_csv(os.path.join(out_dir, 'factors_train_70.csv'), index=False)
    factors_test.to_csv(os.path.join(out_dir, 'factors_test_20.csv'), index=False)
    factors_val.to_csv(os.path.join(out_dir, 'factors_val_10.csv'), index=False)
    print("  -> Saved factors splits to data_splits/factors_*.csv")

    # 2. Process port_3x2 (Primary Test Assets: 750 portfolios)
    port_3x2_path = os.path.join(rep_dir, 'port_3x2.csv')
    port_3x2_id_path = os.path.join(rep_dir, 'port_3x2_id.csv')
    port_3x2_raw = pd.read_csv(port_3x2_path, header=None)
    # Col 0 is date, remaining are returns
    port_3x2_ret = port_3x2_raw.iloc[:, 1:].values
    # Excess returns
    port_3x2_excess = port_3x2_ret - rf[:, np.newaxis]

    port_3x2_id = pd.read_csv(port_3x2_id_path)
    # Portfolios with min_stk >= 10
    include_3x2_idx = np.where(port_3x2_id.iloc[:, 3].values >= 10)[0]
    cols_3x2 = []
    for idx in include_3x2_idx:
        cols_3x2.extend(range(idx * 6, (idx + 1) * 6))

    port_3x2_filtered = pd.DataFrame(port_3x2_excess[:, cols_3x2])
    port_3x2_filtered.columns = [f"port_3x2_{i+1}" for i in range(port_3x2_filtered.shape[1])]
    port_3x2_filtered.insert(0, 'Date', dates)

    port_3x2_train = port_3x2_filtered.iloc[idx_train]
    port_3x2_test = port_3x2_filtered.iloc[idx_test]
    port_3x2_val = port_3x2_filtered.iloc[idx_val]

    port_3x2_train.to_csv(os.path.join(out_dir, 'port_3x2_excess_train_70.csv'), index=False)
    port_3x2_test.to_csv(os.path.join(out_dir, 'port_3x2_excess_test_20.csv'), index=False)
    port_3x2_val.to_csv(os.path.join(out_dir, 'port_3x2_excess_val_10.csv'), index=False)
    print(f"  -> Saved port_3x2 excess return splits (750 assets) to data_splits/port_3x2_excess_*.csv")

    # 3. Process port_5x5 (Robustness Portfolios: 1,825 portfolios)
    port_5x5_path = os.path.join(rep_dir, 'port_5x5.csv')
    port_5x5_id_path = os.path.join(rep_dir, 'port_5x5_id.csv')
    port_5x5_raw = pd.read_csv(port_5x5_path, header=None)
    port_5x5_ret = port_5x5_raw.iloc[:, 1:].values
    port_5x5_excess = port_5x5_ret - rf[:, np.newaxis]

    port_5x5_id = pd.read_csv(port_5x5_id_path)
    include_5x5_idx = np.where(port_5x5_id.iloc[:, 3].values >= 10)[0]
    cols_5x5 = []
    for idx in include_5x5_idx:
        cols_5x5.extend(range(idx * 25, (idx + 1) * 25))

    port_5x5_filtered = pd.DataFrame(port_5x5_excess[:, cols_5x5])
    port_5x5_filtered.columns = [f"port_5x5_{i+1}" for i in range(port_5x5_filtered.shape[1])]
    port_5x5_filtered.insert(0, 'Date', dates)

    port_5x5_train = port_5x5_filtered.iloc[idx_train]
    port_5x5_test = port_5x5_filtered.iloc[idx_test]
    port_5x5_val = port_5x5_filtered.iloc[idx_val]

    port_5x5_train.to_csv(os.path.join(out_dir, 'port_5x5_excess_train_70.csv'), index=False)
    port_5x5_test.to_csv(os.path.join(out_dir, 'port_5x5_excess_test_20.csv'), index=False)
    port_5x5_val.to_csv(os.path.join(out_dir, 'port_5x5_excess_val_10.csv'), index=False)
    print(f"  -> Saved port_5x5 excess return splits (1825 assets) to data_splits/port_5x5_excess_*.csv")

    # 4. Process port202 (Giglio & Xiu Anomaly Portfolios: 202 assets)
    port202_path = os.path.join(rep_dir, 'port202.csv')
    port202_raw = pd.read_csv(port202_path, header=None)
    port202_ret = port202_raw.iloc[:, 1:].values / 100.0  # converted from percent to decimal
    port202_excess = port202_ret - rf[:, np.newaxis]

    port202_df = pd.DataFrame(port202_excess)
    port202_df.columns = [f"port_202_{i+1}" for i in range(port202_df.shape[1])]
    port202_df.insert(0, 'Date', dates)

    port202_train = port202_df.iloc[idx_train]
    port202_test = port202_df.iloc[idx_test]
    port202_val = port202_df.iloc[idx_val]

    port202_train.to_csv(os.path.join(out_dir, 'port_202_excess_train_70.csv'), index=False)
    port202_test.to_csv(os.path.join(out_dir, 'port_202_excess_test_20.csv'), index=False)
    port202_val.to_csv(os.path.join(out_dir, 'port_202_excess_val_10.csv'), index=False)
    print(f"  -> Saved port_202 excess return splits (202 assets) to data_splits/port_202_excess_*.csv")

    # 5. Metadata manifest
    manifest = {
        "dataset_name": "ADML Replication Data Splits",
        "reference_paper": "Identifying Factors via Automatic Debiased Machine Learning (Maasoumi et al., 2023)",
        "total_months": int(T_total),
        "split_ratios": {"training": 0.70, "testing": 0.20, "validation": 0.10},
        "splits": {
            "training_70": {
                "months": int(n_train),
                "percentage": float(round(n_train / T_total * 100, 2)),
                "start_date": int(train_dates[0]),
                "end_date": int(train_dates[1]),
                "row_indices": [0, n_train - 1]
            },
            "testing_20": {
                "months": int(n_test),
                "percentage": float(round(n_test / T_total * 100, 2)),
                "start_date": int(test_dates[0]),
                "end_date": int(test_dates[1]),
                "row_indices": [n_train, n_train + n_test - 1]
            },
            "validation_10": {
                "months": int(n_val),
                "percentage": float(round(n_val / T_total * 100, 2)),
                "start_date": int(val_dates[0]),
                "end_date": int(val_dates[1]),
                "row_indices": [n_train + n_test, T_total - 1]
            }
        },
        "datasets": {
            "factors": {"columns": 150, "features": "150 equity risk factor returns + RF + Date"},
            "port_3x2_excess": {"columns": 750, "features": "750 bivariate sorted portfolios with >= 10 stocks"},
            "port_5x5_excess": {"columns": 1825, "features": "1825 fine-sorted portfolios with >= 10 stocks"},
            "port_202_excess": {"columns": 202, "features": "202 anomaly portfolios from Giglio and Xiu (2021)"}
        }
    }

    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=4)
    print("  -> Saved manifest.json")
    print("=" * 80)
    print("DATA SPLIT COMPLETE SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()
