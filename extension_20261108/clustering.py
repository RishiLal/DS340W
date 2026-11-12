"""
Hierarchical clustering (Ward.D on 1-correlation) and theme payoff construction.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def rscript():
    configured = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if configured:
        return configured
    candidates = sorted(Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"))
    if candidates:
        return str(candidates[-1])
    raise FileNotFoundError("Install R and add Rscript to PATH or set RSCRIPT.")

def run_r_ward_d_clustering(cor_mat, k=13):
    """
    Run R's hclust(as.dist(1 - cor), method = 'ward.D') and cutree(..., k = 13)
    to guarantee exact equivalence to JKP's Ward.D on 1-correlation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cor_csv = tmp_path / "cor.csv"
        out_csv = tmp_path / "clusters.csv"
        
        # Save correlation matrix
        np.savetxt(cor_csv, cor_mat, delimiter=",")
        
        r_script = """
        args <- commandArgs(trailingOnly = TRUE)
        cor <- as.matrix(read.csv(args[1], header = FALSE))
        d <- as.dist(1 - cor)
        hc <- hclust(d, method = 'ward.D')
        cl <- cutree(hc, k = as.integer(args[3]))
        write.csv(cl, args[2], row.names = FALSE)
        """
        script_file = tmp_path / "script.R"
        script_file.write_text(r_script)
        
        res = subprocess.run([rscript(), "--vanilla", str(script_file), str(cor_csv), str(out_csv), str(k)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"R clustering failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        
        cluster_labels = pd.read_csv(out_csv)["x"].values
        assert len(cluster_labels) == cor_mat.shape[0]
        assert len(np.unique(cluster_labels)) == k
        return cluster_labels

def fit_theme_representation(train_factors, train_market, full_factors, full_market, k=13):
    """
    Fit transformations, scales, and Ward.D clustering on train data,
    then construct theme payoffs F and test payoffs X for full data.
    
    Args:
        train_factors: DataFrame (n_train x 153)
        train_market: Series (n_train)
        full_factors: DataFrame (n_full x 153)
        full_market: Series (n_full)
        k: number of clusters (13)
        
    Returns:
        F_full: ndarray (n_full x 14) theme payoffs (13 clusters + scaled market)
        X_full: ndarray (n_full x 154) test payoffs (153 scaled factors + scaled market)
        scales: dict of training standard deviations
        cluster_labels: ndarray (153,) cluster assignments for the 153 factors
    """
    # 1. Training standard deviations (ddof=1)
    factor_scales = train_factors.std(ddof=1).values
    mkt_scale = train_market.std(ddof=1)
    assert (factor_scales > 0).all(), "Zero standard deviation found in training factors"
    assert mkt_scale > 0, "Zero standard deviation in training market"
    
    # 2. Market-adjust each factor on training data
    mkt_var = train_market.var(ddof=1)
    covs = np.array([train_factors.iloc[:, i].cov(train_market) for i in range(train_factors.shape[1])])
    betas = covs / mkt_var
    adj_train = train_factors.values - np.outer(train_market.values, betas)
    
    # 3. Correlation matrix of market-adjusted returns
    cor_mat = pd.DataFrame(adj_train).corr().values
    
    # 4. Ward.D clustering via R
    cluster_labels = run_r_ward_d_clustering(cor_mat, k=k)
    
    # 5. Build full test payoffs X (scaled by training std)
    # Dimension: n_full x 154
    scaled_factors_full = full_factors.values / factor_scales[np.newaxis, :]
    scaled_market_full = (full_market.values / mkt_scale)[:, np.newaxis]
    X_full = np.hstack([scaled_factors_full, scaled_market_full])
    assert X_full.shape == (full_factors.shape[0], full_factors.shape[1] + 1)
    
    # 6. Build full theme payoffs F (13 equal-weight cluster averages of scaled factors + scaled market)
    # Dimension: n_full x 14
    F_themes = np.zeros((full_factors.shape[0], k))
    for c_idx, c_label in enumerate(sorted(np.unique(cluster_labels))):
        member_mask = (cluster_labels == c_label)
        # Equal-weight average of scaled member returns
        F_themes[:, c_idx] = scaled_factors_full[:, member_mask].mean(axis=1)
        
    F_full = np.hstack([F_themes, scaled_market_full])
    assert F_full.shape == (full_factors.shape[0], k + 1)
    
    scales_info = {
        "factor_scales": factor_scales,
        "market_scale": mkt_scale,
        "factor_names": full_factors.columns.tolist()
    }
    
    return F_full, X_full, scales_info, cluster_labels
