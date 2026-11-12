"""Reproduce the post-hoc state-dependent SDF comparison from its fixed protocol."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from extension_20261108.clustering import fit_theme_representation
from extension_20261108.data import compute_volatility_state, load_screened_universe
from extension_20261108.models import (
    compute_unconditional_basis_points, evaluate_sdf_moments,
    solve_augmented_least_squares, solve_ridge_sdf,
)
from extension_20261108.tuning import get_penalty_grid, run_goldilocks_selection

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "reference" / "primary"


def json_file(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def build(panel, market, history, n, end):
    """Learn on rows [:n] and apply through end, preserving original payoff order."""
    dates = panel.index[:end].tolist()
    h, _, state = compute_volatility_state(history, dates, dates[:n])
    f, x, scales, labels = fit_theme_representation(
        panel.iloc[:n], market.iloc[:n], panel.iloc[:end], market.iloc[:end]
    )
    h = h.to_numpy()
    q = np.column_stack([x, h[:, None] * x])
    basis = np.column_stack([f, h[:, None] * f])
    np.testing.assert_array_equal(basis[:, :14], f)
    meta = dict(names=scales["factor_names"], scales=scales["factor_scales"].tolist(),
                market_scale=float(scales["market_scale"]), clusters=labels.tolist(),
                cutoff=state["c_threshold"], train_high_fraction=state["p_H_train_frequency"])
    return basis, q, h, meta


def perturb(panel, market, history, cutoff):
    p, m, hist = panel.copy(), market.copy(), history.copy()
    for data in (p, m, hist):
        data.loc[data.index > cutoff] *= -3
    return p, m, hist


def fit_curves_and_check(panel, market, history):
    grid = get_penalty_grid()
    curves, audits = [], []
    for n in (210, 280, 350, 420):
        end = min(n + 70, 420)
        z, q, h, meta = build(panel, market, history, n, end)
        # Perturb all subsequent data, including a fold's scoring observations.
        z2, q2, _, meta2 = build(*perturb(panel, market, history, panel.index[n-1]), n, end)
        assert meta2 == meta
        np.testing.assert_array_equal(z2[:n], z[:n])
        np.testing.assert_array_equal(q2[:n], q[:n])
        # After the end of scoring even the scored payoffs must be invariant.
        z3, q3, _, meta3 = build(*perturb(panel, market, history, panel.index[end-1]), n, end)
        assert meta3 == meta
        np.testing.assert_array_equal(z3, z)
        np.testing.assert_array_equal(q3, q)
        assert set(np.sign(h)) == {-1., 1.}
        probe = np.linspace(-0.2, 0.4, 28)
        predictions = np.sum((probe[:14] + h[:, None] * probe[14:]) * z[:, :14], axis=1)
        np.testing.assert_allclose(z @ probe, predictions, rtol=0, atol=1e-12)
        constrained = z.copy()
        constrained[:, 14:] = 0
        losses = []
        max_solver = 0.
        for lam in grid:
            b = solve_ridge_sdf(q[:n], z[:n], lam)
            other = solve_augmented_least_squares(q[:n], z[:n], lam)
            max_solver = max(max_solver, float(np.linalg.norm(b-other)))
            np.testing.assert_array_equal(b, solve_ridge_sdf(q2[:n], z2[:n], lam))
            # Use the same column layout for the left block in both products.
            one = solve_ridge_sdf(q[:n], z[:n, :14], lam)
            reduced = solve_ridge_sdf(q[:n], constrained[:n], lam)
            np.testing.assert_allclose(reduced, np.r_[one, np.zeros(14)], rtol=0, atol=1e-12)
            np.testing.assert_allclose(constrained @ reduced, z[:, :14] @ one, rtol=0, atol=1e-12)
            if end > n:
                loss = evaluate_sdf_moments(b, z[n:], q[n:])["loss"]
                assert loss == evaluate_sdf_moments(b, z3[n:], q3[n:])["loss"]
                losses.append(loss)
        assert max_solver < 1e-10, (n, max_solver)
        audits.append(dict(prefix=n, end=end, worst_solver_l2=max_solver))
        if losses:
            curves.append(losses)
    pooled = np.sqrt(np.mean(curves, axis=0))
    gold = run_goldilocks_selection(pooled)
    return dict(grid=grid.tolist(), fold_losses=curves, pooled_rmse=pooled.tolist(),
                minimum=int(np.argmin(pooled)), goldilocks=gold, prefix_checks=audits)


def record_eval(b, z, q, meta):
    ev = evaluate_sdf_moments(np.asarray(b), z, q)
    err = ev["e_bar"]
    ordinary = float(np.sum(err[:154]**2) / 308)
    managed = float(np.sum(err[154:]**2) / 308)
    np.testing.assert_allclose(ordinary + managed, ev["loss"], rtol=0, atol=1e-15)
    bps = compute_unconditional_basis_points(err[:154], meta["scales"], meta["market_scale"])
    return dict(loss=ev["loss"], ordinary=ordinary, managed=managed, rmse=ev["rmse"],
                ordinary_mae_bps=bps["mae_bps"]), ev["MQ"]


def paired_draws(moments, length):
    """Each draw uses one shared vector of contiguous monthly indices."""
    rng = np.random.default_rng(340)
    n = len(next(iter(moments.values())))
    rows = []
    for draw in range(999):
        starts = rng.integers(n - length + 1, size=int(np.ceil(n / length)))
        ix = (starts[:, None] + np.arange(length)).ravel()[:n]
        row = {k: float(np.mean(v[ix].mean(axis=0)**2)) for k, v in moments.items()}
        rows.append(dict(draw=draw+1, block=length, **row))
    return pd.DataFrame(rows)


def run(dest):
    hashes = json.loads((ROOT / "reference/input_and_protocol_hashes.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    panel, market, hist, _, dates = load_screened_universe()
    months = pd.PeriodIndex(dates, freq="M")
    assert months.equals(pd.period_range("1975-01", "2025-12", freq="M"))
    exports = [pd.read_csv(ROOT / "data/splits" / (role + ".csv"), index_col="date")
               for role in ("Training", "Test", "Validation")]
    assert [len(p) for p in exports] == [420, 132, 60]
    pd.testing.assert_frame_equal(pd.concat(exports), panel.assign(mkt_vw_exc=market),
                                  rtol=0, atol=1e-15, check_names=False)
    saved = json.loads((PRIMARY / "frozen_models.json").read_text())
    curve = fit_curves_and_check(panel, market, hist)
    json_file(dest / "tuning_and_checks.json", curve)
    z, q, _, fit_meta = build(panel, market, hist, 420, 420)
    variants = {"U": saved["U_G"]["lambda"], "C1": saved["C_G"]["lambda"],
                "C2_matched": saved["C_G"]["lambda"],
                "C2_G": curve["grid"][curve["goldilocks"]["best_penalty_idx"]],
                "C2_M": curve["grid"][curve["minimum"]]}
    coef = {k: saved[s]["b"] for k, s in (("U","U_G"), ("C1","C_G"))}
    for k in ("C2_matched", "C2_G", "C2_M"):
        coef[k] = solve_ridge_sdf(q, z, variants[k]).tolist()
    for k in ("U", "C1"):
        checks = q[:, :154] if k == "U" else q
        np.testing.assert_allclose(solve_ridge_sdf(checks, z[:, :14], variants[k]), coef[k],
                                   rtol=0, atol=1e-12)
    json_file(dest / "fits.json", dict(lambdas=variants, coefficients=coef, transforms=fit_meta))
    z, q, h, meta = build(panel, market, hist, 420, 612)
    assert meta == fit_meta
    results, contrasts, all_draws = [], [], []
    for period, start, end, reference in (("Test",420,552,"test"), ("Validation",552,612,"validation")):
        prior = json.loads((PRIMARY / (reference + "_evaluation.json")).read_text())
        rows, moments = {}, {}
        for name, b in coef.items():
            score_basis = z[start:end, :len(b)]
            metrics, moments[name] = record_eval(b, score_basis, q[start:end], meta)
            if name in ("U", "C1"):
                base = prior["models"]["U_G" if name == "U" else "C_G"]
                np.testing.assert_allclose(moments[name].mean(axis=0), base["e_bar"], rtol=0, atol=1e-12)
            rows[name] = metrics
            results.append(dict(period=period, model=name, penalty=variants[name],
                                high=int(sum(h[start:end] > 0)), low=int(sum(h[start:end] < 0)), **metrics))
        for length in (12, 6):
            draws = paired_draws(moments, length)
            draws.insert(0, "period", period)
            all_draws.append(draws)
            pairs = [(k, "C1") for k in coef] + [("C2_G","U"), ("C2_G","C2_M"), ("C2_G","C2_matched")]
            for a, b in pairs:
                low, high = np.quantile(draws[a]-draws[b], [.025, .975])
                contrasts.append(dict(period=period, model=a, comparator=b, block=length,
                                      contrast=rows[a]["loss"]-rows[b]["loss"], lower=low, upper=high))
    pd.DataFrame(results).to_csv(dest / "models.csv", index=False)
    pd.DataFrame(contrasts).to_csv(dest / "contrasts.csv", index=False)
    pd.concat(all_draws).to_csv(dest / "paired_draws.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/diagnostic")
    target = parser.parse_args().out.resolve()
    if target.exists():
        parser.error("Choose a new output directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as tmp:
        stage = Path(tmp) / "results"
        stage.mkdir()
        run(stage)
        stage.rename(target)
    print(f"Verified both periods and saved results to {target}")

