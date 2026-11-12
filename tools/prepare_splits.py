"""Build the assignment exports and lag history directly from verified inputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extension_20261108.data import load_screened_universe

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    panel, market, history, names, dates = load_screened_universe()
    months = pd.PeriodIndex(dates, freq="M")
    assert months.equals(pd.period_range("1975-01", "2025-12", freq="M"))
    joined = panel.assign(mkt_vw_exc=market)
    config = json.loads((ROOT / "protocol/config.json").read_text())
    ledger = []
    for key, role in (("training", "Training"), ("development_test", "Test"),
                      ("final_validation", "Validation")):
        spec = config["sample_split"][key]
        part = joined.loc[spec["start"]:spec["end"]]
        assert len(part) == spec["months"] and part.shape[1] == 154
        target = destination / (role + ".csv")
        part.to_csv(target, index_label="date")
        exported = pd.read_csv(target, index_col="date")
        reference = pd.read_csv(ROOT / "data/splits" / target.name, index_col="date")
        pd.testing.assert_frame_equal(exported, part, check_names=False, rtol=0, atol=1e-15)
        pd.testing.assert_frame_equal(exported, reference, rtol=0, atol=1e-15)
        ledger.append(dict(role=role, first=part.index[0], last=part.index[-1],
                           months=len(part), series=part.shape[1]))
    warmup = history.loc["1974-01-01":"1974-12-31"]
    assert len(warmup) == 12 and np.isfinite(warmup).all()
    assert pd.PeriodIndex(warmup.index, freq="M").equals(pd.period_range("1974-01", "1974-12", freq="M"))
    warmup.to_csv(destination / "Market_warmup_1974.csv", index_label="date")
    (destination / "manifest.json").write_text(json.dumps(
        dict(partitions=ledger, ordered_assets=names + ["mkt_vw_exc"],
             warmup_months=12, csv_comparison_absolute_tolerance=1e-15),
        indent=2))
    return ledger

