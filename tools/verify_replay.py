"""Check newly produced artifacts against the fixed local research release."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def compare(expected, actual, key):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), key
        for name in expected:
            compare(expected[name], actual[name], f"{key}/{name}")
    elif isinstance(expected, list):
        assert len(expected) == len(actual), key
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f"{key}/{i}")
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-9, err_msg=key)
    else:
        assert expected == actual, key


def verify(output_root=ROOT / "runs"):
    results = []
    for group in ("primary", "diagnostic"):
        for ref in sorted((ROOT / "reference" / group).iterdir()):
            if ref.name in ("config.json", "summary_table.md"):
                continue
            out = output_root / group / ref.name
            identical = ref.read_bytes() == out.read_bytes()
            if not identical:
                if ref.suffix == ".csv":
                    pd.testing.assert_frame_equal(pd.read_csv(ref), pd.read_csv(out),
                                                  check_exact=False, atol=1e-11, rtol=1e-9)
                else:
                    compare(json.loads(ref.read_text()), json.loads(out.read_text()), ref.name)
            status = "byte-identical" if identical else "within tolerance"
            print(group, ref.name, status)
            results.append(dict(group=group, file=ref.name, status=status))
    return results


if __name__ == "__main__":
    verify()


