"""Generate research tables from current numerical outputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def write_table(path, title, frame, digits=8):
    # Explicit formatting keeps these reports independent of optional table libraries.
    lines = ["# " + title, "", "| " + " | ".join(frame.columns) + " |",
             "|" + "|".join(["---"] * len(frame.columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(run):
    dest = run / "tables"
    dest.mkdir()
    fits = read(run / "primary/frozen_models.json")
    transforms = read(run / "primary/training_transforms.json")
    diagnostic = pd.read_csv(run / "diagnostic/models.csv")
    all_contrasts = pd.read_csv(run / "diagnostic/contrasts.csv")
    tuning = read(run / "primary/tuning_curves.json")
    bootstrap = read(run / "primary/bootstrap_results.json")
    selected = tuning["selection_summary"]
    grid = np.asarray(tuning["curves_data"]["lambdas"])
    for period, filename in (("Test", "test"), ("Validation", "validation")):
        scores = read(run / "primary" / (filename + "_evaluation.json"))
        state = diagnostic[diagnostic.period.eq(period)].iloc[0]
        rows = []
        for model, metrics in scores["models"].items():
            errors = np.asarray(metrics["e_bar"])
            ordinary = float(np.sum(errors[:154] ** 2) / 308)
            managed = float(np.sum(errors[154:] ** 2) / 308)
            np.testing.assert_allclose(ordinary + managed, metrics["loss"], atol=1e-15, rtol=0)
            rows.append(dict(model=model, coefficients=len(fits[model]["b"]),
                             penalty=fits[model]["lambda"], months=scores["num_months"],
                             high_months=int(state.high), low_months=int(state.low),
                             loss=metrics["loss"], ordinary=ordinary, managed=managed,
                             rmse=metrics["rmse"], ordinary_mae_bps=metrics["mae_bps"]))
        primary = pd.DataFrame(rows)
        primary.to_csv(dest / (filename + "_primary.csv"), index=False)
        write_table(dest / (filename + "_primary.md"),
                    period + " primary experiment (14 coefficients in each model)", primary)
        d = diagnostic[diagnostic.period.eq(period)].copy()
        d.insert(2, "coefficients", [14 if k in ("U", "C1") else 28 for k in d.model])
        d.to_csv(dest / (filename + "_diagnostic.csv"), index=False)
        write_table(dest / (filename + "_diagnostic.md"), period + " post-hoc diagnostic", d)

    contrasts = []
    for period in ("Test", "Validation"):
        scores = read(run / "primary" / (period.lower() + "_evaluation.json"))
        fits_equal = all(fits[a]["b"] == fits[b]["b"] for a, b in (("U_G", "U_M"), ("C_G", "C_M")))
        assert fits_equal, "Test-period paired primary draws require equal M/G fits in this release."
        for block in (12, 6):
            c1_minus_u = all_contrasts[
                all_contrasts.period.eq(period) & all_contrasts.model.eq("U") &
                all_contrasts.comparator.eq("C1") & all_contrasts.block.eq(block)
            ].iloc[0]
            if period == "Validation":
                key = "primary_block_12" if block == 12 else "sensitivity_block_6"
                ci = bootstrap[key]["ci_intervals"]["delta_goldilocks"]
                np.testing.assert_allclose([-c1_minus_u.upper, -c1_minus_u.lower],
                                           [ci["p025"], ci["p975"]], rtol=0, atol=1e-15)
            for rule in ("G", "M"):
                contrasts.append(dict(period=period, comparison="C_" + rule + " minus U_" + rule,
                                      block=block, difference=scores["delta_goldilocks" if rule == "G" else "delta_minimum"],
                                      lower=-c1_minus_u.upper, upper=-c1_minus_u.lower))
            contrasts.append(dict(period=period, comparison="tuning interaction",
                                  block=block, difference=scores["interaction"], lower=0., upper=0.))
    primary_contrasts = pd.DataFrame(contrasts)
    primary_contrasts.to_csv(dest / "primary_contrasts.csv", index=False)
    write_table(dest / "primary_contrasts.md", "Primary contrasts and paired percentile intervals", primary_contrasts)
    all_contrasts.to_csv(dest / "diagnostic_contrasts.csv", index=False)

    methods = {
        "primary_coefficients": {k: len(v["b"]) for k, v in fits.items()},
        "ordinary_test_assets": len(transforms["factor_names"]) + 1,
        "theme_count": len(set(transforms["cluster_labels"])),
        "primary_evaluation_moments": 308,
        "training_fitted_cutoff": transforms["volatility_metadata"]["c_threshold"],
        "training_high_frequency": transforms["volatility_metadata"]["p_H_train_frequency"],
        "penalty_choices": {},
    }
    for key, choice in selected.items():
        detail = dict(penalty=choice["lambda_val"], index=choice["lambda_idx"])
        if "goldilocks_details" in choice:
            start = choice["goldilocks_details"]["best_window_idx"]
            detail["winning_window"] = grid[start:start+3].tolist()
        methods["penalty_choices"][key] = detail
    (dest / "method_details.json").write_text(json.dumps(methods, indent=2))
    text = """# Methods and interpretation for the generated tables

The primary experiment compares two fitting dictionaries crossed with two tuning rules.
Every primary SDF has 14 coefficients for 13 training-fitted themes and the market.
U fits the 154 ordinary payoff restrictions. C fits both ordinary and volatility-managed
restrictions. All four models score the same 308 payoffs. The fixed intercept is one.

The state uses the sample standard deviation of the preceding 12 monthly market excess
returns, ending before the return month. Its cutoff is the training two-thirds quantile.
The indicator is centered and standardized using its training frequency. Each factor
is divided by its own training standard deviation before averaging within a theme.
The training objective divides the sum of squared mean moments by 154 in both fits.
Evaluation divides by 308 after taking each discounted payoff's period mean and squaring.

Goldilocks adds the min-max-normalized mean and variance across each three-penalty window.
It chooses the lowest-score window, then the lowest-error penalty inside that window.
First index breaks ties. It uses expanding forward folds inside 1975–2009 Training.
The primary hypotheses predict a negative conditional-minus-ordinary contrast under
Goldilocks and a negative interaction relative to minimum-error tuning.

The post-hoc diagnostic appends hF to the original F, giving the SDF 28 coefficients.
It uses the same standardized binary state and the same 308 evaluation payoffs.
Its main contrast compares separately Goldilocks-tuned C2 with static conditional C1.
The matched comparison uses C1's exact saved penalty. Both later periods were already
observed before this diagnostic was proposed. Intervals condition on the frozen fits.

Twelve-month moving blocks are primary and six-month blocks are sensitivity, each using
999 paired monthly resamples at seed 340. Primary contrast intervals here are recovered
from the same diagnostic draws for U and C1. Equal M/G fitted coefficients were checked
before applying those draws to both rules and recording their exactly zero interaction.
Ordinary and managed columns each use denominator 308, so they sum to total loss.
Ordinary MAE is restored to raw return units and then converted to monthly basis points.

The inputs and numerical output files beside this folder are the sources of the tables.
The prior Gemini audit prose misdescribed the state, selector and parameter counts.
Use these regenerated tables with the frozen protocols when writing the manuscript.
"""
    val = read(run / "primary/validation_evaluation.json")
    increase = 100 * val["delta_goldilocks"] / val["models"]["U_G"]["loss"]
    lead = all_contrasts[
        all_contrasts.period.eq("Validation") & all_contrasts.model.eq("C2_G") &
        all_contrasts.comparator.eq("C1") & all_contrasts.block.eq(12)
    ].iloc[0]
    text += (f"\nIn the 2021–2025 primary sample, managed restrictions raised squared pricing loss "
             f"by {increase:.2f}%. The tuning interaction was {val['interaction']:.8f}. "
             f"The separately tuned diagnostic changed loss by {lead.contrast:+.8f}, "
             f"with a 12-month paired interval [{lead.lower:+.8f}, {lead.upper:+.8f}]. "
             "Neither primary directional hypothesis received support. These observed "
             "outcomes and intervals leave the population direction uncertain.\n")
    (dest / "METHODS_AND_RESULTS.md").write_text(text, encoding="utf-8")

