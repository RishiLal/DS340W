# Volatility Conditioning and US Factor Pricing (DS340W Extension & Replication)

This repository contains Rishi Lal's DS340W pricing extension, its required portfolio-return data, and a separate workflow for the original JKP benchmark. The extension studies 153 US factor portfolios plus the market. After installing Python, R, and the pinned Python dependencies, run `python pipeline.py` to regenerate its models, tables, and figures.

- **Central Research Plan & Results:** [RESEARCH_PLAN.md](RESEARCH_PLAN.md)
- **Start here — download, run, and find the results:** [QUICKSTART.md](QUICKSTART.md)
- **Data Inventory & Provenance:** [DATA_AND_SOURCES.md](DATA_AND_SOURCES.md)
- **Software Versions:** [ENVIRONMENT.md](ENVIRONMENT.md) and [requirements.txt](requirements.txt)
- **Run and GitHub Publication Guide:** [RUN_AND_PUBLISH.md](RUN_AND_PUBLISH.md)
- **Latest Local Verification:** [RELEASE_VERIFICATION.md](RELEASE_VERIFICATION.md)
- **Publication Figures & Index:** [reference/figures/FIGURES.md](reference/figures/FIGURES.md)

---

## 1. Project Overview & Findings

The project evaluates out-of-sample pricing accuracy using the updated US factor returns from Jensen, Kelly, and Pedersen (2023), extended through December 2025. It compares an unconditional SDF ($U$, 14 coefficients) against a static conditional SDF ($C1$, 14 coefficients) incorporating 154 volatility-managed restrictions (Chen, Pelger, and Zhu, 2024), evaluated under both minimum-error and Goldilocks window stability tuning (Ciganovic, D'Amario, and Tancioni, 2026), followed by an exploratory post-hoc diagnostic model allowing time-varying state interactions ($C2$, 28 coefficients).

### Key Scientific Findings:
- **Primary Finding (Validation 2021–2025, 60 months):** Conditioning **raised** observed squared pricing loss by **+7.34%**, increasing from $0.015347$ to $0.016473$ ($\Delta = +0.001126$). Although the error on volatility-managed positions fell slightly ($0.003448 \to 0.003399$), this minor improvement was outweighed by an increase in pricing error on the 154 ordinary asset restrictions ($0.011899 \to 0.013075$).
- **Statistical Uncertainty:** The 12-month moving-block paired 95% bootstrap confidence interval is $[-0.000278, +0.007289]$ (6-month sensitivity: $[-0.000562, +0.005995]$). Because the interval includes zero, the sample difference does not establish a general population effect.
- **Tuning Agreement & Interaction:** Both Minimum-Error and Goldilocks window tuning selected identical shrinkage penalties ($\lambda = 1.2589$ for $U$, $\lambda = 2.5119$ for $C1$), resulting in an exact zero tuning interaction ($I = 0.000000$ [0.000000, 0.000000]).
- **Exploratory Diagnostic (28 coefficients):** Adding $h_{t-1} F_t$ interactions yielded higher validation loss ($0.016530$ at the matched penalty $\lambda = 2.5119$; $0.017743$ when re-tuned at the grid boundary $\lambda = 10.0$). The 95% intervals cross zero ($[-0.000472, +0.003473]$ and $[-0.000600, +0.008900]$), leaving the population direction uncertain.
- **Scope Note:** Metrics measure out-of-sample asset pricing error on discounted payoff moments, not future stock return predictability, trading strategy profits, or cumulative investment wealth.

---

## 2. Supported Environment & Verification Status

- **Host Platform (Tested):** Windows 11 Enterprise (x86_64, Windows-11-10.0.26200-SP0)
- **Python Version (Tested):** Python 3.12.13 (CPython AMD64)
- **Key Python Packages:** `numpy==2.5.4`, `pandas==3.1.1`, `matplotlib==3.11.2`, `python-dateutil==4.0.0`, `tzdata==2026.6` (plus pinned transitive dependencies in `requirements.txt`)
- **R Version (Tested):** R 4.6.1 (2026-06-24) / `Rscript.exe`
- **R Usage:** Base R `hclust(..., method = 'ward.D')` and `cutree(..., k = 13)` only; no external R libraries are required for the extension pipeline.
- **Hardware Requirements:** Standard x86_64 CPU; no GPU, cloud connection, or WRDS credentials required.
- **Portability Statement:** *Verified on this Windows laptop using isolated environments; independent-machine and other-OS execution not tested.*

---

## 3. Installation & Setup

### Windows PowerShell Setup (Tested)

Install Python 3.12 and R first (tested versions are listed above). Open PowerShell in the folder containing `pipeline.py` and `requirements.txt`, whether downloaded or cloned. Run:

```powershell
# 1. Create an isolated virtual environment
python -m venv .venv

# 2. Install pinned dependencies (using explicit venv interpreter)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Verify unit tests (all 16 tests, 0 skipped)
.\.venv\Scripts\python.exe -m unittest -v extension_20261108.test_pipeline
```

*Note: In Windows instructions, explicit interpreter paths (e.g. `.\.venv\Scripts\python.exe`) are used to eliminate any requirement for `Activate.ps1` or changes to PowerShell execution policies.*

### macOS / Linux Setup (Untested Equivalent)

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Install dependencies
./.venv/bin/python -m pip install -r requirements.txt

# 3. Run unit tests
./.venv/bin/python -m unittest -v extension_20261108.test_pipeline
```

### R Configuration & Discovery

The extension automatically discovers `Rscript` if it is on system `PATH` or located under standard Windows installation paths (e.g. `C:\Program Files\R\R-*\bin\Rscript.exe`). If Rscript is installed in a non-standard location:

- **PowerShell:** `$env:RSCRIPT = "C:\Path\To\R\bin\Rscript.exe"`
- **Bash:** `export RSCRIPT="/path/to/R/bin/Rscript"`

---

## 4. One-Command Extension Execution

The entire extension pipeline runs from a single command:

```powershell
# Default run (creates a fresh timestamped directory under runs/):
.\.venv\Scripts\python.exe pipeline.py

# Explicit clean output directory:
.\.venv\Scripts\python.exe pipeline.py --out runs/my_run
```

Existing output directories are strictly rejected with exit code 2 to prevent accidental data overwrites.

The default output folder is inside the folder containing `pipeline.py`, even if the command is launched from elsewhere. An explicit relative `--out` path is relative to your terminal's current directory. For example, from `C:\Users\Alex\Desktop\DS340W-reproduce`, `--out runs/my_run` saves everything under `C:\Users\Alex\Desktop\DS340W-reproduce\runs\my_run\`. Results are local files; they are not automatically uploaded to GitHub or opened in an app.

### What `pipeline.py` Does (All 8 Stages):
1. **`preflight`:** Validates assertion enablement, discovers Rscript, and logs versions.
2. **`tests`:** Executes 16 tests covering input hashes, screening, chronological splits, information locality, solver equivalence, fitting weights, Goldilocks tie-breaking, paired bootstrap alignment, portable figure generation, selector reactivity, missing-metadata rejection, both penalty-grid boundaries, and Git/R handling. Figure fixtures regenerate learned training metadata from the bundled inputs; no previous run folder is required.
3. **`splits`:** Rebuilds `Training.csv` (420 mo), `Test.csv` (132 mo), `Validation.csv` (60 mo), and 1974 market warmup from the bundled author-supplied portfolio-return archives.
4. **`primary`:** Fits 4 primary models, computes 3-fold forward tuning curves, scores Test and Validation periods, and draws 999 moving-block bootstrap resamples.
5. **`diagnostic`:** Fits the 28-coefficient state-dependent diagnostic models, scores both holdout periods, and generates paired contrast draws.
6. **`numerical_comparison`:** Verifies ten generated numerical artifacts against frozen references. It reports byte identity when available; otherwise it checks structure and numerical agreement with absolute tolerance 1e-11 and relative tolerance 1e-9. Different environments may differ in insignificant floating-point digits. Reference matching supplements the methodology tests; it does not by itself establish scientific validity.
7. **`tables`:** Exports Markdown and CSV research tables to `tables/`.
8. **`figures`:** Generates vector PDF and 300 DPI PNG publication figures, structured `figures_data.json`, `figure_index.json`, and `FIGURES.md`.

### Measured Runtimes:
- **Dependency Installation:** Depends on download speed and package caches; allow several minutes.
- **Full Pipeline Execution:** Usually under one minute on the tested laptop; see the latest measured run in [RELEASE_VERIFICATION.md](RELEASE_VERIFICATION.md).

### Recognizing a Successful Run:
A successful execution prints:
```text
Running preflight
Running tests
Running splits
Running primary
Running diagnostic
Running numerical_comparison
Running tables
Running figures
All stages passed. Results and tables are in <dest>
```
and exits with code `0`. Inspect `<dest>/run_manifest.json` for `"status": "passed"`, elapsed runtime, stage summaries, and source/output SHA-256 digests.

### Generated Artifacts in the Output Directory:
```text
runs/<run_name>/
├── run_manifest.json                     # Complete execution provenance and hashes
├── logs/                                 # Detailed per-stage stdout/stderr logs
│   ├── preflight.log, tests.log, splits.log, primary.log,
│   ├── diagnostic.log, numerical_comparison.log, tables.log, figures.log
├── splits/                               # Reconstructed factor panels
│   ├── Training.csv, Test.csv, Validation.csv, Market_warmup_1974.csv, manifest.json
├── primary/                              # Primary model fits and evaluations
│   ├── frozen_models.json, tuning_curves.json, test_evaluation.json,
│   ├── validation_evaluation.json, bootstrap_results.json, summary_table.md
├── diagnostic/                           # Diagnostic model fits and contrasts
│   ├── fits.json, models.csv, contrasts.csv, paired_draws.csv, tuning_and_checks.json
├── tables/                               # Manuscript tables and method summaries
│   ├── METHODS_AND_RESULTS.md, method_details.json, primary_contrasts.md,
│   ├── test_primary.md, validation_primary.md, test_diagnostic.md, validation_diagnostic.md
└── figures/                              # Publication figures and plotting data
    ├── figure1_loss_decomposition.pdf, figure1_loss_decomposition.png
    ├── figure2_contrasts_uncertainty.pdf, figure2_contrasts_uncertainty.png
    ├── figure3_tuning_curves.pdf, figure3_tuning_curves.png
    ├── figures_data.json, figure_index.json, FIGURES.md
```

---

## 5. Publication Figures

Figures consume run numerical artifacts directly and use a headless backend (`matplotlib.use('Agg')`):

1. **Figure 1: Decomposition of Out-of-Sample Squared Pricing Loss**
   - *PDF:* [`figures/figure1_loss_decomposition.pdf`](reference/figures/figure1_loss_decomposition.pdf) | *PNG:* [`figures/figure1_loss_decomposition.png`](reference/figures/figure1_loss_decomposition.png)
   - Stacked bars decomposing loss into ordinary (154 assets) and volatility-managed (154 assets) contributions across Development Test (2010–2020) and Final Validation (2021–2025). Shows that managed error reduction was outweighed by ordinary error expansion in Validation.
2. **Figure 2: Validation Loss Differences and 95% Bootstrap Intervals**
   - *PDF:* [`figures/figure2_contrasts_uncertainty.pdf`](reference/figures/figure2_contrasts_uncertainty.pdf) | *PNG:* [`figures/figure2_contrasts_uncertainty.png`](reference/figures/figure2_contrasts_uncertainty.png)
   - Point estimates and 95% moving-block paired bootstrap intervals (12-month primary, 6-month sensitivity) for $C - U$, $C2\text{-matched} - C1$, and $C2\_G - C1$. Negative differences favor the first-named model. Shows that all intervals span zero.
3. **Figure 3: Training Forward-Fold Tuning Curves and Penalty Selection**
   - *PDF:* [`figures/figure3_tuning_curves.pdf`](reference/figures/figure3_tuning_curves.pdf) | *PNG:* [`figures/figure3_tuning_curves.png`](reference/figures/figure3_tuning_curves.png)
   - Three-panel display showing pooled validation RMSE across 31 log-spaced ridge penalties for $U$, $C1$, and $C2$. Shaded regions highlight winning Goldilocks windows; markers show exact agreement between minimum-error and Goldilocks selectors ($\lambda = 1.259, 2.512, 10.0$).

---

## 6. Troubleshooting & Common Errors

1. **`error: Run Python with assertions enabled.`**
   - *Cause:* Python was invoked with `-O` or `-OO`.
   - *Resolution:* Run standard Python without optimization flags (`python pipeline.py`).
2. **`error: Choose a new output directory.`**
   - *Cause:* The directory specified by `--out` already exists.
   - *Resolution:* Supply a fresh, non-existent path (e.g. `--out runs/my_run_02`).
3. **`FileNotFoundError: R executable (Rscript) not found.`**
   - *Cause:* R is not installed or Rscript is not in PATH or standard Program Files.
   - *Resolution:* Install R from CRAN, add to PATH, or set `$env:RSCRIPT = "C:\Path\To\Rscript.exe"`.
4. **Running from an Unzipped Archive (Git-Free Execution):**
   - `pipeline.py` detects when Git metadata or the Git binary is absent. It records `git_available: false` and `commit: unavailable` in `run_manifest.json` without failing, retaining full source file hashing.

---

## 7. Data Sources, Scope & Attribution

### Bundled Extension Inputs:
- `evidence/JKP_current_US_factors.zip` (2.67 MB, SHA-256: `f766aa2c...`)
- `evidence/JKP_current_US_market.zip` (19.7 KB, SHA-256: `a90c1f2c...`)
- **Vintage:** December 2025 frozen vintage from [jkpfactors.com](https://jkpfactors.com/data).
- **Format:** Capped-value-weighted monthly excess returns in USD, decimal units, factor direction convention already applied.
- **License:** Noncommercial reuse under CC BY-NC 4.0 with attribution to Jensen, Kelly, and Pedersen (2023).

### Parent Paper Replication (Separate Workflow):
The parent publication archive `evidence/JKP_original_data.zip` (286 MB, SHA-256: `74e8b394...`) is excluded from Git tracking due to file size. To reproduce the original JKP US benchmark:

```powershell
git submodule update --init
.\.venv\Scripts\python.exe replicate.py acquire --archive evidence/JKP_original_data.zip
.\.venv\Scripts\python.exe demo.py --archive evidence/JKP_original_data.zip
```

Skip acquisition if you already have the exact archive; `demo.py` verifies its members before running. The submodule command is for Git clones and is optional for this demo, which uses the verified archive's analysis code. On first use the demo installs its additional R packages into `env/R-library/`. The earlier parent package versions are in [evidence/R_versions_reference.csv](evidence/R_versions_reference.csv); unlike the extension's Python requirements, this is an inventory rather than a dependency lockfile. See [ENVIRONMENT.md](ENVIRONMENT.md).

The demo prints the three benchmark checks and saves `local/last_verified_US_run.json`. A later successful demo replaces that summary. It does not generate the extension's tables or figures.

- **Scope Distinction:**
  - `python pipeline.py` reproduces the **new DS340W extension methodology** (volatility conditioning, managed portfolios, Goldilocks tuning, state interactions, uncertainty).
  - `python demo.py` reproduces the **parent empirical Bayes factor benchmark** (matching published Table I US $\tau_c = 0.35\%$, $\tau_w = 0.21\%$, and 98/119 replicated factors).
  - Multi-hour historical global simulations (5+ hours) and proprietary WRDS security-level microdata are outside the scope recomputed by the default commands. Retained historical validation checks are preserved in `reference/parent/`.

---

## 8. License & Reuse Notice

The underlying historical and updated factor data are © 2026 Jensen, Kelly, and Pedersen and provided under CC BY-NC 4.0. Upstream replication code in `upstream/ReplicationCrisis` is subject to the authors' original distribution terms. The DS340W extension research code, protocols, and artifacts are prepared for academic evaluation. No open-source software license has been assigned to the project's own code; the third-party data terms remain separate.
