# Software versions and setup

Use Python 3.12 and R. The verified Windows environment uses **Python 3.12.13** and **R 4.6.1 (2026-06-24)**. These are tested versions, not a claim that every newer version or operating system has been verified.

## Main extension: exact Python dependencies

[requirements.txt](requirements.txt) pins every installed runtime dependency. Install it in a fresh environment using the commands in [QUICKSTART.md](QUICKSTART.md).

| Package | Pinned version |
|---|---|
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| matplotlib | 3.11.2 |
| python-dateutil | 4.0.0 |
| tzdata | 2026.6 |
| contourpy | 1.4.0 |
| cycler | 0.12.1 |
| fonttools | 4.68.0 |
| kiwisolver | 1.5.1 |
| packaging | 26.3 |
| pillow | 13.0.0 |
| pyparsing | 3.3.3 |

The extension uses base R's Ward.D hierarchical clustering and does not require additional R packages. A CPU is sufficient. Python package installation needs internet access; the extension analysis subsequently reads the bundled data locally. Each run records the actual Python, R and package versions and the requirements-file SHA-256 in `run_manifest.json`.

## Separate original-paper benchmark

The parent workflow uses additional R packages. [tools/setup_R.R](tools/setup_R.R) installs them into the project's isolated `env/R-library/` directory. [evidence/R_versions_reference.csv](evidence/R_versions_reference.csv) records the package versions from the earlier verified parent environment. It is a version inventory, **not an automatically restored R lockfile**: CRAN dependencies are resolved at installation time, while the `drlib` GitHub dependency is pinned to a commit.

The setup writes its actual installed inventory to `evidence/R_packages.csv` and its R session details to `evidence/R_sessionInfo.txt`. These generated files and the R library are ignored by Git. Parent results must pass their numerical assertions; changing reference values to accommodate a failure is not an acceptable repair.

On November 20 the automatic setup installed a fresh isolated R library, and the US benchmark passed all three numerical checks. Its [170-package inventory, including base R](reference/parent/R_versions_20261120.csv) and [benchmark result](reference/parent/fresh_US_20261120.json) are included.

## Portability and results

Windows is the execution-tested platform. Independent hardware and macOS/Linux remain untested. The pipeline compares ten numerical artifacts using absolute tolerance `1e-11` and relative tolerance `1e-9` if their bytes differ. Figures may have platform-specific font or metadata differences.

Default extension output is `runs/pipeline_<UTC timestamp>/` inside this project. `--out runs/my_run` resolves relative to the terminal's current directory; run it from the project root as documented. Existing destinations are rejected. The parent demonstration writes `local/last_verified_US_run.json` inside this project; a later successful parent demo replaces that summary.
