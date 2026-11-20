# Reproduce the results

These steps run the new DS340W methodology and generate its tables and three figures. All required portfolio-return data are included. The original JKP benchmark has a separate command in [RUN_AND_PUBLISH.md](RUN_AND_PUBLISH.md).

## 1. Download the correct version

On GitHub, select the release branch `codex/jkp-release` (or the exact commit linked in the submission). Choose **Code → Download ZIP**, then extract it. Open a terminal in the extracted folder that contains `pipeline.py` and `requirements.txt`.

If you use Git, after the release is published:

```powershell
git clone --branch codex/jkp-release https://github.com/RishiLal/DS340W.git DS340W-reproduce
Set-Location DS340W-reproduce
```

## 2. Install prerequisites

Install **Python 3.12** and **R**. The tested versions are Python 3.12.13 and R 4.6.1 on Windows. The extension uses base R only; no extra R packages, GPU, or data-service account are needed.

Check the Python interpreter before continuing:

```powershell
python --version
```

It should report Python 3.12. If it reports a different version, use the executable for your Python 3.12 installation in the next command.

## 3. Set up and run

In PowerShell, from the folder containing `pipeline.py`, run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe pipeline.py
```

Dependency installation needs internet access and can take several minutes. Once installed, the analysis runs locally using the included data. Recent full runs took about 30–40 seconds on the tested laptop; other machines may take longer. You do not need to activate the environment or change PowerShell's execution policy.

For later runs, only the last command is needed. Every run gets a new output folder.

## 4. Confirm success

The terminal should finish with **All stages passed** and print the output folder under `runs/`.

- `run_manifest.json` should have `"status": "passed"`.
- `logs/tests.log` should report **16 tests**, `OK`, and no skips.
- `logs/numerical_comparison.log` should report **ten matching files**, either byte-identical or within the documented numerical tolerance.

A nonzero exit or failed check needs investigation. Do not ignore it or change the expected results to make it pass.

## 5. Open the results

In the new run folder:

For example, if you extracted the project to `C:\Users\Alex\Desktop\DS340W-reproduce`, the default results will be under `C:\Users\Alex\Desktop\DS340W-reproduce\runs\pipeline_<timestamp>\`. The terminal prints the exact path. Open that folder in File Explorer; CSV files open in a spreadsheet app, PDFs in a PDF viewer, and Markdown files in a text editor. Nothing is automatically uploaded to GitHub. You can choose a predictable folder by running `.\.venv\Scripts\python.exe pipeline.py --out runs/my_run` from the project folder, provided that folder does not already exist.

| What to read | File or folder |
|---|---|
| Main final-period comparison | `tables/validation_primary.md` |
| Exploratory model comparison | `tables/validation_diagnostic.md` |
| Differences and confidence intervals | `tables/primary_contrasts.csv`, `tables/diagnostic_contrasts.csv` |
| Figures ready to insert into the paper | `figures/` — PDF and PNG versions |
| Figure captions and data sources | `figures/FIGURES.md` |
| Fitted models and detailed results | `primary/`, `diagnostic/` |

For a quick numerical check, final-period loss should be approximately **0.01534692 for U** and **0.01647338 for C1**, a **7.34% increase**. The primary bootstrap interval includes zero. These measure pricing errors, not investment returns or future-return prediction accuracy.

## If something fails

- **Rscript not found:** install R, or set its actual executable path before running. For the tested Windows installation:

```powershell
$env:RSCRIPT = 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe'
.\.venv\Scripts\python.exe pipeline.py
```

- **Missing Python package:** rerun the dependency-installation command using `.venv\Scripts\python.exe`.
- **Output directory already exists:** use the default command, or supply a new unused directory with `--out runs/another_run`.
- **Data hash or numerical comparison failed:** retain the logs and check the release version, input files, and package versions. Do not substitute newer data.

The extension works without Git metadata, including from an extracted ZIP. Windows execution is verified; macOS/Linux instructions are provided in the longer guide but have not been execution-tested. Small floating-point differences are checked against the stated tolerance; identical image bytes or fonts on every machine are not promised.
