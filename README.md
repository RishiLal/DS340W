# Volatility conditioning and US factor pricing

Start with the [central research plan, results and three-day checklist](RESEARCH_PLAN.md). The [input inventory and attribution](DATA_AND_SOURCES.md) identifies the original author code, both data vintages and the three assignment CSVs.

## Replay the extension

Run these commands from the release root. The reference system used Windows, Python 3.12.13, NumPy 2.5.4, pandas 3.1.1 and R 4.6.1. The extension uses base R for Ward.D. Set the environment variable RSCRIPT to its executable if Rscript is outside PATH and outside a standard Windows installation. Use regular Python with assertions enabled. In PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest extension_20261108.test_pipeline
.\.venv\Scripts\python.exe -m extension_20261108.run_extension
.\.venv\Scripts\python.exe -m diagnostic.run
.\.venv\Scripts\python.exe tools/verify_replay.py
~~~

On Unix use `.venv/bin/python` in place of the Windows interpreter. Fresh primary outputs go to `runs/primary`; the diagnostic requires a fresh `runs/diagnostic` or an explicit unused `--out` directory. The verifier compares default outputs to committed references. A successful check prints the comparison for each file. JSON and CSV structure are checked on numerical fallback with absolute tolerance 1e-11 and relative tolerance 1e-9. The diagnostic additionally checks fit isolation, original moments and solver agreement before saving both period tables and draws. See [primary results](reference/primary/summary_table.md) and [diagnostic contrasts](reference/diagnostic/contrasts.csv).

## Run the original JKP benchmark

Initialize the read-only upstream source pointer with `git submodule update --init`. The native computation uses the authors’ historical ZIP. The original driver, R scripts, two metadata workbooks and processed data are materialized together by the prepare command.

~~~powershell
python replicate.py acquire --archive evidence/JKP_original_data.zip
python demo.py --archive evidence/JKP_original_data.zip
~~~

The demo installs its own R packages if needed. You can pass `--rscript` and `--rlib` for an existing installation. See [observed R package versions](evidence/R_versions_reference.csv). The setup script installs available package versions from CRAN and pins the drlib Git revision. It is not a complete R dependency lock. The last verified US fit used 10,000 bootstrap draws and took 132 seconds including archive verification with packages installed. Its three comparisons are saved under `local/last_verified_US_run.json`.

## Broader parent workflow

Use a fresh run directory for a cold fit. The native workflow uses the project-local `env/R-library`. Supply the R executable path explicitly if needed. From PowerShell, with Rscript on PATH and the Python environment active:

~~~powershell
Rscript --vanilla tools/setup_R.R .
python replicate.py prepare --archive evidence/JKP_original_data.zip --run runs/native
python replicate.py run --run runs/native
python replicate.py compare --run runs/native
Rscript --vanilla tools/rerender_figures.R . runs/native
~~~

The native run command writes timestamped invocation logs and can resume in the same prepared directory with the same inputs and environment. Verify the JSON comparisons after a successful exit. Rendering uses the completed checkpoints. The compatibility layer loads the broom tidier, uses a scalar mean in one plotting helper and explicitly prints figures. In the historical run the largest simulation alone took about 5 h 16 min. The original native Table I and US/world checks are in [the comparison ledger](reference/parent/TableI_comparison.csv). Only content rendering was checked for the complete set of 37 figure PDFs. Use the published captions for figure numbering.

