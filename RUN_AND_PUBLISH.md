# Run the study and publish the reviewed release

For the shortest reader instructions, use [QUICKSTART.md](QUICKSTART.md). The rest of this guide also covers data provenance, the parent benchmark, and the owner's publication steps.

## What can be reproduced

The default command recomputes the new methodology: input checks, splits, primary and exploratory models, tuning, bootstrap uncertainty, tables, and three figures. It fits models from the bundled author-supplied portfolio returns. Saved reference results are used for verification, not substituted for model fitting.

The study has 153 factor portfolios plus the market, not 154 individual stock-characteristic factors. Training, development, and final evaluation periods are fixed by the protocol. Fixed settings such as the seed, dates, and penalty grid are intentional research choices. Learned coefficients, state cutoffs, selected penalties, losses, and intervals are computed from data.

The original JKP benchmark is a separate command below. The default pipeline does not rerun all parent-paper analyses or reconstruct returns from underlying security-level records.

## Reader: download and run

Use the published release branch or commit identified in the submission. An older default branch may contain a different project. On GitHub, select that branch before choosing Code > Download ZIP, then extract the archive. Alternatively, clone the published branch with Git. GitHub documents [cloning a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

Install Python 3.12 and R. Tested versions: Python 3.12.13 and R 4.6.1 on Windows. The extension only needs base R for Ward.D clustering. Internet access is needed to install dependencies; afterwards the extension uses the bundled data without downloads, credentials, or a GPU.

Open PowerShell in the extracted/cloned folder containing `pipeline.py`. Check that `python --version` is Python 3.12, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe pipeline.py
```

No PowerShell activation script or execution-policy change is required. If Rscript cannot be found, set `RSCRIPT` to the executable in your actual R installation before running:

```powershell
$env:RSCRIPT = 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe'
```

The last command creates a new timestamped directory under `runs/`. To name it yourself, use a directory that does not already exist:

```powershell
.\.venv\Scripts\python.exe pipeline.py --out runs/my_verified_run
```

The macOS/Linux equivalents use `python3 -m venv .venv` and `.venv/bin/python`, with Rscript on PATH. Those platforms have not been execution-tested for this release.

### Check success and find the results

The terminal must finish with `All stages passed`. In the printed output directory:

- `run_manifest.json`: status, stage timings, code/data hashes, requirements hash, and installed package versions.
- `logs/tests.log`: 16 tests, zero failures, zero skips.
- `logs/numerical_comparison.log`: ten matching numerical artifacts (byte-identical or within the documented tolerance).
- `tables/validation_primary.csv` and `validation_diagnostic.csv`: final-period model comparison.
- `tables/primary_contrasts.csv` and `diagnostic_contrasts.csv`: paired differences and uncertainty.
- `tables/test_primary.csv` and `test_diagnostic.csv`: development-period results.
- `figures/`: three vector PDFs, three PNGs, plotted values, captions, and an index.
- `primary/` and `diagnostic/`: fitted coefficients, selected penalties, tuning curves, evaluations, and bootstrap results.

Expected final-period losses, rounded for checking: U 0.01534692; C1 0.01647338; C2-matched 0.01652978; C2-G/M 0.01774292. C1's point estimate is about 7.34% higher than U. The primary confidence interval includes zero. These are pricing errors, not trading returns or future-return prediction accuracy.

Numerical reproducibility has been verified on isolated source copies/environments on one Windows laptop. Do not promise byte-identical output on every laptop. The numerical checker permits only the documented floating-point tolerance; a failure needs investigation, not wider tolerances. Figure fonts/PDF metadata and run timestamps can differ without changing the result values.

## Data and original author code

The current factor archive (about 2.67 MB), market archive (about 20 KB), protocols, and required reference/split files are included. They are sufficient for the extension. See [DATA_AND_SOURCES.md](DATA_AND_SOURCES.md) for hashes, definitions, attribution, and distribution terms.

The parent paper's historical archive is separate (about 286 MB) and is acquired directly from the authors with verification. It is intentionally excluded from ordinary Git tracking. For the original US benchmark, after the extension setup:

```powershell
.\.venv\Scripts\python.exe replicate.py acquire --archive evidence/JKP_original_data.zip
.\.venv\Scripts\python.exe demo.py --archive evidence/JKP_original_data.zip
```

This requires internet access and additional R packages installed by the parent workflow. It does not rerun the multi-hour global simulations. The upstream author source is pinned as a Git submodule; for a Git clone, `git submodule update --init` obtains it. A ZIP does not populate submodule contents. The extension does not require the submodule; the historical archive supplies the parent analysis inputs/code used by its separate command.

Skip the acquisition command if the archive already exists. The demo saves its summary to `local/last_verified_US_run.json`, replacing the previous summary only after a successful run. See [ENVIRONMENT.md](ENVIRONMENT.md) for the separate parent R package inventory and its installation limitations.

## Owner: prepare the reviewed GitHub version

No publication is performed by this guide or by the verification command. The current local branch is `DS340W-project-code`; its configured origin is `https://github.com/RishiLal/DS340W.git`. Verify both before using the publication commands. Do not publish the surrounding legacy DS340W folder as this release.

1. Review [RELEASE_VERIFICATION.md](RELEASE_VERIFICATION.md), the figures, and `git diff`. Check that the package contains the intended source, data, documentation, and new files. No open-source license is currently assigned to the project's own code; preserve third-party notices and terms.
2. Stage only the intended release changes. The current cleanup's intended files are:

```powershell
git status --short
git branch --show-current
git remote -v
git add README.md RESEARCH_PLAN.md QUICKSTART.md RUN_AND_PUBLISH.md RELEASE_VERIFICATION.md
git add pipeline.py requirements.txt tools/build_results.py tools/build_figures.py
git add extension_20261108/test_pipeline.py reference/figures reference/release_verification_20261118.json
git add reference/release_verification_20261119.json
git add ENVIRONMENT.md reference/release_verification_20261120.json
git add reference/parent/fresh_US_20261120.json reference/parent/R_versions_20261120.csv
git diff --cached --stat
git diff --cached --check
git diff --cached
```

The required input archives, numerical references, split CSVs, and submodule pointer are already tracked. Verify that the staged list does not include environments, old runs, credentials, unrelated documents, or the historical ZIP. The older audit is preserved in the ignored local `runs/publication_check_20261120/` evidence folder.

In **GitHub Desktop**, choose repository **DS340W**, then choose **Current worktree > jkp_20261111**. Confirm the branch is **DS340W-project-code**. Review the same intended changes, enter the commit summary below, and commit. Use **Publish branch** for its first push, or **Push origin** for subsequent commits. The main worktree contains the legacy project and thousands of unrelated local audit files; it is not the release package.

3. Create the local commit after reviewing the staged content:

```powershell
git commit -m "Finalize reproducible JKP extension and figures"
```

4. After you explicitly decide to publish, push the named release branch (do not force-push):

```powershell
git push -u origin DS340W-project-code
```

GitHub's [push documentation](https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository) explains the remote/branch arguments. If the push is rejected because the remote has newer work, inspect and reconcile it; do not overwrite it.

5. Check the repository on GitHub. Confirm public access in a signed-out browser, inspect the release branch's README/data/figures, and record its commit hash. Share a link to this branch or commit so readers do not land on a legacy default branch. Make it the default branch only as a deliberate repository-management decision.
6. Finally, clone the published branch into another fresh directory and follow the reader commands above. Record the returned manifest and numerical comparison. This verifies that every necessary file actually reached GitHub, rather than only existing locally.

Example clone command, usable after the branch has been published:

```powershell
git clone --branch DS340W-project-code https://github.com/RishiLal/DS340W.git DS340W-reproduce
Set-Location DS340W-reproduce
```

Keep the published commit, submitted manuscript, and figures linked to the same verified numerical release. No scientific settings need to change for publication.
