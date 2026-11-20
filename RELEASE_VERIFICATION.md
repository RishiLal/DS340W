# Local release verification — rechecked November 20, 2026

The current release passed a fresh installation and reproduction check for both the extension and the original US benchmark. These checks were completed before publication; they do not certify every historical parent-paper analysis or execution on an independent computer.

## Fresh setup and publication readiness — November 20

- Copied the intended distribution into a Git-free folder with spaces in its path. Created a new Python environment, installed every pin in `requirements.txt`, and passed `pip check`.
- The default extension command passed all eight stages in **40.297 seconds**, including **16 tests with zero skips**. All ten model-result artifacts were **byte-identical** to the frozen references. Verified all 47 output hashes and 23 source/input hashes independently.
- Rebuilt all three split exports. Their values match the distributed exports within the existing absolute tolerance of `1e-15`; maximum differences were below `1e-16`. CSV text need not be byte-identical. The figure reference hashes verified, and regenerated plotted values were byte-identical.
- Installed the parent workflow's R dependencies into a **new project-local R library**. The separate US demo completed in **161.23 seconds**, excluding package installation, using the original 10,000 draws. It reproduced tau_c **0.35088%**, tau_w **0.20946%**, and **98/119** factors (**82.35294%**), all within the published rounded precision. An initial optional-library probe failed because that older personal library lacked `directlabels`; the documented fresh automatic setup subsequently succeeded.
- Reused the local historical ZIP and verified all 21 members before fitting. Its SHA-256 matches the recorded original archive. The remote download endpoint returned HTTP 200 with ZIP content type and the expected 286,249,432-byte length; the entire archive was not downloaded again.
- Added [ENVIRONMENT.md](ENVIRONMENT.md) and clarified the actual output locations, isolated Python commands, separate R requirements, and parent output replacement behavior. Preserved the older untracked audit under the ignored local evidence directory.

Machine-readable evidence: [November 20 verification](reference/release_verification_20261120.json), [fresh parent result](reference/parent/fresh_US_20261120.json), and [actual parent R versions](reference/parent/R_versions_20261120.csv). Durable local output and logs are in `runs/publication_check_20261120/`. No model definitions, seeds, numerical references, or acceptance tolerances changed.

This is verified reproducibility on one Windows laptop, including fresh package installation. It is not a guarantee for every laptop or operating system. The historical global simulations and security-level analyses are outside this new check. The parent R package file is an inventory, not a lockfile.

## Fresh setup check — November 19

Followed the [quickstart](QUICKSTART.md) using a new Git-free source copy in a path containing spaces, a newly created Python environment, and a fresh installation of the pinned dependencies. `pip check` passed. The default `pipeline.py` command completed all eight stages in **29.579 seconds**, with **16 tests passed and zero skipped**. All ten numerical artifacts were byte-identical to the reference results, all 47 output hashes verified, and the 23 recorded source/input hashes matched the current release. No methodology or numerical-reference changes were needed.

This check used Python 3.12.13 and the existing R 4.6.1 installation on the same Windows laptop. It does not establish execution on another operating system or computer. The new short guide includes setup, the run command, success checks, output locations, and troubleshooting without requiring PowerShell activation.

- Machine-readable evidence: [reference/release_verification_20261119.json](reference/release_verification_20261119.json).
- Local full run: `runs/publication_recheck_20261119/`.
- Local installation and terminal logs: `runs/publication_recheck_20261119_setup/`.

## Previous verified result — November 18

| Check | Observed result |
|---|---|
| Clean source distribution | New Git-free copy; path contained spaces; execution started outside the project |
| Environment | Isolated Python environment installed and verified earlier on this Windows laptop; `pip check` passed |
| Runtime | 32.469 seconds |
| Tests | 16 passed, zero failed, zero skipped |
| Pipeline | All eight stages passed |
| Numerical reference comparison | All ten numerical artifacts byte-identical |
| Output integrity | All 47 recorded hashes verified |
| Dependency provenance | Requirements hash and installed package inventory recorded |
| Figures | All three PNGs visually inspected; PDFs, PNGs, plotted data, and captions generated together |
| Output protection | Existing destination rejected with exit 2 |
| Assertion protection | Optimized Python rejected with exit 2 before creating an output directory |
| Missing R | Exit 1, failed manifest, and actionable failure log |
| Whitespace | `git diff --check` passed |

The numerical findings, including the 7.34% final-period loss increase and bootstrap intervals including zero, are unchanged. Core fitting, tuning, data processing, uncertainty settings, and frozen numerical references were not changed by this cleanup.

## Corrections completed

- Missing `training_transforms.json` now raises an error before tables are generated. Learned cutoff/frequency values are never filled in from constants.
- Portable figure fixtures generate their training metadata from the bundled data rather than depend on an ignored recording directory.
- Selected penalties, windows, interaction values, and numerical figure captions use run artifacts. Lower, interior, and upper grid selections receive distinct descriptions.
- Added regression tests for missing training metadata, a lower-boundary selection, and Git metadata inherited from an enclosing repository.
- Git provenance is accepted only when the repository top-level directory matches the project root. ZIP downloads still work without Git.
- Updated the reader instructions, publication guide, requirements provenance, and reference figures.

Fixed dates, dimensions, seeds, and penalty-grid settings remain intentional parts of the frozen experiment. Numerical references and expected values in regression tests are verification fixtures, not substitutes for computing the results. The default pipeline recomputes model fits and numerical outputs from the author-supplied portfolio-return inputs.

## Evidence and files

- Machine-readable check record and source hashes: [reference/release_verification_20261118.json](reference/release_verification_20261118.json).
- Publication figure copies, captions and source values: [reference/figures/FIGURES.md](reference/figures/FIGURES.md).
- Figure-copy hashes and generator identity: [reference/figures/PROVENANCE.json](reference/figures/PROVENANCE.json).
- Reader and owner instructions: [RUN_AND_PUBLISH.md](RUN_AND_PUBLISH.md).
- Local full execution evidence: `runs/publication_verified_20261118/`.
- Local negative-check evidence: `runs/publication_negative_checks_20261118/`.

The `runs/` folders are intentionally ignored and are regenerated by readers. The machine-readable record retains the checked source identity and summary without depending on a temporary environment path. The current base commit predates the cleanup; use the source hashes until the reviewed changes are committed.

The publication file set includes the existing tracked source/data/protocol/reference files plus the revised pipeline, tests, reports, plotting module, figure copies, and guides. The old `FINAL_CODE_AUDIT.md` is preserved locally at `runs/publication_check_20261120/previous_FINAL_CODE_AUDIT.md` and is not part of the published package.

## Limits and remaining publication actions

Execution was verified on this Windows laptop using isolated copies and environments. Independent hardware and macOS/Linux execution have not been tested. The numerical verifier supports documented floating-point tolerances; byte-identical PDFs or fonts on different platforms are not promised.

All required extension data are bundled. The historical parent archive is downloaded separately with checksum verification, and author code is retained through a pinned submodule/archive workflow. The default command does not reproduce every result in the parent paper.

After publication, clone the published branch and run the reader guide once to verify the remotely distributed file set. These local pre-publication checks do not establish remote availability. No open-source license has been assigned to the project's own code; third-party terms remain separate.
