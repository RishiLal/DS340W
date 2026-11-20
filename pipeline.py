"""Build verified JKP splits, fits, diagnostic and research tables in one fresh run."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from diagnostic.run import run as diagnostic
from extension_20261108.clustering import rscript
from extension_20261108.run_extension import main as primary
from tools.build_figures import build_figures
from tools.build_results import build
from tools.prepare_splits import prepare
from tools.verify_replay import verify

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def git(args):
    try:
        top = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=False
        )
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != ROOT.resolve():
            return None
        res = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(ROOT), *args],
            capture_output=True, text=True, check=False
        )
        if res.returncode == 0:
            return res.stdout.strip()
        return None
    except (FileNotFoundError, PermissionError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="New output directory. Existing paths are rejected.")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run Python with assertions enabled.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = (args.out or ROOT / "runs" / ("pipeline_" + stamp)).resolve()
    if dest.exists():
        parser.error("Choose a new output directory.")
    dest.mkdir(parents=True)
    logs = dest / "logs"
    logs.mkdir()
    commit_raw = git(["rev-parse", "HEAD"])
    tree_raw = git(["status", "--porcelain", "--untracked-files=normal"])
    manifest = dict(status="running", started_utc=stamp,
                    commit=commit_raw if commit_raw is not None else "unavailable (no git metadata or executable)",
                    working_tree=tree_raw if tree_raw is not None else "unavailable",
                    git_available=commit_raw is not None,
                    python=platform.python_version(), platform=platform.platform(),
                    numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
                    requirements_sha256=digest(ROOT / "requirements.txt"),
                    installed_packages={
                        dist.metadata["Name"]: dist.version
                        for dist in sorted(importlib.metadata.distributions(), key=lambda d: (d.metadata["Name"] or "").lower())
                        if dist.metadata["Name"]
                    },
                    stages=[])
    sources = sorted([ROOT / "pipeline.py", ROOT / "requirements.txt",
                      *ROOT.glob("tools/*.py"),
                      *ROOT.glob("extension_20261108/*.py"), *ROOT.glob("diagnostic/*.py"),
                      *ROOT.glob("protocol/*"), *ROOT.glob("evidence/JKP_current_US*.zip")])
    manifest["source_and_input_sha256"] = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    path = dest / "run_manifest.json"

    def save():
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def preflight():
        try:
            r_exec = rscript()
            manifest["R"] = subprocess.check_output([r_exec, "--version"], text=True).strip()
        except FileNotFoundError as err:
            raise FileNotFoundError(
                "R executable (Rscript) not found.\n"
                "Actionable setup instructions:\n"
                "  1. Install R (version 4.0+) from https://cloud.r-project.org/\n"
                "  2. Add Rscript to your system PATH (e.g. C:\\Program Files\\R\\R-4.6.1\\bin), OR\n"
                "  3. Set the RSCRIPT environment variable to the Rscript executable path\n"
                "     (e.g. in PowerShell: $env:RSCRIPT = 'C:\\Program Files\\R\\R-4.6.1\\bin\\Rscript.exe')"
            ) from err

    steps = [
        ("preflight", preflight),
        ("tests", lambda: subprocess.run(
            [sys.executable, "-m", "unittest", "extension_20261108.test_pipeline"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)),
        ("splits", lambda: prepare(dest / "splits")),
        ("primary", lambda: primary(dest / "primary")),
        ("diagnostic", lambda: diagnostic(dest / "diagnostic", dest / "primary", dest / "splits")),
        ("numerical_comparison", lambda: verify(dest)),
        ("tables", lambda: build(dest)),
        ("figures", lambda: build_figures(dest)),
    ]
    save()
    started = time.monotonic()
    try:
        for name, operation in steps:
            stage = dict(name=name, status="running")
            manifest["stages"].append(stage)
            save()
            then = time.monotonic()
            print(f"Running {name}", flush=True)
            if name == "diagnostic":
                (dest / "diagnostic").mkdir()
            with (logs / (name + ".log")).open("w", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    result = operation()
                    if isinstance(result, subprocess.CompletedProcess):
                        print(result.stdout)
                        result.check_returncode()
                    elif result is not None:
                        stage["result"] = result
            stage.update(status="passed", elapsed_seconds=round(time.monotonic() - then, 3))
            save()
        for name, expected in manifest["source_and_input_sha256"].items():
            if digest(ROOT / name) != expected:
                raise RuntimeError(f"Source or input changed during this run: {name}")
        manifest["outputs_sha256"] = {
            p.relative_to(dest).as_posix(): digest(p)
            for p in sorted(dest.rglob("*")) if p.is_file() and p != path
        }
        manifest["status"] = "passed"
    except Exception:
        manifest["status"] = "failed"
        if manifest["stages"] and manifest["stages"][-1]["status"] == "running":
            manifest["stages"][-1]["status"] = "failed"
        (logs / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        save()
    print(f"All stages passed. Results and tables are in {dest}")


if __name__ == "__main__":
    main()
