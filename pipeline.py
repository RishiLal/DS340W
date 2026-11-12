"""Build verified JKP splits, fits, diagnostic and research tables in one fresh run."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from diagnostic.run import run as diagnostic
from extension_20261108.clustering import rscript
from extension_20261108.run_extension import main as primary
from tools.build_results import build
from tools.prepare_splits import prepare
from tools.verify_replay import verify

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def git(args):
    return subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(ROOT), *args], text=True
    ).strip()


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
    manifest = dict(status="running", started_utc=stamp, commit=git(["rev-parse", "HEAD"]),
                    working_tree=git(["status", "--porcelain", "--untracked-files=normal"]),
                    python=platform.python_version(), platform=platform.platform(),
                    numpy=np.__version__, pandas=pd.__version__, stages=[])
    sources = sorted([ROOT / "pipeline.py", *ROOT.glob("tools/*.py"),
                      *ROOT.glob("extension_20261108/*.py"), *ROOT.glob("diagnostic/*.py"),
                      *ROOT.glob("protocol/*"), *ROOT.glob("evidence/JKP_current_US*.zip")])
    manifest["source_and_input_sha256"] = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    path = dest / "run_manifest.json"

    def save():
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def preflight():
        manifest["R"] = subprocess.check_output([rscript(), "--version"], text=True).strip()

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
        manifest["stages"][-1]["status"] = "failed"
        (logs / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        save()
    print(f"All stages passed. Results and tables are in {dest}")


if __name__ == "__main__":
    main()
