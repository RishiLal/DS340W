"""Run the original US empirical Bayes specification from a verified fresh extraction."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from replicate import acquire, prepare

ROOT = Path(__file__).resolve().parent

def r_executable():
    found = shutil.which('Rscript')
    if found:
        return found
    choices = sorted(Path('C:/Program Files/R').glob('R-*/bin/Rscript.exe'), reverse=True)
    if choices:
        return str(choices[0])
    raise SystemExit('Install R and add Rscript to PATH, or pass --rscript.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--rscript')
    parser.add_argument('--rlib', type=Path)
    args = parser.parse_args()
    r = args.rscript or r_executable()
    archive = args.archive or ROOT/'evidence/JKP_original_data.zip'
    if not archive.exists() and args.archive:
        raise SystemExit(f'Archive missing: {archive}')
    if not archive.exists():
        archive = ROOT/'local/JKP_original_data.zip'
        if not archive.exists():
            acquire(archive)
    libs = [ROOT/'env/R-library']
    lib = args.rlib or next((p for p in libs if p.is_dir()), None)
    if lib is None:
        subprocess.run([r,'--vanilla',str(ROOT/'tools/setup_R.R'),str(ROOT)],check=True)
        lib = libs[0]
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='jkp_demo_') as tmp:
        work = Path(tmp)/'run'
        print('Verifying all source files and extracting a fresh working copy...',flush=True)
        prepare(archive.resolve(), work)
        output = ROOT/'local'
        output.mkdir(exist_ok=True)
        subprocess.run([r,'--vanilla',str(ROOT/'tools/us_benchmark.R'),str(work),str(lib.resolve())],
                       cwd=work,check=True)
        report = json.loads((work/'result.json').read_text())
        report['elapsed_seconds_including_input_verification'] = round(time.monotonic()-start,2)
        path = output/'last_verified_US_run.json'
        path.write_text(json.dumps(report,indent=2))
        print(f'Verified in {report["elapsed_seconds_including_input_verification"]} seconds. Evidence: {path}',flush=True)

if __name__=='__main__':
    main()
