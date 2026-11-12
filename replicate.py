"""Prepare, run and inspect the original JKP analysis from any directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
URL = 'https://www.dropbox.com/sh/wcrjok1qyxtrasi/AABZ90GDCUvIzDzijt8Qoo3ha?dl=1'

def prepare(archive: Path, destination: Path):
    """Verify every archived file against the pinned file-level hashes before fitting."""
    if destination.exists():
        raise FileExistsError(f'Choose a new run directory: {destination}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    reference = json.loads((ROOT/'evidence/JKP_snapshot_manifest.json').read_text())
    reference = {k.replace(chr(92),'/'):v for k,v in reference.items()}
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        staging=Path(temp)/'run'
        staging.mkdir()
        with zipfile.ZipFile(archive) as z:
            files=[item for item in z.infolist() if not item.is_dir()]
            names=[item.filename for item in files]
            if len(names)!=len(set(names)) or set(names)!=set(reference):
                raise ValueError('Archive members differ from the pinned paper release')
            for item in files:
                member=PurePosixPath(item.filename)
                if member.is_absolute() or '..' in member.parts: raise ValueError(member)
                with z.open(item) as source:
                    digest=hashlib.file_digest(source,'sha256').hexdigest()
                if digest!=reference[item.filename]['sha256']: raise ValueError(f'Hash mismatch: {member}')
                if member.parts[0]=='Data': target=staging.joinpath(*member.parts)
                elif member.parts[0]=='Code - Analysis' or member.suffix=='.xlsx': target=staging/member.name
                else: continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(item) as source,target.open('wb') as output:
                    shutil.copyfileobj(source,output)
        for name in ['Objects','Figures','numeric']: (staging/name).mkdir()
        staging.rename(destination)

def acquire(path):
    path=path.resolve(); path.parent.mkdir(exist_ok=True,parents=True)
    if path.exists(): raise FileExistsError(path)
    partial=path.with_suffix('.download')
    with urllib.request.urlopen(URL,timeout=120) as source, partial.open('xb') as target:
        shutil.copyfileobj(source,target)
    with zipfile.ZipFile(partial) as z:
        if z.testzip(): raise ValueError('Downloaded ZIP failed CRC verification')
    partial.rename(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('acquire').add_argument('--archive',required=True,type=Path)
    prep=sub.add_parser('prepare')
    prep.add_argument('--archive',required=True,type=Path)
    prep.add_argument('--run',required=True,type=Path)
    run=sub.add_parser('run')
    run.add_argument('--run',required=True,type=Path)
    run.add_argument('--rscript',default=shutil.which('Rscript') or 'C:/Program Files/R/R-4.6.1/bin/Rscript.exe')
    for command in ['status','compare']: sub.add_parser(command).add_argument('--run',required=True,type=Path)
    args=p.parse_args()
    if args.command=='acquire': return acquire(args.archive)
    destination=args.run.resolve()
    if args.command=='prepare': return prepare(args.archive.resolve(),destination)
    if args.command=='status': print((destination/'runtime_status.json').read_text()); return
    if args.command=='compare':
        from tools.compare_native import compare
        print(json.dumps(compare(destination),indent=2)); return
    if not (destination/'main.R').is_file(): raise ValueError('Prepare the run directory first')
    from datetime import datetime
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    with (destination/f'invocation_{stamp}.log').open('xb') as log:
        subprocess.run([args.rscript,'--vanilla',str(ROOT/'tools/run_compatible.R'),str(ROOT),str(destination)],
            cwd=destination,stdout=log,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__': main()
