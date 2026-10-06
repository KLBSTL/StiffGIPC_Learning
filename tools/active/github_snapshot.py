"""Create a source-only, recoverable Git snapshot before the authorized cleanup.

No deletion, credential handling, Git mutation or network calls. Large generated
data remains outside Git; its report and identities are retained separately.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SKIP={'.git','__pycache__','build','builds','Output','outputs','.vs','node_modules'}
CODE={'.cu','.cuh','.cpp','.cc','.c','.h','.hpp','.inl','.inc','.py','.ps1','.sh',
      '.cmake','.md','.txt','.json','.toml','.yml','.yaml','.bat','.def','.rc','.natvis'}
PRIVATE=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{35,}|\bAKIA[A-Z0-9]{16}\b')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def selected():
    files=set()
    for folder in ('sources','tools','configs','manifests','references'):
        base=ROOT/folder
        if not base.exists():continue
        for p in base.rglob('*'):
            rel=p.relative_to(base)
            if not p.is_file() or SKIP.intersection(rel.parts):continue
            if folder=='sources' and 'Assets' in rel.parts:continue
            if p.suffix.lower() in CODE or p.name in {'LICENSE','.gitignore','.clang-format','.editorconfig'}:
                files.add(p)
    files.update(p for p in ROOT.glob('*.md'))
    files.update((ROOT/'reports').rglob('*.md'))
    for name in ('EXPERIMENT_INDEX.json','EXPERIMENT_DECISIONS.json',
                 'ipc_revision_20261005_quality_protocol.json',
                 'contact_pool_quality_results_20261006.json',
                 'contact_pool_resource_analysis_20261006.json',
                 'ipc_contact_pool_20261006_verification.json'):
        p=ROOT/'reports/active'/name
        if p.exists():files.add(p)
    for p in (ROOT/'builds/active/provenance').glob('*/manifest.json'):
        files.add(p)
    files.add(ROOT/'builds/active/manifest.json')
    # One input tree per currently required source provider. Git deduplicates
    # identical mesh blobs; existing manifest paths stay reproducible.
    for source in ('stiff_base','stiff_perf_v50'):
        files.update(p for p in (ROOT/'sources'/source/'Assets').rglob('*')
                     if p.is_file() and not SKIP.intersection(p.parts))
    return sorted(files)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--destination',type=Path,required=True)
    args=ap.parse_args();dest=args.destination.resolve()
    workspace=ROOT.parent.resolve()
    assert dest.is_relative_to(workspace) and dest!=workspace and dest!=ROOT
    assert (dest/'.git').is_dir(),'Expected newly cloned Git repository'
    paths=selected();rows=[]
    for p in paths:
        assert p.resolve().is_relative_to(ROOT) and not p.is_symlink(),str(p)
        target=dest/p.relative_to(ROOT)
        assert not target.exists(),str(target)
        size=p.stat().st_size
        assert size<95*1024*1024,'Git oversized file: '+str(p)
        if p.suffix.lower() in CODE or size<1024*1024:
            assert not PRIVATE.search(p.read_bytes()),'Secret signature found in '+str(p)
        rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':size,'sha256':sha(p)})
    for p,row in zip(paths,rows):
        target=dest/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
        assert sha(target)==row['sha256']
    record={'schema':1,'purpose':'Pre-cleanup source recovery; generated runs and binaries excluded',
            'original_root':str(ROOT),'files':rows,'file_count':len(rows),
            'bytes':sum(r['bytes'] for r in rows),'secret_signature_scan':'passed'}
    (dest/'PRE_CLEANUP_SNAPSHOT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('file_count','bytes','secret_signature_scan')}))

if __name__=='__main__':main()
