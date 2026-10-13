"""Stream a verified session and build receipts into a new download archive."""
import argparse
import gzip
from pathlib import Path
import tarfile
from protocol import ROOT,read,write,sha,require
from run_suite import verify


def collect(session,manifest,output):
    session=session.resolve();require(session.is_relative_to(ROOT/'runs'),'Session outside payload/runs')
    sealed=read(manifest);verify(sealed);require(not output.exists(),'Result archive already exists')
    require(not output.resolve().is_relative_to(session),'Archive must be outside the collected session')
    paths=[p for p in session.rglob('*') if p.is_file()]+[manifest,*ROOT.glob('PAYLOAD_MANIFEST*.json'),
        *ROOT.glob('RUNNER_OVERLAY*.json'),*(ROOT/'build').glob('autodl_programs*.json')]
    for program in sealed['programs'].values():
        paths.append(ROOT/program['exe']['path'])
        paths.extend(ROOT/r['path'] for r in program['build_evidence'] if Path(r['path']).suffix in ('.log','.txt','.json'))
        test_log=(ROOT/program['exe']['path']).parent/'ctest.log'
        if test_log.exists():paths.append(test_log)
    paths=sorted(set(paths))
    for p in paths:require(not p.is_symlink() and p.resolve().is_relative_to(ROOT),'Linked/external result rejected')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as tar:
        for p in paths:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    write(Path(str(output)+'.json'),dict(path=output.name,bytes=output.stat().st_size,sha256=sha(output),
        source_session=str(session.relative_to(ROOT)),files=len(paths),manifest_sha256=sha(manifest),
        uploaded=False,performance_certified=False,physical_quality_certified=False))
    print('Result archive ready:',output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();collect(a.session,a.manifest,a.output)
