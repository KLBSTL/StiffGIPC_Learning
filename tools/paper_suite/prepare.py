"""Create a portable audited source/input bundle; never connect to AutoDL."""
import argparse
import gzip
import json
from pathlib import Path
import tarfile
from protocol import ROOT,SCENES,plan,input_inventory,read,write,sha,require
from run_suite import source_files,records


def prepare(output):
    catalog=[dict(scene=p.stem,sha256=sha(p),bytes=p.stat().st_size) for p in sorted((ROOT/'Assets/benchmark_scenes').glob('*.json'))]
    packet=dict(plan=plan(),selected_inputs=input_inventory(),historical_catalog=catalog,
        selected_resolution='low',source_records=records(source_files()),
        reference_images_are_format_examples_only=True,remote_results_pending=True)
    write(output,packet)
    print(json.dumps(dict(historical_scenes=len(catalog),selected_scenes=len(SCENES),full_runs=105,frames_per_run=100,repeats=3,
        matched_steps=24,checkpoint_seeds=2,output=str(output))))


def bundle(output):
    require(not output.exists(),'Bundle exists; never replace')
    files=source_files();manifest=dict(schema='paper_suite_payload.v1',files=records(files),plan=plan())
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as tar:
        for p in files:
            info=tar.gettarinfo(str(p),arcname='paper_suite/'+p.relative_to(ROOT).as_posix())
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
            info.mode=0o755 if p.suffix=='.sh' else 0o644
            with p.open('rb') as f:tar.addfile(info,f)
        import io
        b=(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode('utf8')
        info=tarfile.TarInfo('paper_suite/PAYLOAD_MANIFEST.json');info.size=len(b);info.mode=0o644;tar.addfile(info,io.BytesIO(b))
    write(Path(str(output)+'.json'),dict(path=output.name,bytes=output.stat().st_size,sha256=sha(output),files=len(files),
        uploaded=False,remote_execution=False,scope='Current dirty working tree including existing IPC work; selected common assets and frozen baseline; no builds/results/git/secrets'))
    print(json.dumps(dict(bundle=str(output),bytes=output.stat().st_size,sha256=sha(output),files=len(files))))


def verify_payload():
    manifest=read(ROOT/'PAYLOAD_MANIFEST.json')
    require(manifest['plan']==plan(),'Payload plan differs')
    require(records(source_files())==manifest['files'],'Payload source or asset inventory differs')
    print(json.dumps(dict(payload_verified=True,files=len(manifest['files']))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    for name in ('plan','bundle'):
        a=s.add_parser(name);a.add_argument('--output',type=Path,required=True)
    s.add_parser('verify-payload');args=p.parse_args()
    if args.action=='plan':prepare(args.output)
    elif args.action=='bundle':bundle(args.output)
    else:verify_payload()
