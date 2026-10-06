"""Separate accepted-path CCD audit; never use these runs for timing."""
import argparse
import collections
import json
import subprocess
from config import ROOT, read, sha
from prepare_cloth_benchmark import SCENES, task


def prepare():
    runs=[]
    for key,scene in SCENES.items():
        for arm in ('stiff','toi_warp'):
            row=task(scene,key,arm,'audit')
            row['config']['diagnostics']=['substeps']
            runs.append(row)
    path=ROOT/'configs/active/cloth_restrict_audit.json'
    with path.open('x') as f:
        json.dump({'report':'reports/active/CLOTH_RESTRICT_AUDIT_BATCH.json',
                   'stop_on_failure':False,'runs':runs},f,indent=2)
    print(json.dumps({'plan':str(path),'runs':len(runs)}))


def verify(output):
    if output.exists():raise FileExistsError(output)
    matrix=read(ROOT/'reports/active/CLOTH_RESTRICT_AUDIT_BATCH.json')
    exe=ROOT/'builds/validator/Release/diagnose_first_path.exe'
    report={'validator_sha256':sha(exe),'scope':'Separate diagnostic trajectories; timing trajectories have no substeps.',
            'physical_quality_certified':False,'runs':[]}
    for row in matrix['runs']:
        run=ROOT/'runs/active'/row['name'];trace=run/'trace';result=read(run/'result.json')
        rec={'run':row['name'],'scene_key':row['scene_key'],'variant':row['variant'],
             'result':result,'coverage_passed':False}
        report['runs'].append(rec)
        if result['status']!='completed' or result['recorded_frames']!=100:
            rec['failure']='Incomplete simulation'
            output.write_text(json.dumps(report,indent=2));continue
        groups=collections.defaultdict(list)
        for p in sorted((trace/'substeps').glob('safe_*.bin')):
            groups[int(p.stem.split('_')[1])].append(p)
        assert sorted(groups)==list(range(100)),(row['name'],sorted(groups))
        segments=0;counts=[]
        for frame,paths in sorted(groups.items()):
            assert [int(p.stem.split('_')[2]) for p in paths]==list(range(len(paths)))
            assert paths[0].read_bytes()==(trace/f'state_{frame:04d}.bin').read_bytes()
            assert paths[-1].read_bytes()==(trace/f'state_{frame+1:04d}.bin').read_bytes()
            segments+=len(paths)-1;counts.append(len(paths)-1)
        rec.update(coverage_passed=True,accepted_segments=segments,stationary_bridges=99,segments_per_frame=counts)
        target=ROOT/'reports/active'/f"CCD_{row['name']}.json"
        command=[str(exe),str(trace),str(target),'substeps','--stable-nh1']
        if target.exists():raise FileExistsError(target)
        rec['command']=command
        try:
            with target.with_suffix('.log').open('x') as log:
                proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            rec['exit_code']=proc.returncode
            ccd=read(target);assert ccd['paths_checked']==segments+99
            rec['ccd']=ccd
        except subprocess.TimeoutExpired:
            rec['failure']='CPU CCD audit exceeded predeclared 300 seconds'
        output.write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({k:rec[k] for k in ('run','coverage_passed','accepted_segments','ccd','failure') if k in rec}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--output')
    args=p.parse_args()
    if args.prepare:prepare()
    else:
        if not args.output:p.error('--output is required for verification')
        verify(ROOT/args.output)
