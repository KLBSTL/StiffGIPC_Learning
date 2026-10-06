"""Read-only local evidence checks; CPU residuals and exported accepted-path CCD."""
import argparse,collections,csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from analyze_perf_v36 import matrix_from_snapshot

ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def stages(path):
    pending={};totals=collections.defaultdict(float);counts=collections.Counter();last=None
    if not path.exists():return {}
    for line in path.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        last=r;stage=r['stage'];key=(r['frame'],r['direction'],r.get('index',-1),stage.rsplit('_',1)[0])
        if stage.endswith('_begin'):
            assert key not in pending,(path,key)
            pending[key]=r
        elif stage.endswith('_end'):
            start=pending.pop(key);duration=r['steady_seconds']-start['steady_seconds'];assert duration>=0
            name=key[-1];totals[name]+=duration;counts[name]+=1
    return {'inclusive_seconds':dict(totals),'completed_calls':dict(counts),'unfinished':list(pending.values()),'last':last,
            'scope':'Nested inclusive times; PCG loop includes Graph preparation. Diagnostic synchronization and IO alter timings.'}

def fixed(path):
    study=read(path);prefix=Path(str(path)[:-len('_study.json')]);meta,A,b=matrix_from_snapshot(prefix)
    assert study['system_unchanged'] and study['primary_restored_bitwise']
    checks=[]
    for phase,tol in [(1,1e-4),(3,1e-16)]:
        for mode,label in enumerate(['host','graph']):
            x=np.fromfile(f'{prefix}_p{phase}_m{mode}_x.bin',dtype='<f8')
            residual=float(np.linalg.norm(b-A@x)/max(np.linalg.norm(b),1e-300))
            rows=[r for r in study['runs'] if r['rho_tolerance']==tol and r['mode']==label]
            assert len(rows)==2 and np.isfinite(residual)
            checks.append({'rho_tolerance':tol,'mode':label,'cpu_true_residual':residual,
                'gpu_true_residual':rows[0]['true_relative_residual'],
                'iterations':[r['iterations'] for r in rows],
                'max_repeat_relative':max(r['repeat_difference']['relative'] for r in rows),
                'max_host_relative':max(r['host_difference']['relative'] for r in rows),
                'failures':sum(bool(r['error']) or r['limit'] for r in rows)})
    strict=[r for r in checks if r['rho_tolerance']==1e-16]
    return {'path':str(path),'dofs':meta['dofs'],'checks':checks,'system_unchanged':True,'primary_restored_bitwise':True,
        'strict_gate_passed':all(r['cpu_true_residual']<=1e-8 and r['max_repeat_relative']<=1e-6 and not r['failures'] for r in strict),
        'operator_repeats':study['operator_repeats'],'operator_mean_us':study['operator_mean_us']}

def outer_stages(run):
    path=run/'stage.jsonl'
    if not path.exists():return {}
    rows=[]
    lines=path.read_text().splitlines()
    for i,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:assert i==len(lines)-1
    assert all(r['sequence']==i+1 for i,r in enumerate(rows))
    pcg=[r['pcg'] for r in rows if r['stage']=='pcg_end']
    return {'last_event':rows[-1] if rows else None,'completed_pcg':len(pcg),
        'max_iterations':max((r['iterations'] for r in pcg),default=0),
        'limits':sum(bool(r.get('iteration_limit')) for r in pcg),'breakdowns':sum(bool(r.get('breakdown')) for r in pcg),
        'max_pairs':max((r.get('pairs',0) for r in rows),default=0),
        'max_trial_motion_m':max((r.get('max_motion_m',0) for r in rows),default=0)}

def analyze(run,ccd):
    req=read(run/'requested.json');res=read(run/'result.json');manifest=read(ROOT/'manifests/perf_v53_local.json')
    assert req['source_digest']==manifest['source_digest'] and req['runner_sha256']==sha(ROOT/'tools/run_perf_v53.py')
    assert req['exe_sha256']==manifest['binaries']['builds/local-v53/Release/gipc.exe']['sha256']
    output={'run':run.name,'result':res,'identity_verified':True,'linear_stages':stages(run/'linear_stage.jsonl'),
        'persisted_stage_stats':outer_stages(run)}
    stats=run/'output/stats.json';frames=read(stats)['frames'] if stats.exists() else []
    pcg=[n['pcg'] for f in frames for n in f.get('newton',[]) if 'pcg' in n]
    offset=res['resume_completed_physical_frames']
    for i,f in enumerate(frames):
        if 'toi_frame_friction_snapshot' in f:assert f['toi_frame_friction_snapshot']['physical_frame']==offset+i
    output['flushed_stats']={'frames':len(frames),'pcg_calls':len(pcg),'max_iterations':max((r['iterations'] for r in pcg),default=0),
        'limit_hits':sum(bool(r.get('iteration_limit')) for r in pcg),'breakdowns':sum(bool(r.get('breakdown')) for r in pcg),
        'frame_exits':[f.get('toi_exit') for f in frames]}
    output['fixed_systems']=[fixed(p) for p in sorted((run/'fixed').glob('*_study.json'))]
    restore=run/'output/checkpoint_restore.json'
    if restore.exists():output['restore']=read(restore)
    output['checkpoints']={}
    for folder in ['checkpoint','final_checkpoint']:
        cp=run/folder/'checkpoint.json'
        if cp.exists():
            meta=read(cp);assert meta['binary_id']==req['exe_sha256']
            output['checkpoints'][folder]={'completed_physical_frames':meta['completed_physical_frames'],
                'members':{p.name:sha(p) for p in sorted(cp.parent.glob('*.bin'))},'metadata_sha256':sha(cp)}
    if req['substeps'] and (run/'trace/substeps').exists():
        groups=collections.defaultdict(list)
        for p in sorted((run/'trace/substeps').glob('safe_*.bin')):groups[int(p.stem.split('_')[1])].append(p)
        accepted=0;coverage=[];completed=res.get('recorded_frames',0)
        for absolute,paths in sorted(groups.items()):
            local=absolute-offset;assert local>=0
            assert [int(p.stem.split('_')[2]) for p in paths]==list(range(len(paths)))
            assert paths[0].read_bytes()==(run/f'trace/state_{local:04d}.bin').read_bytes()
            if local<completed:assert paths[-1].read_bytes()==(run/f'trace/state_{local+1:04d}.bin').read_bytes()
            accepted+=len(paths)-1
            coverage.append({'physical_frame_zero_based':absolute,'run_frame_zero_based':local,'segments':len(paths)-1,'completed':local<completed})
        output['path_coverage']={'frames':coverage,'accepted_segments':accepted,'stationary_bridges':max(0,len(groups)-1)}
        if ccd and accepted:
            target=ROOT/'reports'/f'CCD_{run.name}.json';exe=ROOT/'builds/validator/Release/diagnose_first_path.exe'
            command=[str(exe),str(run/'trace'),str(target),'substeps','--stable-nh1']
            if not target.exists():
                with target.with_suffix('.log').open('w') as log:
                    proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                output['ccd_exit_code']=proc.returncode
            report=read(target);assert report['paths_checked']==accepted+len(groups)-1
            output['ccd']={'report':report,'command':command,'validator_sha256':sha(exe)}
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--ccd',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    manifest=read(ROOT/'manifests/perf_v53_local.json')
    for f in manifest['files']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
    for path,entry in manifest['binaries'].items():assert sha(ROOT/path)==entry['sha256']
    rows=[]
    for name in a.runs:
        row=analyze(ROOT/'runs/local'/name,a.ccd);rows.append(row)
        a.output.write_text(json.dumps({'source_files_verified':len(manifest['files']),'runs':rows},indent=2,allow_nan=False))
        print(json.dumps({'run':name,'status':row['result']['status'],'pcg':row['flushed_stats'],
            'fixed_gates':[s['strict_gate_passed'] for s in row['fixed_systems']],'ccd':row.get('ccd',{}).get('report')}),flush=True)
if __name__=='__main__':main()
