"""Finite mixed-bunny ablations on preserved base/v54 binaries; no solver mutation."""
import argparse, csv, hashlib, json, math, os, shutil, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FEATURES={'refit':'GIPC_CCD_BVH_REFIT','batch':'GIPC_BATCHED_ENERGY','reuse':'GIPC_ENERGY_REUSE'}
PRESETS={
    'base':{}, 'host':{}, 'graph':{'graph':True},
    'refit':{'refit':True}, 'batch':{'batch':True}, 'reuse':{'reuse':True},
    'combined':{'graph':True,'refit':True,'batch':True,'reuse':True},
    'cholesky':{'cholesky':True},
    'toi':{'graph':True,'refit':True,'batch':True,'reuse':True,'cholesky':True,'toi':True},
}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gpu():
    q='utilization.gpu,memory.free,temperature.gpu,power.draw,clocks.sm'
    s=subprocess.check_output(['nvidia-smi','--query-gpu='+q,'--format=csv,noheader,nounits'],text=True)
    return dict(zip(['utilization','free_mib','temperature_c','power_w','sm_mhz'],map(float,s.strip().split(','))))
def main():
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=PRESETS,required=True)
    p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=100)
    p.add_argument('--timeout',type=int,default=120);p.add_argument('--audit',action='store_true')
    a=p.parse_args();assert 0<a.steps<=100 and 0<a.timeout<=180
    assert Path(a.name).name==a.name and a.name not in ('.','..')
    cfg=PRESETS[a.preset];base=a.preset=='base'
    manifest_path=ROOT/'manifests'/('perf_v34.json' if base else 'perf_v54_local.json')
    m=read(manifest_path);files=[x for x in m['files'] if not base or x['path'].startswith('sources/stiff_base/')]
    assert len(files)==(274 if base else 294)
    for x in files:assert sha(ROOT/x['path'])==x['sha256'],x['path']
    exe=ROOT/'builds'/('local-base' if base else 'local-v54')/'Release/gipc.exe'
    exe_sha=sha(exe);assert exe_sha==m['binaries'][exe.relative_to(ROOT).as_posix()]['sha256']
    out=ROOT/'runs/local'/a.name;assert not out.exists()
    assert shutil.disk_usage(ROOT).free>4*1024**3
    before=[gpu()];time.sleep(1);before.append(gpu())
    budget=min(.75*before[-1]['free_mib'],before[-1]['free_mib']-1536);assert budget>=1024
    out.mkdir();(out/'output').mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
    env.update(GIPC_SCENE='bunny_cloth_bunny_l',GIPC_STEPS=str(a.steps),GIPC_DT='.01',
        GIPC_NEWTON_TOL='.01',GIPC_PCG_TOL='1e-4',GIPC_ACCEL_SUITE='0',
        GIPC_CONTACT_BACKEND='toi_al' if cfg.get('toi') else 'ipc',
        GIPC_PCG_EXECUTION='conditional_graph' if cfg.get('graph') else 'host',
        GIPC_PCG_PRECONDITIONER='mas',GIPC_MAS_CHOLESKY=str(int(cfg.get('cholesky',False))),
        GIPC_MAS_WIDE_APPLY='0',GIPC_MAS_INVERSE64='0',GIPC_PCG_FUSED_DIAG_UPDATE='0',
        GIPC_CCD_PAIR_LIMIT='10000000',GIPC_TOI_POLICY='robust',GIPC_TOI_INNER_EXIT='native',
        GIPC_TOI_CHOOSE_START=str(int(cfg.get('toi',False))),
        GIPC_TOI_RESTART_FULL_STEP_GUARD=str(int(cfg.get('toi',False))),
        GIPC_TOI_REDUCED_SLACK='0',GIPC_TOI_SAFE_INJECTIVITY='auto',
        GIPC_TOI_VOLUME_BOUND='normalized',GIPC_TOI_ROBUST_VELOCITY_TOL='.05',
        GIPC_TOI_TRIAL_INJECTIVITY='0',GIPC_TOI_REUSE_INITIAL_ASSEMBLY='1',
        GIPC_DEFER_STATS='1',GIPC_TRACE_STRIDE='1',GIPC_TRACE_PHYSICS='0',
        GIPC_TRACE_DIR=str(out/'trace'),GIPC_DUMP_STATE=str(out/'final.bin'),GIPC_OUTPUT_PATH=str(out/'output'))
    for k,v in FEATURES.items():env[v]=str(int(cfg.get(k,False)))
    if a.audit:
        env.update(GIPC_TRACE_SUBSTEPS=str(out/'trace/substeps'),GIPC_AUDIT_PCG='1',
            GIPC_AUDIT_REFIT='1',GIPC_AUDIT_ENERGY='1')
    requested=vars(a)|{'configuration':cfg,'environment':{k:v for k,v in env.items() if k.startswith('GIPC_')},
        'manifest':str(manifest_path),'verified_source_files':len(files),'source_digest':m['source_digest'],
        'exe_sha256':exe_sha,'runner_sha256':sha(Path(__file__)),'gpu_before':before,
        'timing_scope':'shared desktop GPU; no exclusive-performance certification','memory_budget_mib':budget}
    (out/'requested.json').write_text(json.dumps(requested,indent=2))
    start=time.monotonic();samples=[];status='running'
    with (out/'run.log').open('wb') as log:
        proc=subprocess.Popen([str(exe)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/'process.json').write_text(json.dumps({'pid':proc.pid,'exe':str(exe)}))
        while proc.poll() is None:
            g=gpu();samples.append(g)
            reason='timeout' if time.monotonic()-start>a.timeout else 'disk_reserve' if shutil.disk_usage(out).free<1024**3 else 'memory_budget' if g['free_mib']<768 or before[-1]['free_mib']-g['free_mib']>budget else None
            if reason:status=reason;proc.terminate();break
            time.sleep(1)
        proc.wait()
    if status=='running':status='completed' if proc.returncode==0 else 'failed'
    result={'status':status,'exit_code':proc.returncode,'wall_seconds':time.monotonic()-start,
        'timing_is_diagnostic':True,'audit':a.audit,'recorded_frames':0,'gpu_samples':samples}
    f=out/'trace/frames.csv'
    if f.exists():
        with f.open() as stream:rows=list(csv.DictReader(stream))
        result.update(recorded_frames=len(rows),solver_seconds=sum(float(x['solver_ms']) for x in rows)/1000)
    if (out/'final.bin').exists():
        import struct
        result['finite']=all(math.isfinite(x[0]) for x in struct.iter_unpack('<d',(out/'final.bin').read_bytes()))
    assert sha(exe)==exe_sha
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='gpu_samples'}),flush=True)
    return int(status!='completed' or result['recorded_frames']!=a.steps)
if __name__=='__main__':raise SystemExit(main())
