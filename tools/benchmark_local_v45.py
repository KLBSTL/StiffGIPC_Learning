"""Local staged checkpoint validation and bounded branches from identical saved state."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'runs/local'
PREFIX='v45r2_'; REPORT_PREFIX='V45R2_'
p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['smoke','fixed_smoke','contact_seed','contact_restart','reach35','default','strict','velocity']);a=p.parse_args()
report=ROOT/'reports'/f'{REPORT_PREFIX}{a.phase}.json';assert not report.exists()
records=[]
def run(name,scene,steps,extra=(),expected=0,timeout=180):
    command=[sys.executable,str(ROOT/'tools/run_perf_v45.py'),'--platform','local','--arm','base_toi_graph',
        '--scene',scene,'--preconditioner','mas','--mas-cholesky','1','--steps',str(steps),'--name',PREFIX+name,
        '--suite','1','--dt','.01','--robust-velocity-tol','.05','--trace','--audit-pcg','--failure-system',
        '--diagnostic-toi','--quality-only','--timeout',str(timeout),'--stage-from-frame','1','--linear-stages',
        '--ccd-pair-limit','10000000',*map(str,extra)]
    with (ROOT/'reports'/f'{REPORT_PREFIX}runner_{name}.log').open('w') as log:
        proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout+90)
    folder=RUNS/(PREFIX+name)
    result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {'status':'launcher_failed'}
    record={'name':name,'command':command,'runner_code':proc.returncode,'result':result}
    records.append(record);report.write_text(json.dumps(records,indent=2))
    print(json.dumps(record),flush=True)
    if expected==0 and proc.returncode:raise RuntimeError('Required local check failed: '+name)
    if expected==4 and result.get('exit_code')!=4:raise RuntimeError('Invalid checkpoint was not rejected: '+name)
    return folder
def compare_state(reference,continued,label):
    result={'label':label,'physical_relative_gate':1e-8,'physical_max_abs_gate':1e-8,'buffers':[]}
    for name in ['vertices','old_vertices','velocities','x_tilde','abd_q','abd_q_prev','abd_q_tilde','abd_q_v']:
        x=np.fromfile(reference/'final_checkpoint'/(name+'.bin'),dtype='<f8')
        y=np.fromfile(continued/'final_checkpoint'/(name+'.bin'),dtype='<f8')
        assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        maximum=float(np.max(np.abs(x-y))) if x.size else 0.
        relative=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30)) if x.size and np.linalg.norm(x)>0 else maximum
        result['buffers'].append({'name':name,'bitwise_equal':bool(np.array_equal(x.view('u8'),y.view('u8'))),'max_abs':maximum,'relative_l2':relative})
    result['passed']=all(x['max_abs']<=1e-8 and x['relative_l2']<=1e-8 for x in result['buffers'])
    (ROOT/'reports'/f'{REPORT_PREFIX}CONTINUE_{label}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    assert result['passed'],'Continuation differs beyond predeclared physical tolerance; retain failure'
if a.phase=='smoke':
    seed=run('seed','cloth_hang_l',3,['--checkpoint-frame','3','--checkpoint-final','--substeps'])
    resumed=run('resume','cloth_hang_l',1,['--checkpoint-load',seed/'checkpoint','--checkpoint-final','--substeps'])
    compare_state(seed,resumed,'hang')
    bad=RUNS/(PREFIX+'corrupt_checkpoint');assert not bad.exists();shutil.copytree(seed/'checkpoint',bad)
    file=bad/'velocities.bin';data=bytearray(file.read_bytes());assert data;data[0]^=1;file.write_bytes(data)
    corrupt=run('reject_corrupt','cloth_hang_l',1,['--checkpoint-load',bad],expected=4)
    assert 'Checkpoint member integrity mismatch' in (corrupt/'run.log').read_text(errors='replace')
    mismatch=run('reject_physics','cloth_hang_l',1,['--checkpoint-load',seed/'checkpoint','--dt','.02'],expected=4)
    assert 'Checkpoint scene/physics context mismatch' in (mismatch/'run.log').read_text(errors='replace')
elif a.phase=='fixed_smoke':
    run('fixed_smoke','cloth_hang_l',1,['--checkpoint-load',RUNS/(PREFIX+'seed/checkpoint'),'--fixed-study','--compact-study',
        '--fixed-study-frames','3','--fixed-study-directions','1','--checkpoint-final'])
elif a.phase=='contact_seed':
    run('contact_seed','bunny_cloth_bunny_l',30,['--checkpoint-frame','30','--checkpoint-final','--stage-from-frame','28'],timeout=240)
elif a.phase=='contact_restart':
    out=run('contact_restart','bunny_cloth_bunny_l',1,['--checkpoint-load',RUNS/(PREFIX+'contact_seed/checkpoint'),'--checkpoint-final','--substeps'])
    compare_state(RUNS/(PREFIX+'contact_seed'),out,'contact')
elif a.phase=='reach35':
    run('reach35','bunny_cloth_bunny_l',6,['--checkpoint-load',RUNS/(PREFIX+'contact_seed/checkpoint'),'--checkpoint-frame','35','--substeps'],expected=None,timeout=180)
else:
    extra=['--checkpoint-load',RUNS/(PREFIX+'reach35/checkpoint'),'--substeps']
    if a.phase=='strict':extra += ['--pcg-tol','1e-16']
    elif a.phase=='velocity':extra += ['--inner-exit','velocity_only']
    else:extra += ['--fixed-study','--compact-study','--fixed-study-frames','35','--fixed-study-directions','1']
    run('branch_'+a.phase,'bunny_cloth_bunny_l',1,extra,expected=None,timeout=180)
