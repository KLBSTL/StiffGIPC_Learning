"""Bounded local diagnostics. Never mutate the native solver configuration."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','graph','host']);a=p.parse_args()
report=ROOT/'reports'/f'V49_{a.phase}.json';assert not report.exists();records=[]
def run(name,scene,steps,extra=(),arm='base_toi_graph',timeout=180):
    command=[sys.executable,str(ROOT/'tools/run_perf_v49.py'),'--platform','local','--arm',arm,'--scene',scene,
        '--steps',str(steps),'--name','v49_'+name,'--timeout',str(timeout),'--trace','--substeps','--physics','--audit-pcg',
        '--quality-only','--mas-cholesky','1','--preconditioner','mas','--suite','1','--dt','.01','--pcg-tol','1e-4',
        '--robust-velocity-tol','.05','--inner-exit','native','--failure-system','--ccd-pair-limit','10000000',*extra]
    with (ROOT/'reports'/f'V49_{name}_runner.log').open('x') as log:
        proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout+90)
    folder=ROOT/'runs/local'/('v49_'+name);result=json.loads((folder/'result.json').read_text())
    records.append({'run':folder.name,'command':command,'returncode':proc.returncode,'result':result})
    report.write_text(json.dumps(records,indent=2));print(json.dumps(records[-1]),flush=True)
    return folder,result
if a.phase=='smoke':
    off,ro=run('smoke_off','cloth_hang_l',3,timeout=30)
    on,rn=run('smoke_on','cloth_hang_l',3,['--direction-event','--event-from','1','--event-first','1','--event-second','2','--event-velocity','1e-12'],timeout=30)
    assert ro['status']==rn['status']=='completed'
    x=np.fromfile(off/'final.bin',dtype='<f8');y=np.fromfile(on/'final.bin',dtype='<f8')
    comparison={'max_abs':float(np.max(np.abs(x-y))),'relative_l2':float(np.linalg.norm(x-y)/np.linalg.norm(x)),
        'bitwise':bool(np.array_equal(x.view('u8'),y.view('u8'))),'physical_gate':1e-8,'fixture_threshold_only':True}
    comparison['passed']=comparison['max_abs']<=1e-8 and comparison['relative_l2']<=1e-8
    (ROOT/'reports/V49_SMOKE_COMPARISON.json').write_text(json.dumps(comparison,indent=2))
    assert comparison['passed']
    events=list((on/'events').glob('*_event.json'));assert 1<=len(events)<=4
else:
    run('bunny_'+a.phase,'bunny_cloth_bunny_l',40,['--direction-event','--stage-from-frame','30'],
        arm='base_toi_graph' if a.phase=='graph' else 'base_toi')
