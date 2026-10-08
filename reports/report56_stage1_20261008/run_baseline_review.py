"""Three baseline calibrations + two reviews; material diagnostics only."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
SESSION=ROOT/'runs/report56_stage1_20261008'
for folder in ('bench','diagnostic','full_eval','local'):
    sys.path.insert(0,str(ROOT/'tools'/folder))
from config import expand,read
from local_identity import verify,record
from linux_runner import write_new,verify_files
from pool_metrics import metrics,export_evidence
from windows_runner import execute,gpu_lock

SCENES=('cloth_sphere7_l','cloth_fixed_bunny_l')
KEYS=('max_stretch','p99_stretch','fixed_drift_m')
def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def material(folder):
    verify_files(folder,read(folder/'evidence.json')['files'])
    if not export_evidence(folder,120,'state')['passed']:
        raise RuntimeError('Incomplete baseline positions')
    data=metrics(folder)
    return {key:data[key] for key in KEYS}|{'finite':data['finite'],
        'pcg_reported_failures':data['pcg_failures'],'evidence':record(folder/'evidence.json')}
def main():
    identity=read(SESSION/'REFERENCE_IDENTITY.json')['programs']['base']
    verify(identity['sources']+identity['build_evidence']+[identity['exe']]+identity['dlls'])
    protocol={'steps':120,'dt':.01,'absolute_numerical_tolerance':1e-10,
        'relative_numerical_tolerance':1e-10,'scenes':{},'program':identity['exe'],
        'scope':'Freeze whole-segment material ranges before review. Numerical tolerance is not deformation budget. No statistical, actual-velocity, or accepted-path CCD certification.'}
    for scene in SCENES:
        rows=[material(SESSION/'reference'/f'{scene}_base_r{i}') for i in (1,2,3)]
        protocol['scenes'][scene]={'calibration':rows,'ranges':{
            k:[min(r[k] for r in rows),max(r[k] for r in rows)] for k in KEYS}}
    write_new(REPORT/'BASELINE_MATERIAL_PROTOCOL.json',protocol)
    ledger={'status':'running','protocol':record(REPORT/'BASELINE_MATERIAL_PROTOCOL.json'),
        'rows':[],'quality_certified':False,'performance_certified':False}
    write_new(SESSION/'BASELINE_REVIEW_LEDGER.json',ledger)
    with gpu_lock(ROOT):
        for repeat in (4,5):
            for scene in SCENES:
                name=f'{scene}_base_r{repeat}'
                task={'name':name,'binary':'base','config':expand({'scene':scene,
                    'steps':120,'timeout_seconds':180,'preset':'host'})}
                result=execute(SESSION/'baseline_review',task,identity,0)
                row={'name':name,'scene':scene,'result':result};ledger['rows'].append(row)
                if result['status']!='completed':
                    ledger['status']='failed';save(SESSION/'BASELINE_REVIEW_LEDGER.json',ledger)
                    raise RuntimeError('Review failed; evidence retained, no retry')
                row['material']=material(SESSION/'baseline_review'/name)
                row['checks']={}
                for k,bounds in protocol['scenes'][scene]['ranges'].items():
                    actual=row['material'][k];upper=bounds[1];tol=1e-10+1e-10*abs(upper)
                    row['checks'][k]={'actual':actual,'upper':upper,'excess':actual-upper,
                        'numerical_tolerance':tol,'passed':actual<=upper+tol}
                row['passed']=all(c['passed'] for c in row['checks'].values())
                save(SESSION/'BASELINE_REVIEW_LEDGER.json',ledger)
    ledger['status']='completed';ledger['baseline_ranges_reproduced']=all(r['passed'] for r in ledger['rows'])
    ledger['decision']='Reference reproduced; full certification gaps remain.' if ledger['baseline_ranges_reproduced'] else \
        'Protocol cannot certify. Frozen bounds retained; review/candidate data do not expand them.'
    save(SESSION/'BASELINE_REVIEW_LEDGER.json',ledger);save(REPORT/'BASELINE_REVIEW.json',ledger)
    print(json.dumps({'status':ledger['status'],'baseline_ranges_reproduced':ledger['baseline_ranges_reproduced']}),flush=True)
if __name__=='__main__':main()
