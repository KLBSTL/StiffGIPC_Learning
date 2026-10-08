"""Two observer-off controls followed by two bounded private-probe diagnostics."""
from pathlib import Path
import json
import sys
from run_stage1 import ROOT,REPORT,SESSION,SCENES,config,guard_summary,save
for folder in ('bench','diagnostic','full_eval','local'):
    sys.path.insert(0,str(ROOT/'tools'/folder))
from local_identity import active_identity,record,verify
from linux_runner import write_new
from windows_runner import execute,gpu_lock

def main():
    build=ROOT/'build/report56_stage1_verified_20261008'
    identity=active_identity(ROOT,build,build/'Release/gipc.exe',build/'build.log')
    write_new(SESSION/'PROBE_IDENTITY.json',identity)
    ledger={'status':'running','program_sha256':identity['exe']['sha256'],
        'planned_runs':4,'rows':[],'tools':[record(__file__),record(REPORT/'run_stage1.py'),
            record(ROOT/'tools/local/windows_runner.py'),record(ROOT/'tools/bench/config.py'),
            record(ROOT/'tools/bench/validate_run.py')],
        'scope':'Shared WDDM controls and separate instrumented observations; no speed/quality certification.'}
    write_new(SESSION/'PROBE_LEDGER.json',ledger)
    with gpu_lock(ROOT):
        for stage,diagnostics in (('observer_off',False),('probe',True)):
            for scene in SCENES:
                task={'name':scene,'binary':'active','config':config(scene,'all',diagnostics)}
                result=execute(SESSION/stage,task,identity,0)
                row={'stage':stage,'scene':scene,'result':result,
                    'summary':guard_summary(SESSION/stage/scene,result)}
                ledger['rows'].append(row)
                if not row['summary']['completed'] or not row['summary'].get('guards_passed',False):
                    ledger['status']='failed';save(SESSION/'PROBE_LEDGER.json',ledger)
                    raise RuntimeError('Probe/control failed; evidence retained, no automatic retry')
                save(SESSION/'PROBE_LEDGER.json',ledger)
    verify(identity['sources']+identity['build_evidence']+identity['objects']+[identity['exe']]+identity['dlls'])
    ledger['status']='completed';save(SESSION/'PROBE_LEDGER.json',ledger)
    print(json.dumps({'status':ledger['status'],'runs':len(ledger['rows']),
        'program_sha256':identity['exe']['sha256']}),flush=True)
if __name__=='__main__':main()
