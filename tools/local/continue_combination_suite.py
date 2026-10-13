"""Explicitly finish the six-scene subset after user defers Stack; no retry/overwrite."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/paper_suite'),str(ROOT/'tools/bench')]
from run_combination_suite import plan,checkpoint_env,identities,power,observation
from config import read
from linux_runner import require,write_new
from local_identity import record,verify
from pool_metrics import export_evidence
import windows_runner


def run(session):
    packet=read(session/'identities.json');ids=identities()
    require(ids==packet['programs'] and power()==packet['power'],'Program identity/power changed')
    verify([packet['controller']]);previous=read(session/'performance_ledger.json')
    require(not previous['complete'],'Original stage already complete')
    require(not (session/'performance_delivery_ledger.json').exists(),'Continuation already exists')
    failures=[r for r in previous['runs'] if 'observation' not in r]
    require(len(failures)==1 and failures[0]['task']['scene_key']=='Stack' and failures[0]['result']['status']=='memory_budget',
        'Unexpected original failure; do not silently omit it')
    selected=[s for s in plan()['scenes'] if s!='Stack']
    tasks=[t for t in plan()['stages']['performance'] if t['scene_key'] in selected]
    prior={r['task']['name']:r for r in previous['runs'] if 'observation' in r}
    require(all(name in {t['name'] for t in tasks} for name in prior),'Unexpected prior successful run')
    receipt=dict(authorized_by_user='Stack暂缓，先交付六个场景',selected_scenes=selected,excluded_scenes=['Stack'],
        controller=record(__file__),prior_ledger=record(session/'performance_ledger.json'),
        reuse_complete_runs=len(prior),remaining_runs=len(tasks)-len(prior),retry=False,
        all_original_failures_retained=True,original_plan=packet['plan'])
    write_new(session/'continuation_manifest.json',receipt)
    rows=[];original=windows_runner.environment
    with windows_runner.gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
        for t in tasks:
            if t['name'] in prior:
                r=prior[t['name']];require(r['task']==t,'Prior task changed')
                require(read(session/t['name']/'result.json')['status']=='completed','Prior native status changed')
                rows.append(r);continue
            require(not (session/t['name']).exists(),'Unexpected output exists; no overwrite')
            identity=ids[t['identity_key']]
            windows_runner.environment=lambda c,out,t=t,identity=identity:checkpoint_env(c,out,t,identity,session)
            try:result=windows_runner.execute(session,t,identity,0)
            finally:windows_runner.environment=original
            row=dict(task=t,result=result);rows.append(row)
            if result['status']=='completed':
                try:
                    row['observation']=observation(session/t['name'],t,result)
                    if identity['capabilities'].get('actual_velocity'):
                        require(export_evidence(session/t['name'],100,'velocity')['passed'],'Incomplete actual velocity')
                except Exception as error:row['analysis_error']=f'{type(error).__name__}: {error}'
            with (session/'performance_continuation_journal.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            if 'observation' not in row:break
            m=row['observation'];print(json.dumps(dict(run=t['name'],seconds=m['TimeTot'],N=m['N_tot'],P=m['P_tot'])),flush=True)
        ledger=dict(stage='performance_delivery',runs=rows,skipped=tasks[len(rows):],selected_scenes=selected,excluded_scenes=['Stack'],
            scope_reduction=receipt,complete=len(rows)==len(tasks)==90 and all('observation' in r for r in rows),power_after=power())
        write_new(session/'performance_delivery_ledger.json',ledger)
    require(ledger['power_after']==packet['power'],'Power changed')
    for identity in ids.values():verify(identity['sources']+identity['build_evidence']+[identity['exe']]+identity['dlls']+identity.get('objects',[]))
    print(json.dumps(dict(complete=ledger['complete'],runs=len(rows),formal_frames=100*len(rows),excluded=['Stack'])),flush=True)
    return 0 if ledger['complete'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    a=p.parse_args();raise SystemExit(run(a.session.resolve()))
