"""Seal this finite round, preserving every historical entry and decision."""
import argparse
import datetime
from config import ROOT, read, sha, digest
from ipc_benchmark import write
from contact_pool import TAG, PROTOCOL, PROTOCOL_SHA, path_for

ANCHOR=ROOT/f'reports/active/{TAG}_index_anchor.json'
INDEX=ROOT/'reports/active/EXPERIMENT_INDEX.json'
DECISIONS=ROOT/'reports/active/EXPERIMENT_DECISIONS.json'

def anchor():
    index=read(INDEX);decisions=read(DECISIONS)
    assert len(index['entries'])==616 and TAG not in decisions
    write(ANCHOR,{'count':616,'entries_digest':digest(index['entries']),
        'decisions_digest':digest(decisions),'decisions':decisions,
        'index_file_sha256':sha(INDEX),'decisions_file_sha256':sha(DECISIONS)})
    print('Historical index anchored: 616 entries.')

def finish():
    a=read(ANCHOR);index=read(INDEX);decisions=read(DECISIONS)
    assert len(index['entries'])==a['count'] and digest(index['entries'])==a['entries_digest']
    assert digest(decisions)==a['decisions_digest'] and TAG not in decisions
    assert sha(PROTOCOL)==PROTOCOL_SHA
    manifest_path=ROOT/'builds/active/manifest.json';manifest=read(manifest_path)
    exe=ROOT/'builds/active/Release/gipc.exe';identity=sha(exe)
    assert manifest['status']=='completed'
    assert manifest['binaries'][exe.relative_to(ROOT).as_posix()]['sha256']==identity
    for row in manifest['files']:
        assert sha(ROOT/row['path'])==row['sha256'],row['path']
    files=[ANCHOR,manifest_path,PROTOCOL]
    runs=[];unattempted=[];analysis={};planned=0
    for stage in ('guards','screen','cost','full'):
        plan_path=path_for('configs',stage)
        if not plan_path.exists():continue
        plan=read(plan_path);assert plan['native_identity']==identity
        planned+=len(plan['runs'])
        files.append(plan_path);batch_path=ROOT/plan['report']
        batch=read(batch_path) if batch_path.exists() else {'runs':[]}
        if batch_path.exists():
            files.append(batch_path);assert batch['plan_sha256']==sha(plan_path)
        attempted={r['name'] for r in batch['runs']}
        unattempted.extend(r['name'] for r in plan['runs'] if r['name'] not in attempted)
        for t in batch['runs']:
            folder=ROOT/'runs/active'/t['name'];req=read(folder/'requested.json');r=read(folder/'result.json')
            assert t['result']==r
            if t['binary']=='active':assert req['exe_sha256']==identity
            saved=folder/'build_manifest.json'
            assert req['manifest_sha256']==sha(saved)
            saved_manifest=read(saved)
            saved_exe=next(v for k,v in saved_manifest['binaries'].items() if k.endswith('gipc.exe'))
            assert req['exe_sha256']==saved_exe['sha256']
            if r['status']=='completed':assert r['recorded_frames']==req['expanded_config']['steps']
            runs.append({'name':t['name'],'status':r['status'],'frames':r['recorded_frames'],
                'scene':req['expanded_config']['scene'],'solver_seconds':r.get('solver_seconds'),
                'diagnostic':r['heavy_diagnostics'],'exe_sha256':req['exe_sha256']})
            files.extend((folder/'requested.json',folder/'result.json',saved))
        report_path=path_for('reports',stage)
        if report_path.exists():
            analysis[stage]=read(report_path);files.append(report_path)
    fixture_path=ROOT/f'runs/active/{TAG}_fixtures/result.json'
    fixture=read(fixture_path);assert fixture['exe_sha256']==identity;files.append(fixture_path)
    for name in ('CONTACT_POOL_RESULTS_20261006.md','contact_pool_quality_review_20261006.md',
                 'NEXT_CONTACT_POOL_PLAN_20261006.md','contact_pool_resource_analysis_20261006.json',
                 'contact_pool_quality_results_20261006.json','contact_pool_quality_results_20261006.md'):
        files.append(ROOT/'reports/active'/name)
    # Validate every evidence file before updating either historical registry.
    file_records=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in files]
    ids={r['id'] for r in index['entries']}
    for r in runs:
        assert r['name'] not in ids
        folder='runs/active/'+r['name']+'/'
        index['entries'].append({'id':r['name'],'origin':TAG,'status':'pending' if r['status']=='completed' else 'failed',
            'completion':r['status'],'frames':r['frames'],'scene':r['scene'],'solver_seconds':r['solver_seconds'],
            'performance_certified':False,'quality_certified':False,
            'reason':'Finite shared-desktop component diagnostic; see detailed component decision.',
            'evidence':[folder+'requested.json',folder+'result.json',folder+'build_manifest.json']})
    index['entries'].append({'id':TAG+'_fixtures','origin':TAG,
        'status':'accepted' if fixture['passed'] else 'failed','status_scope':'GPU numerical fixtures only',
        'performance_certified':False,'quality_certified':False,'evidence':[fixture_path.relative_to(ROOT).as_posix()]})
    gate=all(analysis.get(s,{}).get('stage_gate_passed',False) for s in ('guards','screen','cost'))
    numerical_failure=not fixture['passed'] or any(
        r.get('pool',{}).get('totals',{}).get('validation_failed',0)
        for r in analysis.get('guards',{}).get('runs',[]))
    decisions[TAG]={'status':'rejected' if numerical_failure else 'pending',
        'status_scope':'Finite component diagnostic; performance and physical acceptance remain pending',
        'default_promoted':False,'performance_certified':False,'quality_certified':False,
        'full_run_gate_passed':gate,
        'stage_gates':{s:analysis.get(s,{}).get('stage_gate_passed') for s in ('guards','screen','cost')},
        'reason':'Missing speed evidence or physical compatibility support cannot promote a default; resource failures are retained.',
        'decision_evidence':'reports/active/CONTACT_POOL_RESULTS_20261006.md'}
    index['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    assert digest(index['entries'][:a['count']])==a['entries_digest']
    assert all(decisions[k]==v for k,v in a['decisions'].items())
    # Machine-maintained JSON, preserving each historical object verbatim.
    for path,value in ((INDEX,index),(DECISIONS,decisions)):
        path.write_text(__import__('json').dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    result={'delivery_verified':True,'exe_sha256':identity,'compiler_units':len(manifest['compiler_inputs']),
        'changed_objects':len(manifest['changed_objects']),'fixtures_passed':fixture['passed'],
        'historical_count':a['count'],'historical_entries_digest':a['entries_digest'],
        'historical_decisions_digest':a['decisions_digest'],'index_final_count':len(index['entries']),
        'planned_runs':planned,'attempted_runs':len(runs),'completed_runs':sum(r['status']=='completed' for r in runs),
        'unattempted':unattempted,'runs':runs,'default_promoted':False,
        'performance_certified':False,'quality_certified':False,
        'files':file_records,
        'index_sha256':sha(INDEX),'decisions_sha256':sha(DECISIONS)}
    write(ROOT/f'reports/active/{TAG}_verification.json',result)
    print(__import__('json').dumps({k:result[k] for k in ('delivery_verified','attempted_runs','completed_runs','index_final_count')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['anchor','finish']);a=p.parse_args()
    (anchor if a.action=='anchor' else finish)()
