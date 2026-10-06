"""Verify and index this finite round without rewriting historical decisions."""
import csv
import datetime
import json
from config import ROOT, read, sha, digest
from ipc_benchmark import write
from contact_optimization import TAG, PROTOCOL, PROTOCOL_SHA, window

def finish():
    assert sha(PROTOCOL)==PROTOCOL_SHA
    manifest_path=ROOT/'builds/active/manifest.json';manifest=read(manifest_path)
    exe=ROOT/'builds/active/Release/gipc.exe'
    identity=sha(exe)
    assert manifest['status']=='completed'
    assert manifest['binaries'][exe.relative_to(ROOT).as_posix()]['sha256']==identity
    required=['collision/ipc_bounded_ccd.h','collision/ipc_bounded_ccd.inl',
        'collision/ipc_bounded_ccd_observer.h','core/GIPC.cu','solver/MASPreconditioner.cu']
    for logical in required:
        path=ROOT/'sources/stiff_active/StiffGIPC'/logical
        row=next(r for r in manifest['files'] if (ROOT/r['path']).resolve()==path.resolve())
        assert row['sha256']==sha(path), 'Current source differs from tested build: '+logical
    fixture=ROOT/'runs/active'/f'{TAG}_fixtures/result.json'
    f=read(fixture);assert f['exe_sha256']==identity and f['passed']
    evidence=[manifest_path,fixture,PROTOCOL]
    runs=[];planned=[];unattempted=[]
    stages=['guards','screen','profile']
    if (ROOT/f'configs/active/{TAG}_full.json').exists():stages.append('full')
    baseline=ROOT/'configs/active/ipc_contact_20261006_baseline.json'
    for path in [baseline,*[ROOT/f'configs/active/{TAG}_{s}.json' for s in stages]]:
        plan=read(path);batch_path=ROOT/plan['report'];batch=read(batch_path)
        assert batch['plan_sha256']==sha(path)
        evidence += [path,batch_path]
        planned.extend(t['name'] for t in plan['runs'])
        attempted={t['name'] for t in batch['runs']}
        unattempted += [t['name'] for t in plan['runs'] if t['name'] not in attempted]
        for task in batch['runs']:
            folder=ROOT/'runs/active'/task['name'];req=read(folder/'requested.json');result=read(folder/'result.json')
            assert result==task['result']
            saved=read(folder/'build_manifest.json')
            assert req['manifest_sha256']==sha(folder/'build_manifest.json')
            assert req['exe_sha256']==saved['binaries'][next(p for p in saved['binaries'] if p.endswith('gipc.exe'))]['sha256']
            if path!=baseline and task['binary']=='active':assert req['exe_sha256']==identity
            if result['status']=='completed':assert result['recorded_frames']==req['expanded_config']['steps']
            runs.append({'name':task['name'],'scene':req['expanded_config']['scene'],
                'status':result['status'],'frames':result['recorded_frames'],
                'solver_seconds':result.get('solver_seconds'),'exe_sha256':req['exe_sha256'],
                'request_sha256':sha(folder/'requested.json'),'result_sha256':sha(folder/'result.json'),
                'diagnostic':result.get('heavy_diagnostics',False)})
            evidence += [folder/'requested.json',folder/'result.json',folder/'build_manifest.json']
    screen=read(ROOT/f'reports/active/{TAG}_screen_analysis.json')
    guards=read(ROOT/f'reports/active/{TAG}_guards_analysis.json')
    assert guards['attempted']==3 and all(r['status']=='completed' and r['hard_checks_passed'] for r in guards['runs'])
    audit_calls=sum(r['bounded']['validation_calls'] for r in guards['runs'])
    assert audit_calls>0 and not sum(r['bounded']['validation_failed'] for r in guards['runs'])
    posthard=[]
    for stage in stages:
        analysis_path=ROOT/f'reports/active/{TAG}_{stage}_analysis.json'
        analysis=read(analysis_path);evidence.append(analysis_path)
        for row in analysis['runs']:
            if row['status']!='completed':continue
            abd=[f['abd_min_J'] for f in row['frames'] if f['abd_min_J'] is not None]
            passed=all(j>0 for j in abd) and row['fixed_drift_m']<=1e-10
            assert passed, 'ABD flip or fixed drift: '+row['name']
            posthard.append({'name':row['name'],'abd_min_J':min(abd) if abd else None,
                'fixed_drift_m':row['fixed_drift_m'],'passed':passed})
    for name in ('CONTACT_OPTIMIZATION_PLAN_20261006.md','CONTACT_OPTIMIZATION_RESULTS_20261006.md',
                 'NEXT_CONTACT_POOL_PLAN_20261006.md',
                 'contact_cost_rank_20261006_agent.json','contact_baseline_quality_20261006_agent.json',
                 'bounded_ccd_mechanism_20261006_agent.json','contact_linear_cost_20261006_agent.json'):
        evidence.append(ROOT/'reports/active'/name)
    for scene in ('hang','fixed'):
        folder=ROOT/'runs/active'/f'{TAG}_profile_{scene}_off_nodes'
        evidence.extend(folder/name for name in ('cost_export_identity.json','activity_analysis.json','linear_profile.json'))
    promoted=False
    index_path=ROOT/'reports/active/EXPERIMENT_INDEX.json'
    decisions_path=ROOT/'reports/active/EXPERIMENT_DECISIONS.json'
    index=read(index_path);decisions=read(decisions_path)
    old_entries=list(index['entries']);old_decisions=dict(decisions)
    assert len(old_entries)==594 and TAG not in decisions
    ids={e['id'] for e in old_entries}
    for r in runs:
        assert r['name'] not in ids
        prefix='runs/active/'+r['name']+'/'
        index['entries'].append({'id':r['name'],'origin':TAG,'scene':r['scene'],
            'status':'pending' if r['status']=='completed' else 'failed','completion':r['status'],
            'performance_certified':False,'quality_certified':False,
            'reason':'Shared desktop diagnostic; numerical coverage and physical certification are separate.',
            'frames':r['frames'],'solver_seconds':r['solver_seconds'],
            'evidence':[prefix+'requested.json',prefix+'result.json',prefix+'build_manifest.json']})
    index['entries'].append({'id':TAG+'_fixtures','origin':TAG,'status':'accepted',
        'status_scope':'GPU fixtures and historical numerical regressions only',
        'performance_certified':False,'quality_certified':False,'evidence':[fixture.relative_to(ROOT).as_posix()]})
    decision={'status':'pending' if screen['full_performance_gate_passed'] else 'rejected',
        'status_scope':'Default promotion and full-performance gate',
        'default_promoted':promoted,'performance_certified':False,'quality_certified':False,
        'reason':'Short-window net >=5% gate evaluated separately for both contact scenes; no relaxed stops.',
        'decision_evidence':f'reports/active/{TAG}_screen_analysis.json',
        'full_performance_gate_passed':screen['full_performance_gate_passed']}
    decisions[TAG]=decision
    index['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    assert index['entries'][:594]==old_entries
    assert all(decisions[k]==v for k,v in old_decisions.items())
    verification={'delivery_verified':True,'exe_sha256':identity,'manifest_sha256':sha(manifest_path),
        'compiler_units':len(manifest['compiler_inputs']),'changed_objects':len(manifest['changed_objects']),
        'quality_protocol_sha256':PROTOCOL_SHA,'fixtures_passed':True,'guard_audit_calls':audit_calls,
        'planned_runs':len(planned),'attempted_runs':len(runs),
        'completed_runs':sum(r['status']=='completed' for r in runs),'unattempted':unattempted,
        'historical_index_count':594,'historical_entries_digest':digest(old_entries),
        'historical_decisions_digest':digest(old_decisions),'index_final_count':len(index['entries']),
        'default_promoted':False,'performance_certified':False,'quality_certified':False,
        'runs':runs,'posthard_checks':posthard,
        'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in evidence]}
    # Generated index maintenance preserves every historical object verbatim.
    index_path.write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    decisions_path.write_text(json.dumps(decisions,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    verification['index_sha256']=sha(index_path);verification['decisions_sha256']=sha(decisions_path)
    write(ROOT/f'reports/active/{TAG}_verification.json',verification)
    print(json.dumps({k:verification[k] for k in ('delivery_verified','attempted_runs','completed_runs',
        'guard_audit_calls','index_final_count','exe_sha256')}))

if __name__=='__main__':finish()
