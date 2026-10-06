"""Close the single candidate; preserve historical index entries and failures."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from config import ROOT,read,sha,digest
from ipc_benchmark import write

TAG='ipc_legacy_ordered_20261005'

def main():
    folder=ROOT/'reports/active';analysis=read(folder/f'{TAG}_analysis.json')
    fixtures=read(ROOT/f'runs/active/{TAG}_fixtures/result.json')
    manifest=read(ROOT/'builds/active/manifest.json');exe=ROOT/'builds/active/Release/gipc.exe'
    expected=manifest['binaries'][exe.relative_to(ROOT).as_posix()]['sha256']
    assert sha(exe)==expected==fixtures['exe_sha256']
    assert fixtures['passed'] and len(fixtures['checks'])==8
    assert analysis['completed_system_numerical_checks_passed'] and not analysis['cost_gate_passed']
    assert not analysis['promotion_allowed'] and not analysis['default_changed']
    assert len(analysis['systems'])==2 and len(analysis['unavailable_systems'])==2
    assert not analysis['full_fixed_system_coverage_passed']
    protocol=read(folder/f'{TAG}_protocol.json')
    assert sha(folder/'ipc_revision_20261005_quality_protocol.json')==protocol['old_quality_protocol_sha256']
    assert len(manifest['compiler_inputs'])==37 and len(manifest['changed_objects'])==14
    identities=[];includes=set()
    for unit in manifest['compiler_inputs']:
        for key in ('source','object'):
            item=unit[key];assert sha(ROOT/item['path'])==item['sha256']
            identities.append({'kind':key,'path':item['path'],'passed':True})
        for item in unit['project_include_inputs']:
            assert sha(ROOT/item['path'])==item['sha256'];includes.add(item['path'])
    required=['solver/legacy_restrict_options.h','solver/legacy_ordered_restrict.cuh',
              'linear_system/solver/pcg_legacy_restrict_study.inl']
    assert all(any(p.endswith(suffix) for p in includes) for suffix in required)
    assert not (ROOT/'runs/active/.gpu.lock').exists()
    comparisons=[]
    new=ROOT/f'runs/active/{TAG}_disabled_f2';old=ROOT/'runs/active/ipc_mas_stage_20261005_disabled_f2'
    for frame in range(3):
        x=np.fromfile(new/f'trace/state_{frame:04d}.bin','<f8');y=np.fromfile(old/f'trace/state_{frame:04d}.bin','<f8')
        assert x.shape==y.shape
        comparisons.append({'frame':frame,'max_position_component_difference_m':float(np.abs(x-y).max())})
    assert comparisons[0]['max_position_component_difference_m']==0
    assert max(p['max_position_component_difference_m'] for p in comparisons)<1e-10
    checks=analysis['run_checks'];assert all(p['exe_sha256']==expected for p in checks)
    disabled=next(p for p in checks if p['run']==new.name)
    assert disabled['production_iterations']==[6,16,6,7,18]
    verification={'delivery_verified':True,'scope':'Completed diagnostics/regression, not full coverage, quality or performance acceptance',
        'completed_systems':2,'unavailable_systems':analysis['unavailable_systems'],
        'fixed_system_pcg_solves':32,'full_M_probes':48,'ordered_stage_actions':16,
        'existing_fixture_checks_passed':8,'configuration_tests_passed':11,'configuration_subtests_passed':55,
        'exe_sha256':expected,'manifest_sha256':sha(ROOT/'builds/active/manifest.json'),
        'source_object_checks':identities,'new_kernel_and_study_in_compile_dependency_closure':True,
        'default_off_short_prefix':comparisons,'old_quality_protocol_unchanged':True,
        'production_thresholds_changed':False,'default_promoted':False,'cost_gate_passed':False,
        'gpu_lock_present':False,'subagent_read_only_review':'No blocking code issue; restoration and cost limits explicitly documented',
        'evidence_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [folder/f'{TAG}_analysis.json',ROOT/f'runs/active/{TAG}_fixtures/result.json']}}
    write(folder/f'{TAG}_verification.json',verification)
    index_path=folder/'EXPERIMENT_INDEX.json';index=read(index_path);prior=index['entries'];prior_hash=digest(prior)
    decisions_path=folder/'EXPERIMENT_DECISIONS.json';decisions=read(decisions_path)
    evidence=['reports/active/IPC_LEGACY_ORDERED_RESULTS_20261005.md',f'reports/active/{TAG}_verification.json']
    additions=[]
    names=[s['name'] for s in read(ROOT/f'configs/active/{TAG}.json')['runs']]+[TAG+'_fixtures']
    existing={row['id'] for row in prior}
    for name in names:
        assert name not in existing
        run=ROOT/'runs/active'/name;result=read(run/'result.json');req_path=run/'requested.json'
        req=read(req_path) if req_path.exists() else {};config=req.get('expanded_config',{})
        fixture=name.endswith('_fixtures');failed=not fixture and result['status']!='completed'
        decision={'status':'failed' if failed else 'accepted',
            'status_scope':'Resource budget failure; selected fixed system unavailable' if failed else 'Diagnostic/fixture completion only',
            'performance_certified':False,'quality_certified':False,
            'reason':'Memory reserve preserved; no retry. Candidate not promoted.' if failed else 'Completed numerical/regression diagnosis; cost gate fails and material quality unverified.',
            'decision_evidence':evidence[0]}
        decisions[name]=decision
        additions.append({'id':name,'origin':'active_fixture' if fixture else 'active_run',
            'scene':config.get('scene'),'frames':result.get('recorded_frames'),
            'completion':result.get('status'),'solver_seconds':result.get('solver_seconds'),
            'config_sha256':req.get('config_sha256'),'exe_sha256':req.get('exe_sha256',result.get('exe_sha256')),
            'diagnostics':config.get('diagnostics'),'evidence':[p.relative_to(ROOT).as_posix() for p in [req_path,run/'result.json'] if p.exists()]+evidence,
            **decision})
    candidate=TAG+'_candidate';assert candidate not in existing
    decision={'status':'rejected','status_scope':'Execution optimization promotion only; not numerical failure',
        'performance_certified':False,'quality_certified':False,
        'reason':'Completed systems pass finite numerical checks but no 15% complete-linear cost benefit evidence; two systems resource-incomplete. Default remains atomic.',
        'decision_evidence':evidence[0]}
    decisions[candidate]=decision;additions.append({'id':candidate,'origin':'active_candidate_decision','evidence':evidence,**decision})
    assert digest(prior)==prior_hash
    index['entries']=prior+additions;index['updated_utc']=datetime.now(timezone.utc).isoformat()
    assert len({r['id'] for r in index['entries']})==len(index['entries'])
    assert digest(index['entries'][:len(prior)])==prior_hash
    decisions_path.write_text(json.dumps(decisions,indent=2,allow_nan=False),encoding='utf-8')
    index_path.write_text(json.dumps(index,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'delivery_verified':True,'cost_gate_passed':False,'new_index_entries':len(additions),
        'old_index_entries_preserved':len(prior),'total_index_entries':len(index['entries']),
        'completed_fixed_systems':2,'resource_incomplete_systems':2,'existing_fixtures':8}))

if __name__=='__main__':main()
