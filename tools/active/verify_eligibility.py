"""Complete one component's evidence chain; preserve every prior decision."""
import io
import json
import unittest
from datetime import datetime, timezone
from config import ROOT, read, sha, digest, matches_requested
from ipc_benchmark import write
from eligibility_round import TAG, checked
from verify_ipc_light_delivery import CountedResult
import test_contracts


def main():
    folder = ROOT / 'reports/active'
    target = folder / f'{TAG}_verification.json'
    assert not target.exists()
    protocol_path = folder / f'{TAG}_protocol.json'
    protocol = read(protocol_path)
    identity = sha(ROOT / 'builds/active/Release/gipc.exe')
    manifest_path = ROOT / 'builds/active/manifest.json'
    manifest = read(manifest_path)
    assert identity == protocol['identity'] == manifest['binaries']['builds/active/Release/gipc.exe']['sha256']
    assert sha(manifest_path) == protocol['manifest_sha256']
    assert manifest['status'] == 'completed' and len(manifest['compiler_inputs']) == 37
    tests = unittest.TextTestRunner(stream=io.StringIO(), resultclass=CountedResult).run(unittest.defaultTestLoader.loadTestsFromModule(test_contracts))
    assert tests.wasSuccessful() and tests.testsRun == 16 and tests.subtests == 92
    identity_checks = 0
    for row in manifest['files']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
        identity_checks += 1
    for unit in manifest['compiler_inputs']:
        for row in (unit['source'], unit['object'], *unit['project_include_inputs']):
            assert sha(ROOT / row['path']) == row['sha256'], row['path']
            identity_checks += 1
    link_text='\n'.join(c for r in manifest['link_evidence'] for c in r['commands']).upper()
    assert all(unit['object']['path'].replace('\\','/').rsplit('/',1)[-1].upper() in link_text
               for unit in manifest['compiler_inputs'])
    quality_path = folder / 'ipc_revision_20261005_quality_protocol.json'
    assert sha(quality_path) == protocol['old_quality_sha256'] == '1cfb9045d6988fa606b8bb76afdd71e20661ef602c84296c49cdda9a0d07d78f'
    fixture_path = ROOT / f'runs/active/{TAG}_fixtures/result.json'
    fixture = read(fixture_path)
    assert fixture['passed'] and fixture['exe_sha256'] == identity
    assert len(fixture['checks']) == 9 and all(r['passed'] for r in fixture['checks'])
    eligibility_fixture = next(r['result'] for r in fixture['checks'] if r['case'] == 'bvh_eligibility')
    assert eligibility_fixture['pass']
    assert len(eligibility_fixture['metadata_cases'])==48 and len(eligibility_fixture['query_cases'])==10
    assert all(c['pass'] for c in eligibility_fixture['metadata_cases']+eligibility_fixture['query_cases'])
    analysis_path = folder / f'{TAG}_hang_subset_analysis.json'
    evidence_path = folder / f'{TAG}_hang_subset_evidence.json'
    analysis, evidence = read(analysis_path), read(evidence_path)
    assert evidence['analysis_sha256'] == sha(analysis_path)
    selection_path = folder / f'{TAG}_selection.json'
    selection = read(selection_path)
    review_path=folder/f'{TAG}_review.json'
    review=read(review_path)
    assert review['evidence_sha256']==sha(evidence_path)
    direction_path=folder/f'{TAG}_direction_audit.json'
    direction=read(direction_path)
    assert direction['current_candidate']['source_sha256']==sha(evidence_path)
    for row in direction['source_calls']:
        assert sha(ROOT/row['path'])==row['current_sha256']
    for row in direction['captures']:
        assert sha(ROOT/row['sqlite'])==row['sqlite_sha256']
    rows, skipped = [], []
    artifact_paths = [protocol_path, analysis_path, evidence_path, selection_path, fixture_path, manifest_path, quality_path,review_path,
                      folder/f'{TAG}_analysis.json',folder/f'{TAG}_evidence.json',direction_path]
    for stage in ('guards','screen','profile','full'):
        plan_path = ROOT / f'configs/active/{TAG}_{stage}.json'
        batch_path = folder / f'{TAG}_{stage}_batch.json'
        if not batch_path.exists():
            if stage=='full':
                assert not selection['eligible']
                continue
            assert stage in ('screen','profile') and selection['disposition']=='blocked_prerequisite'
            source_plan=read(plan_path)
            source_sha=sha(plan_path)
            artifact_paths.append(plan_path)
            plan_path=ROOT/f'configs/active/{TAG}_{stage}_hang_subset.json'
            batch_path=folder/f'{TAG}_{stage}_hang_subset_batch.json'
            subset=read(plan_path)
            assert subset['source_plan_sha256']==source_sha==protocol['plan_sha256'][f'configs/active/{TAG}_{stage}.json']
            assert subset['runs']==[r for r in source_plan['runs'] if r['scene_key']=='hang']
            assert subset['guard_sha256']==sha(folder/f'{TAG}_guards_batch.json')
            skipped += [{'name':r['name'],'stage':stage,'reason':'Fixed/mixed guard incomplete; resource-limited subset only'}
                        for r in source_plan['runs'] if r['scene_key']!='hang']
        plan, batch = read(plan_path), read(batch_path)
        expected = (selection['full_plan_sha256'] if stage=='full' else
                    sha(plan_path) if 'hang_subset' in plan_path.name else protocol['plan_sha256'][plan_path.relative_to(ROOT).as_posix()])
        assert sha(plan_path) == batch['plan_sha256'] == expected
        planned = {r['name']:r for r in plan['runs']}
        assert len(planned)==len(plan['runs'])
        assert set(planned)=={r['name'] for r in batch['runs']}|{r['name'] for r in batch['skipped']}
        for r in batch['runs']:
            assert all(r[k]==v for k,v in planned[r['name']].items())
            run = ROOT/'runs/active'/r['name']
            req, result = read(run/'requested.json'), read(run/'result.json')
            assert result==r['result'] and matches_requested(req['expanded_config'],r['config'])
            assert req['config_sha256']==digest(req['expanded_config'])
            if r['binary']=='active':
                assert req['exe_sha256']==identity
                assert req['manifest_sha256']==sha(run/'build_manifest.json')==sha(manifest_path)
            check = checked(r,result)
            assert check['passed']==r['checks']['passed']
            assert check.get('eligibility_counters')==r['checks'].get('eligibility_counters')
            if stage=='profile' and check['passed']:
                exported = read(run/'cost_export_identity.json')
                assert exported['plan_sha256']==sha(plan_path)
                assert exported['capture_sha256']==sha(run/'nsight.nsys-rep')
            rows.append(r|{'stage':stage})
        skipped += [r|{'stage':stage} for r in batch['skipped']]
        artifact_paths += [plan_path,batch_path]
    assert len([r for r in rows if r['stage']=='guards'])==2
    assert sum(r['checks']['passed'] for r in rows if r['stage']=='guards')==1
    assert len(rows)==10 and sum(r['result']['status']=='completed' for r in rows)==9
    assert len(skipped)==9
    assert not (ROOT/'runs/active/.gpu.lock').exists()
    assert not any(d[k] for d in (protocol,selection,analysis,evidence,review) for k in ('performance_certified','quality_certified','default_promoted'))
    report_path = folder / 'IPC_ELIGIBILITY_RESULTS_20261006.md'
    artifact_paths += [report_path, folder/'IPC_ELIGIBILITY_PLAN_20261006.md',folder/'COMPONENT_REVIEW_PROTOCOL.md',ROOT/'STATUS.md',
                       ROOT/'tools/active/eligibility_round.py',ROOT/'tools/active/eligibility_evidence.py',ROOT/'tools/active/verify_eligibility.py',
                       ROOT/'tools/active/review_eligibility.py',ROOT/'tools/active/fixtures.py',
                       ROOT/'sources/stiff_active/StiffGIPC/collision/query_eligibility.md']
    failed_build_log=ROOT/'builds/active/provenance/20261006T035027_481734Z_ipc_eligibility_20261006/build.log'
    artifact_paths.append(failed_build_log)
    index_path, decisions_path = folder/'EXPERIMENT_INDEX.json',folder/'EXPERIMENT_DECISIONS.json'
    index, decisions = read(index_path),read(decisions_path)
    prior, old_decisions = index['entries'],dict(decisions)
    assert len(prior)==570
    prior_digest, decisions_digest = digest(prior),digest(old_decisions)
    additions = []
    def add(name,origin,status,reason,**fields):
        assert name not in decisions and name not in {r['id'] for r in prior+additions}
        decision = {'status':status,'status_scope':'Finite local eligibility component diagnostic', 'reason':reason,
                    'performance_certified':False,'quality_certified':False,'default_promoted':False,
                    'decision_evidence':report_path.relative_to(ROOT).as_posix()}
        decisions[name]=decision
        additions.append({'id':name,'origin':origin,'evidence':[report_path.relative_to(ROOT).as_posix(),target.relative_to(ROOT).as_posix()],**decision,**fields})
    for r in rows:
        add(r['name'],'active_run','accepted' if r['checks']['passed'] else 'failed',
            'Complete diagnostic; physics/performance certification pending' if r['checks']['passed'] else str(r['checks']['failures']),
            stage=r['stage'],frames=r['result']['recorded_frames'],completion=r['result']['status'],material_100f_passed=r['checks'].get('material_100f_passed'))
    for r in skipped: add(r['name'],'active_unattempted','pending',r['reason'],stage=r['stage'],completion='not_executed')
    add(TAG+'_fixtures','active_fixture','accepted','Independent summary and eight prior GPU checks passed',exe_sha256=identity)
    add(TAG+'_candidate','active_candidate_decision','rejected',
        'Hang 43f paired median 0.962011 fails performance screen; fixed/mixed guards incomplete, stop expansion',implemented=True)
    add(TAG+'_first_build','active_build','failed','NVCC/MSVC fixture constexpr capture compile error retained',build_log_sha256=sha(failed_build_log))
    add(TAG+'_final_build','active_build','accepted','Fixture type-trait repair; 37 units, 11 changed objects, linked input identities verified',exe_sha256=identity)
    add(TAG+'_delivery','active_delivery','accepted','Implementation and finite partial tests analyzed; general quality/performance acceptance pending',exe_sha256=identity)
    index['entries']=prior+additions
    index['updated_utc']=datetime.now(timezone.utc).isoformat()
    assert digest(index['entries'][:570])==prior_digest
    assert digest({k:decisions[k] for k in old_decisions})==decisions_digest
    report = {'delivery_verified':True,'exe_sha256':identity,'source_object_identity_checks':identity_checks,
              'compiler_units':37,'changed_objects':len(manifest['changed_objects']),'config_tests':tests.testsRun,'config_subtests':tests.subtests,
              'gpu_fixture_cases':9,'attempted':len(rows),'completed':sum(r['result']['status']=='completed' for r in rows),
              'failed':sum(not r['checks']['passed'] for r in rows),'skipped':len(skipped),'historical_entries_and_decisions_preserved':True,
              'old_quality_protocol_unchanged':True,'prior_index_count':570,'total_index_count':len(index['entries']),
              'prior_index_digest':prior_digest,'prior_decisions_digest':decisions_digest,
              'performance_certified':False,'quality_certified':False,'default_promoted':False,
              'evidence_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in artifact_paths}}
    decisions_path.write_text(json.dumps(decisions,indent=2,allow_nan=False),encoding='utf-8')
    index_path.write_text(json.dumps(index,indent=2,allow_nan=False),encoding='utf-8')
    write(target,report)
    print(json.dumps({k:report[k] for k in ('delivery_verified','attempted','completed','failed','skipped','total_index_count','default_promoted')}))


if __name__=='__main__':main()
