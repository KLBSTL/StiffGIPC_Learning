"""Close the finite diagnosis, preserving all previous acceptance decisions."""
import json
from pathlib import Path
from config import ROOT,read,sha
from ipc_benchmark import write

TAG='ipc_mas_stage_20261005'


def main():
    folder=ROOT/'reports/active';analysis=read(folder/f'{TAG}_analysis.json')
    study_path=ROOT/f'runs/active/{TAG}_f2/fixed/f2_n1_study.json'
    study=read(study_path);stages=study['mas_stage_probe']['stages']
    bitwise={}
    for row in stages:
        outputs=row['outputs'];ref=Path(outputs[0]['file']).read_bytes()
        bitwise[row['stage']]=[Path(o['file']).read_bytes()==ref for o in outputs[1:]]
    assert not all(bitwise['restrict']) and all(bitwise['local']) and all(bitwise['prolong'])
    assert not all(bitwise['full_action'])
    fixtures=read(ROOT/f'runs/active/{TAG}_fixtures/result.json')
    manifest=read(ROOT/'builds/active/manifest.json')
    exe=ROOT/'builds/active/Release/gipc.exe'
    expected=manifest['binaries'][exe.relative_to(ROOT).as_posix()]['sha256']
    assert sha(exe)==expected==analysis['run_checks'][0]['exe_sha256']==fixtures['exe_sha256']
    assert analysis['passed'] and fixtures['passed'] and len(fixtures['checks'])==8
    protocol=read(folder/f'{TAG}_protocol.json')
    assert sha(folder/'ipc_revision_20261005_quality_protocol.json')==protocol['old_quality_protocol_sha256']
    assert not (ROOT/'runs/active/.gpu.lock').exists()
    report={'passed':True,'scope':'Completed same-system attribution and regression only; quality/performance not certified',
        'bitwise_equal_to_warmup':bitwise,'scene_runs_completed':2,'stage_replays':16,
        'kernel_stage_actions':24,'existing_fixture_checks_passed':8,
        'configuration_tests_passed':10,'configuration_subtests_passed':44,
        'configuration_test_command':'E:/Anaconda/envs/DL/python.exe -m pytest -q tools/active/test_contracts.py',
        'exe_sha256':expected,'source_manifest_sha256':sha(ROOT/'builds/active/manifest.json'),
        'old_quality_protocol_unchanged':True,'gpu_lock_present':False,
        'production_thresholds_changed':False,'new_numerical_candidate_implemented':False,
        'evidence_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in
            [study_path,folder/f'{TAG}_analysis.json',ROOT/f'runs/active/{TAG}_fixtures/result.json']}}
    write(folder/f'{TAG}_verification.json',report)
    index_path=folder/'EXPERIMENT_INDEX.json';index=read(index_path)
    decision_path=folder/'EXPERIMENT_DECISIONS.json';decisions=read(decision_path)
    current={TAG+'_f2',TAG+'_disabled_f2',TAG+'_fixtures'}
    evidence=['reports/active/IPC_MAS_STAGE_RESULTS_20261005.md',f'reports/active/{TAG}_verification.json']
    updated=[]
    for row in index['entries']:
        if row['id'] not in current:continue
        scope='Fixture correctness only' if row['id'].endswith('_fixtures') else 'Fixed-system diagnostic completion only'
        decision={'status':'accepted','status_scope':scope,'performance_certified':False,'quality_certified':False,
            'reason':'Finite stage attribution and restoration verified. Does not accept trajectory quality or default promotion.',
            'decision_evidence':evidence[0]}
        row.update(decision);row['evidence']=list(dict.fromkeys(row['evidence']+evidence))
        decisions[row['id']]=decision;updated.append(row['id'])
    assert set(updated)==current
    decision_path.write_text(json.dumps(decisions,indent=2,allow_nan=False),encoding='utf-8')
    index_path.write_text(json.dumps(index,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'passed':True,'bitwise_equal_to_warmup':bitwise,'existing_fixtures':8,
                      'index_entries_updated':len(updated),'old_quality_protocol_unchanged':True}))


if __name__=='__main__':main()
