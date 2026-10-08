"""CPU-only downloaded-report verification; never launches a simulation."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def same(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-11)

def main():
    results=read(ROOT/'RESULTS.json');seal=read(ROOT/'SEAL.json');build=read(ROOT/'BUILD_IDENTITY.json')
    status=read(ROOT/'STATUS.json');csv_rows=list(csv.DictReader((ROOT/'TIMINGS.csv').open()))
    assert results['recorded_tasks']==results['planned_tasks']==len(seal['tasks'])==len(csv_rows)==21
    assert len(results['scenes'])==7 and status['stage']=='completed'
    assert status['recorded_runs']==status['completed_native_runs']==status['hard_checks_passed']==21
    assert build['passed'] and build['sources_unchanged'] and build['assets_equivalence']['passed']
    assert build['builds']['active']['counts']=={'tu':37,'cuda':33,'cxx':4}
    assert build['builds']['base']['counts']=={'tu':35,'cuda':31,'cxx':4}
    assert seal['build_identity_sha256']==sha(ROOT/'BUILD_IDENTITY.json')
    assert build['temporal_freshness']['actual_supervisor_start_available'] is False
    assert 'Filesystem birth time' in build['temporal_freshness']['timestamp_source']
    analysis_rows=[]
    for index,task in enumerate(seal['tasks'],1):
        row=read(ROOT/'runs'/f'{index:02d}.json');a=row['analysis'];r=row['result'];c=task['config'];t=a['timing'];csv_row=csv_rows[index-1]
        assert row['task']==task and row['seal_sha256']==sha(ROOT/'SEAL.json')
        assert (c['steps'],c['dt'],c['pcg_rho_tol'],c['ipc_cumulative_tol'],c['ipc_min_updates'])==(120,.01,1e-4,.01,6)
        assert c['backend']=='ipc' and c['mas']=='legacy' and c['ipc_termination']=='legacy'
        assert c['timeout_seconds']==180 and c['pcg_graph_chunk']==1 and not c['diagnostics']
        assert r['status']=='completed' and r['exit_code']==0 and r['recorded_frames']==120
        assert r['finite'] and a['hard_checks_passed'] and not a['failures']
        assert not r['heavy_diagnostics'] and r['wall_seconds']<=180
        assert not any(s['foreign_compute_pids'] for s in r['gpu_samples'])
        m=seal[task['binary']+'_manifest'];program=a['program_identity']
        assert program['source_digest']==m['source_digest'] and program['exe_sha256']==m['exe_sha256']
        assert m['exe_sha256']==build['builds'][task['binary']]['exe']['sha256']
        assert program['config_sha256']==task['expanded_config_sha256']
        assert csv_row['scene']==task['scene_key'] and csv_row['arm']==task['arm']
        assert csv_row['status']=='completed' and csv_row['hard_checks_passed']=='True'
        assert same(csv_row['solver_seconds'],t['solver_seconds']) and same(r['solver_seconds'],t['solver_seconds'])
        assert same(csv_row['wall_seconds'],t['process_wall_seconds'])
        assert same(csv_row['core_event_seconds'],t['core_cuda_event_seconds'])
        assert int(csv_row['pcg_iterations'])==a['work']['pcg_iterations']
        assert int(csv_row['linear_directions'])==a['work']['linear_directions']
        for dest,key in [('assembly','assembly'),('linear','pcg'),('ccd','ccd'),('line_search','line_search'),('state_update','state_update')]:
            assert same(csv_row[dest+'_seconds'],t['phase_ms'][key]/1000)
        analysis_rows.append(a)
    for scene in results['scenes']:
        rows=scene['runs'];arms={r['arm']:r for r in rows}
        assert len(rows)==3 and set(arms)=={'base','graph','all'}
        assert all(r in analysis_rows for r in rows)
        assert len(scene['input_comparisons'])==2 and all(p['passed'] for p in scene['input_comparisons'])
        s={arm:r['timing']['solver_seconds'] for arm,r in arms.items()};ratios=scene['ratios']
        assert ratios['base']==1 and same(ratios['graph'],s['base']/s['graph'])
        assert same(ratios['all'],s['base']/s['all']) and same(ratios['other_components'],s['graph']/s['all'])
        assert not scene['performance_certified'] and not scene['quality_certified']
        assert all(m['frames']==121 for m in scene['state_comparisons']['base/all']['position'].values())
    files=[ROOT/name for name in ('RESULTS.json','SEAL.json','BUILD_IDENTITY.json','TIMINGS.csv','STATUS.json','REPORT.md','ANALYSIS.md','PLAN.md','DOWNLOAD_RECEIPT.json')]
    packet={'passed':True,'cpu_only':True,'scenes':7,'complete_120_frame_runs':21,
            'configuration_program_input_and_ratio_checks':True,'quality_certified':False,'performance_certified':False,
            'raw_trajectory_rehash_scope':'Original raw was verified on AutoDL by the sealed controller. This offline check uses downloaded reports; raw trajectories not downloaded.',
            'actual_supervisor_start_available':False,'files':[{'path':p.name,'sha256':sha(p)} for p in files],
            'verifier_sha256':sha(Path(__file__))}
    (ROOT/'VERIFICATION.json').write_text(json.dumps(packet,indent=2))
    print(json.dumps({k:packet[k] for k in ('passed','cpu_only','scenes','complete_120_frame_runs')}))

if __name__=='__main__':main()
