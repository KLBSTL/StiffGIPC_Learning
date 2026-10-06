"""Verify frozen source/run identities and exact coverage of exported accepted paths."""
import csv
import hashlib
import json
from pathlib import Path
from report_local_perf_v33 import work

ROOT=Path(__file__).resolve().parents[1];data=ROOT/'downloads/autodl_perf_v41_20261003'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(data/'manifests/perf_v41_autodl.json')
for f in manifest['files']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
assert sha(ROOT/'tools/run_perf_v41.py')==sha(data/'tools/run_perf_v41.py')
rows=[]
for run in sorted((data/'runs/autodl').glob('autodl_perf_v41_*')):
    req=read(run/'requested.json');res=read(run/'result.json');frames,counts=work(run)
    assert req['source_digest']==manifest['source_digest']
    assert req['runner_sha256']==sha(data/'tools/run_perf_v41.py')
    assert req['exe_sha256'] in [x['sha256'] for x in manifest['binaries'].values()]
    assert req['dt']==.01 and req['tol']==.01 and req['pcg_tol']==1e-4 and req['suite']=='1'
    completed=res['recorded_frames'];assert len(frames)==completed+(res['status']!='completed')
    timing=list(csv.DictReader((run/'trace/frames.csv').open()));assert len(timing)==completed
    assert abs(sum(float(t['solver_ms']) for t in timing)/1000-res['solver_seconds'])<1e-8
    pcg=[n['pcg'] for f in frames for n in f.get('newton',[]) if 'pcg' in n]
    assert set(counts['execution'])=={req['execution']}
    for i,f in enumerate(frames):
        assert f['toi_policy']=='robust' and f['toi_penalty_scope']=='movable'
        assert f['robust_trial_velocity_tol']==.05
        assert f['toi_frame_friction_snapshot']['captured_this_solve']
        assert f['toi_frame_friction_snapshot']['physical_frame']==i
        assert all(t.get('candidate_query_policy')=='current_trial_shared_full_ccd' for t in f['toi'] if 'alpha' in t)
    row={'run':run.name,'result':res,'work':counts,'max_iterations':max(s['iterations'] for s in pcg),
         'breakdowns':[s for s in pcg if s.get('breakdown')],'requested_inverse64':req['mas_inverse64']}
    if req['substeps']:
        accepted=0
        for i,f in enumerate(frames):
            steps=[t for t in f['toi'] if 'alpha' in t]
            states=sorted((run/'trace/substeps').glob(f'safe_{i:04d}_*.bin'))
            assert len(states)==len(steps)+1
            assert states[0].read_bytes()==(run/f'trace/state_{i:04d}.bin').read_bytes()
            if i<completed:assert states[-1].read_bytes()==(run/f'trace/state_{i+1:04d}.bin').read_bytes()
            accepted+=len(steps)
        short=run.name.removeprefix('autodl_perf_v41_');ccd=read(ROOT/'reports'/f'CCD_V41_{short}.json')
        assert ccd['paths_checked']==accepted+len(frames)-1
        validator='diagnose_first_path' if 'bunny' in short else 'validate_path'
        exe=ROOT/f'builds/validator/Release/{validator}.exe'
        row['path_verification']={'accepted_segments':accepted,'stationary_bridges':len(frames)-1,
            'all_exported_accepted_paths_covered':True,'completed_endpoints_bitwise':True,'ccd':ccd,
            'validator_sha256':sha(exe),'command':[str(exe),str(run/'trace'),str(ROOT/'reports'/f'CCD_V41_{short}.json'),'substeps','--stable-nh1']}
    physics=run/'trace/physical_energy.csv'
    if physics.exists():
        energy=list(csv.DictReader(physics.open()));row['physics_rows']=len(energy)
        row['physics_first']=energy[0] if energy else None;row['physics_last']=energy[-1] if energy else None
    rows.append(row)
fixture=read(ROOT/'reports/MAS_FIXTURE_V41_CPU.json')
checks=read(ROOT/'downloads/fixtures_v41_20261003/reports/components.json')
component={k:{'count':len(v),'passed':all(t['passed'] for t in v)} for k,v in checks.items() if isinstance(v,list)}
result={'source_files_verified':len(manifest['files']),'source_digest':manifest['source_digest'],
        'binaries':manifest['binaries'],'runs':rows,'component_checks':component,
        'fixture_graph_gate':{x['case']:x['fixture']['full64_graph_passed'] for x in fixture},
        'full_trajectory_repair_passed':False,'performance_certified':False,
        'results_archive_sha256':sha(ROOT/'downloads/results_v41.tar.gz')}
target=ROOT/'reports/VERIFICATION_V41_20261003.json';assert not target.exists()
target.write_text(json.dumps(result,indent=2));print(json.dumps({'verified_source_files':len(manifest['files']),'verified_runs':len(rows),'components':component,'fixture_graph_gate':result['fixture_graph_gate']}))
