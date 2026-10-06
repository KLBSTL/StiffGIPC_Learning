"""Identity and evidence gates for v38, without simulation certification."""
import hashlib
import json
from pathlib import Path

r=Path(__file__).resolve().parents[1]
data=r/'downloads/mas_replay_v38'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(r/'manifests/mas_replay_v38_repaired.json')
assert all(sha(r/f['path'])==f['sha256'] for f in manifest['files'])
old=read(r/'manifests/perf_v37.json')
assert all(sha(r/f['path'])==f['sha256'] for f in old['files'])
analysis=read(r/'reports/MAS_REPLAY_V38_EXTENDED_ANALYSIS.json')
rows=[];binary_hashes=set();total_solves=0
for run in sorted((data/'runs').iterdir()):
    req=read(run/'requested.json');result=read(run/'result.json');done=read(run/'completion.json')
    assert done['exit_code']==0 and len(result['solves'])==9 and len(result['operators'])==10
    assert result['native_inverse_recomputed_bitwise']
    assert req['manifest_sha256']==sha(r/'manifests/mas_replay_v38_repaired.json')
    assert req['runner_sha256']==sha(data/'tools/run_replay_v38.py')
    binary_hashes.add(req['binary_sha256']);total_solves+=len(result['solves'])
    source_run='autodl_perf_v37_mas_smoke' if run.name.startswith('smoke') else 'autodl_perf_v37_bunny_mas'
    source=r/'downloads/autodl_perf_v37_20261003/runs/autodl'/source_run
    assert all(sha(source/name)==h for name,h in req['input_sha256'].items())
    rows.append({'run':run.name,'solves':len(result['solves']),'rho_stops':sum(x['reason']=='rho_stop' for x in result['solves']),
                 'iteration_limits':sum(x['reason']=='iteration_limit' for x in result['solves']),
                 'breakdowns':sum(x['reason']=='breakdown' for x in result['solves'])})
assert len(binary_hashes)==1 and len(rows)==14 and total_solves==126
assert len(analysis)==14
for row in analysis:
    assert max(s['cpu_gpu_residual_disagreement'] for s in row['solves'])<1e-7
    if row['mode']=='captured_r64' or (row['mode']=='full64' and row['case']!='failure'):
        assert max(x['relative'] for x in row['cpu_hierarchy_operator_checks'])<1e-6
    if row['case']=='failure' and row['mode']=='full64':
        assert max(x['relative'] for x in row['same_inverse_cpu_hierarchy_checks'])<1e-6
    if row['case']=='smoke' and row['mode']=='captured_native':assert row['max_vs_v37']<5e-6
    if row['case']=='failure' and row['mode'] in ['captured_wide','inverse64_r32','captured_r64','full64']:
        assert all(s['reason']=='rho_stop' for s in row['solves'])
archive=r/'downloads/mas_replay_v38_results.tar.gz'
assert sha(archive)=='97b18b489be05e11d7786352388683585a726cb8606872e21b5008c3346008bc'
out={'groups':rows,'total_solves':total_solves,'binary_sha256':next(iter(binary_hashes)),
     'source_files_verified':len(manifest['files']),'v37_files_verified':len(old['files']),
     'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
     'cpu_check_max_residual_disagreement':max(s['cpu_gpu_residual_disagreement'] for row in analysis for s in row['solves']),
     'scope':'Frozen-system evidence only. No full simulation or performance certification.',
     'tools':{p.name:sha(p) for p in (r/'tools').glob('*replay_v38.py')},
     'analysis_sha256':sha(r/'reports/MAS_REPLAY_V38_EXTENDED_ANALYSIS.json'),
     'inverse_dump_sha256':sha(data/'reports/failure_inverse64.bin'),
     'inverse_dump_source_sha256':sha(r/'tools/dump_inverse_v38.cu'),
     'full64_cpu_reference_inverse_operator_gate_1e6':max(x['relative'] for row in analysis if row['case']=='failure' and row['mode']=='full64' for x in row['cpu_hierarchy_operator_checks'])<=1e-6}
target=r/'reports/MAS_REPLAY_V38_VERIFICATION.json';assert not target.exists()
assert out['inverse_dump_sha256']=='6a9d7e1e4d112098d1e1033c90e73a1aa54352735bfd1517eeceb1e289c21f13'
target.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
