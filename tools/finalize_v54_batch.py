"""Final finite-run/identity/CCD assertions, preserving failed sphere gate."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads((ROOT/p).read_text())
sha=lambda p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
m=read('manifests/perf_v54_local.json')
for f in m['files']:assert sha(f['path'])==f['sha256']
for p,e in m['binaries'].items():assert sha(p)==e['sha256']
rows=[];paths=0
for prefix,expected in [('V54_BATCH',7),('V54_PERTURB',4)]:
    matrix=read(f'reports/{prefix}_MATRIX.json');v=read(f'reports/{prefix}_VERIFICATION.json');assert len(matrix)==len(v['runs'])==expected
    for r in v['runs']:
        c=r['ccd']['report'];assert c['passed'] and c['conservative_collision_flags']==0 and r['ccd_exit_code']==0
        assert r['result']['status']=='completed' and r['result']['finite'] and r['identity_verified']
        assert r['flushed_stats']['limit_hits']==0 and r['flushed_stats']['breakdowns']==0
        assert c['paths_checked']==r['path_coverage']['accepted_segments']+r['path_coverage']['stationary_bridges']
        paths+=c['paths_checked']
    rows+=matrix
gate=read('reports/V54_BATCH_SPHERE_GATE.json');assert not gate['passed']
assert all(c['A']['passed'] and c['b_rhs']['passed'] for c in gate['comparisons'])
cpu=read('reports/V54_BATCH_SPHERE_SYSTEM.json');assert all(s['cpu_residual']<=1e-10 for s in cpu['systems'])
summary={'source_files_verified':len(m['files']),'completed_runs':len(rows),'exported_frames':sum(r['result']['recorded_frames'] for r in rows),
    'ccd_paths':paths,'ccd_passed':True,'pcg_limits_or_breakdowns':0,'sphere_continuation_gate_passed':False,
    'sphere_first_A_b_equivalent':True,'accurate_fixed_CPU_reference_passed':True,
    'cpu_max_relative_residual':max(s['cpu_residual'] for s in cpu['systems']),
    'physical_truth_certified':False,'performance_certified':False,'solver_changed':False}
out=ROOT/'reports/V54_BATCH_FINAL_CHECK.json';assert not out.exists();out.write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
