import hashlib,json
from pathlib import Path
import numpy as np
from analyze_perf_v36 import matrix_from_snapshot
from analyze_perf_v37 import difference
r=Path(__file__).resolve().parents[1];d=r/'downloads/mas_replay_v38'
prefix=r/'downloads/autodl_perf_v37_20261003/runs/autodl/autodl_perf_v37_bunny_mas/mas_audit_failure'
_,matrix,b=matrix_from_snapshot(prefix)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((r/'manifests/mas_replay_v38_strict.json').read_text())
assert all(sha(r/f['path'])==f['sha256'] for f in manifest['files'])
records=[]
for run in sorted((d/'runs').glob('strict_m*')):
    req=json.loads((run/'requested.json').read_text());a=json.loads((run/'result.json').read_text())
    assert json.loads((run/'completion.json').read_text())['exit_code']==0
    assert req['manifest_sha256']==sha(r/'manifests/mas_replay_v38_strict.json')
    assert all(sha(prefix.parent/name)==h for name,h in req['inputs'].items())
    first={};rows=[]
    for s in a['solves']:
        x=np.fromfile(run/s['file'],'<f8');first.setdefault(s['tolerance'],x)
        residual=float(np.linalg.norm(b-matrix@x)/np.linalg.norm(b))
        repeat=difference(first[s['tolerance']],x)['relative']
        assert s['reason']=='rho_stop' and residual<1e-6 and repeat<1e-6
        rows.append({**s,'cpu_true_residual':residual,'repeat_relative_solution':repeat,
                     'cpu_gpu_residual_difference':abs(residual-s['true_relative_residual'])})
    records.append({'mode':a['mode'],'binary_sha256':req['binary_sha256'],'solves':rows})
assert len(records)==2 and sum(len(x['solves']) for x in records)==12
archive=r/'downloads/mas_replay_v38_followup.tar.gz'
assert sha(archive)=='3c11bfc60272c7acc171464168a952535e48086ae7898e188f2e43345dc05540'
out={'scope':'Same failed A/b/M hierarchy, tolerance-only follow-up; no simulation certification.',
     'runs':records,'total_v38_solves':138,'all_strict_true_residual_and_repeat_gates_1e6_passed':True,
     'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
     'analysis_script_sha256':sha(Path(__file__))}
cross=[]
for i in range(3,6):
    for j in range(3,6):
        x=np.fromfile(d/f'runs/strict_m4/x_{i}.bin','<f8');y=np.fromfile(d/f'runs/strict_m5/x_{j}.bin','<f8')
        cross.append(difference(x,y)['relative'])
out['cross_mode_max_solution_difference_rho_1e16']=max(cross)
p=r/'reports/MAS_REPLAY_V38_STRICT_ANALYSIS.json';assert not p.exists();p.write_text(json.dumps(out,indent=2))
for a in records:
    for tol in [1e-14,1e-16]:
        s=[x for x in a['solves'] if x['tolerance']==tol]
        print(a['mode'],tol,'max_residual',max(x['cpu_true_residual'] for x in s),'max_repeat',max(x['repeat_relative_solution'] for x in s))
