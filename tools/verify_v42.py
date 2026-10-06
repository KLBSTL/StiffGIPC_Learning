"""Freeze/provenance checks and explicit combined fixed-system gates."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
main=ROOT/'downloads/replay_v42_20261003';strict=ROOT/'downloads/replay_v42_strict_20261003';initial=ROOT/'downloads/replay_v42_initial_20261003'
manifest=read(main/'manifests/mas_replay_v42a.json')
for f in manifest['files']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
assert sha(ROOT/'sources/mas_replay_v42/native_kernels.cuh')==sha(ROOT/'sources/mas_replay_v40/native_kernels.cuh')
assert sha(ROOT/'tools/benchmark_replay_v42.py')==manifest['runner_sha256']==sha(main/'tools/benchmark_replay_v42.py')
assert sha(ROOT/'tools/strict_replay_v42.py')==sha(strict/'tools/strict_replay_v42.py')
first=read(initial/'manifests/mas_replay_v42.json')
for f in first['files']:
    path=initial/f['path'] if f['path'].startswith('sources/mas_replay_v42/') else ROOT/f['path']
    assert sha(path)==f['sha256'],f['path']
binary=(main/'reports/binary.sha256').read_text().split()[0]
commands=read(main/'reports/replay_matrix.json');extra=read(strict/'reports/strict16_matrix.json')
assert len(commands)==15 and len(extra)==5
for r in commands+extra:assert r['exit_code']==0 and r['binary_sha256']==binary
cpu=read(ROOT/'reports/REPLAY_V42_CPU.json');tight=read(ROOT/'reports/REPLAY_V42_STRICT_CPU.json')
solves=[s for r in cpu for s in r['solves']]+[s for r in tight for s in r['solves']]
assert len(solves)==70
gates=[]
for row in tight:
    control=next(r for r in cpu if r['case']==row['case'] and r['mode']==7)
    for s in row['solves']:assert s['tolerance']==1e-16 and s['iteration_limit']==int(.3*60858)
    status={k:control[k] for k in ['factor_passed','cpu_operator_passed','operator_repeat_passed','operator_bitwise']}
    status.update({k:row[k] for k in ['factor_bitwise_same_as_main','all_probes_bitwise_same_as_main','all_rho_stop','strict_residual_passed','repeat_passed']})
    gates.append({'case':row['case'],'checks':status,'passed':all(status.values())})
result={'source_files_verified':len(manifest['files']),'initial_source_files_verified':len(first['files']),
        'native_kernels_unchanged_from_v40':True,'binary_sha256':binary,'main_invocations':len(commands),'supplemental_invocations':len(extra),
        'solves':len(solves),'rho_stops':sum(s['reason']=='rho_stop' for s in solves),'breakdowns':sum(s['reason']=='breakdown' for s in solves),
        'iteration_limits':sum(s['reason']=='iteration_limit' for s in solves),
        'max_cpu_gpu_residual_difference':max(s['cpu_gpu_residual_difference'] for s in solves),
        'candidate_gates_at_rho1e16':gates,'full_simulation_verified':False,'performance_certified':False,
        'archives':{name:sha(ROOT/'downloads'/name) for name in ['replay_v42_results.tar.gz','replay_v42_strict.tar.gz','initial_v42.tar.gz']}}
target=ROOT/'reports/VERIFICATION_V42_20261003.json';assert not target.exists();target.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
