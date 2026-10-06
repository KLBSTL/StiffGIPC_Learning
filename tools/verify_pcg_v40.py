import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
control=ROOT/'downloads/pcg_v40_20261003';replay=ROOT/'downloads/replay_v40_20261003'
manifest=read(control/'manifests/perf_v39_autodl.json')
for f in manifest['files']:assert sha(ROOT/f['path'])==f['sha256']
fixed=read(replay/'manifests/mas_replay_v40.json')
for f in fixed['files']:assert sha(ROOT/f['path'])==f['sha256']
assert sha(ROOT/'sources/mas_replay_v40/native_kernels.cuh')==sha(ROOT/'sources/mas_replay_v38/native_kernels.cuh')
commands=read(control/'reports/PCG_V40_CONTROLS.json')
for row in commands:
    run=control/'runs/autodl'/('autodl_pcg_v40_'+row['name']);req=read(run/'requested.json')
    assert req['source_digest']==manifest['source_digest']
    assert req['runner_sha256']==sha(ROOT/'tools/run_perf_v39.py')
    assert req['dt']==.01 and req['tol']==.01 and req['scene']=='bunny_cloth_bunny_l'
    assert row['recorded_frames']==len(list(__import__('csv').DictReader((run/'trace/frames.csv').open())) )
    frames=read(run/'output/stats.json')['frames']
    limits=[n['pcg'] for f in frames for n in f.get('newton',[]) if n.get('pcg',{}).get('iteration_limit')]
    assert len(limits)==len(row['limit_events'])
cpu=read(ROOT/'reports/PCG_V40_REPLAY_CPU.json');solves=[s for row in cpu for s in row['solves']]
out=ROOT/'reports/PCG_V40_VERIFICATION.json';assert not out.exists()
out.write_text(json.dumps({'simulator_source_files':len(manifest['files']),'replay_source_files':len(fixed['files']),
    'native_kernels_unchanged_from_v38':True,'controls':len(commands),'fixed_solves':len(solves),
    'base_completed_100':sum(row['name'].startswith('base') and row['recorded_frames']==100 and not row['limit_events'] for row in commands),
    'rho_stops':sum(s['reason']=='rho_stop' for s in solves),'guarded_breakdowns':sum(s['reason']=='breakdown' for s in solves),
    'cpu_gpu_residual_max_abs_difference':max(s['cpu_gpu_residual_difference'] for s in solves),
    'controls_archive_sha256':sha(ROOT/'downloads/controls_v40.tar.gz'),
    'replay_archive_sha256':sha(ROOT/'downloads/replay_v40_results.tar.gz'),
    'simulator_binary_identities':manifest['binaries'],
    'replay_binary_sha256':(replay/'reports/binary.sha256').read_text().split()[0]},indent=2))
print(out.read_text())
