"""Identity and exact accepted-path export coverage, including failed partial frames."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
data=ROOT/'downloads/autodl_perf_v39_20261003'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(data/'manifests/perf_v39_autodl.json')
for f in manifest['files']:
    assert sha(ROOT/f['path'])==f['sha256'],f['path']
assert sha(ROOT/'tools/run_perf_v39.py')==sha(data/'tools/run_perf_v39.py')
records=[]
for name,validator in [('wide_hang_paths','validate_path'),('wide_bunny','diagnose_first_path'),('wide_bunny_strict','diagnose_first_path')]:
    run=data/'runs/autodl'/('autodl_perf_v39_'+name)
    req=read(run/'requested.json');res=read(run/'result.json')
    assert req['source_digest']==manifest['source_digest']
    assert req['runner_sha256']==sha(data/'tools/run_perf_v39.py')
    frames=read(run/'output/stats.json')['frames']
    completed=res['recorded_frames'];accepted=0
    assert len(frames)==completed+(res['status']!='completed')
    for i,f in enumerate(frames):
        steps=[t for t in f['toi'] if 'alpha' in t]
        states=sorted((run/'trace/substeps').glob(f'safe_{i:04d}_*.bin'))
        assert len(states)==len(steps)+1,(name,i,len(states),len(steps))
        assert states[0].read_bytes()==(run/f'trace/state_{i:04d}.bin').read_bytes()
        if i<completed:
            assert states[-1].read_bytes()==(run/f'trace/state_{i+1:04d}.bin').read_bytes()
        accepted+=len(steps)
    bridges=max(0,len(frames)-1)
    ccd=read(ROOT/'reports'/f'CCD_V39_{name}.json')
    assert ccd['paths_checked']==accepted+bridges
    executable=ROOT/'builds/validator/Release'/f'{validator}.exe'
    records.append({'name':name,'completed_frames':completed,'recorded_frame_attempts':len(frames),
                    'partial_final_frame':len(frames)>completed,'accepted_segments':accepted,
                    'stationary_bridges':bridges,'checked_segments':ccd['paths_checked'],
                    'all_exported_accepted_paths_covered':True,'completed_frame_endpoints_bitwise':True,
                    'ccd_passed':ccd['passed'],'conservative_flags':ccd['conservative_collision_flags'],
                    'validator_sha256':sha(executable),'command':[str(executable),str(run/'trace'),
                        str(ROOT/'reports'/f'CCD_V39_{name}.json'),'substeps','--stable-nh1']})
target=ROOT/'reports/VERIFICATION_V39_20261003.json';assert not target.exists()
target.write_text(json.dumps({'source_files_verified':len(manifest['files']),
    'source_digest':manifest['source_digest'],'binary_identities':manifest['binaries'],
    'runner_verified':True,'paths':records},indent=2))
for row in records:print(json.dumps(row))
