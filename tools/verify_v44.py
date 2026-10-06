"""Verify downloaded identities and trace/diagnostic contracts without claiming trajectory equivalence."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];data=ROOT/'downloads/autodl_perf_v44_20261004'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read(data/'manifests/perf_v44_autodl.json');identity=read(data/'reports/FINAL_IDENTITY_V44.json')
for x in m['files']:assert sha(ROOT/x['path'])==x['sha256'],x['path']
for x in identity['files']:assert sha(data/x['path'])==x['sha256'],x['path']
for p,x in m['binaries'].items():assert sha(data/p)==x['sha256'],p
runs=[]
for run in sorted((data/'runs/autodl').glob('autodl_perf_v44_*')):
    req=read(run/'requested.json');res=read(run/'result.json')
    assert req['source_digest']==m['source_digest']
    assert req['runner_sha256']==sha(data/'tools/run_perf_v44.py')
    assert req['exe_sha256']==m['binaries']['builds/autodl-stiff_perf_v44/gipc']['sha256']
    assert req['dt']==.01 and req['tol']==.01 and req['pcg_tol']==1e-4 and req['mas_cholesky']=='1'
    stats=read(run/'output/stats.json');frames=stats['frames']
    assert len(frames)==res['recorded_frames']
    assert req['stage_from_frame'] or not (run/'stage.jsonl').exists()
    runs.append({'run':run.name,'status':res['status'],'completed_frames':len(frames),'stage_from_frame':req['stage_from_frame']})
repeat=read(data/'reports/SMOKE_REPEAT_V44.json')
assert repeat['original_bitwise_gate_passed'] is False
report={'sources_verified':len(m['files']),'downloaded_files_verified':len(identity['files']),
    'source_digest':m['source_digest'],'binary':m['binaries']['builds/autodl-stiff_perf_v44/gipc'],
    'runs':runs,'original_bitwise_gate_passed':False,'performance_certified':False,'full_path_ccd_verified':False}
(ROOT/'reports/VERIFICATION_V44_20261004.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
