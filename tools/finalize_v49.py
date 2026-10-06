"""Freeze final verification and cleanup references without changing runs."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=ROOT/'reports/FINAL_V49_20261004.json';assert not out.exists()
manifest=read(ROOT/'manifests/perf_v49_local.json')
for f in manifest['files']:assert sha(ROOT/f['path'])==f['sha256']
for name,f in manifest['binaries'].items():assert sha(ROOT/name)==f['sha256']
old=read(ROOT/'manifests/perf_v47_local.json')
for f in old['files']:assert sha(ROOT/f['path'])==f['sha256']
for name,f in old['binaries'].items():assert sha(ROOT/name)==f['sha256']
bunny=read(ROOT/'reports/VERIFICATION_V49_BUNNY.json')['runs'][0]
full=read(ROOT/'reports/FULL_ENERGY_V49_BUNNY.json');smoke=read(ROOT/'reports/FULL_ENERGY_V49_SMOKE.json')
stage=[json.loads(s) for s in (ROOT/'runs/local/v49_bunny_graph/stage.jsonl').read_text().splitlines()]
completed=bunny['result']['recorded_frames'];partial=[s['pcg'] for s in stage if s['frame']>completed and s['stage']=='pcg_end']
result={'cleanup':read(ROOT.parent/'stiffGIPC/docs/cleanup_20261004/result.json'),
    'v47_and_v49_frozen_identity_verified':True,'source_digest':manifest['source_digest'],
    'binaries':manifest['binaries'],'smoke_comparison':read(ROOT/'reports/V49_SMOKE_COMPARISON.json'),
    'components_passed':read(ROOT/'reports/V49_components.json')['passed'],
    'full_energy_smoke_passed':smoke['passed'],'full_energy_bunny_passed':full['passed'],
    'bunny_result':bunny['result'],'flushed_pcg':bunny['flushed_stats'],
    'partial_pcg':{'completed':len(partial),'limit_hits':sum(bool(p.get('iteration_limit')) for p in partial),
        'breakdowns':sum(bool(p.get('breakdown')) for p in partial),'max_iterations':max((p['iterations'] for p in partial),default=0),
        'last_stage':stage[-1] if stage else None},'ccd':bunny['ccd']['report'],
    'full_energy_events':[{k:v for k,v in e.items() if k!='checks'} for e in full['events']],
    'full_energy_check_count':sum(len(e['checks']) for e in full['events']),
    'helper_sha256':{p.name:sha(p) for p in sorted((ROOT/'tools').glob('*v49.py'))}}
late_path=ROOT/'reports/FULL_ENERGY_V49_LATE.json'
if late_path.exists():
    late=read(late_path);verify=read(ROOT/'reports/VERIFICATION_V49_LATE.json')['runs'][0]
    late_stage=[json.loads(s) for s in (ROOT/'runs/local/v49_bunny_late/stage.jsonl').read_text().splitlines()]
    late_partial=[s['pcg'] for s in late_stage if s['frame']>verify['result']['recorded_frames'] and s['stage']=='pcg_end']
    result['late_capture']={'result':verify['result'],'flushed_pcg':verify['flushed_stats'],
        'stage_stats':verify['persisted_stage_stats'],'ccd':verify['ccd']['report'],'passed':late['passed'],
        'partial_pcg':{'completed':len(late_partial),'max_iterations':max((s['iterations'] for s in late_partial),default=0),
            'limit_hits':sum(bool(s.get('iteration_limit')) for s in late_partial),'breakdowns':sum(bool(s.get('breakdown')) for s in late_partial)},
        'events':[{k:v for k,v in e.items() if k!='checks'} for e in late['events']],
        'check_count':sum(len(e['checks']) for e in late['events'])}
reference=ROOT/'reports/REFERENCE_V49_LATE.json'
if reference.exists():result['late_reference']=read(reference)
out.write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in result.items() if k not in ['cleanup','binaries','helper_sha256','flushed_pcg','ccd']},indent=2))
