"""Final evidence inventory; expected scientific gate failures remain explicit."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
data=ROOT/'downloads/autodl_perf_v36_20261003'
target=ROOT/'reports/AUTODL_PERF_V36_FINAL_VERIFICATION.json';assert not target.exists()
frozen={}
for name in ['perf_v34.json','perf_v34_cuda128.json','perf_v36.json']:
    manifest=read(ROOT/'manifests'/name)
    mismatches=[f['path'] for f in manifest['files'] if hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()!=f['sha256']]
    assert not mismatches
    frozen[name]={'files':len(manifest['files']),'mismatches':mismatches}
results=[read(p) for p in (data/'runs/autodl').glob('*/result.json')]
paths=[read(ROOT/'reports'/f'AUTODL_PERF_V36_{name}_PATH_VALIDATION.json') for name in ['SPHERE','HANG','STRICT']]
assert all(p['passed'] and p['frame_bridges_bitwise'] for p in paths)
eq=[read(ROOT/'reports'/f'AUTODL_PERF_V36_GRAPH_EQ_R{i}.json') for i in range(1,4)]
extra=read(data/'reports/AUTODL_PERF_V36_EXTRA_PRECISION.json')['runs'][-1]
fixture=read(data/'builds/fixture.json');assert fixture['passed']
archives={}
for name,expected in [('autodl_perf_v36_20261003_results.tar.gz','8d5a6cdb14d3b0a636dfea69d7471803b9341393564665331dccd7e2237c1f14'),
                      ('extra_precision_v36.tar.gz','11a44175ba9417a29371dcf96dad6f2b08ad25c4da42d6d83f674ddaa3094f9d'),
                      ('strict_compare_v36.tar.gz','9047f6b55a7f973238bd6bfc52b2b93b079df0b553961b464e0320be51efe984')]:
    p=ROOT/'downloads'/name;value=hashlib.sha256(p.read_bytes()).hexdigest();assert value==expected
    archives[name]={'bytes':p.stat().st_size,'sha256':value}
out={'gpu_runs':len(results),'completed_runs':sum(r['status']=='completed' for r in results),
     'preserved_failed_runs':sum(r['status']!='completed' for r in results),'frozen_sources':frozen,
     'fixtures':fixture,'ctest':'1/1 passed; downloaded builds/ctest_v36.log',
     'fixed_systems':13,'fixed_system_measured_solves':408,
     'same_system_1e16_100_frame_audit':extra['maxima'],
     'same_system_graph_host_1e6_passed_in_audited_run':extra['maxima']['same_system_relative_solution_difference']<=1e-6,
     'whole_trajectory_graph_pairs_passed':sum(r['passed'] for r in eq),'whole_trajectory_graph_pairs_total':3,
     'independent_ccd_exported_runs':3,'independent_ccd_segments':sum(p['validator_segments'] for p in paths),
     'accepted_advancement_segments':sum(p['accepted_segments'] for p in paths),
     'stationary_frame_bridges':sum(p['stationary_frame_bridges'] for p in paths),
     'conservative_flags':sum(p['ccd']['conservative_collision_flags'] for p in paths),
     'all_formal_timing_paths_independently_ccd_checked':False,
     'quality_matched_speedup_certified':False,'bunny_100_frames_completed':False,
     'physical_reference_convergence_accepted':False,'production_defaults_changed':False,
     'archives':archives,'remote_final_state':{'active_compute_processes':0,'gpu_utilization_percent':0,'free_vram_mib':24081,'free_disk_approx_gib':4.3}}
target.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
