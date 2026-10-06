"""Verify the same-precision timing comparison and physical-time-aligned references."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from report_perf_v36 import aligned_quality, read
from report_local_perf_v33 import work
from report_v32_autodl import quality

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();data=a.data.resolve()
for entry in read(data/'reports/AUTODL_PERF_V36_STRICT_FILES.json'):
    assert hashlib.sha256((data/entry['path']).read_bytes()).hexdigest()==entry['sha256']
matrix=read(data/'reports/AUTODL_PERF_V36_STRICT_COMPARE.json')['runs']
manifest=read(data/'manifests/perf_v36_autodl.json')
for row in matrix:
    run=data/row['run'];req,res=read(run/'requested.json'),read(run/'result.json')
    assert res['status']=='completed' and res['recorded_frames']==req['steps'] and res['finite']
    assert req['source_digest']==manifest['source_digest']
    assert req['runner_sha256']==hashlib.sha256((data/'tools/run_perf_v36.py').read_bytes()).hexdigest()
    assert req['exe_sha256'] in [v['sha256'] for v in manifest['binaries'].values()]
    frames,counts=work(run)
    assert not any(counts[k] for k in ['pcg_limit_hits','outer_limit_hits','unsafe_native_steps','unfinished_outer'])
    assert abs(sum(float(t['solver_ms']) for t in csv.DictReader((run/'trace/frames.csv').open()))/1000-res['solver_seconds'])<1e-8
    if row['formal_timing']:
        assert req['pcg_tol']==1e-16 and req['dt']==.01 and req['tol']==.01 and req['robust_velocity_tol']==.05
        assert req['preconditioner']=='diag' and req['suite']=='1' and req['fused_diag_update']=='0'
        assert not res['timing_is_diagnostic']
        if req['execution']=='conditional_graph':assert counts['execution']=={'conditional_graph':counts['directions']}
    row['work']=counts
keyed={r['label']:r for r in matrix}
run=lambda label:data/keyed[label]['run']
pairs=[]
for repeat in range(1,4):
    host,graph=f'base_toi_r{repeat}',f'base_toi_graph_r{repeat}'
    q=quality(run(graph),run(host))
    cloth_normalized=q['max_cloth_mass_rms_vs_base_percent']/100
    pairs.append({'repeat':repeat,'host_seconds':keyed[host]['solver_seconds'],'graph_seconds':keyed[graph]['solver_seconds'],
                  'speed_ratio':keyed[host]['solver_seconds']/keyed[graph]['solver_seconds'],
                  'cloth_normalized_max_rms':cloth_normalized,'cloth_gate_1e6':cloth_normalized<=1e-6,
                  'quality':q})
host_med=statistics.median(r['host_seconds'] for r in pairs);graph_med=statistics.median(r['graph_seconds'] for r in pairs)
ref=run('base_dt8')
refs={f'dt{i}_to_dt{j}':aligned_quality(run(f'base_dt{i}'),run(f'base_dt{j}')) for i,j in [(1,2),(2,4),(4,8)]}
refs['dt4_repeat']=aligned_quality(run('base_dt4'),run('base_dt4_repeat'))
out={'scope':'rho=1e-16 TOI host/Graph, same model and parameters. This is not TOI-versus-official-base quality certification.',
     'runs':matrix,'pairs':pairs,'host_median_seconds':host_med,'graph_median_seconds':graph_med,
     'median_time_ratio':host_med/graph_med,'all_pairs_cloth_gate_passed':all(r['cloth_gate_1e6'] for r in pairs),
     'graph_repeat_quality':[aligned_quality(run(f'base_toi_graph_r{i}'),run('base_toi_graph_r1')) for i in range(1,4)],
     'host_repeat_quality':[aligned_quality(run(f'base_toi_r{i}'),run('base_toi_r1')) for i in range(1,4)],
     'strict_reference_convergence':refs,
     'graph_vs_strict_base_dt8':aligned_quality(run('base_toi_graph_r1'),ref)}
a.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['runs','pairs']},indent=2))
