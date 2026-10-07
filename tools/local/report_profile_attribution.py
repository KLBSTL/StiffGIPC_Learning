"""Read-only Nsight attribution for energy launches and whole Graph intervals.

Graph-internal operations are never assigned using the replay CPU scope.
Energy attribution requires one runtime correlation and a nested native NVTX
range. GPU intervals and CPU waits are kept separate and are not added.
"""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from analyze_profile import canonical_frames,clip,row_timing,classify
from local_identity import record
from linux_runner import require,write_new

SESSION=ROOT/'runs/report_execution_20261007';REPORT=ROOT/'reports/report_execution_20261007'

def analyze(folder):
    path=folder/'nsight.sqlite';before=record(path)
    conn=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    conn.row_factory=sqlite3.Row;conn.execute('PRAGMA query_only=ON')
    try:
        tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        def load(name):return [dict(r) for r in conn.execute(f'SELECT * FROM "{name}"')] if name in tables else []
        strings={r['id']:r['value'] for r in load('StringIds')}
        ranges=[r|dict(name=r.get('text') or strings.get(r.get('textId'),'unknown'))
                for r in load('NVTX_EVENTS') if r.get('end') and r['end']>r['start']]
        frame=canonical_frames([r for r in ranges if r['name']=='ipc.physical_frame'])
        require(len(frame)==1,'One selected completed physical frame required')
        windows=[(r['start'],r['end']) for r in frame]
        runtime=load('CUPTI_ACTIVITY_KIND_RUNTIME');correlations=defaultdict(list)
        for r in runtime:correlations[r['correlationId']].append(r)
        kernels=load('CUPTI_ACTIVITY_KIND_KERNEL');owners=defaultdict(list)
        eligible=0;unattributed=0
        for r in kernels:
            if not clip(r['start'],r['end'],windows):continue
            if r.get('graphNodeId'):continue
            eligible+=1;matches=correlations[r['correlationId']]
            if len(matches)!=1:unattributed+=1;continue
            api=matches[0]
            rs=[n for n in ranges if n['name'].startswith('ipc.energy_') and
                n.get('globalTid')==api.get('globalTid') and n['start']<=api['start']<=api['end']<=n['end']]
            if rs:
                owner=min(rs,key=lambda n:n['end']-n['start'])['name']
                owners[owner].append(r)
        graph=load('CUPTI_ACTIVITY_KIND_GRAPH_TRACE')
        memory=load('CUPTI_ACTIVITY_KIND_MEMCPY')+load('CUPTI_ACTIVITY_KIND_MEMSET')
        all_work=kernels+graph+memory
        result=dict(schema='report.profile.attribution.v1',sqlite=before,analyzer=record(Path(__file__)),
            captured_frame_cpu_ms=(windows[0][1]-windows[0][0])/1e6,
            gpu_work_union_including_whole_graph=row_timing(all_work,windows),
            whole_graph=row_timing(graph,windows),outside_graph_kernels=row_timing([r for r in kernels if not r.get('graphNodeId')],windows),
            energy_kernel_owners={name:row_timing(rows,windows) for name,rows in owners.items()},
            all_cub_kernels_upper_bound=row_timing([r for r in kernels if classify(strings.get(r['demangledName'],''))=='reduce.cub_unspecified_operand'],windows),
            spmv_kernels=row_timing([r for r in kernels if classify(strings.get(r['demangledName'],''))=='spmv'],windows),
            outside_graph_kernel_count=eligible,outside_graph_without_unique_runtime_count=unattributed,
            graph_internal_operator_attribution_available='CUPTI_ACTIVITY_KIND_GRAPH_TRACE' not in tables,
            performance_certified=False,
            scope='Captured diagnostic frame only. GPU interval union includes whole Graph trace and clips boundaries. Energy ownership requires unique runtime correlation; Graph nodes remain excluded. CPU/GPU/waits are not additive; no whole-scene speedup inferred.')
    finally:conn.close()
    require(record(path)==before,'SQLite modified during analysis')
    return result

def main():
    outputs={}
    for key,folder in (('sphere','profile_sphere_f49_node'),('fixed','profile_fixed_f65_graph')):
        outputs[key]=analyze(SESSION/folder)
    write_new(REPORT/'PROFILE_ATTRIBUTION.json',outputs)
    print(json.dumps({k:{'cpu_ms':v['captured_frame_cpu_ms'],
        'gpu_union_ms':v['gpu_work_union_including_whole_graph']['union_ms'],
        'graph_ms':v['whole_graph']['union_ms'],'energy_reduce_ms':v['energy_kernel_owners'].get('ipc.energy_batch_reduce'),
        'spmv_ms':v['spmv_kernels']['union_ms'],
        'graph_internal_attribution':v['graph_internal_operator_attribution_available']} for k,v in outputs.items()}))

if __name__=='__main__':main()
