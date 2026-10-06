"""Offline, read-only accounting of one selected CPU/NVTX-only physical frame.

Adapted from the recorded donor analyzers below. No simulator, CUDA calls,
SQLite writes, solver imports, or performance certification. GPU symbol work,
GPU covered time, CPU envelopes and CPU waits remain separate measurements.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

DONORS = {
    'stiff_toi_cudagraph_20260929/tools/active/contact_linear_profile.py':
        'f70bd5af57a4ed9e6bcbd9e387de66adf4ad28d42e48c372067d562d146f9376',
    'stiff_toi_cudagraph_20260929/tools/active/analyze_ipc_light_cost.py':
        '651774f6989a96ffd37023af27e35b86b0e1294355e6e78eadf0646afe94428a',
}
APP_PREFIXES = ('ipc.', 'collision.', 'linear.', 'mas.', 'pcg.', 'graph.', 'abd.',
                'preconditioner.', 'global_preconditioner.', 'local_preconditioner.', 'diagnostic.', 'transfer.')
TABLES = {'kernel': 'CUPTI_ACTIVITY_KIND_KERNEL', 'memcpy': 'CUPTI_ACTIVITY_KIND_MEMCPY',
          'memset': 'CUPTI_ACTIVITY_KIND_MEMSET'}
RULES = [
    ('mas.numeric_scatter', ('prepare_hessian_bcoo_kernel',)),
    ('mas.numeric_reduce', ('prepare_hessian_bcoo_sum_kernel',)),
    ('mas.inverse_prepare', ('__inverse6_P96x96', '__inverse6_P96x96_typed', 'invert_cholesky_factor')),
    ('mas.factor_prepare', ('symmetric_cholesky',)),
    ('mas.restrict', ('__buildMultiLevelR_optimized_new', '__buildMultiLevelR_optimized_new_wide',
                      'deterministic_restrict', 'warp_deterministic_restrict')),
    ('mas.local_action', ('_schwarzLocalXSym6', '_schwarzLocalXSym6_wide', 'cholesky_action', 'factor_inverse_action')),
    ('mas.prolong', ('__collectFinalZ_new', '__collectFinalZ_new_wide')),
    ('mas.prolong_fused_dot', ('mas_collect_final_z_dot',)),
    ('abd.local_action', ('abd_preconditioner_apply_kernel',)),
    ('spmv', ('warp_reduce_sym_spmv_kernel', 'scale_y_kernel', 'fill_y_zero_kernel')),
    ('pcg.dot_partial', ('PCG_vdv_Reduction',)),
    ('pcg.vector_update', ('graph_dx_r', 'update_vector_dx_r', 'update_vector_c')),
    ('pcg.vector_and_condition', ('graph_p_continue',)),
    ('pcg.control_and_validation', ('graph_alpha', 'graph_beta', 'graph_init', 'graph_check_zero_rho', 'audit_residual')),
    ('matrix.convert', ('compute_hash_and_index_kernel', 'set_dst_val_kernel', 'set_row_col_from_unique_key_kernel',
                        'compute_sorted_partition_kernel', 'set_row_col_from_partition_kernel',
                        'setup_ge2sym_kernel', 'finalize_ge2sym_kernel', '_set_hash_value')),
    ('collision.discrete_query', ('_selfQuery_vf', '_selfQuery_ee', '_GroundCollisionDetect')),
    ('collision.swept_query', ('_selfQuery_vf_ccd', '_selfQuery_ee_ccd', '_selfQuery_vf_ccd_pool', '_selfQuery_ee_ccd_pool')),
    ('collision.bvh', ('fill_bvh_indices_kernel', '_calcLeafBvs', '_calcLeafBvs_ccd', '_calcMChash',
                        '_calcLeafNodes', '_calcInternalNodes', '_calcInternalAABB', '_sortBvs')),
    ('contact.gradient_hessian', ('_calBarrierHessian', '_calBarrierGradientAndHessian', '_calBarrierGradient',
                                   '_computeGroundGradientAndHessian', '_computeGroundGradient')),
]
COMPILED = [(group, [re.compile(r'(?<![A-Za-z0-9_])' + re.escape(fn) + r'(?=[(<])')
                     for fn in names]) for group, names in RULES]


def require(ok, message):
    if not ok: raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start: continue
        if result and start <= result[-1][1]: result[-1] = (result[-1][0], max(end, result[-1][1]))
        else: result.append((start, end))
    return result


def clip(start, end, windows):
    # Normalize even caller-supplied nested windows before clipping each row.
    return [(max(start, a), min(end, b)) for a, b in merge(windows) if start < b and end > a]


def timing(intervals, count=None):
    intervals = list(intervals); union = merge(intervals)
    summed = sum(b-a for a,b in intervals); covered = sum(b-a for a,b in union)
    span = union[-1][1]-union[0][0] if union else 0
    return {'count': len(intervals) if count is None else count, 'sum_ms': summed/1e6, 'union_ms': covered/1e6,
            'first_start_ns': union[0][0] if union else None, 'last_end_ns': union[-1][1] if union else None,
            'first_to_last_span_ms': span/1e6, 'gaps_between_activities_ms': (span-covered)/1e6}


def row_timing(rows, windows):
    rows = [r for r in rows if clip(r['start'], r['end'], windows)]
    return timing((part for row in rows for part in clip(row['start'], row['end'], windows)), len(rows))


def intersection(a, b):
    a,b = merge(a),merge(b); i=j=0; result=[]
    while i<len(a) and j<len(b):
        low,high=max(a[i][0],b[j][0]),min(a[i][1],b[j][1])
        if high>low: result.append((low,high))
        if a[i][1]<=b[j][1]: i+=1
        else: j+=1
    return result


def classify(name):
    hits=[group for group,patterns in COMPILED if any(p.search(name) for p in patterns)]
    require(len(hits)<=1, 'Ambiguous symbol classification: '+name)
    if hits: return hits[0]
    if 'cub::' in name and re.search(r'(?<![A-Za-z0-9_])DeviceReduce(?:SingleTile|Kernel)',name):
        return 'reduce.cub_unspecified_operand'
    return 'unknown_kernel'


def canonical_frames(rows):
    """Collapse duplicate/nested same-thread physical ranges; never add them."""
    result=[]
    for row in sorted(rows,key=lambda r:(r['start'],-r['end'])):
        if any(other.get('globalTid')==row.get('globalTid') and other['start']<=row['start']
               and row['end']<=other['end'] for other in result): continue
        require(not any(clip(row['start'],row['end'],[(p['start'],p['end'])]) for p in result),
                'Ambiguous overlapping physical-frame ranges')
        result.append(row)
    return result


def summarize(rows, windows):
    grouped=defaultdict(list)
    for row in rows: grouped[row['category']].append(row)
    all_time=row_timing(rows,windows)
    categories={name:row_timing(members,windows) for name,members in sorted(grouped.items())}
    require(abs(sum(r['sum_ms'] for r in categories.values())-all_time['sum_ms'])<1e-6,
            'GPU category partition is inconsistent')
    return {'all_gpu_activity':all_time,'categories':categories,
            'cross_category_union_overlap_ms':max(0,sum(r['union_ms'] for r in categories.values())-all_time['union_ms']),
            'sum_minus_union_ms':max(0,all_time['sum_ms']-all_time['union_ms'])}


def analyze(sqlite_path, cost_path, frame=57):
    require(sqlite_path.is_file() and cost_path.is_file(), 'SQLite and cost JSONL must exist')
    sources={str(p.resolve()):file_hash(p) for p in (sqlite_path,cost_path)}
    records=[json.loads(line) for line in cost_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    require(records and all(r.get('frame')==frame for r in records), 'Cost JSONL must contain only the selected physical frame')
    require(all(r.get('gpu_events_enabled') is False and r.get('measurement_mode')=='nvtx_cpu_only'
                and r.get('nvtx_available') is True and r.get('sample_kind')=='production' for r in records),
            'Expected native production CPU/NVTX-only records without GPU event timings')
    cost_frames=[r for r in records if r.get('stage')=='ipc.physical_frame']
    require(len(cost_frames)==1 and cost_frames[0].get('parent_scope_id')==0,
            'Exactly one outermost physical-frame cost record is required')
    connection=sqlite3.connect(sqlite_path.resolve().as_uri()+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row; connection.execute('PRAGMA query_only=ON')
    try:
        tables={r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        def load(table):
            return [dict(r) for r in connection.execute(f'SELECT * FROM "{table}"')] if table in tables else []
        strings={r['id']:r['value'] for r in load('StringIds')}
        def name(row,fields):
            for field in fields:
                value=row.get(field)
                if value is not None: return strings.get(value,str(value)) if isinstance(value,int) else str(value)
            return '<unnamed>'
        ranges=[]
        for row in load('NVTX_EVENTS'):
            if row.get('end') is not None and row['end']>row['start']:
                ranges.append(row|{'name':name(row,('text','textId'))})
        raw_frames=[r for r in ranges if r['name']=='ipc.physical_frame']; frames=canonical_frames(raw_frames)
        require(len(frames)==1, 'Exactly one completed ipc.physical_frame NVTX range is required; capture may be truncated')
        windows=[(frames[0]['start'],frames[0]['end'])]
        app=[r for r in ranges if r['name'].startswith(APP_PREFIXES) and clip(r['start'],r['end'],windows)]
        runtime=[]; correlations=defaultdict(list)
        for row in load('CUPTI_ACTIVITY_KIND_RUNTIME'):
            if row.get('end') is None or row['end']<row['start']: continue
            row=row|{'name':name(row,('name','nameId'))}; runtime.append(row)
            if row.get('correlationId') is not None: correlations[row['correlationId']].append(row)
        activities=[]; outside=0
        for kind,table in TABLES.items():
            for row in load(table):
                if row.get('end') is None or row['end']<=row['start']: continue
                if not clip(row['start'],row['end'],windows): outside+=1; continue
                symbol=name(row,('demangledName','shortName','name','nameId')) if kind=='kernel' else kind
                matches=correlations.get(row.get('correlationId'),[])
                api=matches[0] if len(matches)==1 else None
                owners=[r for r in app if api is not None and r.get('globalTid')==api.get('globalTid')
                        and r['start']<=api['start'] and api['end']<=r['end']]
                owner=min(owners,key=lambda r:r['end']-r['start'])['name'] if owners else None
                activities.append(row|{'name':symbol,'kind':kind,'category':classify(symbol) if kind=='kernel' else 'memory.'+kind,
                    'runtime_match':'unique' if api is not None else ('ambiguous' if matches else 'missing'),
                    'runtime_launch_owner':owner})
    finally: connection.close()
    require('CUPTI_ACTIVITY_KIND_KERNEL' in tables and any(r['kind']=='kernel' for r in activities),
            'No actual selected-frame kernel activities; node tracing is required for kernel cost attribution')
    grouped=defaultdict(list)
    for row in activities:
        if row['kind']=='kernel': grouped[row['name']].append(row)
    symbols=[{'symbol':symbol,'category':rows[0]['category'],**row_timing(rows,windows)} for symbol,rows in grouped.items()]
    symbols.sort(key=lambda r:(-r['sum_ms'],r['symbol']))
    stage_groups=defaultdict(list)
    for row in app: stage_groups[row['name']].append(row)
    cpu_stages=[{'stage':stage,'instances':len(rows),'cpu_union_ms':row_timing(rows,windows)['union_ms'],
                 'inclusive_not_additive':True} for stage,rows in sorted(stage_groups.items())]
    selected_api=[r for r in runtime if clip(r['start'],r['end'],windows)]
    waits=[r for r in selected_api if r['name'].startswith(('cudaDeviceSynchronize','cudaStreamSynchronize','cudaEventSynchronize'))]
    memcpy_api=[r for r in selected_api if r['name'].startswith('cudaMemcpy')]
    graph=[r for r in activities if r.get('graphNodeId')]
    graph_timing=row_timing(graph,windows)
    graph_span=[(graph_timing['first_start_ns'],graph_timing['last_end_ns'])] if graph else []
    gpu_intervals=[part for r in activities for part in clip(r['start'],r['end'],windows)]
    busy_in_graph_span=timing(intersection(gpu_intervals,graph_span))['union_ms']
    result={'schema':'local.profile.v1','read_only':True,'selected_physical_frame':frame,
            'performance_certified':False,'sources_sha256':sources,'analyzer_sha256':file_hash(Path(__file__)),
            'donor_sha256':DONORS,'raw_physical_nvtx_ranges':len(raw_frames),'canonical_physical_nvtx_ranges':len(frames),
            'frame_identity_basis':'One native outermost cost record for the selected frame and one canonical completed NVTX physical-frame range. NVTX itself has no numeric frame ID; retain launcher configuration evidence.',
            'missing_activity_tables':[table for table in TABLES.values() if table not in tables],
            'cpu':{'physical_frame_nvtx_wall_ms':(windows[0][1]-windows[0][0])/1e6,
                   'native_physical_frame_cpu_submit_ms':cost_frames[0].get('cpu_submit_ms'),
                   'native_gpu_events_enabled':False,'cost_record_count':len(records),'nvtx_envelopes':cpu_stages,
                   'cuda_runtime_table_available':'CUPTI_ACTIVITY_KIND_RUNTIME' in tables,
                   'runtime_api_union_ms':row_timing(selected_api,windows)['union_ms'] if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables else None,
                   'explicit_wait_api_union_ms':row_timing(waits,windows)['union_ms'] if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables else None,
                   'memcpy_api_union_including_possible_wait_ms':row_timing(memcpy_api,windows)['union_ms'] if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables else None},
            'gpu':summarize(activities,windows),'kernel_symbols':symbols,
            'unknown_kernel':row_timing([r for r in activities if r['category']=='unknown_kernel'],windows),
            'graph':{'graph_node_field_available':any('graphNodeId' in r for r in activities),
                'node_activity':graph_timing,
                'node_without_unique_runtime':row_timing([r for r in graph if r['runtime_match']!='unique'],windows),
                'node_without_runtime_launch_owner':row_timing([r for r in graph if r['runtime_launch_owner'] is None],windows),
                'node_internal_nvtx_operator_unattributed':graph_timing,
                'node_unknown_kernel_symbol':row_timing([r for r in graph if r['category']=='unknown_kernel'],windows),
                'all_gpu_busy_in_first_last_node_span_ms':busy_in_graph_span,
                'node_span_uncovered_by_captured_gpu_ms':max(0,graph_timing['first_to_last_span_ms']-busy_in_graph_span)},
            'boundary_crossing_activity_records':sum((r['start'],r['end'])!=clip(r['start'],r['end'],windows)[0] for r in activities),
            'activity_records_outside_selected_frame':outside,
            'classification_rules':{k:list(v) for k,v in RULES},
            'scope':{'gpu':'Kernel-symbol allowlist classification partitions real clipped activity records. Unknown kernels and unspecified CUB operands remain visible. Category sums count overlapping work; union is covered wall time.',
                     'cpu':'NVTX stage unions are inclusive envelopes, not additive component costs. Parent and child envelopes are never summed into a total.',
                     'waits':'CUDA API intervals include GPU waits and overlap device work. Never add CPU API or NVTX wall time to GPU time.',
                     'graph':'Kernel symbols identify operations independently of NVTX. A replay launch owner never identifies internal graph-node CostScopes; internal NVTX attribution remains unresolved even with a runtime correlation.',
                     'gaps':'Uncovered spans include scheduling, dependencies and unobserved activity; they are not proven removable host overhead or predicted speedup.',
                     'capture':'Only the selected CPU/NVTX frame timeline window is measured. Asynchronous work can cross its boundaries; clipping is not causal ownership of every GPU operation by that frame. Full-scene cost, occupancy, dropped records and capture completeness are not inferred.'}}
    require(all(file_hash(Path(path))==digest for path,digest in sources.items()), 'Profile inputs changed during analysis')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sqlite',type=Path,required=True);parser.add_argument('--cost-jsonl',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--frame',type=int,default=57)
    args=parser.parse_args();require(not args.output.exists(),'Output already exists; preserve earlier analysis')
    result=analyze(args.sqlite,args.cost_jsonl,args.frame);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'output':str(args.output.resolve()),'frame':args.frame,
        'cpu_nvtx_wall_ms':result['cpu']['physical_frame_nvtx_wall_ms'],
        'gpu_sum_ms':result['gpu']['all_gpu_activity']['sum_ms'],'gpu_union_ms':result['gpu']['all_gpu_activity']['union_ms'],
        'unknown_kernel_union_ms':result['unknown_kernel']['union_ms'],
        'graph_internal_stage_unattributed_union_ms':result['graph']['node_internal_nvtx_operator_unattributed']['union_ms'],
        'performance_certified':False}))


if __name__=='__main__':main()
