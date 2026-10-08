"""Read-only CPU interpretation of stage-1 structure and BVH replay evidence.

Writes only this report's new analysis outputs. Missing or failed native runs
remain missing/failed; probe timing is never promoted to production speed.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
SESSION=ROOT/'runs/report56_stage1_20261008'
SCENES=('cloth_sphere7_l','cloth_fixed_bunny_l')
STAGES=('raw_compare','raw_snapshot','converted_compare','converted_snapshot','pair_compare','pair_snapshot')
STATES={'disabled','first','owner_changed','shape_changed','content_changed','hit'}
SHAPE=('owner','input_offset','item_count','output_offset','block_rows','block_cols','device')
CONTEXT=('frame','outer','inner','contact_model_id','linear_system_id')
DISTRIBUTIONS=('node_aabb_tests','internal_pops','leaf_overlaps','narrow_calls','max_pending_stack')

def require(value,message):
    if not value:raise ValueError(message)

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def record(path):
    raw=path.read_bytes()
    return {'path':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def finite_tree(value,path='record'):
    if isinstance(value,float):require(math.isfinite(value),'Nonfinite field: '+path)
    elif isinstance(value,dict):
        for key,item in value.items():finite_tree(item,path+'.'+key)
    elif isinstance(value,list):
        for index,item in enumerate(value):finite_tree(item,f'{path}[{index}]')

def number(value,label):
    require(type(value) in (int,float) and math.isfinite(value) and value>=0,'Invalid nonnegative field: '+label)

def integer(value,label):require(type(value) is int and value>=0,'Invalid integer field: '+label)
def boolean(value,label):require(type(value) is bool,'Invalid boolean field: '+label)
def ratio(n,d):return n/d if d else None

def closed_index(folder):
    entries=read(folder/'evidence.json')['files'];index={r['path']:r for r in entries}
    require(len(entries)==len(index),'Duplicate closed evidence paths')
    for name,row in index.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe evidence path')
        integer(row['bytes'],'evidence bytes')
        require(type(row['sha256']) is str and len(row['sha256'])==64,'Invalid evidence SHA256')
    return index

def closed_bytes(folder,index,name):
    raw=(folder/name).read_bytes();expected=index[name]
    require(len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'],
            'Closed evidence changed: '+str(folder/name))
    return raw

def closed_json(folder,index,name):return json.loads(closed_bytes(folder,index,name).decode('utf-8-sig'))

def vector_difference(a,b,groups):
    require(len(a)==len(b)==24*sum(len(ids) for ids in groups.values()),'Wrong double3 trajectory byte count')
    first=list(struct.iter_unpack('<ddd',a));second=list(struct.iter_unpack('<ddd',b));result={}
    require(all(math.isfinite(x) for v in first+second for x in v),'Nonfinite actual position/velocity export')
    for name,ids in groups.items():
        squared=[];maximum=component_max=0.;changed=0
        for index in ids:
            delta=tuple(x-y for x,y in zip(first[index],second[index]));length=math.hypot(*delta)
            squared.append(length*length);maximum=max(maximum,length)
            component_max=max(component_max,*(abs(v) for v in delta));changed+=any(v!=0 for v in delta)
        result[name]={'vertices':len(ids),'rms_vector_difference':math.sqrt(math.fsum(squared)/len(ids)),
            'maximum_vector_difference':maximum,'maximum_absolute_component_difference':component_max,
            'numerically_different_vertices':changed}
    return result

def trajectory_difference(first,second,index_a,index_b):
    static=('trace/topology.bin','trace/body_ids.bin','trace/boundary_types.bin','trace/masses.bin','trace/metadata.json')
    equality={};buffers={}
    for name in static:
        a=closed_bytes(first,index_a,name);b=closed_bytes(second,index_b,name)
        equality[name]=a==b;buffers[name]=a
    require(all(equality.values()),'Trajectory body/geometry/physical input layouts differ')
    n,faces,tets=struct.unpack('<III',buffers['trace/topology.bin'][:12])
    require(len(buffers['trace/topology.bin'])==12+12*faces+16*tets,'Wrong topology byte count')
    ids=[v[0] for v in struct.iter_unpack('<i',buffers['trace/body_ids.bin'])]
    require(n==len(ids) and len(buffers['trace/boundary_types.bin'])==4*n and len(buffers['trace/masses.bin'])==8*n,
            'Wrong physical attribute count')
    scene=closed_json(first,index_a,'output/scene.json');other=closed_json(second,index_b,'output/scene.json')
    require(scene['case_id']==other['case_id'] and scene['objects']==other['objects'],
            'Trajectory scene/body definitions differ')
    abd=[o for o in scene['objects'] if o['body_type']=='ABD'];fem=[o for o in scene['objects'] if o['body_type']=='FEM']
    require(len(abd)==len(fem)==1 and fem[0]['dimension']==2 and set(ids)=={-1,0},'Unsupported actual body layout')
    metadata=json.loads(buffers['trace/metadata.json'])
    require(sum(v==0 for v in ids)==metadata['abd_point_num'],'ABD metadata/body IDs disagree')
    groups={'ABD:'+abd[0]['name']:[i for i,v in enumerate(ids) if v==0],
            'FEM_2D:'+fem[0]['name']:[i for i,v in enumerate(ids) if v==-1]}
    result={};verified=0
    for kind,unit in (('state','scene_length_unit'),('velocity','scene_length_unit_per_second')):
        missing=[];different=[];frames=[]
        for frame in range(121):
            name=f'trace/{kind}_{frame:04d}.bin'
            if name not in index_a or name not in index_b or not (first/name).is_file() or not (second/name).is_file():
                missing.append(frame);continue
            a=closed_bytes(first,index_a,name);b=closed_bytes(second,index_b,name);verified+=2
            if a!=b:different.append(frame)
            frames.append({'frame':frame,'bitwise_equal':a==b,'by_body':vector_difference(a,b,groups)})
        totals={}
        for name,group in groups.items():
            samples=[r['by_body'][name] for r in frames]
            totals[name]={'vertices':len(group),'rms_vector_difference_over_recorded_frames':
                math.sqrt(math.fsum(r['rms_vector_difference']**2 for r in samples)/len(samples)) if samples else None,
                'maximum_vector_difference':max((r['maximum_vector_difference'] for r in samples),default=None),
                'maximum_frame_rms':max((r['rms_vector_difference'] for r in samples),default=None),
                'maximum_difference_frame':max(frames,key=lambda r:r['by_body'][name]['maximum_vector_difference'])['frame'] if frames else None,
                'frame_120':next((r['by_body'][name] for r in frames if r['frame']==120),None)}
        result[kind]={'units':unit,'missing_frames':missing,'different_frames':different,
            'all_121_exports_bitwise_equal':not missing and not different,'per_frame':frames,'body_totals':totals}
    return {'initial_static_attributes_byte_equal':equality,'body_membership_source':'actual body_ids.bin; ABD count cross-checked against metadata.json',
        'verified_trajectory_files':verified,'comparisons':result,
        'rms_definition':'sqrt(sum of squared 3D vertex differences / body vertex count); all-frame RMS weights all recorded frames equally. Maximum is the largest 3D vertex difference. These are differences, not error against physical ground truth.'}

def shape(row,kind):
    keys=SHAPE if kind!='pair' else ('owner','unique_count','block_rows','block_cols','device')
    result=tuple(row[key] for key in keys)
    return result+(row['unique_count'],) if kind=='converted' else result

def validate_linear(row,frame,pcg,previous):
    finite_tree(row)
    require(row['observed'] is True and row['frame']==frame and row['linear_system_id']>0,
            'Missing/wrong linear-system observation identity')
    for key in CONTEXT:
        require(key in row,'Missing linear context: '+key)
        if key in pcg:require(row[key]==pcg[key],'Probe/PCG context mismatch: '+key)
    for key in (*SHAPE,'unique_count','row_storage','col_storage','observation_index',
                'raw_snapshot_bytes','converted_snapshot_bytes','pair_snapshot_bytes',
                'raw_capacity_bytes','converted_capacity_bytes','pair_capacity_bytes'):
        integer(row[key],key)
    require(row['owner']>0 and row['unique_count']<=row['item_count'],'Invalid structure owner/count')
    require(not row['item_count'] or (row['row_storage']>0 and row['col_storage']>0),'Missing live raw storage')
    for key in ('gap_before','raw_equal','raw_comparison_performed','raw_snapshot_updated',
                'converted_observed','converted_comparison_performed','mapping_equal','partition_equal',
                'converted_snapshot_updated','pair_observed','pair_equal','pair_comparison_performed',
                'pair_snapshot_updated','pair_gap_before'):
        boolean(row[key],key)
    for kind,state_key,flag,gap in (('raw','raw_state','raw_comparison_performed','gap_before'),
                                  ('converted','converted_state','converted_comparison_performed','gap_before'),
                                  ('pair','pair_state','pair_comparison_performed','pair_gap_before')):
        state=row[state_key];require(state in STATES,'Unknown structure state')
        if row[flag] or state in ('hit','content_changed'):
            require(previous is not None and not row[gap] and shape(row,kind)==shape(previous,kind),
                    'Comparison/hit changed owner or shape, or crossed an invalidated gap')
        if state=='owner_changed':
            require(previous is not None and row['owner']!=previous['owner'],'Owner-change state without changed owner')
        if state=='shape_changed':
            require(previous is not None and row['owner']==previous['owner'] and shape(row,kind)!=shape(previous,kind),
                    'Shape-change state without changed same-owner shape')
        if row[flag]:require(state in ('hit','content_changed'),'Comparison lacks final content state')
    require(row['raw_equal']==(row['raw_state']=='hit'),'Raw equality/state mismatch')
    if row['converted_observed']:
        require((row['mapping_equal'] and row['partition_equal'])==(row['converted_state']=='hit'),
                'Converted joint equality/state mismatch')
    if row['pair_observed']:require(row['pair_equal']==(row['pair_state']=='hit'),'Pair equality/state mismatch')
    for prefix,count in (('raw',row['item_count']),('converted',row['item_count']),('pair',row['unique_count'])):
        if prefix=='pair' and not row['pair_observed']:continue
        require(row[prefix+'_snapshot_bytes']==8*count and
                row[prefix+'_capacity_bytes']>=row[prefix+'_snapshot_bytes'],'Invalid snapshot bytes/capacity')
    for key in ('event_setup_cpu_ms','workspace_cpu_ms','probe_cpu_ms'):number(row[key],key)
    for stage in STAGES:
        cost=row[stage]
        for key in ('cpu_ms','gpu_ms','readback_cpu_ms'):number(cost[key],stage+'.'+key)
        for key in ('gpu_valid','completion_waited'):boolean(cost[key],stage+'.'+key)
        require(cost['readback_cpu_ms']<=cost['cpu_ms']+1e-5,'Readback exceeds containing CPU stage')
        if not cost['gpu_valid']:require(cost['gpu_ms']==0,'Invalid GPU interval marked unavailable')
    if previous:require(row['observation_index']>previous['observation_index'] and
                         row['linear_system_id']>previous['linear_system_id'],'Nonmonotonic system observation')

def reuse_summary(rows):
    summary={};previous=None
    definitions=(('raw','raw_state','raw_equal','gap_before'),
                 ('mapping','converted_state','mapping_equal','gap_before'),
                 ('partition','converted_state','partition_equal','gap_before'),
                 ('converted_joint','converted_state',None,'gap_before'),
                 ('canonical_pairs','pair_state','pair_equal','pair_gap_before'))
    for name,state_key,equal_key,gap in definitions:
        observed=(rows if name=='raw' else [row for row in rows if row['pair_observed']] if
                  name=='canonical_pairs' else [row for row in rows if row['converted_observed']])
        excluded=collections.Counter();states=collections.Counter();eligible=hits=empty=compared=0
        previous=None
        for row in observed:
            state=row[state_key];states[state]+=1
            kind='pair' if name=='canonical_pairs' else 'raw' if name=='raw' else 'converted'
            if row[gap]:reason='gap_before'
            elif state=='first':reason='first'
            elif previous is None:reason='window_or_subset_boundary'
            elif state in ('owner_changed','shape_changed','disabled'):reason=state
            elif row['linear_system_id']!=previous['linear_system_id']+1 or row['observation_index']!=previous['observation_index']+1:
                reason='nonadjacent_system_observation'
            elif state not in ('hit','content_changed') or shape(row,kind)!=shape(previous,kind):reason='not_comparable'
            else:reason=None
            equal=row[equal_key] if equal_key else row['mapping_equal'] and row['partition_equal']
            if reason:excluded[reason]+=1
            else:
                eligible+=1;hits+=int(equal)
                empty+=int((row['unique_count'] if kind=='pair' else row['item_count'])==0)
            flag='pair_comparison_performed' if kind=='pair' else 'raw_comparison_performed' if kind=='raw' else 'converted_comparison_performed'
            compared+=int(row[flag]);previous=row
        summary[name]={'observations':len(observed),'states':dict(states),'adjacent_comparable':eligible,
            'adjacent_hits':hits,'adjacent_hit_rate':ratio(hits,eligible),'eligible_empty':empty,
            'explicit_comparisons':compared,'excluded':dict(excluded),
            'hit_fraction_all_observations':ratio(sum(states[s] for s in ('hit',)),len(observed)) if name not in ('mapping','partition') else
                ratio(sum(row[equal_key] for row in observed),len(observed))}
        summary[name]['state_scope']=('converted joint comparator state; individual mapping_equal or partition_equal flags determine this column\'s hits'
            if name in ('mapping','partition') else 'native '+state_key)
    summary['canonical_hit_without_joint_mapping_partition_hit']=sum(
        r['pair_observed'] and r['pair_equal'] and not(r['converted_observed'] and r['mapping_equal'] and r['partition_equal']) for r in rows)
    return summary

def linear_costs(rows):
    result={'observations':len(rows),'probe_cpu_ms':sum(r['probe_cpu_ms'] for r in rows),
        'event_setup_cpu_ms':sum(r['event_setup_cpu_ms'] for r in rows),
        'workspace_cpu_ms':sum(r['workspace_cpu_ms'] for r in rows),'six_stages':{},
        'peak_capacity_bytes':max((sum(r[k] for k in ('raw_capacity_bytes','converted_capacity_bytes','pair_capacity_bytes')) for r in rows),default=None)}
    for stage in STAGES:
        costs=[r[stage] for r in rows];valid=[c for c in costs if c['gpu_valid']]
        result['six_stages'][stage]={'cpu_ms':sum(c['cpu_ms'] for c in costs),
            'gpu_ms_valid_intervals':sum(c['gpu_ms'] for c in valid) if valid else None,
            'gpu_valid_intervals':len(valid),'completion_waited':sum(c['completion_waited'] for c in costs),
            'readback_cpu_ms_subset':sum(c['readback_cpu_ms'] for c in costs)}
    result['scope']='probe_cpu_ms contains stage CPU, workspace and setup costs. Readback is a CPU subset; GPU event spans overlap CPU waits. Do not add these envelopes or parent CostScopes.'
    return result

def validate_distribution(value,n,name):
    integer(value['count'],name+'.count');integer(value['sum'],name+'.sum');number(value['mean'],name+'.mean')
    require(value['count']==n,'Distribution count differs from launch query count')
    if not n:
        require(value['sum']==0 and value['mean']==0 and all(value[k] is None for k in ('p50','p95','p99','max')),
                'Invalid empty distribution');return
    points=[value[k] for k in ('p50','p95','p99','max')]
    for v in points:integer(v,name+'.percentile')
    require(points==sorted(points) and abs(value['mean']-value['sum']/n)<=1e-10*max(1,value['mean']),
            'Invalid percentile order or mean')

def validate_bvh(row):
    finite_tree(row)
    require(row['schema']=='gipc.bvh_query_workload.v1' and row['phase'] in ('DCD','FullCCD') and
            type(row['frame']) is int and 1<=row['frame']<=120,'Invalid BVH record identity')
    for key in ('passed','first_successful_query_only','production_outputs_untouched','tree_storage_untouched'):
        require(row[key] is True,'BVH production/replay guard failed: '+key)
    require(row['performance_certified'] is False,'Probe illegally certifies performance')
    number(row['diagnostic_host_ms'],'diagnostic_host_ms');number(row['dHat'],'dHat')
    if row['phase']=='DCD':require(row['alpha'] is None,'DCD unexpectedly has swept alpha')
    else:number(row['alpha'],'alpha')
    sample=row['sample'];require(sample['typed_multiset_equal'] is True and sample['atomic_array_order_compared'] is False,
        'Production typed tuple equality missing or raw atomic order wrongly used')
    counts=sample['private_counts'];require(len(counts)==5,'Invalid typed counts')
    for value in counts:integer(value,'private_counts')
    require(counts[0]==sample['production_pairs'],'Production/private pair count mismatch')
    if row['phase']=='DCD':require(counts[1]==0 and sum(counts[2:])==counts[0],'DCD type counts do not partition pairs')
    for key in ('private_device_bytes','workload_readback_bytes'):integer(sample[key],key)
    require(len(sample['queries'])==2,'Missing VF/EE workload summaries')
    kinds=set()
    for query in sample['queries']:
        kind=query['kind'];require(kind in (row['phase']+'_VF',row['phase']+'_EE') and kind not in kinds,'Wrong/duplicate query kind');kinds.add(kind)
        n=query['query_count'];integer(n,'query_count')
        for key in DISTRIBUTIONS:validate_distribution(query[key],n,key)
        validate_distribution(query['launch_warp_max_node_aabb_tests'],(n+31)//32,'warp_max')
        require(query['narrow_calls']['sum']<=query['leaf_overlaps']['sum'],'More narrow calls than leaf overlaps')
        if row['phase']=='FullCCD':require(query['narrow_calls']['sum']==0,'FullCCD includes narrow phase calls')
        if n:require(1<=query['max_pending_stack']['p50']<=query['max_pending_stack']['max']<=65 and
                     query['node_aabb_tests']['max']==query['launch_warp_max_node_aabb_tests']['max'],
                     'Invalid stack high-water or warp maximum')
        top=query['heaviest_queries'];require(len(top)==min(16,n),'Incomplete top16')
        threads=[t['launch_thread'] for t in top]
        require(len(set(threads))==len(threads) and all(type(t) is int and 0<=t<n for t in threads),'Invalid top16 thread identity')
        require([t['node_aabb_tests'] for t in top]==sorted((t['node_aabb_tests'] for t in top),reverse=True),'Unordered top16')
        if n:require(top[0]['node_aabb_tests']==query['node_aabb_tests']['max'],'Top16 misses maximum')
        for t in top:
            for key in ('original_id','node_aabb_tests','narrow_calls'):integer(t[key],'top16.'+key)
        require(query['query_node_mean']==query['node_aabb_tests']['mean'] and
                query['launch_warp_max_node_mean']==query['launch_warp_max_node_aabb_tests']['mean'],'Duplicated mean fields disagree')
        mean=query['query_node_mean'];imbalance=query['warp_max_mean_over_query_mean']
        require((imbalance is None) if mean==0 else finite_number(imbalance) and
                abs(imbalance-query['launch_warp_max_node_mean']/mean)<=1e-10*max(1,imbalance),'Invalid warp/query mean ratio')

def finite_number(value):return type(value) in (int,float) and math.isfinite(value)

def bvh_summary(rows):
    by_kind={}
    for row in rows:
        for query in row['sample']['queries']:by_kind.setdefault(query['kind'],[]).append((row,query))
    result={}
    for kind,group in by_kind.items():
        total_queries=sum(q['query_count'] for r,q in group)
        node_tests=sum(q['node_aabb_tests']['sum'] for r,q in group)
        total_warps=sum(q['launch_warp_max_node_aabb_tests']['count'] for r,q in group)
        warp_nodes=sum(q['launch_warp_max_node_aabb_tests']['sum'] for r,q in group)
        nonempty=[(r,q) for r,q in group if q['query_count']]
        peak_ratio=max(nonempty,key=lambda v:v[1]['warp_max_mean_over_query_mean']) if nonempty else None
        peak_p99=max(nonempty,key=lambda v:v[1]['node_aabb_tests']['p99']) if nonempty else None
        result[kind]={'sampled_frames':[r['frame'] for r,q in group],
            'replay_launches':len(group),'query_threads_sampled':sum(q['query_count'] for r,q in group),
            'node_aabb_tests_sampled':node_tests,'query_node_weighted_mean':ratio(node_tests,total_queries),
            'launch_warp_node_weighted_mean':ratio(warp_nodes,total_warps),
            'maximum_query_nodes':max((q['node_aabb_tests']['max'] for r,q in nonempty),default=None),
            'highest_per_replay_query_p99':{'frame':peak_p99[0]['frame'],'p99':peak_p99[1]['node_aabb_tests']['p99']} if peak_p99 else None,
            'highest_per_replay_warp_mean_over_query_mean':{'frame':peak_ratio[0]['frame'],
                'ratio':peak_ratio[1]['warp_max_mean_over_query_mean']} if peak_ratio else None,
            'leaf_overlaps_sampled':sum(q['leaf_overlaps']['sum'] for r,q in group),
            'narrow_calls_sampled':sum(q['narrow_calls']['sum'] for r,q in group),
            'maximum_stack':max((q['max_pending_stack']['max'] for r,q in group if q['query_count']),default=None),
            'maximum_launch_warp_nodes':max((q['launch_warp_max_node_aabb_tests']['max'] for r,q in group if q['query_count']),default=None),
            'per_sample_tail_summary':[{'frame':r['frame'],'linear_system_id':r.get('linear_system_id'),
                'query_count':q['query_count'],'node_distribution':q['node_aabb_tests'],
                'launch_warp_max_node_distribution':q['launch_warp_max_node_aabb_tests'],
                'warp_max_mean_over_query_mean':q['warp_max_mean_over_query_mean'],
                'heaviest_queries':q['heaviest_queries']} for r,q in group]}
    return {'phase_samples':len(rows),'passed_samples':sum(r['passed'] for r in rows),
        'initialization_without_linear_system':[r for r in rows if not r.get('linear_system_id')],
        'sampled_system_observations':sum(bool(r.get('linear_system_id')) for r in rows),
        'diagnostic_host_ms':sum(r['diagnostic_host_ms'] for r in rows),
        'private_device_bytes_peak':max((r['sample']['private_device_bytes'] for r in rows),default=None),
        'workload_readback_bytes_sampled':sum(r['sample']['workload_readback_bytes'] for r in rows),
        'query_kinds':result,
        'scope':'At most one successful replay per frame and DCD/FullCCD phase. Totals cover sampled replays only, not every production query or Newton system. Highest per-replay p99 is not a pooled query percentile. Warp node-count tails do not prove register spill, occupancy loss or removable execution cost.'}

def selected_windows(scene,reference):
    entry=next(s for s in reference['scenes'] if s['scene']==scene);candidates=entry['profile_frame_candidates']
    windows={'all_frames':[1,120],'cold_start':[1,3]}
    for key,name in (('linear_heavy_stable_directions','reference_linear_heavy'),('ccd_line_search_peak','reference_ccd_line_search_heavy')):
        item=candidates.get(key)
        if item:windows[name]=item['window']
    return windows

def cost_scopes(rows):
    ids=[r['scope_id'] for r in rows];require(len(ids)==len(set(ids)),'Repeated CostScope identity')
    known=set(ids);by_stage={}
    for row in rows:
        finite_tree(row);require(row['parent_scope_id']==0 or row['parent_scope_id'] in known,'Missing CostScope parent')
        require(row['schema']=='gipc.cost.v1' and row['diagnostic_only'] is True and row['inclusive'] is True,
                'Wrong CostScope schema or inclusive diagnostic semantics')
        number(row['cpu_submit_ms'],'cpu_submit_ms')
        if row.get('gpu_interval_ms') is not None:number(row['gpu_interval_ms'],'gpu_interval_ms')
        key=(row['stage'],row.get('sample_kind'));by_stage.setdefault(key,[]).append(row)
    return [{'stage':stage,'sample_kind':kind,'calls':len(group),
        'inclusive_cpu_ms':sum(r['cpu_submit_ms'] for r in group),
        'gpu_valid_intervals':sum(r.get('gpu_interval_ms') is not None for r in group),
        'inclusive_gpu_event_ms':sum(r['gpu_interval_ms'] for r in group if r.get('gpu_interval_ms') is not None)
            if any(r.get('gpu_interval_ms') is not None for r in group) else None,
        'scope':'Inclusive CostScope envelopes. Production parents include diagnostic children; entries must not be added across ancestry or to the independent probe costs.'}
        for (stage,kind),group in sorted(by_stage.items(),key=lambda item:str(item[0]))]

def optional_off(folder,scene,request):
    off=folder/scene
    if not (off/'result.json').exists():return {'status':'not_recorded','trajectory_comparison':None}
    result=read(off/'result.json')
    if result['status']!='completed' or result['recorded_frames']!=120:return {'status':'incomplete_or_failed','trajectory_comparison':None}
    if not (off/'evidence.json').exists():return {'status':'not_closed','trajectory_comparison':None}
    closed=closed_index(off)
    req=closed_json(off,closed,'requested.json');result=closed_json(off,closed,'result.json')
    require(closed_json(off,closed,'config_validation.json')['passed'] is True and result['finite'] is True,
            'Observer-off configuration/finite guard failed')
    c=req['expanded_config'];current=request['expanded_config']
    require(req['exe_sha256']==request['exe_sha256'] and req['source_digest']==request['source_digest'],
            'Observer-off uses a different native program')
    require(c['diagnostics']==[] and all(c[key]==value for key,value in current.items() if key not in
            {'diagnostics','cost_frames','cost_events','profile'}),'Observer-off numerical configuration differs')
    on=SESSION/'probe'/scene;difference=trajectory_difference(on,off,closed_index(on),closed)
    return {'status':'recorded','observer_off_solver_seconds_observed':result['solver_seconds'],
        'same_exe_sha256':req['exe_sha256'],'same_source_digest':req['source_digest'],
        'closed_inventory':{'probe':record(on/'evidence.json'),'observer_off':record(off/'evidence.json')},
        'trajectory_comparison':difference['comparisons'],'actual_input_verification':
            {k:v for k,v in difference.items() if k!='comparisons'},
        'scope':'Same-program bitwise trajectory observation only. A difference is not causal proof of production mutation without repeats; equality does not certify independent CCD or physical quality.'}

def historical_repeats(scene,current):
    folders=[SESSION/'reference'/f'{scene}_all_r{i}' for i in (1,2,3)]
    if any(not (p/'evidence.json').is_file() for p in folders):return {'status':'not_recorded','pairs':None}
    indices=[closed_index(p) for p in folders];requests=[];sources=[]
    for folder,index in zip(folders,indices):
        request=closed_json(folder,index,'requested.json');result=closed_json(folder,index,'result.json')
        require(result['status']=='completed' and result['recorded_frames']==120 and result['finite'] is True and
            closed_json(folder,index,'config_validation.json')['passed'] is True,'Invalid historical reference run')
        require(request['expanded_config']['scene']==scene and request['expanded_config']['steps']==120,
                'Wrong historical scene/window')
        requests.append(request);sources.append({'folder':str(folder),'evidence_inventory':record(folder/'evidence.json'),
            'request':record(folder/'requested.json'),'build_manifest':record(folder/'build_manifest.json'),
            'solver_seconds_observed':result['solver_seconds']})
        closed_bytes(folder,index,'build_manifest.json')
    require(all((r['exe_sha256'],r['source_digest'],r['expanded_config'])==
                (requests[0]['exe_sha256'],requests[0]['source_digest'],requests[0]['expanded_config']) for r in requests),
            'Historical all repeats mix native programs/configurations')
    exclude={'diagnostics','cost_frames','cost_events','profile'}
    old=requests[0]['expanded_config'];new=current['expanded_config']
    mismatches={k:{'historical':old[k],'probe':new[k]} for k in old.keys()&new.keys() if k not in exclude and old[k]!=new[k]}
    current_folder=SESSION/'probe'/scene;current_index=closed_index(current_folder)
    actual_initial_equal={name:closed_bytes(folders[0],indices[0],name)==closed_bytes(current_folder,current_index,name)
        for name in ('trace/topology.bin','trace/body_ids.bin','trace/boundary_types.bin','trace/masses.bin',
                     'trace/metadata.json','trace/state_0000.bin','trace/velocity_0000.bin')}
    pairs=[]
    for a,b in ((0,1),(0,2),(1,2)):
        pairs.append({'repeats':[a+1,b+1],'trajectory_difference':trajectory_difference(folders[a],folders[b],indices[a],indices[b])})
    return {'status':'recorded_historical_different_program','exe_sha256':requests[0]['exe_sha256'],
        'source_digest':requests[0]['source_digest'],'different_executable_from_current':requests[0]['exe_sha256']!=current['exe_sha256'],
        'common_non_diagnostic_config_mismatches':mismatches,'actual_initial_input_bytes_equal_to_current':actual_initial_equal,
        'sources':sources,'pairs':pairs,
        'scope':'Three already closed old-executable all-arm repetitions, paired as r1/r2, r1/r3 and r2/r3. Historical repeat-difference scale only; not a same-program observer control, predeclared neutrality threshold, baseline expansion or physical quality certification.'}

def inspect(scene,reference,off_dir):
    folder=SESSION/'probe'/scene
    if not (folder/'result.json').exists():return {'scene':scene,'status':'not_completed','linear':None,'bvh':None}
    result=read(folder/'result.json')
    if result['status']!='completed':return {'scene':scene,'status':'run_failed','native_result':result,'linear':None,'bvh':None}
    required=('requested.json','config_validation.json','output/stats.json','cost.jsonl','bvh_query_probe.jsonl','result.json')
    index=closed_index(folder);inputs={}
    for name in required:
        actual=record(folder/name);require(actual['bytes']==index[name]['bytes'] and actual['sha256']==index[name]['sha256'],
            'Closed probe evidence changed: '+name);inputs[name]=actual
    request=read(folder/'requested.json');config=request['expanded_config'];stats=read(folder/'output/stats.json');frames=stats['frames']
    require(result['recorded_frames']==120 and result.get('finite') is True and len(frames)==120 and
            read(folder/'config_validation.json')['passed'] is True and config['scene']==scene and
            config['steps']==120 and set(config['diagnostics'])=={'cost','structure_probe','bvh_query_probe'},'Incomplete/wrong probe run')
    finite_tree(stats);require(not stats.get('failure'),'Native failure report')
    linear=[];systems=set();system_context={};previous=None
    for frame_id,frame in enumerate(frames,1):
        for newton in frame['newton']:
            if 'pcg' not in newton:continue
            require('linear_structure_probe' in newton,'PCG solve lacks enabled structure observation')
            row=newton['linear_structure_probe'];validate_linear(row,frame_id,newton['pcg'],previous)
            require(row['linear_system_id'] not in systems,'Duplicated production linear system identity')
            systems.add(row['linear_system_id']);system_context[row['linear_system_id']]=row;linear.append(row);previous=row
    require(linear,'No structure observations')
    bvh=[json.loads(line) for line in (folder/'bvh_query_probe.jsonl').read_text().splitlines() if line.strip()]
    keys=[]
    for row in bvh:
        validate_bvh(row);keys.append((row['frame'],row['phase']))
        if row.get('linear_system_id'):
            require(row['linear_system_id'] in systems,'BVH replay refers to absent production system')
            require(all(row[k]==system_context[row['linear_system_id']][k] for k in CONTEXT),
                    'BVH replay/linear-system context mismatch')
    require(len(keys)==len(set(keys)) and bvh,'Repeated per-frame/phase replay or no replay evidence')
    costs=[json.loads(line) for line in (folder/'cost.jsonl').read_text().splitlines() if line.strip()]
    scope_summary=cost_scopes(costs)
    windows=[]
    for name,(low,high) in selected_windows(scene,reference).items():
        lr=[r for r in linear if low<=r['frame']<=high];br=[r for r in bvh if low<=r['frame']<=high]
        windows.append({'name':name,'frames':[low,high],'structure_reuse':reuse_summary(lr),
            'structure_costs':linear_costs(lr),'bvh_replay_samples':len(br),
            'bvh_diagnostic_host_ms':sum(r['diagnostic_host_ms'] for r in br),'bvh_query_summary':bvh_summary(br)})
    return {'scene':scene,'status':'validated_completed','inputs':inputs,'native_solver_seconds_instrumented':result['solver_seconds'],
        'linear':{'system_observations':len(linear),'reuse':reuse_summary(linear),'costs':linear_costs(linear),
            'per_system':linear,'owner_count':len({r['owner'] for r in linear}),
            'numerical_production_buffers_bitwise_unchanged':None,
            'neutrality_scope':'Exact same-owner keys/permutation/partition comparisons and independent snapshots are observed. No per-system Hessian/RHS/M before-after checksum is exported; symbolic hits do not prove numerical or MAS topology reuse.'},
        'bvh':bvh_summary(bvh),'windows':windows,'cost_scope_inclusive_observations':scope_summary,
        'observer_off':optional_off(off_dir,scene,request),'historical_repeat_differences':historical_repeats(scene,request)}

def analyze(off_dir):
    reference=read(REPORT/'REFERENCE_COST_ANALYSIS.json');rows=[]
    for scene in SCENES:
        try:rows.append(inspect(scene,reference,off_dir))
        except (ValueError,KeyError,FileNotFoundError,json.JSONDecodeError) as error:
            rows.append({'scene':scene,'status':'invalid_or_unclosed_evidence','error':type(error).__name__+': '+str(error),'linear':None,'bvh':None})
    return {'schema':'report56_structure_query_analysis.v1','data_complete':all(r['status']=='validated_completed' for r in rows),
        'analyzer':record(Path(__file__)),'reference_cost_selection':record(REPORT/'REFERENCE_COST_ANALYSIS.json'),
        'probe_ledger':record(SESSION/'PROBE_LEDGER.json') if (SESSION/'PROBE_LEDGER.json').is_file() else None,
        'scenes':rows,'performance_certified':False,'physical_quality_certified':False,
        'limits':['Probe timings include diagnostic synchronization and replay; they are not production speed or a removable-cost estimate.',
            'Readback CPU is contained in compare CPU. GPU intervals overlap completion waits. Parent CostScopes include diagnostic children and are never added to probe costs.',
            'Canonical pair equality does not prove reusable MAS topology, mapping, partition, numerical factorization or contact safety.',
            'BVH counters describe one sampled replay per frame/phase. Initialization without linear ID is separate. No extrapolation to all queries, register spill or occupancy.',
            'Observer-off is optional and absent results stay pending. On/off RMS and maxima describe actual output differences, not physical errors or a diagnostic neutrality certificate.',
            'Old-executable all-arm three-repeat differences are historical diagnostic context only, not a same-program control or predeclared neutrality threshold. No baseline velocity reconstruction, quality tolerance change or prior quality-gate promotion.']}

def markdown(data):
    lines=['# 结构与查询探针观察分析','',f"数据完整：{data['data_complete']}。本报告只分析已关闭并通过证据哈希核对的探针输出；诊断计时不作为生产速度。",'']
    for scene in data['scenes']:
        lines+=['## '+scene['scene'],'',f"状态：{scene['status']}。",'']
        if scene['status']!='validated_completed':
            if scene.get('error'):lines.append(scene['error'])
            lines+=['缺失／未关闭／失败的数据没有补零或生成命中率。',''];continue
        linear=scene['linear'];cost=linear['costs'];reuse=linear['reuse']
        lines += [f"观察到{linear['system_observations']}个生产线性系统、{linear['owner_count']}个owner。",'',
            '| 结构 | 相邻可比 | 命中 | 可比命中率 | 命中／全部观测 | 不可比原因 |','|---|---:|---:|---:|---:|---|']
        for kind in ('raw','mapping','partition','converted_joint','canonical_pairs'):
            row=reuse[kind];rate='不可比较' if row['adjacent_hit_rate'] is None else f"{100*row['adjacent_hit_rate']:.3f}%"
            coverage='未观测' if row['hit_fraction_all_observations'] is None else f"{100*row['hit_fraction_all_observations']:.3f}%"
            lines.append(f"| {kind} | {row['adjacent_comparable']} | {row['adjacent_hits']} | {rate} | {coverage} | {json.dumps(row['excluded'],ensure_ascii=False)} |")
        lines += ['',f"canonical pair命中但mapping／partition联合未命中：{reuse['canonical_hit_without_joint_mapping_partition_hit']}个。该计数不能解释为MAS分区或数值因子可复用。",'',
            '相邻可比分母只包括同owner、同shape、无gap的相邻系统；first／shape变化／gap被排除，因此较高的可比命中率不等于大部分系统可复用。', '',
            '| 六阶段 | CPU包含ms | 有效GPU event ms | readback CPU子区间ms |','|---|---:|---:|---:|']
        for stage,value in cost['six_stages'].items():
            gpu='未观测' if value['gpu_ms_valid_intervals'] is None else f"{value['gpu_ms_valid_intervals']:.3f}"
            lines.append(f"| {stage} | {value['cpu_ms']:.3f} | {gpu} | {value['readback_cpu_ms_subset']:.3f} |")
        lines += ['',f"线性探针CPU总包含时间{cost['probe_cpu_ms']:.3f}ms，event setup {cost['event_setup_cpu_ms']:.3f}ms、workspace {cost['workspace_cpu_ms']:.3f}ms均属于其子项。它们不能与六阶段CPU、GPU、readback或父级CostScope重复相加。",'',
            f"BVH验证{scene['bvh']['passed_samples']}/{scene['bvh']['phase_samples']}次私有replay；初始化无linear ID记录{len(scene['bvh']['initialization_without_linear_system'])}条，单列保存在JSON。生产typed tuple multiset相等、production outputs/tree storage untouched标志均检查为true；这不等于独立物理质量认证。",'',
            '| 查询类 | replay次数 | 样本query数 | 样本AABB测试 | 样本leaf overlap | 栈高水位 |','|---|---:|---:|---:|---:|---:|']
        for kind,value in scene['bvh']['query_kinds'].items():
            lines.append(f"| {kind} | {value['replay_launches']} | {value['query_threads_sampled']} | {value['node_aabb_tests_sampled']} | {value['leaf_overlaps_sampled']} | {value['maximum_stack']} |")
        lines += ['', '| 查询类 | 样本加权AABB均值 | 最大单query AABB | 最大逐replay p99／帧 | 最大warp均值/query均值／帧 |',
            '|---|---:|---:|---:|---:|']
        for kind,value in scene['bvh']['query_kinds'].items():
            p99=value['highest_per_replay_query_p99'];warp=value['highest_per_replay_warp_mean_over_query_mean']
            lines.append(f"| {kind} | {value['query_node_weighted_mean']:.3f} | {value['maximum_query_nodes']} | {p99['p99']} / {p99['frame']} | {warp['ratio']:.3f} / {warp['frame']} |")
        lines += ['',f"BVH私有replay诊断host包含时间{scene['bvh']['diagnostic_host_ms']:.3f}ms，私有device峰值{scene['bvh']['private_device_bytes_peak']} bytes，样本readback总量{scene['bvh']['workload_readback_bytes_sampled']} bytes。这里只观测每帧每phase一次；最大逐replay p99不是全部query的汇总p99。",'',
            '| reference选择窗口 | 帧 | 当前系统数 | raw命中/可比 | joint命中/可比 | canonical命中/可比 |','|---|---|---:|---:|---:|---:|']
        for w in scene['windows']:
            wr=w['structure_reuse'];cols=[f"{wr[k]['adjacent_hits']}/{wr[k]['adjacent_comparable']}" for k in ('raw','converted_joint','canonical_pairs')]
            lines.append(f"| {w['name']} | {w['frames'][0]}–{w['frames'][1]} | {w['structure_costs']['observations']} | "+' | '.join(cols)+' |')
        lines += ['', '0/0表示不可比较，不能解释为0%命中。窗口由已完成120帧Graph reference预选；当前探针的系统数、query尾部与计时仅为该窗口观察，不能外推全部查询或寄存器spill。','']
        off=scene['observer_off'];lines += [f"关闭态对照：{off['status']}。线性数值Hessian／RHS／M没有逐系统前后checksum，因此未写成bitwise unchanged通过。",'']
        if off['status']=='recorded':
            lines += [f"同exe/source、非诊断数值配置一致。关闭态solver {off['observer_off_solver_seconds_observed']:.6f}s，开启诊断 {scene['native_solver_seconds_instrumented']:.6f}s；这两次单次WDDM桌面观测不构成性能比较。",'',
                '| 实际导出 | body类型 | 121帧bitwise相等 | 121帧RMS差 | 最大3D差／帧 | frame120 RMS／最大差 |','|---|---|---|---:|---:|---:|']
            for kind,value in off['trajectory_comparison'].items():
                for body,total in value['body_totals'].items():
                    last=total['frame_120']
                    lines.append(f"| {kind} | {body} | {value['all_121_exports_bitwise_equal']} | {total['rms_vector_difference_over_recorded_frames']:.9g} | {total['maximum_vector_difference']:.9g} / {total['maximum_difference_frame']} | {last['rms_vector_difference']:.9g} / {last['maximum_vector_difference']:.9g} |")
            for kind,value in off['trajectory_comparison'].items():
                lines += ['',f"{kind} 缺失帧数{len(value['missing_frames'])}，逐字节不同帧数{len(value['different_frames'])}，首个不同帧{min(value['different_frames'],default=None)}。",'']
            lines += ['RMS按每body的3D顶点差计算；位置使用场景长度单位，实际速度使用长度单位/秒。frame0相等及后续差异是观测事实，单次开关对照不能直接判断探针改写了数值buffer，也不能认证中性或物理质量。','']
        history=scene['historical_repeat_differences'];lines += [f"历史三轮all参考：{history['status']}。",'']
        if history['status']=='recorded_historical_different_program':
            lines += [f"旧exe `{history['exe_sha256']}`，不同于本次程序。仅作为既有重复差异量级的诊断参考，不是本次同程序控制、中性阈值或质量认证。",'',
                '| 旧all重复对 | 实际导出 | body类型 | 121帧RMS差 | 最大3D差 |','|---|---|---|---:|---:|']
            for pair in history['pairs']:
                for kind,value in pair['trajectory_difference']['comparisons'].items():
                    for body,total in value['body_totals'].items():
                        lines.append(f"| r{pair['repeats'][0]}/r{pair['repeats'][1]} | {kind} | {body} | {total['rms_vector_difference_over_recorded_frames']:.9g} | {total['maximum_vector_difference']:.9g} |")
            lines += ['', '历史与本次共同非诊断配置不一致字段：'+json.dumps(history['common_non_diagnostic_config_mismatches'],ensure_ascii=False)+'。实际初始拓扑、body、边界、质量、位置和速度的逐字节对应结果保留在JSON。','']
    lines+=['## 结论边界','']+data['limits']+['','复现：`E:/Anaconda/envs/DL/python.exe reports/report56_stage1_20261008/analyze_structure_queries.py`；可用`--observer-off-dir`接入后续同程序关闭态目录，`--require-complete`要求两场景完整有效。','']
    return '\n'.join(lines)

def self_test():
    def row(index,state='hit',count=2,gap=False):
        return dict(owner=1,input_offset=0,item_count=count,output_offset=2,block_rows=3,block_cols=3,device=0,
            unique_count=1,linear_system_id=index,observation_index=index,gap_before=gap,pair_gap_before=gap,
            raw_state=state,converted_state=state,pair_state=state,raw_equal=state=='hit',mapping_equal=state=='hit',
            partition_equal=state=='hit',pair_equal=state=='hit',converted_observed=True,pair_observed=True,
            raw_comparison_performed=state in ('hit','content_changed'),converted_comparison_performed=state in ('hit','content_changed'),
            pair_comparison_performed=state in ('hit','content_changed'))
    a=row(1,'first');b=row(2);c=row(3,'shape_changed',3);d=row(4,'first',3,True);e=row(5,'content_changed',3)
    e['pair_state']='hit';e['pair_equal']=True
    result=reuse_summary([a,b,c,d,e]);require(result['raw']['adjacent_comparable']==2 and result['raw']['adjacent_hits']==1,'Synthetic denominator failure')
    require(result['raw']['excluded']=={'first':1,'shape_changed':1,'gap_before':1},'Synthetic exclusion failure')
    require(result['canonical_hit_without_joint_mapping_partition_hit']==1,'Canonical/joint false equivalence')
    try:finite_tree({'gpu_ms':float('nan')})
    except ValueError:pass
    else:raise ValueError('Nonfinite field accepted')
    validate_distribution({'count':0,'sum':0,'mean':0,'p50':None,'p95':None,'p99':None,'max':None},0,'empty')
    scopes=cost_scopes([{'scope_id':1,'parent_scope_id':0,'stage':'synthetic','sample_kind':'production',
                         'schema':'gipc.cost.v1','diagnostic_only':True,'inclusive':True,
                         'cpu_submit_ms':2,'gpu_interval_ms':None}])
    require(scopes[0]['inclusive_gpu_event_ms'] is None and scopes[0]['gpu_valid_intervals']==0,
            'Missing GPU cost imputed as zero')
    first=struct.pack('<6d',1,2,3,4,5,6);second=struct.pack('<6d',1,2,3,7,9,6)
    vectors=vector_difference(first,second,{'ABD':[0],'FEM_2D':[1]})
    require(vectors['ABD']['rms_vector_difference']==0 and vectors['FEM_2D']['rms_vector_difference']==5 and
            vectors['FEM_2D']['maximum_vector_difference']==5,'Wrong 3D body RMS/max calculation')
    try:vector_difference(first,struct.pack('<6d',1,2,3,4,float('nan'),6),{'ABD':[0],'FEM_2D':[1]})
    except ValueError:pass
    else:raise ValueError('Nonfinite actual trajectory accepted')
    print(json.dumps({'cpu_self_test':'passed','checks':['adjacent_denominator','first_shape_gap_exclusions','canonical_not_mapping_partition','nonfinite_rejection','empty_distribution','missing_gpu_not_zero','actual_body_vector_rms_max','nonfinite_actual_trajectory_rejection']}))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--observer-off-dir',type=Path,default=SESSION/'observer_off')
    parser.add_argument('--require-complete',action='store_true');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return 0
    data=analyze(args.observer_off_dir);output=REPORT/'STRUCTURE_QUERY_ANALYSIS.json'
    output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8');output.with_suffix('.md').write_text(markdown(data),encoding='utf-8')
    print(json.dumps({'data_complete':data['data_complete'],'scenes':[{k:r[k] for k in ('scene','status')} for r in data['scenes']]}))
    return 2 if args.require_complete and not data['data_complete'] else 0

if __name__=='__main__':raise SystemExit(main())
