"""P3 input/tuple completeness, resource and whole-query cost verification."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from config import read
from linux_runner import require,write_new
from local_identity import record,verify
from run_innovation_p3 import SESSION,SCENES
from m0_analyze import compare,geometry,csv_write
from cpu_fixed_reference import load_snapshot

def ratios(values):
    require(len(values) in (3,7) and all(math.isfinite(v) and v>0 for v in values),
        'Paired screening requires 3 or 7 finite positive ratios')
    logs=[math.log(v) for v in values];center=statistics.mean(logs)
    critical=4.30265273 if len(values)==3 else 2.44691185
    half=critical*statistics.stdev(logs)/math.sqrt(len(logs))
    return dict(pairs=values,geometric_mean=math.exp(center),median=statistics.median(values),
        log_t_95_screening_interval=[math.exp(center-half),math.exp(center+half)])

def inputs(study):
    folder=Path(study['input_directory']);evidence=[]
    for item in study['input_buffers']:
        path=folder/item['file'];require(path.stat().st_size==item['elements']*item['element_bytes'],'Snapshot size differs')
        evidence.append(record(path))
    n=study['edge_count'];edges=np.fromfile(folder/'edges.bin',dtype='<u4').reshape(-1,2)
    x=np.fromfile(folder/'positions.bin',dtype='<f8').reshape(-1,3)
    rest=np.fromfile(folder/'rest.bin',dtype='<f8').reshape(-1,3)
    body=np.fromfile(folder/'body.bin',dtype='<i4');boundary=np.fromfile(folder/'boundary.bin',dtype='<i4')
    boxes=np.fromfile(folder/'boxes.bin',dtype='<f8').reshape(-1,6)
    nodes=np.fromfile(folder/'nodes.bin',dtype='<u4').reshape(-1,4)
    require(len(edges)==n and len(nodes)==len(boxes)==2*n-1,'Captured BVH shape differs')
    require(x.shape==rest.shape and len(x)==len(body)==len(boundary),'Captured point attributes differ')
    require(np.isfinite(x).all() and np.isfinite(rest).all() and np.isfinite(boxes).all(),'Nonfinite query input')
    require(edges.max()<len(x),'Edge endpoint not captured')
    pending=[0];seen=set();leaves=[]
    while pending:
        i=pending.pop();require(i<len(nodes) and i not in seen,'Disconnected/cyclic tree');seen.add(i)
        parent,left,right,element=map(int,nodes[i])
        if element==0xffffffff:
            require(i<n-1 and left<len(nodes) and right<len(nodes) and left!=right,'Invalid internal node')
            require(nodes[left,0]==i and nodes[right,0]==i,'Parent/child identity differs')
            expected=np.r_[np.maximum(boxes[left,:3],boxes[right,:3]),np.minimum(boxes[left,3:],boxes[right,3:])]
            require(np.array_equal(expected,boxes[i]),'Stale internal bounds')
            pending.extend([left,right])
        else:
            require(i>=n-1 and element<n,'Invalid leaf mapping');leaves.append(element)
            vertices=x[edges[element]];expected=np.r_[vertices.max(axis=0),vertices.min(axis=0)]
            require(np.array_equal(expected,boxes[i]),'Leaf bound differs from original primitive geometry')
    require(len(seen)==2*n-1 and sorted(leaves)==list(range(n)),'Tree is not a complete primitive permutation')
    if 'raw_pair_ids' in study:
        raw=np.asarray(study['raw_pair_ids'],dtype=np.int64).reshape(-1,2)
        require(len(raw)==study['workload']['narrow_calls']['sum'],'Raw count differs from original F narrow calls')
        require(np.all(raw[:,0]<raw[:,1]) and np.all(raw>=0) and np.all(raw<n),'Original primitive orientation lost')
        require(len(np.unique(raw,axis=0))==len(raw),'Duplicate original primitive IDs in raw queue')
        a,b=edges[raw[:,0]],edges[raw[:,1]]
        require(np.all((body[a[:,0]]!=body[b[:,0]])|(body[a[:,0]]==-1)),'Body filter differs')
        require(not np.any((a[:,0]==b[:,0])|(a[:,0]==b[:,1])|(a[:,1]==b[:,0])|(a[:,1]==b[:,1])),'Shared topology not excluded')
        require(not np.any(np.all(boundary[np.c_[a,b]]>=2,axis=1)),'All-fixed filter differs')
        aa=x[a];bb=x[b];gap=math.sqrt(study['dHat_squared_distance'])
        require(np.all(bb.min(axis=1)-aa.max(axis=1)<gap) and np.all(aa.min(axis=1)-bb.max(axis=1)<gap),'Nonoverlapping raw candidate')
    return evidence

def analyze(session,output):
    require(not output.exists(),'Fresh analysis required');session=session.resolve()
    seal=read(session/'seal.json');identity=seal['identity']
    verify(identity['sources']+identity['objects']+identity['build_evidence']+[identity['exe']]+identity['dlls'])
    flat=[];queries=[];state_rows=[];geometric=[];data={}
    for stage in ('smoke','prefix','query','matched','fallback','whole','control_query'):
        ledger=read(session/(stage+'_ledger.json'));require(ledger['complete'],'Incomplete finite stage')
        for row in ledger['runs']:
            t=row['task'];folder=session/t['name'];o=row['observation'];r=read(folder/'result.json')
            require(row['passed'] and r['status']=='completed' and r['cleanup_owned_job_empty'],'Invalid run')
            require(read(folder/'config_validation.json')['passed'],'Resolved configuration differs')
            frames=read(folder/'output/stats.json')['frames'];newton=[n for f in frames for n in f['newton']]
            entry=dict(stage=stage,name=t['name'],scene=t['scene'],mode=t['mode'],repeat=t.get('repeat',0),
                steps=t['config']['steps'],N=o['N'],P=o['P'],solver_ms=o['solver_ms'],line_search_ms=o['phase_ms']['line_search'],
                wall_ms=r['wall_seconds']*1000,energy_backtracks=sum(n.get('energy_backtracks',0) for n in newton),
                intersection_backtracks=sum(n.get('intersection_backtracks',0) for n in newton),
                w_queries=sum(f['p3_ee']['w_queries'] for f in frames),
                raw_peak=max(f['p3_ee']['raw_peak'] for f in frames),fallbacks=sum(f['p3_ee']['complete_fused_fallbacks'] for f in frames))
            flat.append(entry);data[(stage,t['scene'],t['mode'],t.get('repeat',0))]=entry
            if stage in ('query','control_query'):
                study=json.loads((folder/'p3_ee.jsonl').read_text().strip());require(study['passed'],'Same query failed')
                evidence=inputs(study)
                speeds=[p['F']['complete_cpu_ms']/p['W']['complete_cpu_ms'] for p in study['pairs']]
                queries.append(dict(scene=t['scene'],mode=t['mode'],name=t['name'],stage=stage,study=study,input_evidence=evidence,
                    complete_query_speedup=ratios(speeds)))
    pairs={}
    for stage in ('matched','whole'):
        groups={}
        for scene in SCENES:
            measures={}
            for metric in ('solver_ms','line_search_ms'):
                measures[metric]=ratios([data[(stage,scene,'fused',r)][metric]/data[(stage,scene,'two_stage',r)][metric] for r in (1,2,3)])
            groups[scene]=measures
            final=1 if stage=='matched' else SCENES[scene][1]
            controls=[session/f'{stage}_{scene.lower()}_fused_r{r}' for r in (1,2,3)]
            diameter=[compare(a,b,final,final) for a,b in itertools.combinations(controls,2)]
            for repeat in (1,2,3):
                candidate=session/f'{stage}_{scene.lower()}_two_stage_r{repeat}'
                state_rows.append(dict(stage=stage,scene=scene,repeat=repeat,
                    comparison=compare(candidate,controls[repeat-1],final,final),control_repeat_diameter=diameter))
                for mode,folder in [('fused',controls[repeat-1]),('two_stage',candidate)]:
                    for g in geometry(folder,session/('prefix_'+scene.lower()),final,SCENES[scene][1]-1 if stage=='matched' else 0):
                        geometric.append(dict(stage=stage,scene=scene,mode=mode,repeat=repeat,**g))
        pairs[stage]=groups
    matrix_comparisons=[];cross_process_inputs=[]
    for scene in SCENES:
        group=[q for q in queries if q['scene']==scene]
        require(len(group)==3,'Missing original pair or F/F calibration')
        signatures=[{Path(r['path']).name:r['sha256'] for r in q['input_evidence']} for q in group]
        # First queried trial geometry is reached AFTER the first linear solve.
        # An exact checkpoint restore does not imply equal later trial states.
        # Retain this distinction; only WITHIN a native replay are all inputs shared.
        for i,j in ((0,1),(0,2),(2,1)):
            a,b=group[i],group[j]
            equal={k:signatures[i][k]==signatures[j][k] for k in signatures[i]}
            require(all(equal[k] for k in ('rest.bin','body.bin','boundary.bin','edges.bin')),'Static scene input differs')
            x,y=[np.fromfile(Path(q['study']['input_directory'])/'positions.bin',dtype='<f8').reshape(-1,3) for q in (a,b)]
            distance=np.linalg.norm(x-y,axis=1)
            cross_process_inputs.append(dict(scene=scene,a=a['name'],b=b['name'],buffer_equal=equal,
                max_coordinate_difference_m=float(np.max(np.abs(x-y))),
                point_rms_difference_m=float(np.sqrt(np.mean(distance**2))),point_max_difference_m=float(distance.max()),
                typed_multiset_equal=a['study']['typed_pairs']==b['study']['typed_pairs'],
                scope='Later trial geometry, not exact common input; native within-query F/W replay IS exact common input'))
        matrices=[]
        for q in group:
            folder=session/q['name']/'fixed'
            studies=list(folder.glob('*_study.json'));require(len(studies)==1,'Missing unique matrix capture')
            prefix=Path(str(studies[0])[:-len('_study.json')]);matrices.append(load_snapshot(prefix))
        for i,j in ((0,1),(0,2),(2,1)):
            (a,b,*_),(c,d,*_)=matrices[i],matrices[j]
            require(a.shape==c.shape and b.shape==d.shape,'Matrix/dof shape changed')
            delta=a-c;denominator=math.sqrt(float(a.multiply(a).sum()))
            matrix_comparisons.append(dict(scene=scene,a=group[i]['name'],b=group[j]['name'],
                A_relative_frobenius=math.sqrt(float(delta.multiply(delta).sum()))/max(denominator,1e-300),
                A_max_absolute=float(np.max(np.abs(delta.data))) if delta.nnz else 0,
                b_relative_L2=float(np.linalg.norm(b-d)/max(np.linalg.norm(b),1e-300)),
                b_max_absolute=float(np.max(np.abs(b-d))),
                scope='First assembled A,b after a common checkpoint; F/F calibrated, NOT assumed equal systems'))
    output.mkdir(parents=True)
    csv_write(output/'runs.csv',flat);csv_write(output/'geometry.csv',geometric)
    write_new(output/'queries.json',queries);write_new(output/'state_differences.json',state_rows)
    write_new(output/'matrix_comparisons.json',matrix_comparisons)
    write_new(output/'cross_process_inputs.json',cross_process_inputs)
    summary=dict(schema='innovation_p3_analysis.v1',seal=record(session/'seal.json'),binary=identity['exe'],
        successful_runs=len(flat),frames=sum(r['steps'] for r in flat),paired=pairs,
        queries=[dict(scene=q['scene'],mode=q['mode'],name=q['name'],speedup=q['complete_query_speedup'],
            queue_bytes=q['study']['queue_device_bytes'],resources=q['study']['resources'],
            workload=q['study']['workload'],gpu_stage_ms=q['study']['gpu_stage_ms']) for q in queries],
        matrix_comparisons=matrix_comparisons,cross_process_inputs=cross_process_inputs,
        diagnostic_only=True,performance_certified=False,physical_quality_certified=False)
    write_new(output/'summary.json',summary)
    print(json.dumps(dict(runs=len(flat),frames=summary['frames'],matrix=matrix_comparisons)))
    for q in queries:print(json.dumps(dict(scene=q['scene'],mode=q['mode'],full_query_speedup=q['complete_query_speedup']['geometric_mean'],
        raw=q['study']['workload']['narrow_calls']['sum'],contacts=q['study']['typed_counts'][0])))
    for stage,groups in pairs.items():print(json.dumps(dict(stage=stage,solver={s:v['solver_ms']['geometric_mean'] for s,v in groups.items()})))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',type=Path,default=SESSION)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();analyze(a.session,a.output)
