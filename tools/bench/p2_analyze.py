"""P2-A measured costs, paired timing and repeated-control state differences."""
from __future__ import annotations
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from config import read
from linux_runner import require,write_new
from local_identity import record,verify
from run_innovation_p2a import SCENES,ARMS,SESSION
from m0_analyze import compare,geometry,csv_write

def timing(rows):
    # Three pairs are screening evidence only. This interval assumes normal log
    # ratios; do not reinterpret it as certification on a noisy WDDM desktop.
    logs=[math.log(r['speedup']) for r in rows]
    mean=statistics.mean(logs);half=4.30265273*statistics.stdev(logs)/math.sqrt(3)
    return dict(pairs=rows,geometric_mean=math.exp(mean),median=statistics.median(r['speedup'] for r in rows),
        log_t_95_screening_interval=[math.exp(mean-half),math.exp(mean+half)],
        limitations='Three paired samples, normal-log assumption, WDDM desktop; not formal performance certification')

def analyze(session,output,include_whole):
    require(not output.exists(),'Fresh analysis directory required')
    seal=read(session/'seal.json');identity=seal['identity']
    scenes={t['scene']:(t['config']['scene'],t['config']['steps']) for t in seal['plans']['prefix']}
    verify(identity['sources']+[identity['exe']]+identity['dlls']+identity['build_evidence']+identity['objects'])
    stages=['smoke','prefix','cost','audit','matched']+(['whole'] if include_whole else [])
    ledgers={s:read(session/(s+'_ledger.json')) for s in stages}
    prefix_root=Path(ledgers['prefix']['inherited_from']['path']).parent if 'inherited_from' in ledgers['prefix'] else session
    require(all(v['complete'] for v in ledgers.values()),'Incomplete stage; no performance result')
    flat=[];by_stage={}
    for stage,ledger in ledgers.items():
        by_stage[stage]={}
        stage_root=Path(ledger['inherited_from']['path']).parent if 'inherited_from' in ledger else session
        for row in ledger['runs']:
            t=row['task'];folder=stage_root/t['name'];result=read(folder/'result.json')
            require(result['status']=='completed' and result['cleanup_owned_job_empty'],'Invalid native result')
            power=read(folder/'power_samples.json')
            require(power['expected']==seal['power_guid'] and power['after']==seal['power_guid']
                and all(r['guid']==seal['power_guid'] for r in power['samples']),'Power mismatch')
            require(read(folder/'config_validation.json')['passed'],'Resolved native config mismatch')
            frames=read(folder/'output/stats.json')['frames'];newton=[n for f in frames for n in f['newton']]
            o=row['observation'];entry=dict(stage=stage,name=t['name'],scene=t['scene'],arm=t['arm'],
                repeat=t.get('repeat',0),steps=t['config']['steps'],N=o['N'],P=o['P'],solver_ms=o['solver_ms'],
                line_search_ms=o['phase_ms']['line_search'],wall_ms=result['wall_seconds']*1000,
                energy_evaluations=sum(n.get('energy_evaluations',0) for n in newton),
                energy_backtracks=sum(n.get('energy_backtracks',0) for n in newton),
                intersection_backtracks=sum(n.get('intersection_backtracks',0) for n in newton),
                alpha_min=min(o['accepted_alpha'],default=None),exits=o['exits'],
                phases=o['phase_ms'],abd_audit_count=row.get('abd_audit_count',0))
            flat.append(entry);by_stage[stage][(t['scene'],t['arm'],t.get('repeat',0))]=entry
    paired={}
    for stage in ['matched']+(['whole'] if include_whole else []):
        groups={}
        for scene in scenes:
            for arm in (['scene','abd','both'] if stage=='matched' else ['both']):
                measures={}
                for metric in ('solver_ms','line_search_ms'):
                    pairs=[]
                    for repeat in (1,2,3):
                        b=by_stage[stage][(scene,'control',repeat)];a=by_stage[stage][(scene,arm,repeat)]
                        pairs.append(dict(repeat=repeat,control_ms=b[metric],candidate_ms=a[metric],
                            speedup=b[metric]/a[metric],control_N=b['N'],candidate_N=a['N'],
                            control_P=b['P'],candidate_P=a['P']))
                    measures[metric]=timing(pairs)
                groups[scene+'/'+arm]=measures
        paired[stage]=groups
    costs=[];cost_bounds=[]
    for row in ledgers['cost']['runs']:
        t=row['task'];records=[json.loads(s) for s in (session/t['name']/'cost.jsonl').read_text().splitlines()]
        if t['arm']=='control':
            ancestors={r['scope_id']:r for r in records}
            def in_line_search(row):
                seen=set();parent=row['parent_scope_id']
                while parent in ancestors:
                    require(parent not in seen,'Cyclic cost parent identity');seen.add(parent)
                    node=ancestors[parent]
                    if node['stage']=='ipc.line_search':return True
                    parent=node['parent_scope_id']
                return False
            line=sum(r['cpu_submit_ms'] for r in records if r['stage']=='ipc.line_search')
            removable=('bvh.scene_readback','abd.kinetic_readback','abd.shape_readback')
            windows=sum(r['cpu_submit_ms'] for r in records if r['stage'] in removable and in_line_search(r))
            wait=sum(r['cpu_submit_ms'] for r in records if r['stage']=='ipc.line_search.entry_wait')
            cost_bounds.append(dict(scene=t['scene'],line_search_cpu_ms=line,
                eligible_readback_cpu_windows_ms=windows,optimistic_window_fraction=windows/line,
                entry_wait_cpu_ms=wait,
                interpretation='Observed host-window bookkeeping only; includes mandatory GPU work, not a general theorem or pure memcpy cost'))
        buckets=defaultdict(list)
        for r in records:
            if r['stage'].startswith(('bvh.','abd.','ipc.energy','ipc.line_search')):buckets[r['stage']].append(r)
        for stage,values in sorted(buckets.items()):
            costs.append(dict(scene=t['scene'],arm=t['arm'],scope=stage,calls=len(values),
                cpu_inclusive_ms=sum(v['cpu_submit_ms'] for v in values),gpu_interval_ms=None,
                parent_scope_ids=sorted(set(v['parent_scope_id'] for v in values))))
    differences=[];geometry_rows=[]
    for stage in ['matched']+(['whole'] if include_whole else []):
        for scene,(_,physical) in scenes.items():
            final=1 if stage=='matched' else physical
            controls=[session/f'{stage}_{scene.lower()}_control_r{r}' for r in (1,2,3)]
            for repeat,folder in enumerate(controls,1):
                for g in geometry(folder,prefix_root/('prefix_'+scene.lower()),final,
                    physical-1 if stage=='matched' else 0):
                    geometry_rows.append(dict(stage=stage,scene=scene,arm='control',repeat=repeat,**g))
            diameter=[compare(a,b,final,final) for a,b in itertools.combinations(controls,2)]
            for arm in (['scene','abd','both'] if stage=='matched' else ['both']):
                for repeat in (1,2,3):
                    folder=session/f'{stage}_{scene.lower()}_{arm}_r{repeat}'
                    value=compare(folder,controls[repeat-1],final,final)
                    differences.append(dict(stage=stage,scene=scene,arm=arm,repeat=repeat,
                        comparison=value,control_repeat_diameter=diameter,
                        scope='Actual positions/velocities vs paired control; no certified physical reference'))
                    for g in geometry(folder,prefix_root/('prefix_'+scene.lower()),final,
                        physical-1 if stage=='matched' else 0):
                        geometry_rows.append(dict(stage=stage,scene=scene,arm=arm,repeat=repeat,**g))
    output.mkdir(parents=True)
    csv_write(output/'runs.csv',[{k:v for k,v in r.items() if k not in ('phases','exits')} for r in flat])
    csv_write(output/'costs.csv',[{k:v for k,v in r.items() if k!='parent_scope_ids'} for r in costs])
    csv_write(output/'geometry.csv',geometry_rows)
    write_new(output/'state_differences.json',differences)
    summary=dict(schema='innovation_p2a_analysis.v1',seal=record(session/'seal.json'),binary=identity['exe'],
        power_guid=seal['power_guid'],completed_runs=len(flat),frames=sum(r['steps'] for r in flat),
        paired=paired,costs=costs,cost_bounds=cost_bounds,abd_bitwise_audits=sum(r['abd_audit_count'] for r in flat),
        included_whole=include_whole,formal_performance_certified=False,physical_quality_certified=False,
        diagnostic_only=True,limitations=['Short event-selected scenes, three timing pairs','Shared WDDM desktop load',
            'Native MAS atomic reduction changes directions across repeats',
            'M0 Mixed nonpositive tet Jacobians are inherited; this patch does not repair material/injectivity'])
    write_new(output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('completed_runs','frames','abd_bitwise_audits')}))
    for stage,groups in paired.items():
        for group,measure in groups.items():
            print(json.dumps(dict(stage=stage,group=group,solver=measure['solver_ms']['geometric_mean'],
                line_search=measure['line_search_ms']['geometric_mean'])))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,default=SESSION)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--include-whole',action='store_true')
    a=p.parse_args();analyze(a.session,a.output,a.include_whole)
