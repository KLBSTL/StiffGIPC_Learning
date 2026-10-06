"""Export timings separately from physics/trajectory qualification."""
import argparse,csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARMS=['base','base_graph','base_toi','base_toi_graph']

def median_range(values):
    return {'median':statistics.median(values),'min':min(values),'max':max(values),'values':values} if values else None

def details(path):
    data=json.loads((path/'output/stats.json').read_text());frames=data.get('frames',[])
    pcg=[n['pcg'] for f in frames for n in f.get('newton',[]) if 'pcg' in n]
    with (path/'trace/frames.csv').open() as file:times=[float(r['solver_ms'])/1000 for r in csv.DictReader(file)]
    return frames,pcg,times

def main():
    p=argparse.ArgumentParser();p.add_argument('matrix',type=Path);p.add_argument('--output-prefix',type=Path,required=True);a=p.parse_args()
    matrix=json.loads(a.matrix.read_text());root=a.matrix.parent.parent;rows=[];summaries=[]
    for run in matrix['runs']:
        if run['stage']!='measure':continue
        result=run['result'];row={'scene':run['scene'],'arm':run['arm'],'name':run.get('name'),'status':result['status'],
            'solver_seconds':result.get('solver_seconds'),'frames':result.get('recorded_frames'),
            'observed_gpu_increment_mib':result.get('observed_gpu_increment_peak_mib'),
            'quality_matched_speedup':'N/A: separate quality gates pending'}
        path=root/run.get('name','missing')
        if (path/'output/stats.json').exists() and (path/'trace/frames.csv').exists():
            frames,pcg,times=details(path)
            row.update(direction_solves=len(pcg),pcg_iterations=sum(x.get('iterations',0) for x in pcg),
                pcg_limit_hits=sum(x.get('iteration_limit',False) for x in pcg),
                outer_limit_hits=sum(f.get('newton_exit')=='iteration_limit' or f.get('toi_exit')=='iteration_limit' for f in frames),
                graph_captures=max((x.get('graph_captures_total',0) for x in pcg),default=0),
                graph_invalidations=max((x.get('graph_invalidations_total',0) for x in pcg),default=0),
                max_active_contacts=max((t.get('active_self',0)+t.get('active_ground',0) for f in frames for t in f.get('toi',[])),default=0),
                failure_message=data_failure(path))
        rows.append(row)
    for scene in matrix['config']['scenes']:
        scene_rows=[r for r in rows if r['scene']==scene]
        summary={'scene':scene,'timings':{},'diagnostic_speed_ratios':{},'paired_phase_times':[]}
        for arm in ARMS:
            values=[r['solver_seconds'] for r in scene_rows if r['arm']==arm and r['status']=='completed' and r['frames']==matrix['config']['steps'] and r.get('pcg_limit_hits',0)==0 and r.get('outer_limit_hits',0)==0]
            summary['timings'][arm]=median_range(values)
        for arm in ARMS[1:]:
            ratios=[]
            for rep in range(1,matrix['config'].get('repeats',3)+1):
                suffix=f'_r{rep:02d}'
                base=next((r for r in scene_rows if r['arm']=='base' and r.get('name','').endswith(suffix)),None)
                candidate=next((r for r in scene_rows if r['arm']==arm and r.get('name','').endswith(suffix)),None)
                if not base or not candidate or any(r['status']!='completed' or r['frames']!=matrix['config']['steps'] or r.get('pcg_limit_hits',0) or r.get('outer_limit_hits',0) for r in [base,candidate]):continue
                ratios.append(base['solver_seconds']/candidate['solver_seconds'])
                bf,_,bt=details(root/base['name']);_,_,ct=details(root/candidate['name'])
                if len(bt)!=len(ct):continue
                masks={'geometric_contact':[f.get('contact_geometry',{}).get('geometric_contact',False) for f in bf],
                    'native_near_contact':[f.get('contact_geometry',{}).get('native_narrow_self_pairs',0)+f.get('contact_geometry',{}).get('native_narrow_ground_pairs',0)>0 for f in bf]}
                masks['no_native_near_contact']=[not x for x in masks['native_near_contact']]
                for phase,mask in masks.items():
                    frames=[i+1 for i,x in enumerate(mask) if x]
                    b=sum(t for t,m in zip(bt,mask) if m);c=sum(t for t,m in zip(ct,mask) if m)
                    summary['paired_phase_times'].append({'arm':arm,'repeat':rep,'phase':phase,'frame_ids':frames,'frame_count':len(frames),'base_seconds':b,'candidate_seconds':c,'diagnostic_ratio':b/c if c else None})
            summary['diagnostic_speed_ratios'][arm]=median_range(ratios)
        summaries.append(summary)
    report={'matrix_status':matrix['status'],'config':matrix['config'],'timing_scope':'solver window, capture/rebuild included; serialization and state export excluded',
        'qualification':'These are diagnostic ratios; unpassed or pending trajectory, continuous CCD, and physics gates forbid quality-matched claims.',
        'phase_definition':{'geometric_contact':'native narrow-phase minimum distance <= 1e-4 initial bbox diagonal',
                            'native_near_contact':'native distance-tested interaction pairs exist; this is not an exact zero-distance contact classifier'},
        'runs':rows,'scenes':summaries}
    a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    a.output_prefix.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fields=sorted(set(k for r in rows for k in r))
    with a.output_prefix.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'matrix_status':matrix['status'],'recorded_measurements':len(rows),'scenes_with_complete_four_arms':sum(all(s['timings'][arm] for arm in ARMS) for s in summaries)}))

def data_failure(path):
    data=json.loads((path/'output/stats.json').read_text())
    return data.get('failure',data.get('exception',''))

if __name__=='__main__':main()
