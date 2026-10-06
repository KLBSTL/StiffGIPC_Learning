import argparse,csv,json
from pathlib import Path
import numpy as np

def analyze(root):
    result=json.loads((root/'result.json').read_text())
    requested=json.loads((root/'requested.json').read_text())
    data=json.loads((root/'output/stats.json').read_text())
    frames=data.get('frames',[])
    pcg=[n['pcg'] for f in frames for n in f.get('newton',[]) if 'pcg' in n]
    toi=[t for f in frames for t in f.get('toi',[])]
    rows=list(csv.DictReader((root/'trace/frames.csv').open()))
    result.update(run=root.name,arm=requested['arm'],scene=requested.get('scene'),steps=requested['steps'],
        audit=any(requested.get(k,False) for k in ['audit_pcg','audit_graph','profile','substeps']),
        direction_solves=len(pcg),pcg_iterations=sum(p.get('iterations',0) for p in pcg),
        pcg_limit_hits=sum(p.get('iteration_limit',False) for p in pcg),
        graph_captures=max((p.get('graph_captures_total',0) for p in pcg),default=0),
        graph_invalidations=max((p.get('graph_invalidations_total',0) for p in pcg),default=0),
        graph_cache_hits=sum(p.get('graph_cache_hit',False) for p in pcg),
        toi_outer_iterations=len(toi),
        min_accepted_alpha=min((t['alpha'] for t in toi),default=None),
        max_active_contacts=max((t['active_self']+t['active_ground'] for t in toi),default=0),
        normal_force_proxy_first_frame=next((i+1 for i,f in enumerate(frames) if any(t.get('physical_contact_force_proxy') for t in f.get('toi',[]))),None),
        candidate_first_frame=next((int(r['frame']) for r in rows if int(r['candidate_pairs'])+int(r['ground_candidates'])>0),None),
        failure=data.get('failure'),exit_reasons={k:sum((f.get('toi_exit') or f.get('newton_exit'))==k for f in frames) for k in sorted(set(f.get('toi_exit') or f.get('newton_exit') or 'missing' for f in frames))},
        max_same_system_solution_difference=max((p.get('same_system_relative_solution_difference',0) for p in pcg),default=None),
        max_true_relative_residual=max((p.get('true_relative_residual',0) for p in pcg),default=None))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reports=[analyze(r) for r in a.runs];a.output.write_text(json.dumps(reports,indent=2),encoding='utf-8')
    for r in reports:print(json.dumps(r))
