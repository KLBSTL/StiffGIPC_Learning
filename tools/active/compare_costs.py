"""Production cost intervals with diagnostic subtrees removed, including work counts."""
import argparse
import json
from config import ROOT, read,sha

def row(run):
    path=run/'cost.jsonl';events=[json.loads(line) for line in path.read_text().splitlines()]
    frames=sorted({r['frame'] for r in events})
    total=sum(r['gpu_interval_ms'] for r in events if r['stage']=='linear.total_including_diagnostics')
    probe=sum(r['gpu_interval_ms'] for r in events if r['stage']=='diagnostic.operator_probe_including_audit')
    stages={}
    for e in events:
        if e['sample_kind']=='production' and e['gpu_interval_ms'] is not None:
            stages[e['stage']]=stages.get(e['stage'],0)+e['gpu_interval_ms']
    systems=[n for i,f in enumerate(read(run/'output/stats.json')['frames'],1) if i in frames for n in f['newton'] if 'pcg' in n]
    iterations=sum(n['pcg']['iterations'] for n in systems)
    return {'run':run.name,'trace_sha256':sha(path),'frames':frames,'systems':len(systems),'iterations':iterations,
            'inclusive_linear_ms':total,'diagnostic_probe_ms':probe,'linear_excluding_probe_ms':total-probe,
            'linear_ms_per_pcg_iteration':(total-probe)/iterations,'production_inclusive_stages_ms':stages}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('serial');p.add_argument('warp');p.add_argument('--output',required=True);a=p.parse_args()
    x,y=[row(ROOT/n) for n in (a.serial,a.warp)]
    assert x['frames']==y['frames']
    result={'diagnostic_only':True,'performance_certified':False,
            'scope':'Selected windows, continuous from zero. CUDA elapsed includes host gaps and trace overhead. Trajectories and work may differ; same-system protected pairs are separate evidence.',
            'arms':[x,y],'aggregate_linear_ratio':x['linear_excluding_probe_ms']/y['linear_excluding_probe_ms'],
            'per_iteration_normalized_ratio':x['linear_ms_per_pcg_iteration']/y['linear_ms_per_pcg_iteration']}
    with (ROOT/a.output).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='arms'}))
