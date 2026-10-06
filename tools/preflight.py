"""Conservative size estimates before allocating a scene on the local GPU."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def estimate_scene(p,budget):
    d=json.loads(p.read_text());c=d['initial_counts'];nv=c['vertex_count'];nt=c['tet_count'];nf=c['triangle_count']
    # Allocate enough for the initial collision reserve, double staging,
    # conversion scratch, mass/material state and a conservative MAS reserve.
    # All tets are charged as FEM; fixed ABD cases are overestimated on purpose.
    fixed=10*nt+36*nf+nv+10*c['abd_body_count']
    live=fixed+1_600_000
    matrix=(2*live*80+live*28)/2**20
    other=(nv*8192+nt*1024+nf*512)/2**20+400
    estimate=matrix+other
    return {'case_id':p.stem,'vertices':nv,'triangles':nf,'tets':nt,
        'estimated_reserve_mib':round(estimate),'local_budget_mib':round(budget),
        'pilot_decision':'small_pilot' if estimate<=budget else 'autodl_first',
        'estimate_is_not_a_peak_measurement':True}

if __name__=='__main__':
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    free=int(gpu.strip());budget=min(.75*free,free-1536)
    rows=[estimate_scene(p,budget) for p in sorted((ROOT/'sources/stiff_base/Assets/benchmark_scenes').glob('*.json'))]
    out={'free_mib':free,'budget_mib':budget,'cases':rows}
    (ROOT/'reports/local_preflight.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'local_pilot_cases':[r['case_id'] for r in rows if r['pilot_decision']=='small_pilot'],
                  'autodl_first_cases':[r['case_id'] for r in rows if r['pilot_decision']=='autodl_first'],'budget_mib':budget}))
