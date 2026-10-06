"""Summarize selected complete-energy/contact/shape diagnostics, without certification."""
import argparse
import collections
import json
from config import ROOT, read, sha
from validate_run import validate

STAGES=('model_begin','trial_before_dual','dual_update','safe_accepted')

def analyze(folder):
    gate=validate(folder)
    if not gate['passed']:raise ValueError('Configuration/execution gate failed: '+folder.name)
    path=folder/'outer_probe.jsonl'
    events=[json.loads(line) for line in path.read_text().splitlines()]
    groups=collections.defaultdict(dict)
    for e in events:
        group=groups[e['frame'],e['outer']]
        if e['stage'] in group:raise ValueError('Duplicate outer probe stage')
        group[e['stage']]=e
    stats=read(folder/'output/stats.json')['frames'];rows=[]
    for (frame,outer),group in sorted(groups.items()):
        if set(group)!=set(STAGES):raise ValueError('Incomplete outer probe')
        b,t,d,s=[group[k] for k in STAGES]
        if len({e['contact_model_id'] for e in group.values()})!=1:
            raise ValueError('Probe spans different contact models')
        if not t['physics_state_unchanged']:raise ValueError('Energy probe changed protected state')
        n=t['last_newton'];r=stats[frame-1]['toi'][outer]
        if (n['frame'],n['outer'])!=(frame,outer):raise ValueError('Newton identity mismatch')
        if abs(t['parts_sum_minus_complete'])>1e-12*max(1,abs(t['complete_energy'])):
            raise ValueError('Energy components do not match full objective')
        rows.append({'frame':frame,'outer':outer,'contact_model_id':t['contact_model_id'],
            'inner_iterations':t['inner_iterations'],'exit':n['inner_exit_reason'],
            'last_raw_direction_velocity_m_s':n['trial_newton_axis_velocity_m_s'],
            'alpha':s['alpha'],'remaining_beta':s['beta'],
            'contacts':t['contact']['contacts'],'initial_guess':r.get('initial_guess_selection'),
            'safe_begin':b['safe_shape'],'warm_begin':b['warm_shape'],
            'trial':t['trial_shape'],'safe_end':s['safe_shape'],
            'plane_change':b['plane_change'],'contact_before_dual':t['contact'],
            'contact_after_dual':d['contact'],'contact_at_safe':s['solved_contact_model_at_safe'],
            'active_added':s['active_added'],'active_removed':s['active_removed'],
            'complete_energy':t['complete_energy'],'energy_parts':t['energy_parts'],
            'parts_sum_minus_complete':t['parts_sum_minus_complete'],
            'last_energy_before':n['trial_energy_before'],'last_energy_after':n['trial_energy_after'],
            'selected_exit_probe_active':n.get('full_step_exit_probe_active',False)})
    work=[]
    for i,f in enumerate(stats,1):
        systems=[n for n in f['newton'] if 'pcg' in n]
        work.append({'frame':i,'directions':len(systems),
            'pcg_iterations':sum(n['pcg']['iterations'] for n in systems),
            'outers':len(f.get('toi',[])),
            'one_direction_outers':sum(o['inner_iterations']==1 for o in f.get('toi',[]))})
    return {'name':folder.name,'scope':'selected diagnostic; not complete KKT or same-state counterfactual certification',
        'configuration_passed':True,'probe_sha256':sha(path),'event_count':len(events),
        'model_count':len(rows),'protected_energy_probes':len(rows),
        'energy_parts_max_abs_error':max(abs(r['parts_sum_minus_complete']) for r in rows),
        'one_direction_full_step_exits':sum(r['inner_iterations']==1 and r['exit']=='full_step' for r in rows),
        'one_direction_velocity_exits':sum(r['inner_iterations']==1 and r['exit']=='velocity_converged' for r in rows),
        'observed_frames':sorted({r['frame'] for r in rows}),
        'requested_exit_probe':read(folder/'requested.json')['expanded_config'].get('full_step_exit_probe'),
        'rows':rows,'work_by_frame':work,'performance_certified':False,'quality_certified':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    data={'scope':'outer mechanism diagnosis only','runs':[analyze(ROOT/n) for n in a.runs]}
    with (ROOT/a.output).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
    print(json.dumps({'runs':[{k:v for k,v in r.items() if k not in ('rows','work_by_frame')} for r in data['runs']]}))
