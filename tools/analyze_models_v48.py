"""Cross-check CPU terms against committed GPU events; preserve failed gates."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from event_model_v48 import EventModel
from analyze_perf_v36 import matrix_from_snapshot
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def analyze(prefix):
    m=EventModel(prefix);e=m.e;meta,A,b=matrix_from_snapshot(prefix);z=m.load('solution')
    checks=[]
    def check(name,actual,expected,atol=1e-9,rtol=1e-8):
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected)),initial=0.))
        scale=float(np.max(np.abs(expected))) if np.size(expected) else 0.
        passed=bool(np.isfinite(error) and error<=atol+rtol*scale)
        checks.append({'name':name,'max_abs_error':error,'expected_scale':scale,'atol':atol,'rtol':rtol,'passed':passed})
    check('generalized_physical_direction',m.physical(z),m.d,1e-10,1e-12)
    check('exported_abd_dq',z[:12*len(m.q)],m.dq.ravel(),1e-12,1e-12)
    check('contact_kind_counts',np.bincount(m.c['kind'],minlength=3),e['contact_summary']['pt_ee_ground'],0,0)
    check('slack_at_assembly',m.c['slack'],np.maximum(0,m.constraint(m.x)-m.c['lambda']/m.mu),1e-12,1e-12)
    after=m.load('after_vertices').reshape(-1,3)
    predicted=m.x-e['line_search_r']*m.d
    predicted[m.fixed]=m.x[m.fixed] # Native step_forward does not move fixed FEM vertices.
    check('physical_endpoint',after,predicted,1e-10,1e-12)
    terms={};gradients={}
    for name,fn,key in [('al',m.al,'al_contact'),('fem',m.fem,'fem')]:
        energy,g,per=fn(m.x,True);gradients[name]=m.lift(g)
        end=fn(after);check(name+'_gpu_energy_before',energy,e['energy_components_before'][key])
        check(name+'_gpu_energy_after',end,e['energy_components_after'][key])
        dd=float(np.sum(g*-m.d));curvature=m.al_curvature(m.d) if name=='al' else None
        check(name+'_generalized_dot',gradients[name]@(-z),dd,1e-9,1e-10)
        finite=[]
        for displacement in [1e-5,1e-6,1e-7]:
            h=displacement/max(float(np.max(np.abs(m.d))),1.)
            plus=fn(m.x-h*m.d);minus=fn(m.x+h*m.d)
            fd=(plus-minus)/(2*h)
            check(name+'_fd_'+str(displacement),fd,dd,1e-9,1e-6)
            gp=fn(m.x-h*m.d,True)[1];gm=fn(m.x+h*m.d,True)[1]
            second=float(np.sum((gp-gm)*(-m.d))/(2*h))
            if name=='al':check('al_curvature_'+str(displacement),second,curvature,1e-7,1e-6)
            finite.append({'physical_max_probe_m':h*float(np.max(np.abs(m.d))),'h':h,'energy_derivative':fd,'gradient_derivative':second})
        terms[name]={'energy_before':energy,'energy_after':end,'energy_delta':end-energy,
            'negative_direction_derivative':dd,'gradient_norm':float(np.linalg.norm(gradients[name])),
            'analytic_curvature':curvature,'finite_differences':finite}
    other=b-gradients['al']-gradients['fem']
    total_dd=float(-b@z);xax=float(z@(A@z));r=e['line_search_r']
    row={'event':prefix.name,'frame':e['frame'],'outer':e['outer'],'inner':e['inner'],'checks':checks,
        'passed':all(c['passed'] for c in checks),'terms':terms,'b_norm':float(np.linalg.norm(b)),
        'assembled_negative_direction_derivative':total_dd,'assembled_xAx':xax,
        'unseparated_other_gradient_norm':float(np.linalg.norm(other)),
        'unseparated_other_negative_direction_derivative':float(-other@z),
        'total_energy_delta':e['energy_after']-e['energy_before'],
        'model_delta_terms':{'linear_al':r*terms['al']['negative_direction_derivative'],
            'linear_fem':r*terms['fem']['negative_direction_derivative'],'linear_other':float(-r*other@z),
            'quadratic_al':.5*r*r*terms['al']['analytic_curvature'],
            'quadratic_remaining_assembled':.5*r*r*(xax-terms['al']['analytic_curvature'])},
        'contact_state':{'mu':m.mu,'penalty_mu_zero_count':int(np.sum(m.c['penalty_mu']==0)),
            'slack_positive_count':int(np.sum(m.c['slack']>0)),
            'min_constraint':float(m.constraint(m.x).min()) if len(m.c) else None,'min_lambda':float(m.c['lambda'].min()) if len(m.c) else None,
            'max_lambda':float(m.c['lambda'].max()) if len(m.c) else None,'gamma_range':[float(m.c['gamma'].min()),float(m.c['gamma'].max())] if len(m.c) else None,
            'anchor_safe_max_abs':float(np.max(np.abs((m.c['anchor']-m.load('safe').reshape(-1,3)[m.cids])[m.active]),initial=0.))} }
    row['sha256']={p.name:sha(p) for p in prefix.parent.glob(prefix.name+'_*')}
    if m.run.name=='v47_bunny_graph' and prefix.name=='e4':
        reference=np.fromfile(ROOT/'reports/REFERENCE_V47_E4.bin',dtype='<f8')
        row['reference_residual']=float(np.linalg.norm(b-A@reference)/np.linalg.norm(b))
        check('reference_residual',row['reference_residual'],0,1e-8,0)
        probes=[]
        for label,q in [('original',z),('accurate',reference)]:
            d=m.physical(q)
            for alpha in [1e-6,1e-4,.001,.00390625,.0078125,.015625,.03125,.0625]:
                x=m.x-alpha*d;ea=m.al(x);ef=m.fem(x)
                probes.append({'direction':label,'alpha':alpha,'al_delta':ea-terms['al']['energy_before'],
                    'fem_delta':ef-terms['fem']['energy_before'],
                    'al_plus_fem_delta':ea+ef-terms['al']['energy_before']-terms['fem']['energy_before'],
                    'scope':'two energy components only; fixed contact planes, slack and multipliers'})
        row['reference_probes']=probes
        row['passed']=all(c['passed'] for c in checks)
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    manifest=json.loads((ROOT/'manifests/perf_v47_local.json').read_text())
    for f in manifest['files']:assert sha(ROOT/f['path'])==f['sha256']
    for path,f in manifest['binaries'].items():assert sha(ROOT/path)==f['sha256']
    old=json.loads((ROOT/'reports'/f'EVENTS_{a.run}_CPU.json').read_text())
    folder=ROOT/'runs/local'/a.run/'events'
    for event in old['events']:
        for name,digest in event['sha256'].items():assert sha(folder/name)==digest
    out={'schema':'v48-independent-al-fem-1','run':a.run,'source_digest':manifest['source_digest'],
        'frozen_source_binary_events_verified':True,'events':[],
        'scope':'Independent fixed AL and homogeneous FEM energy/gradient. Other terms are not independently decomposed.'}
    for path in sorted(folder.glob('*_event.json')):
        row=analyze(path.parent/path.name.removesuffix('_event.json'));out['events'].append(row)
        a.output.write_text(json.dumps(out,indent=2,allow_nan=False))
        print(json.dumps({'event':row['event'],'passed':row['passed'],'failures':[c for c in row['checks'] if not c['passed']],
            'derivatives':{k:v['negative_direction_derivative'] for k,v in row['terms'].items()},
            'other_derivative':row['unseparated_other_negative_direction_derivative']}),flush=True)
    out['passed']=all(r['passed'] for r in out['events'])
    out['helper_sha256']={p.name:sha(p) for p in [Path(__file__),ROOT/'tools/event_model_v48.py']}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False))
    return 0 if out['passed'] else 1
if __name__=='__main__':sys.exit(main())
