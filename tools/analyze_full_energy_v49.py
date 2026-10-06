"""Independent CPU verification of full-energy probes and contact transitions."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from event_model_v48 import EventModel,CONTACT
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def closest_distances(c):
    """Independent constrained least-squares candidates, not GPU branch logic."""
    a=c['anchor'];out=np.zeros(len(c))
    def segment(p,x,y):
        v=y-x;u=np.clip(np.sum((p-x)*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-300),0,1)
        return np.linalg.norm(p-x-u[:,None]*v,axis=1)
    mask=c['kind']==0;p,x,y,z=a[mask].transpose(1,0,2);e=y-x;f=z-x;w=p-x
    ee=np.sum(e*e,axis=1);ff=np.sum(f*f,axis=1);ef=np.sum(e*f,axis=1)
    ew=np.sum(e*w,axis=1);fw=np.sum(f*w,axis=1);den=ee*ff-ef*ef
    good=den>1e-28*np.maximum(ee*ff,1e-300);den=np.where(good,den,1.)
    u=(ff*ew-ef*fw)/den;v=(ee*fw-ef*ew)/den
    dist=np.linalg.norm(w-u[:,None]*e-v[:,None]*f,axis=1)
    dist=np.where(good&(u>=0)&(v>=0)&(u+v<=1),dist,np.inf)
    out[mask]=np.minimum.reduce([dist,segment(p,x,y),segment(p,y,z),segment(p,z,x)])
    mask=c['kind']==1;x,y,z,w=a[mask].transpose(1,0,2);e=y-x;f=w-z;v=x-z
    ee=np.sum(e*e,axis=1);ff=np.sum(f*f,axis=1);ef=np.sum(e*f,axis=1)
    ev=np.sum(e*v,axis=1);fv=np.sum(f*v,axis=1);den=ee*ff-ef*ef
    good=den>1e-28*np.maximum(ee*ff,1e-300);den=np.where(good,den,1.)
    s=(ef*fv-ff*ev)/den;t=(ee*fv-ef*ev)/den
    dist=np.linalg.norm(v+s[:,None]*e-t[:,None]*f,axis=1)
    dist=np.where(good&(s>=0)&(s<=1)&(t>=0)&(t<=1),dist,np.inf)
    out[mask]=np.minimum.reduce([dist,segment(x,z,w),segment(y,z,w),segment(z,x,y),segment(w,x,y)])
    return out

def analyze(prefix):
    m=EventModel(prefix);e=m.e;checks=[]
    def check(name,x,y,atol=1e-10,rtol=1e-8):
        x=np.asarray(x);y=np.asarray(y);err=float(np.max(np.abs(x-y),initial=0))
        scale=float(np.max(np.abs(y),initial=0));passed=np.isfinite(err) and err<=atol+rtol*scale
        checks.append({'name':name,'max_abs_error':err,'scale':scale,'atol':atol,'rtol':rtol,'passed':bool(passed)})
    assert e['capture_state_unchanged'] and e['post_probe_state_unchanged'] and e['full_probe_state_restored']
    points=e['full_energy_probes'];assert len(points)==2
    check('accepted_probe_center',points[0]['alpha'],e['line_search_r'],0,0)
    check('half_probe_center',points[1]['alpha'],e['line_search_r']/2,0,0)
    check('accepted_energy_reproduction',points[0]['energy'],e['energy_after'])
    max_derivative_disagreement=0.
    for n,point in enumerate(points):
        samples=[point]+[s[k] for s in point['derivatives'] for k in ['plus','minus']]
        for j,sample in enumerate(samples):
            x=m.x-sample['alpha']*m.d;x[m.fixed]=m.x[m.fixed]
            comp=sample['components'];check(f'energy_sum_{n}_{j}',sum(comp.values()),sample['energy'])
            check(f'cpu_al_{n}_{j}',m.al(x),comp['al_contact'])
            check(f'cpu_fem_{n}_{j}',m.fem(x),comp['fem'])
        for j,s in enumerate(point['derivatives']):
            check(f'derivative_arithmetic_{n}_{j}',(s['plus']['energy']-s['minus']['energy'])/(2*s['h']),s['derivative'],0,0)
        d0,d1=[s['derivative'] for s in point['derivatives']]
        disagreement=abs(d0-d1)/max(1.,abs(d0),abs(d1));max_derivative_disagreement=max(max_derivative_disagreement,disagreement)
        check(f'derivative_two_scale_{n}',d0,d1,1e-6,1e-4)
    update_prefix=Path(e['contact_update_prefix']);update_path=Path(str(update_prefix)+'_update.json')
    coverage={'update_completed':update_path.exists(),'next_linearization_completed':False}
    if update_path.exists():
        meta=read(update_path);before=np.fromfile(str(update_prefix)+'_update_before.bin',dtype=CONTACT)
        after=np.fromfile(str(update_prefix)+'_update_after.bin',dtype=CONTACT)
        x=np.fromfile(str(update_prefix)+'_update_vertices.bin',dtype='<f8').reshape(-1,3)
        assert len(before)==len(after)==meta['contacts'] and meta['contact_bytes']==CONTACT.itemsize
        active_slots=np.arange(4)[None,:]<np.where(before['kind']==2,1,4)[:,None]
        ids=np.where(active_slots,before['ids'],0);g=before['grad']*active_slots[:,:,None]
        value=before['offset']+np.einsum('nki,nki->n',g,x[ids]-before['anchor'])
        mu=np.where(before['penalty_mu']>0,before['penalty_mu']*meta['mu']/meta['initial_mu'],meta['mu'])
        slack=np.maximum(0,value-before['lambda']/mu);active=slack==0
        lam=np.where(active,before['lambda']-mu*value,0.)
        if meta['robust_port']:
            age=np.where(active,0,np.minimum(before['release_age']+1,26));gamma=np.where(active,1.,.9**age)
        else:age=before['release_age'];gamma=np.where(active,1.,before['gamma']*.9)
        for name,expected in [('slack',slack),('lambda',lam),('gamma',gamma),('release_age',age)]:check('update_'+name,after[name],expected,1e-12,1e-10)
        for name in ['kind','ids','grad','anchor','offset','penalty_mu','coordinates']:check('update_preserves_'+name,after[name],before[name],0,0)
        if not meta['independent_friction']:
            for name in ['friction_lambda','friction_gamma']:check('update_preserves_'+name,after[name],before[name],0,0)
        nxt=Path(str(update_prefix)+'_next_linearized.json')
        if nxt.exists():
            meta2=read(nxt);host=np.fromfile(str(update_prefix)+'_next_host.bin',dtype=CONTACT)
            nextc=np.fromfile(str(update_prefix)+'_next_linearized.bin',dtype=CONTACT)
            safe=np.fromfile(str(update_prefix)+'_next_safe.bin',dtype='<f8').reshape(-1,3)
            assert len(nextc)==meta2['contacts']==len(host)
            key=lambda c:(int(c['kind']),*map(int,c['ids']))
            old={key(c):c for c in after};new={key(c):c for c in host};linear={key(c):c for c in nextc}
            assert len(old)==len(after) and len(new)==len(host) and len(linear)==len(nextc) and new.keys()==linear.keys()
            eligible={k for k,c in old.items() if (c['release_age']<=25 if meta['robust_port'] else c['gamma']>=.01)}
            common=eligible&new.keys();removed=old.keys()-new.keys();added=new.keys()-common
            readded=(old.keys()&new.keys())-common
            for name in ['lambda','gamma','release_age','slack']:
                check('retained_'+name,[new[k][name] for k in common],[old[k][name] for k in common],0,0)
                check('linearize_preserves_'+name,[linear[k][name] for k in new],[new[k][name] for k in new],0,0)
            if meta['robust_port']:assert all(old[k]['release_age']>25 for k in removed)
            for name,default in [('lambda',0.),('gamma',1.),('slack',0.),('release_age',0)]:
                check('new_contact_'+name,[new[k][name] for k in added],np.full(len(added),default),0,0)
            mask=np.arange(4)[None,:]<np.where(nextc['kind']==2,1,4)[:,None]
            check('next_anchor_safe',nextc['anchor'][mask],safe[nextc['ids'][mask]],0,0)
            dist=closest_distances(nextc);selfc=nextc['kind']!=2
            check('independent_closest_distance',nextc['offset'][selfc]+meta2['delta'],dist[selfc],1e-10,1e-8)
            ground=nextc['kind']==2;scene_meta=read(m.run/'trace/metadata.json')
            normal=np.array(scene_meta['ground_normal']);off=scene_meta['ground_offset']
            check('ground_offset',nextc['offset'][ground],nextc['anchor'][ground,0]@normal-off-meta2['delta'])
            coverage.update(next_linearization_completed=True,retained=len(common),removed=len(removed),added=len(added),
                released_and_readded=len(readded),before_contacts=len(before),next_contacts=len(nextc))
    return {'event':prefix.name,'frame':e['frame'],'outer':e['outer'],'inner':e['inner'],
        'full_state_restored':True,'checks':checks,'passed':all(c['passed'] for c in checks),
        'accepted_r':e['line_search_r'],'energy_before':e['energy_before'],
        'accepted_energy':points[0]['energy'],'half_energy':points[1]['energy'],
        'half_is_lower':points[1]['energy']<points[0]['energy'],
        'extra_decrease_from_half':points[0]['energy']-points[1]['energy'],
        'accepted_derivative':points[0]['derivatives'][-1]['derivative'],
        'half_derivative':points[1]['derivatives'][-1]['derivative'],
        'max_scaled_derivative_disagreement':max_derivative_disagreement,'contact_coverage':coverage}

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();manifest=read(ROOT/'manifests/perf_v49_local.json')
    for f in manifest['files']:assert sha(ROOT/f['path'])==f['sha256']
    for path,f in manifest['binaries'].items():assert sha(ROOT/path)==f['sha256']
    folder=ROOT/'runs/local'/a.run/'events';prior=read(ROOT/'reports'/f'EVENTS_{a.run}_CPU.json')
    for e in prior['events']:
        for path,digest in e['sha256'].items():assert sha(folder/path)==digest
    report={'run':a.run,'events':[],'scope':'Frozen complete energy probes; independent AL/FEM and contact update checks. Not a solver modification.'}
    for path in sorted(folder.glob('*_line_search.json')):
        row=analyze(path.parent/path.name.removesuffix('_line_search.json'));report['events'].append(row)
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in row.items() if k!='checks'}),flush=True)
    assert report['events'];report['passed']=all(e['passed'] for e in report['events'])
    report['helper_sha256']={p.name:sha(p) for p in [Path(__file__),ROOT/'tools/event_model_v48.py']}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False));return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
