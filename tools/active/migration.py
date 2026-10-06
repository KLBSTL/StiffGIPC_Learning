"""Compare an active trace against a frozen executable's own repeat envelope."""
import argparse
import itertools
import json
import numpy as np
from config import ROOT, read, sha

def differences(a,b,start,end):
    d=a/'trace';meta=read(d/'metadata.json');raw=np.fromfile(d/'topology.bin','<u4')
    nv,nf,nt=map(int,raw[:3]);tets=raw[3+3*nf:].reshape(-1,4);tet=np.zeros(nv,bool);tet[tets.ravel()]=True
    mass=np.fromfile(d/'masses.bin','<f8');abd=meta['abd_point_num'];fixed=np.fromfile(d/'boundary_types.bin','<i4')==1
    masks={'cloth':~tet&~fixed,'fem':tet&(np.arange(nv)>=abd)&~fixed,'abd':(np.arange(nv)<abd)&~fixed}
    worst={k:0. for k in masks}
    for i in range(start,end+1):
        x=np.fromfile(a/'trace'/f'state_{i:04d}.bin','<f8').reshape(-1,3)
        y=np.fromfile(b/'trace'/f'state_{i:04d}.bin','<f8').reshape(-1,3)
        for key,mask in masks.items():
            if np.any(mask):worst[key]=max(worst[key],float(np.sqrt(np.average(np.sum((x[mask]-y[mask])**2,axis=1),weights=mass[mask]))))
    return worst

def verify(reference_names,candidate_names,windows):
    refs=[ROOT/'runs/active'/n for n in reference_names];candidates=[ROOT/'runs/active'/n for n in candidate_names]
    rows=[]
    for run in refs+candidates:
        result=read(run/'result.json')
        assert result['status']=='completed',run
        for file in ('state_0000.bin','topology.bin','metadata.json','boundary_types.bin','masses.bin'):
            assert (refs[0]/'trace'/file).read_bytes()==(run/'trace'/file).read_bytes()
    for start,end in windows:
        pairs=[differences(a,b,start,end) for a,b in itertools.combinations(refs,2)]
        envelope={k:max(p[k] for p in pairs) for k in pairs[0]}
        for candidate in candidates:
            delta=differences(refs[0],candidate,start,end)
            rows.append({'run':candidate.name,'window':[start,end],'frozen_repeat_envelope_rms_m':envelope,
                         'candidate_delta_rms_m':delta,
                         'passes':all(delta[k]<=envelope[k]+1e-12 for k in delta)})
    return {'scope':'Migration only; not comparison to Stiff physical accuracy.',
            'floating_comparison_floor_m':1e-12,'references':reference_names,'candidates':candidate_names,
            'passed':all(r['passes'] for r in rows),'checks':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--references',nargs='+',required=True)
    p.add_argument('--candidates',nargs='+',required=True);p.add_argument('--windows',default='1-3,24-26,33-35')
    p.add_argument('--output',required=True);a=p.parse_args()
    windows=[tuple(map(int,w.split('-'))) for w in a.windows.split(',')]
    result=verify(a.references,a.candidates,windows);path=ROOT/a.output
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result));raise SystemExit(int(not result['passed']))
