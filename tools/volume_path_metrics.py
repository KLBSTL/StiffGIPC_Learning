"""Independent cubic determinant audit of a selected accepted-path interval.

Only volume is checked here. Collision, trajectory agreement, and completion
remain separate gates, including for a run that stopped partway through a frame.
"""
import argparse
import json
from pathlib import Path
import numpy as np

def triple(a,b,c):
    return np.einsum('ij,ij->i',a,np.cross(b,c))

def edges(x,tets):
    q=x[tets]
    return q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('--first-frame',type=int,default=0)
    p.add_argument('--last-frame',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    trace=args.run/'trace'
    raw=np.fromfile(trace/'topology.bin',dtype='<u4')
    nv,nf,nt=map(int,raw[:3]);tets=raw[3+nf*3:].reshape(nt,4)
    initial=np.fromfile(trace/'state_0000.bin',dtype='<f8').reshape(nv,3)
    rest=triple(*edges(initial,tets))
    if np.any(rest==0):raise ValueError('Zero initial tetrahedron determinant')
    paths=[f for f in sorted((trace/'substeps').glob('safe_*.bin'))
           if args.first_frame<=int(f.stem.split('_')[1])<=args.last_frame]
    if len(paths)<2:raise ValueError('Need at least two selected accepted states')
    start=np.fromfile(paths[0],dtype='<f8').reshape(nv,3)
    minimum=float('inf');nonpositive=0;finite=bool(np.isfinite(start).all());where={}
    for path in paths[1:]:
        end=np.fromfile(path,dtype='<f8').reshape(nv,3)
        finite=finite and bool(np.isfinite(end).all())
        a,b,c=edges(start,tets);u,v,w=edges(end,tets)
        da,db,dc=u-a,v-b,w-c
        d0=triple(a,b,c)/rest
        d1=(triple(da,b,c)+triple(a,db,c)+triple(a,b,dc))/rest
        d2=(triple(da,db,c)+triple(da,b,dc)+triple(a,db,dc))/rest
        d3=triple(da,db,dc)/rest
        def evaluate(t):return ((d3*t+d2)*t+d1)*t+d0
        local=np.minimum(d0,evaluate(1))
        cubic=np.abs(d3)>1e-30
        disc=4*d2*d2-12*d3*d1
        valid=cubic & (disc>=0)
        for sign in [-1,1]:
            root=np.divide(-2*d2+sign*np.sqrt(np.maximum(0,disc)),6*d3,
                           out=np.zeros_like(d3),where=valid)
            use=valid & (root>0) & (root<1)
            local=np.minimum(local,np.where(use,evaluate(root),np.inf))
        quadratic=(~cubic) & (np.abs(d2)>1e-30)
        root=np.divide(-d1,2*d2,out=np.zeros_like(d2),where=quadratic)
        use=quadratic & (root>0) & (root<1)
        local=np.minimum(local,np.where(use,evaluate(root),np.inf))
        idx=int(local.argmin())
        if local[idx]<minimum:
            minimum=float(local[idx]);where={'segment_end':path.name,'tet_index':idx}
        nonpositive+=int((local<=0).sum())
        start=end
    result={'run':args.run.name,'scope':'selected actual accepted substeps; independent cubic determinant only',
            'first_frame':args.first_frame,'last_frame':args.last_frame,
            'first_state':paths[0].name,'last_state':paths[-1].name,
            'segments_checked':len(paths)-1,'tetrahedra':nt,'finite':finite,
            'relative_jacobian_path_min':minimum,'minimum_location':where,
            'nonpositive_segment_tet_count':nonpositive,
            'volume_gate_for_selected_interval':finite and nonpositive==0,
            'full_quality_gate':'not evaluated; collision and run completion are separate'}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
