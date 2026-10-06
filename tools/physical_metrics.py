"""Geometry/kinematic diagnostics; does not claim material energy equivalence."""
import argparse,json,csv
from pathlib import Path
import numpy as np

def snapshot(run):
    d=run/'trace';files=sorted(d.glob('state_*.bin'));x0=np.fromfile(files[0],dtype='<f8').reshape(-1,3)
    x=np.fromfile(files[-1],dtype='<f8').reshape(-1,3);prev=np.fromfile(files[-2],dtype='<f8').reshape(-1,3)
    metadata=json.loads((run/'requested.json').read_text());scene=json.loads((run/'output/scene.json').read_text())
    dt=metadata.get('dt') or scene['effective_run']['dt']
    delta_frames=int(files[-1].stem[-4:])-int(files[-2].stem[-4:])
    velocity=(x-prev)/(dt*delta_frames);mass=np.fromfile(d/'masses.bin',dtype='<f8')
    fixed=np.fromfile(d/'boundary_types.bin',dtype='<i4')==1
    raw=np.fromfile(d/'topology.bin',dtype='<u4');nv,nf,nt=raw[:3];faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
    in_tet=np.zeros(nv,dtype=bool);in_tet[tets.ravel()]=True;cloth_faces=faces[~in_tet[faces].any(axis=1)]
    def tet_det(state):
        q=state[tets];return np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))
    jacobian=tet_det(x)/tet_det(x0) if len(tets) else np.array([1.])
    edges=np.sort(np.concatenate([cloth_faces[:,[0,1]],cloth_faces[:,[1,2]],cloth_faces[:,[2,0]]]),axis=1)
    edges=np.unique(edges,axis=0);rest=np.linalg.norm(x0[edges[:,0]]-x0[edges[:,1]],axis=1)
    stretch=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/rest if len(edges) else np.array([1.])
    moving=~fixed;kinetic=.5*np.sum(mass[moving]*np.sum(velocity[moving]**2,axis=1))
    momentum=np.sum(mass[moving,None]*velocity[moving],axis=0)
    trace_meta=json.loads((d/'metadata.json').read_text())
    abd_count=trace_meta.get('abd_point_num',0)
    body=np.fromfile(d/'body_ids.bin',dtype='<i4');affine=[]
    for body_id in np.unique(body[:abd_count]):
        ids=np.flatnonzero(body[:abd_count]==body_id)
        design=np.column_stack([np.ones(len(ids)),x0[ids]])
        fitted=design@np.linalg.lstsq(design,x[ids],rcond=None)[0]
        affine.append({'body_id':int(body_id),'vertices':len(ids),'max_affine_fit_error_m':float(np.linalg.norm(x[ids]-fitted,axis=1).max(initial=0))})
    last_frame=int(files[-1].stem[-4:])
    result={'run':run.name,'dt':dt,'time':last_frame*dt,'requested_time':metadata['steps']*dt,'state_frame':last_frame,
            'run_status':json.loads((run/'result.json').read_text())['status'],'finite':bool(np.isfinite(x).all()),
            'fixed_max_displacement_m':float(np.linalg.norm(x[fixed]-x0[fixed],axis=1).max(initial=0)),
            'free_ground_min_gap_m':float((x[moving,1]+1).min(initial=np.inf)),
            'cloth_edge_stretch_max':float(stretch.max()),'cloth_edge_stretch_p95':float(np.percentile(stretch,95)),
            'tet_relative_jacobian_min':float(jacobian.min()),'tet_final_nonpositive_jacobians':int((jacobian<=0).sum()),
            'kinetic_energy_from_position_differences_J':float(kinetic),'momentum_from_position_differences':momentum.tolist(),
            'abd_affine_geometry_checks':affine,
            'abd_check_scope':'initial-to-final affine geometry fit; internal q/dq state is not exported',
            'scope':'final geometry and backward-difference velocity; material/bending energy not computed'}
    if (d/'physical_energy.csv').exists():
        rows=list(csv.DictReader((d/'physical_energy.csv').open()))
        result['fem_cloth_bending_potential_J']={k:float(rows[-1][k]) for k in ['fem_potential','cloth_potential','bending_potential']}
        result['scope']='final geometry, backward-difference velocity, and original FEM/cloth/bending energy kernels; ABD energy excluded'
    return result,x,velocity,mass,moving,x0

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result,x,v,m,mask,x0=snapshot(a.run)
    if a.reference:
        ref,y,w,_,_,y0=snapshot(a.reference)
        if not np.array_equal(x0,y0) or abs(result['time']-ref['time'])>1e-10:raise ValueError('Initial states or physical windows differ')
        result['reference']=ref
        scale=float(np.linalg.norm(np.ptp(x0,axis=0)));rms=float(np.sqrt(np.average(np.sum((x[mask]-y[mask])**2,axis=1),weights=m[mask])))
        vrms=float(np.sqrt(np.average(np.sum((v[mask]-w[mask])**2,axis=1),weights=m[mask])))
        result.update(position_rms_free_m=rms,position_rms_over_scale=rms/scale,velocity_rms_free_m_s=vrms,
                      velocity_rms_over_scale_per_second=vrms/scale)
        reference_velocity=float(np.sqrt(np.average(np.sum(w[mask]**2,axis=1),weights=m[mask])))
        result['velocity_error_relative_reference']=vrms/max(1e-12,reference_velocity)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
