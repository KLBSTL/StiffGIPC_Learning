"""M0 exact data audit and bounded quality diagnostics; never certify speed or physics."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'/name) for name in ('local','bench')]
from config import read
from m0_offline import record,require
from cpu_fixed_reference import load_snapshot
from audit_recycling_reference import preconditioner


def state(folder,frame,velocity=False):
    name='velocity' if velocity else 'state'
    value=np.fromfile(folder/f'trace/{name}_{frame:04d}.bin',dtype='<f8')
    require(value.size>0 and value.size%3==0 and np.isfinite(value).all(),'Invalid real state/velocity export')
    return value.reshape(-1,3)


def layout(folder):
    raw=np.fromfile(folder/'trace/topology.bin',dtype='<u4')
    require(len(raw)>=3,'Missing topology header')
    nv,nf,nt=map(int,raw[:3])
    require(len(raw)==3+3*nf+4*nt,'Topology size mismatch')
    faces=raw[3:3+3*nf].reshape(nf,3);tets=raw[3+3*nf:].reshape(nt,4)
    require((not faces.size or faces.max()<nv) and (not tets.size or tets.max()<nv),'Topology index outside vertices')
    boundary=np.fromfile(folder/'trace/boundary_types.bin',dtype='<i4')
    require(len(boundary)==nv,'Boundary count differs')
    abd=read(folder/'trace/metadata.json')['abd_point_num']
    require(0<=abd<=nv,'ABD range differs')
    # Fixed ABD bodies may keep point BoundaryType==Free; their body boundary
    # is a separate native constraint. Do not dilute free-vertex RMS with them.
    fixed=boundary!=0
    abd_objects=[o for o in read(folder/'output/scene.json')['objects'] if o['body_type']=='ABD']
    if abd:
        require(abd_objects,'ABD metadata without corresponding scene objects')
        body_fixed={o.get('fixed_mode','none')=='all' for o in abd_objects}
        require(len(body_fixed)==1,'Mixed fixed/free ABD requires exported body boundary mapping')
        fixed[:abd]=next(iter(body_fixed))
    volumetric=np.zeros(nv,bool);volumetric[tets.ravel()]=True
    masks=dict(free=~fixed,cloth_dynamic=~volumetric&(np.arange(nv)>=abd)&~fixed,
        fem=volumetric&(np.arange(nv)>=abd),abd=np.arange(nv)<abd,fixed=fixed)
    return nv,faces,tets,boundary,abd,masks


def volumes(x,tets):
    q=x[tets]
    return np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))/6


def jacobians(current,rest,tets):
    denominator=volumes(rest,tets)
    require(np.isfinite(denominator).all() and np.all(denominator!=0),'Degenerate rest tetrahedron')
    value=volumes(current,tets)/denominator
    require(np.isfinite(value).all(),'Nonfinite geometric Jacobian')
    return value


def geometry(folder,rest_folder,frames,offset=0):
    nv,faces,tets,boundary,abd,masks=layout(folder)
    require((folder/'trace/topology.bin').read_bytes()==(rest_folder/'trace/topology.bin').read_bytes(),
        'Rest/current topology differs')
    rest=state(rest_folder,0)
    require(rest.shape==(nv,3),'Rest positions count differs')
    volumetric=np.zeros(nv,bool);volumetric[tets.ravel()]=True
    cloth_faces=faces[~volumetric[faces].any(axis=1)]
    edges=np.unique(np.sort(np.concatenate([cloth_faces[:,[0,1]],cloth_faces[:,[1,2]],cloth_faces[:,[2,0]]]),axis=1),axis=0)
    lengths=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
    require(np.all(lengths>0),'Zero rest cloth edge')
    fem_tets=np.all(tets>=abd,axis=1)
    result=[]
    for f in range(1,frames+1):
        x=state(folder,f);require(x.shape==rest.shape,'Position count differs')
        stretch=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/lengths
        j=jacobians(x,rest,tets) if len(tets) else np.empty(0)
        result.append(dict(local_frame=f,physical_frame=f+offset,
            max_cloth_stretch=float(stretch.max()) if len(stretch) else None,
            p99_cloth_stretch=float(np.quantile(stretch,.99)) if len(stretch) else None,
            fixed_drift_m=float(np.linalg.norm(x[masks['fixed']]-rest[masks['fixed']],axis=1).max()) if masks['fixed'].any() else None,
            fem_min_J=float(j[fem_tets].min()) if fem_tets.any() else None,
            fem_nonpositive=int((j[fem_tets]<=0).sum()),rest_reference=record(rest_folder/'trace/state_0000.bin')))
    return result


def compare(a,b,fa,fb):
    require((a/'trace/topology.bin').read_bytes()==(b/'trace/topology.bin').read_bytes(),'Comparison topology differs')
    nv,_,_,_,_,masks=layout(a)
    result={}
    for kind,velocity in [('position',False),('actual_velocity',True)]:
        if velocity and not ((a/f'trace/velocity_{fa:04d}.bin').is_file() and (b/f'trace/velocity_{fb:04d}.bin').is_file()):
            result[kind]=dict(available=False,reason='No actual velocity export in frozen B0')
            continue
        x,y=state(a,fa,velocity),state(b,fb,velocity)
        require(x.shape==y.shape==(nv,3),'Comparison state sizes differ')
        distance=np.linalg.norm(x-y,axis=1)
        scale=1000 if not velocity else 1
        result[kind]=dict(available=True,unit='mm' if not velocity else 'm/s',
            partitions={name:dict(count=int(mask.sum()),rms=scale*float(np.sqrt(np.mean(distance[mask]**2))),
                max=scale*float(distance[mask].max())) for name,mask in masks.items() if mask.any()})
    return result


def fixed_reference():
    folder=ROOT/'runs/innovation_m0_fixed_20261014/fixed_bunny_f14/fixed'
    files=list(folder.glob('*_study.json'));require(len(files)==1,'Missing unique matrix capture')
    file=files[0];prefix=Path(str(file)[:-len('_study.json')]);study=read(file)
    require(study['system_unchanged'] and study['primary_restored_bitwise'],'Diagnostic mutated production state')
    matrix,rhs,active,zero,evidence,detail=load_snapshot(prefix)
    apply,operator_evidence=preconditioner(prefix,read(Path(str(prefix)+'_meta.json')))
    denominator=float(rhs@apply(rhs));require(denominator>0,'Nonpositive common reference norm')
    rows=[]
    for phase,rho in [(1,1e-4),(3,1e-16)]:
        for mode,label in [(0,'host'),(1,'graph')]:
            p=Path(str(prefix)+f'_p{phase}_m{mode}_x.bin')
            x=np.fromfile(p,dtype='<f8')
            require(x.shape==rhs.shape and np.isfinite(x).all(),'Invalid native saved solution')
            residual=rhs-matrix@x
            relative=float(np.linalg.norm(residual)/max(np.linalg.norm(rhs),np.finfo(float).tiny))
            native=[r for r in study['runs'] if r['rho_tolerance']==rho and r['mode']==label]
            require(len(native)==2 and all(not r['limit'] and not r['error'] for r in native),'Failed/missing matrix solves')
            require(math.isclose(relative,native[0]['true_relative_residual'],rel_tol=1e-5,abs_tol=1e-10),
                'CPU full-A residual differs from independent native probe')
            weighted=float(residual@apply(residual));require(weighted>=0,'Negative weighted true residual')
            rows.append(dict(rho_tolerance=rho,mode=label,iterations=[r['iterations'] for r in native],
                true_L2_relative=relative,true_common_weighted_relative=math.sqrt(weighted/denominator),
                solve_ms=[r['solve_ms'] for r in native],solution=record(p)))
    return dict(system=detail,input_evidence=evidence,preconditioner_evidence=operator_evidence,rows=rows,
        cpu_native_residual_crosscheck_passed=True,
        scope='Full saved A; fixed reference B from saved FP32 MAS coefficients. CPU reduction order differs from GPU. Not an error bound on physical state.')


def csv_write(path,rows):
    require(bool(rows),'Empty output table')
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/innovation_m0_20261014/analysis')
    args=parser.parse_args();require(not args.output.exists(),'Fresh analysis output required')
    sessions=[ROOT/'runs/innovation_m0_20261014',ROOT/'runs/innovation_m0_boxes_hard_20261014']
    runs=[];quality=[];origins=[];shadow=[];inputs=[]
    archived_runner=record(sessions[0]/'windows_runner_at_run.py')
    for session in sessions:
        packet=read(session/'seal.json')
        for stage in ('smoke','prefix','matched','reference'):
            ledger=session/(stage+'_ledger.json')
            if not ledger.exists():continue
            data=read(ledger);require(data['complete'],'Incomplete native stage')
            inputs.append(record(ledger))
            if 'runs' not in data:
                require(stage=='smoke' and session==sessions[1] and
                    data['inherited_same_binary_smoke']==record(sessions[0]/'smoke_ledger.json'),
                    'Missing run list without verified inherited smoke')
                continue
            for row in data['runs']:
                t=row['task'];folder=session/t['name'];req=read(folder/'requested.json');result=read(folder/'result.json')
                require(result['status']=='completed' and result['cleanup_owned_job_empty'],'Run failed or Job not empty')
                require(req['exe_sha256']==packet['programs'][t['binary']]['exe']['sha256'],'Wrong native executable')
                expected_frames=t['config']['steps'];require(result['recorded_frames']==expected_frames,'Frame coverage differs')
                frame_context=read(folder/'trace/frame_context.json') if t['binary']=='active' else dict(completed_physical_frames_at_start=0)
                offset=frame_context['completed_physical_frames_at_start']
                if session==sessions[0]:require(req['runner_sha256']==archived_runner['sha256'],'Primary runner source archive differs')
                actual_origin=offset==0
                origins.append(dict(run=t['name'],requested_from_zero=req['from_zero'],actual_from_zero=actual_origin,
                    physical_frame_offset=offset,legacy_metadata_disagrees=req['from_zero']!=actual_origin))
                power=read(folder/'power_samples.json')
                require(power['after']==packet['power_guid'] and all(s['guid']==packet['power_guid'] for s in power['samples']),
                    'Observed power scheme differs')
                observation=row['observation']
                runs.append(dict(session=session.name,run=t['name'],stage=stage,scene=t['scene'],binary=t['binary'],
                    rule=t.get('rule_label',t.get('reference_label',t['config']['ipc_termination'])),
                    repeat=t.get('repeat',1),steps=expected_frames,target_frame=t.get('target_frame'),
                    N=observation['N'],P=observation['P'],solver_ms=observation['solver_ms'],wall_s=result['wall_seconds'],
                    rho=t['config']['pcg_rho_tol'],outer_tol=t['config']['ipc_newton_tol'],
                    **{p+'_ms':observation['phase_ms'][p] for p in ('assembly','pcg','ccd','line_search','state_update')}))
                rest_folder=folder if stage!='matched' else session/t['checkpoint_from'].split('/')[0]
                quality.extend(dict(run=t['name'],scene=t['scene'],**v) for v in geometry(folder,rest_folder,expected_frames,offset))
                if stage=='matched':
                    frames=read(folder/'output/stats.json')['frames']
                    obs=[n['ipc_residual'] for f in frames for n in f['newton'] if 'ipc_residual' in n]
                    require(all(not o['compensated_exit_allowed'] or o['gated_exit_allowed'] for o in obs),
                        'Compensated exit without direct gate')
                    direct=[o['accepted_updates'] for o in obs if o['gated_exit_allowed']]
                    comp=[o['accepted_updates'] for o in obs if o['compensated_exit_allowed']]
                    shadow.append(dict(run=t['name'],scene=t['scene'],rule=t.get('rule_label'),
                        physical_frame=t['target_frame'],first_direct_allowed=min(direct) if direct else None,
                        first_compensated_allowed=min(comp) if comp else None,
                        history_exit_vetoes=sum(bool(o['history_exit_veto']) for o in obs),
                        common_prefix_scope='Both readiness predicates observed on this single actual accepted prefix',
                        no_compensated_earlier_than_direct=True))
    require(len(runs)==60 and sum(r['steps'] for r in runs)==256,'Incomplete primary/refinement matrix')
    # Quantify cold reference repeat noise and compare every candidate against
    # both stricter repeats. All share the exact restored physical input.
    matched=[]
    repeat_dispersion=[]
    for session in sessions:
        ledger=read(session/'matched_ledger.json')['runs']
        for scene in sorted({r['task']['scene'] for r in ledger}):
            chosen=[r for r in ledger if r['task']['scene']==scene]
            refs=[r for r in chosen if r['task']['rule_label']=='strict']
            require(len(refs)==2,'Two reference repeats required')
            repeat_dispersion.append(dict(scene=scene,session=session.name,kind='cold_inner_only_reference',
                comparison=compare(session/refs[0]['task']['name'],session/refs[1]['task']['name'],1,1)))
            for row in chosen:
                t=row['task'];folder=session/t['name']
                matched.append(dict(scene=scene,session=session.name,run=t['name'],rule=t['rule_label'],repeat=t['repeat'],
                    target_frame=t['target_frame'],comparisons=[compare(folder,session/r['task']['name'],1,1) for r in refs],
                    reference_scope='rho1e-6, same outer .01; a numerical reference, not exact physics'))
    parent=sessions[0]
    ref_ledger=read(parent/'reference_ledger.json')['runs']
    outer=[]
    for label in ('nominal','inner','inner_outer','tighter'):
        chosen=[r for r in ref_ledger if r['task']['reference_label']==label]
        require(len(chosen)==2,'Missing outer calibration repeats')
        dispersion=compare(parent/chosen[0]['task']['name'],parent/chosen[1]['task']['name'],6,6)
        tight=[r for r in ref_ledger if r['task']['reference_label']=='tighter']
        outer.append(dict(label=label,repeat_difference=dispersion,
            against_tighter=[compare(parent/a['task']['name'],parent/b['task']['name'],6,6) for a,b in itertools.product(chosen,tight)],
            scope='Same pristine scene, physical time .24s; preliminary reference convergence, no checkpoint context bypass'))
    baseline_pairs=[]
    for scene in ('boxes','mixed','bunny'):
        a=parent/f'prefix_{scene}_base';b=parent/f'prefix_{scene}_legacy'
        equality={name:(a/'trace'/name).read_bytes()==(b/'trace'/name).read_bytes()
            for name in ('topology.bin','state_0000.bin','masses.bin','boundary_types.bin','body_ids.bin')}
        require(all(equality.values()),'B0/B1 initial problem differs')
        count=next(r['steps'] for r in runs if r['run']==f'prefix_{scene}_base')
        baseline_pairs.append(dict(scene=scene,initial_problem_bytes_equal=equality,
            end_state=compare(a,b,count,count),speedup_certified=False))
    fixed=fixed_reference()
    args.output.mkdir(parents=True)
    csv_write(args.output/'local_runs.csv',runs)
    # Serialize nested rest identities separately instead of writing dict repr.
    csv_write(args.output/'geometric_quality.csv',[{k:v for k,v in q.items() if k!='rest_reference'} for q in quality])
    csv_write(args.output/'origin_audit.csv',origins)
    csv_write(args.output/'same_prefix_gate_audit.csv',shadow)
    report=dict(schema='innovation_m0_analysis.v1',controller=record(Path(__file__)),input_ledgers=inputs,
        native_runs=61,native_frames=257,ordinary_matrix_runs=len(runs),ordinary_matrix_frames=256,
        fixed_capture_gate=read(ROOT/'runs/innovation_m0_fixed_20261014/capture_gate.json'),
        archived_primary_runner=archived_runner,initial_problem_comparisons=baseline_pairs,
        matched_comparisons=matched,reference_repeat_dispersion=repeat_dispersion,outer_calibration=outer,
        fixed_system=fixed,geometry_rest_scope='J and stretch use pristine scene rest positions, including for cold restored states',
        legacy_origin_metadata_mismatches=sum(r['legacy_metadata_disagrees'] for r in origins),
        corrected_origin_integration=read(sessions[1]/'origin_metadata_gate.json'),
        current_res_scope='Maximum absolute raw ABD generalized-gradient scalar and free-FEM gradient component, normalized by the frozen reference. Not mass-normalized physical residual or velocity error.',
        quality_protocol=dict(status='bounded calibration completed; global numerical/physical certification pending',
            execution_candidates='Require accepted/failed branch agreement, complete typed contributions and preserved full CCD; no added physical deformation.',
            numerical_candidates='Use common explicit L2/B_ref linear residual and independently calibrated outer solution. Reference dispersion must be below the accuracy to be distinguished.',
            mixed='SNK1 has finite polynomial energy for negative J and its native IPC injective step is commented out. Nonpositive J is real geometric inversion; physical acceptability is not certified.',
            collision='Native FullCCD retained; independent path CCD certification still pending'),
        performance_certified=False,physical_quality_certified=False,all_native_jobs_closed=True)
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(analysis_complete=True,native_runs=61,native_frames=257,
        initial_problem_equality=True,legacy_origin_metadata_mismatches=report['legacy_origin_metadata_mismatches'],
        fixed_residual_crosscheck=True,performance_certified=False)))


if __name__=='__main__':main()
