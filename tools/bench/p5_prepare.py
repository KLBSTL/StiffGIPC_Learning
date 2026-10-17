"""Audit sealed M3 and prepare one immutable Mixed pair for a GPU cost probe."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import json
from pathlib import Path
import time

import numpy as np

from m3_numeric import ROOT,FrozenSystem,pcg,pod_basis,orthogonal,allow_transfer,rel,Balanced
from local_identity import record,verify
from linux_runner import write_new,require

OLD=ROOT/'reports/innovation_m3_20261015'
REPORT=ROOT/'reports/innovation_p5_gpu_20261015'
OUTPUT=ROOT/'runs/innovation_p5_gpu_20261015/inputs'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def replay_iterates(system,trace,solution):
    """Use saved source recurrence only; no target arrays enter this function."""
    x=np.zeros_like(system.b);r=system.b.copy();z=system.apply(r);p=z.copy()
    all_x=[];slots=[];rng=np.random.default_rng(20261015);writes=[]
    for i,item in enumerate(trace):
        all_x.append(x.copy())
        if i<128:
            slots.append(i);writes.append((i,i))
        else:
            j=int(rng.integers(0,i-16+1))
            if j<112:
                slots[16+j]=i;writes.append((i,16+j))
        ap=system.A@p;alpha=item['alpha'];x+=alpha*p;r-=alpha*ap
        if i+1<len(trace):
            z=system.apply(r);p=z+(trace[i+1]['rho']/item['rho'])*p
    require(np.array_equal(x,solution),'Independent source recurrence failed bitwise replay')
    return np.column_stack(all_x),np.asarray(slots,dtype=np.int32),np.asarray(writes,dtype=np.int32)


def main():
    require(not OUTPUT.exists(),'Fresh GPU input required');OUTPUT.mkdir(parents=True)
    start=time.perf_counter();old_seal=read(OLD/'DELIVERY_SEAL.json');verify(old_seal['files'])
    source_prefix=ROOT/'runs/innovation_m3_20261015/adjacent_mixed/fixed/f15_n1'
    target_prefix=ROOT/'runs/innovation_m3_20261015/adjacent_mixed/fixed/f15_n2'
    source=FrozenSystem(source_prefix);target=FrozenSystem(target_prefix)
    allow_transfer(source.meta,target.meta)
    require(np.array_equal(source.active,target.active) and np.array_equal(source.soft,target.soft)
            and np.array_equal(source.abd,target.abd),'Source/target coordinate layout changed')
    require(source.active.all() and target.active.all(),'Probe ABI requires no eliminated coordinates')
    old_case=read(OLD/'analysis_v2/Mixed/adjacent_f15_n1/result.json')
    old_packet=read(old_case['source_packet']['path']);verify([old_packet['bases']])
    result,x,history=pcg(source.A,source.b,source.apply,source.apply,collect=True)
    require(result['converged'] and result['residual_restarts']==0 and result['iterations']==137,
            'Source ordinary-accuracy recurrence changed')
    saved=np.load(old_case['methods'][0]['solution']['path'],allow_pickle=False)
    require(np.array_equal(x,saved),'Recomputed source solution differs from sealed M3')
    X,slot_iterations,writes=replay_iterates(source,result['trace'],x)
    history_indices=np.array([h[0] for h in history]);require(np.array_equal(np.sort(slot_iterations),history_indices),
                                                            'Reservoir replay selection changed')
    for item in history:
        require(np.array_equal(X[:,item[0]],item[3]),'Reservoir history iterate mismatch')
    raw,info=pod_basis(source.A,history,x)
    with np.load(old_packet['bases']['path'],allow_pickle=False) as archive:
        old_pod=archive['pod'].copy()
    subspaces=[]
    for rank in (8,16):
        z,_=orthogonal(raw[:,:rank]);old_z,_=orthogonal(old_pod[:,:rank])
        error=rel(z,old_z@(old_z.T@z));require(error<1e-10,'Sealed source POD space changed')
        subspaces.append(dict(rank=rank,source_projection_error=error))
    # S before/after the reservoir's deterministic column permutation gives
    # exactly the same A-energy covariance. Keep physical bank order for GPU.
    S=x[:,None]-X[:,slot_iterations]
    sorted_S=x[:,None]-np.column_stack([h[3] for h in history])
    require(np.array_equal(S[:,np.argsort(slot_iterations)],sorted_S),'GPU history-bank permutation is wrong')
    records=[]
    def output(name,value,dtype='<f8',order='F'):
        path=OUTPUT/(name+'.bin');array=np.asarray(value,dtype=dtype)
        path.write_bytes(array.ravel(order=order).tobytes())
        records.append(record(path));return dict(file=path.name,shape=list(array.shape),dtype=dtype,order=order)
    arrays={}
    def put(name,*args,**kwargs):arrays[name]=output(name,*args,**kwargs)
    for label,system in (('source',source),('target',target)):
        put(label+'_rows',system.A.indptr,'<i4');put(label+'_cols',system.A.indices,'<i4')
        put(label+'_values',system.A.data);put(label+'_rhs',system.b)
    put('iterates',X);put('final_source_x',x);put('history_writes',writes,'<i4',order='C')
    put('expected_history',X[:,slot_iterations]);put('expected_S',S)
    require(len(target.ops)==2,'Probe supports one MAS and one ABD component only')
    abd=[v for v in target.ops if v[0]=='abd'];mas=[v for v in target.ops if v[0]=='mas']
    require(len(abd)==len(mas)==1,'Probe requires one local operator of each kind')
    _,abd_offset,inverse=abd[0];_,mas_offset,(nodes,R,D)=mas[0]
    require(len(inverse)==1,'Mixed probe must have one 12-DOF ABD')
    Rt=R.T.tocsr()
    for name,matrix in (('R',R),('RT',Rt)):
        put(name+'_rows',matrix.indptr,'<i4');put(name+'_cols',matrix.indices,'<i4')
        put(name+'_values',matrix.data)
    # Each symmetric dense inverse is stored column-major individually.
    put('mas_inverse',D.transpose(0,2,1),order='C')
    put('abd_inverse',inverse[0])
    rng=np.random.default_rng(20261015);probes=np.column_stack([target.b,rng.normal(size=len(x)),rng.normal(size=len(x))])
    put('probes',probes)
    for rank in (8,16):
        z,_=orthogonal(raw[:,:rank]);op=Balanced(target.A,target.apply,z)
        put(f'expected_Z_{rank}',op.Z);put(f'expected_W_{rank}',op.W)
        put(f'expected_B_{rank}',np.column_stack([op(v) for v in probes.T]))
    put('expected_MAS',np.column_stack([target.apply(v) for v in probes.T]))
    write_new(OUTPUT/'manifest.json',dict(schema='p5_pod_gpu_input.v1',arrays=arrays,records=records,
        n=len(x),source_nnz=source.A.nnz,target_nnz=target.A.nnz,history_width=len(history),
        source_iterations=result['iterations'],history_copy_calls=len(writes),ranks=[8,16],
        mas_nodes=nodes,mas_rows=R.shape[0],mas_blocks=len(D),mas_offset=mas_offset,abd_offset=abd_offset,
        source_context={k:source.meta[k] for k in ('frame','inner','linear_system_id')},
        target_context={k:target.meta[k] for k in ('frame','inner','linear_system_id')},
        source_tolerance=.01,target_solution_used=False,
        scope='One Mixed pair; coefficient FP64 linear MAS reference; no native PCG/GPU integration',
        parent=record(OLD/'DELIVERY_SEAL.json')))
    write_new(REPORT/'REVIEW.json',dict(passed=True,source=record(Path(__file__)),parent=record(OLD/'DELIVERY_SEAL.json'),
        parent_files_verified=len(old_seal['files']),source_solution_bitwise_equal=True,
        source_recurrence_bitwise_equal=True,reservoir_bitwise_equal=True,POD_subspaces=subspaces,
        common_residual=result['relative_l2'],common_MAS_relative=result['common_MAS_relative'],
        input_manifest=record(OUTPUT/'manifest.json'),seconds=time.perf_counter()-start,
        native_solver_modified=False,implementation_errors_found_in_exercised_M3_path=False,
        scope='This is evidence for exercised paths; not proof that all possible M3 inputs are supported'))
    print(json.dumps(dict(review_passed=True,history=len(history),copies=len(writes),subspaces=subspaces,
                          input_bytes=sum(r['bytes'] for r in records))),flush=True)


if __name__=='__main__':main()
