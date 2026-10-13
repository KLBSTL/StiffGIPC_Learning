"""Diagnostic-only precision controls; never edit or rebuild native source."""
from pathlib import Path
import argparse
import json
import os
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
for folder in ('bench','diagnostic','local','full_eval'):
    sys.path.insert(0,str(ROOT/'tools'/folder))
from config import read,expand
from linux_runner import require,write_new
from local_identity import record,verify,verify_active_sources,tree


def cpu(out):
    import numpy as np
    from audit_contact_reference import cg
    from audit_recycling_reference import preconditioner
    from cpu_fixed_reference import load_snapshot,fnv1a64
    prefixes=[ROOT/'runs/deep_toi_native_allowed_20261009/linear113/events'/e for e in ('e1','e2')]
    prefixes.append(ROOT/'runs/audit_v3_adjacent_systems/fixed_ipc/fixed/f57_n1')
    reports=[]
    for prefix in prefixes:
        matrix,rhs,active,_,evidence,detail=load_snapshot(prefix)
        require(active.all(),'Zero rows require explicit treatment')
        meta=read(Path(str(prefix)+'_meta.json'))
        native,operator_evidence,operators=preconditioner(prefix,meta,with_data=True)
        offset,data=next((offset,data) for kind,offset,data in operators if kind=='mas')
        nodes,restriction,stored,fine,coarse=data
        owner=next(o for o in meta['local_preconditioners'] if o['kind']=='MAS_full_owned_buffers')
        path=Path(str(prefix)+'_mas_d_inverseMatMas.bin');raw=path.read_bytes()
        expected=owner['buffers']['d_inverseMatMas']
        require(len(raw)==expected['count']*expected['item_bytes'] and fnv1a64(raw)==expected['fnv1a64'],'Factor input mismatch')
        packed=np.frombuffer(raw,dtype='<f8').reshape(-1,136,3,3).transpose(0,1,3,2)
        h=np.zeros(stored.shape,dtype=np.float64)
        for i in range(16):
            for j in range(i,16):
                tile=packed[:,16*i-i*(i+1)//2+j]
                h[:,3*i:3*i+3,3*j:3*j+3]=tile
                if i!=j:h[:,3*j:3*j+3,3*i:3*i+3]=tile.transpose(0,2,1)
        diagonal=np.arange(48);padding=h[:,diagonal,diagonal]==0
        h[:,diagonal,diagonal]=np.where(padding,1,h[:,diagonal,diagonal])
        require(np.linalg.eigvalsh(h).min()>0,'Nonpositive local matrix')
        exact=np.linalg.inv(h)
        def action(mask,coefficients):
            rdtype=np.float64 if mask&1 else np.float32
            ldtype=np.float64 if mask&2 else np.float32
            pdtype=np.float64 if mask&4 else np.float32
            rmap=restriction.astype(rdtype);coeff=coefficients.astype(ldtype)
            def apply(vector):
                output=vector.copy()
                for kind,local_offset,local_data in operators:
                    if kind=='abd':
                        count=len(local_data)*12
                        output[local_offset:local_offset+count]=np.einsum('bij,bj->bi',local_data,
                            vector[local_offset:local_offset+count].reshape(-1,12)).ravel()
                r=rmap@vector[offset:offset+3*nodes].reshape(nodes,3).astype(rdtype)
                z=np.einsum('bij,bj->bi',coeff,r.astype(ldtype).reshape(-1,48)).reshape(-1,3)
                value=z[fine].astype(pdtype)
                for level in range(coarse.shape[1]):value+=z[coarse[:,level]].astype(pdtype)
                output[offset:offset+3*nodes]=value.ravel()
                return output
            return apply
        cases=[]
        labels={0:'legacy',1:'restrict64',2:'local64',4:'prolong64',3:'restrict_local64',
            5:'restrict_prolong64',6:'local_prolong64',7:'wide_apply64'}
        for mask in (0,1,2,4,3,5,6,7,8):
            name='inverse_storage_and_apply64' if mask==8 else labels[mask]
            apply=action(7,exact) if mask==8 else action(mask,stored)
            result=cg(matrix,rhs,apply,stop_apply=native)
            result['name']=name;cases.append({k:v for k,v in result.items() if k!='history'})
            print(json.dumps(dict(system=prefix.name,case=cases[-1])),flush=True)
        require(cases[0]['iterations']==cg(matrix,rhs,native,stop_apply=native)['iterations'],'Legacy surrogate differs')
        reports.append(dict(prefix=str(prefix),system=detail,evidence=evidence,operator_evidence=operator_evidence,
            local_factor_input=record(path),cases=cases))
    write_new(out/'analysis.json',dict(controller=record(__file__),systems=reports,
        scope='Same captured A,b, hierarchy and factors; stage casts isolated in CPU arithmetic. Common original MAS weighted stopping norm, previous-rho timing. Different CPU/CUDA reduction orders. No native timing or physics claim.',
        native_performance_certified=False,physical_quality_certified=False))


def native(out,identity_path):
    import windows_runner
    from toi_pcg_audit import summary
    from quality_analysis import input_comparison
    from pool_metrics import state_comparison
    identity=read(identity_path);verify_active_sources(identity)
    verify(identity['build_evidence']+identity.get('objects',[])+[identity['exe']]+identity['dlls'])
    source_before=tree(ROOT/'StiffGIPC')+[record(ROOT/'CMakeLists.txt')]
    controller=record(__file__);tasks=[]
    for index,arm in enumerate(('legacy','wide','inverse64','inverse64','wide','legacy'),1):
        config=expand(dict(preset='combined',scene='cloth_fixed_bunny_l',steps=120,timeout_seconds=180,
            mas='legacy',fem_direct_pullback=True,mas_pivot_only_sync=True,discrete_bvh_refit=True))
        extra=dict(GIPC_IPC_PCG_FORCING='off',GIPC_MAS_WIDE_APPLY=str(int(arm=='wide')),
            GIPC_MAS_INVERSE64=str(int(arm=='inverse64')))
        tasks.append(dict(name=f'{index}_{arm}',arm=arm,binary='active',config=config,extra_environment=extra))
    write_new(out/'identity.json',dict(program=identity,controller=controller,tasks=tasks,
        native_source_before=source_before,scope='Existing precision flags only; current IPC forcing off; no native changes. Complete 120-frame IPC scene; paired order. Shared WDDM timing diagnostic.'))
    ledger=[];original=windows_runner.environment;reference=None
    original_validator=windows_runner.validate
    def precision_validator(path):
        # The portable schema supports legacy/cholesky only. Retain every check;
        # bind these three precision expectations to explicitly recorded native
        # flags rather than its legacy-only defaults. Do not relax any tolerance.
        check=original_validator(path);request=read(path/'requested.json')
        require(request['expanded_config']['mas']=='legacy','Precision overlay needs legacy hierarchy')
        env=request['environment'];inverse=env['GIPC_MAS_INVERSE64']=='1'
        expected={'mas.inverse64':inverse,
            'mas.wide_apply':inverse or env['GIPC_MAS_WIDE_APPLY']=='1',
            'mas.pivot_only_sync_effective':env['GIPC_MAS_PIVOT_ONLY_SYNC']=='1' and not inverse}
        for name,value in expected.items():
            records=[r for r in check['checks'] if r['check']==name]
            require(len(records)==1,'Missing/duplicate precision contract')
            row=records[0];row['standard_expected']=row['expected'];row['expected']=value
            row['passed']=row['actual']==value
        check['passed']=all(r['passed'] for r in check['checks'])
        check['precision_expectations_from_explicit_flags']=True
        return check
    with windows_runner.gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
        for task in tasks:
            def environment(config,path,extra=task['extra_environment']):
                return dict(original(config,path),**extra)
            windows_runner.environment=environment
            windows_runner.validate=precision_validator
            try:result=windows_runner.execute(out,task,identity,0)
            finally:
                windows_runner.environment=original
                windows_runner.validate=original_validator
            row=dict(task=task,result=result);ledger.append(row)
            if result['status']!='completed':break
            try:
                row['summary']=summary(out/task['name'],result)
                resolved=read(out/task['name']/'resolved_config.json')
                require(resolved['ipc_pcg_forcing']['mode']=='off','Adaptive tolerance was not disabled')
                require(resolved['mas']['wide_apply']==(task['arm']!='legacy') and
                    resolved['mas']['inverse64']==(task['arm']=='inverse64'),'Effective precision differs')
                row['effective_precision']=resolved['mas']
                if reference is None:reference=out/task['name']
                row['initial_input']=input_comparison(reference,out/task['name'])
                require(row['initial_input']['passed'],'Initial input mismatch')
                row['trajectory']=state_comparison(reference,out/task['name'])
            except Exception as error:
                row['analysis_error']=str(error);break
            print(json.dumps(dict(name=task['name'],summary=row['summary'])),flush=True)
    verify(source_before+[controller])
    write_new(out/'ledger.json',dict(runs=ledger,skipped=tasks[len(ledger):],native_source_unchanged=True))
    return 0 if len(ledger)==len(tasks) and all('summary' in r and 'analysis_error' not in r for r in ledger) else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('cpu','native'));parser.add_argument('--session',required=True)
    parser.add_argument('--identity',type=Path)
    args=parser.parse_args();require(args.session.replace('_','').isalnum(),'Invalid session')
    out=ROOT/'runs'/args.session;require(not out.exists(),'Fresh directory required');out.mkdir(parents=True)
    (out/'controller_at_run.py').write_bytes(Path(__file__).read_bytes())
    if args.action=='cpu':cpu(out);return 0
    require(args.identity is not None,'Frozen identity required')
    return native(out,args.identity)


if __name__=='__main__':raise SystemExit(main())
