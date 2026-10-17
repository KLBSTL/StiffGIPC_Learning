"""Independent full-vector checks and adjacent default/user workspace pairs."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_workspace_apply import RAW,REPORT,BUILD,read
from local_identity import verify,tree
from linux_runner import require,write_new
from m3_numeric import FrozenSystem,rel
import numpy as np
from scipy.linalg import cho_solve

def csv_new(path,rows):
    require(rows and not path.exists(),'Empty/existing table');keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    folder=RAW/'analysis';require(not folder.exists(),'Analysis already exists');folder.mkdir()
    tables=REPORT/'tables';tables.mkdir();manifest=read(RAW/'input_manifest.json');packet=read(RAW/'runtime_seal.json')
    identity=read(BUILD/'identity.json')
    verify(identity['sources']+identity['build_evidence']+identity['runtime']+[identity['exe'],packet['identity'],packet['runner'],packet['manifest'],packet['protocol']]+packet['compiled']+packet['helpers']+manifest['original_inputs'])
    for mode in ['contracts','measure']:require(read(RAW/mode/'result.json')['passed'],'Owned run did not pass')
    contract=read(RAW/'contracts/contracts.json');measured=read(RAW/'measure/measurements.json')
    require(len(contract['cases'])==6 and len(measured['cases'])==4 and len(measured['samples'])==1296,'Incomplete result')
    expected=[]
    for case in manifest['cases']:
        if case['probe']=='zero':continue
        for rnd in range(3):
            for group in range(6):
                for order in range(9):
                    code=(order+group*5+rnd*3)%9
                    for position in range(2):
                        user=(rnd+group+order+position)%2
                        expected.append((case['name'],rnd+1,group+1,order+1,position+1,manifest['methods'][code//3],manifest['executions'][code%3],manifest['workspaces'][user]))
    fields=['name','round','group','order','position','method','execution','workspace']
    require(expected==[tuple(r[k] for k in fields) for r in measured['samples']],'Measured order differs')
    systems={};reference={};construction=[]
    for case in manifest['cases']:
        if case['scene'] not in systems:systems[case['scene']]=FrozenSystem(case['prefix'])
        s=systems[case['scene']];n=case['n'];Z=np.fromfile(case['Z'],dtype='<f8').reshape(n,8,order='F')
        W=np.fromfile(case['W'],dtype='<f8').reshape(n,8,order='F');L=np.fromfile(case['L'],dtype='<f8').reshape(8,8,order='F')
        r=np.fromfile(case['files']['r'],dtype='<f8');base=s.full_apply(r);a=cho_solve((L,True),Z.T@r)
        u=s.full_apply(r-W@a);value=u+Z@(a-cho_solve((L,True),W.T@u))
        require(rel(W,s.full_A@Z)<1e-9 and rel(L@L.T,.5*(Z.T@W+W.T@Z))<1e-9,'Original CSR/coarse check failed')
        require(rel(base,np.fromfile(case['files']['MAS_expected'],dtype='<f8'))<=1e-12
            and rel(value,np.fromfile(case['files']['balanced_expected'],dtype='<f8'))<=1e-12,'Parent CPU expectation differs')
        reference[case['name']]=dict(MAS=base,reference=value,fused=value)
        construction.append(dict(name=case['name'],n=n,CSR_W_relative=rel(W,s.full_A@Z),rhs_norm=float(np.linalg.norm(r))))
    audit=[];graphs=[];warm={};case_map={c['name']:c for c in manifest['cases']}
    def check(name,method,path,kind,**extra):
        v=np.fromfile(path,dtype='<f8');target=reference[name][method]
        require(v.shape==target.shape and np.isfinite(v).all(),'Missing/nonfinite output')
        error=rel(v,target);require(error<=1e-3,'Independent CPU action failed')
        if name.endswith('_zero'):require(np.count_nonzero(v)==0,'Zero contract nonzero')
        audit.append(dict(name=name,method=method,kind=kind,CPU_relative=error,output=path,**extra));return v
    for phase,data in [('contracts',contract),('measure',measured)]:
        for case in data['cases']:
            name=case['name'];require(len(case['warm'])==18 and len(case['graphs'])==12,'Missing warm/graphs')
            require(all(case[k] for k in ['restoration_passed','post_graph_exception_restored','workspace_guards_passed','default_pool_restored','same_handle_and_stream']),'Contract failed')
            require(case['workspace_bytes']==4*1024*1024 and case['guard_bytes_each']==256,'Workspace dimensions differ')
            require(len(case['faults'])==2 and all(r['restored'] for r in case['faults']),'Missing workspace fault path')
            for row in case['warm']:
                v=check(name,row['method'],row['output'],phase+'_warm',execution=row['execution'],workspace=row['workspace'])
                warm[(phase,name,row['method'],row['execution'],row['workspace'])]=v
                host=warm[(phase,name,row['method'],'host128','default')]
                require(rel(v,host)<=1e-3,'Workspace/Graph warm differs from default Host')
                if name.endswith('_rhs'):
                    old=np.fromfile(case_map[name]['historical_b'][row['method']],dtype='<f8');require(rel(v,old)<=1e-3,'Historical calibration failed')
            for row in case['graphs']:
                types=row['types_cuda_enum'];require(row['nodes']>0 and sum(types.values())==row['nodes'],'Graph evidence invalid')
                allocations=types.get('10',0);frees=types.get('11',0)
                target=0 if row['workspace']=='user4MiB' or row['method']=='MAS' else 2*row['calls']
                require(allocations==frees==target,'Node gate failed')
                graphs.append(dict(phase=phase,name=name,method=row['method'],workspace=row['workspace'],calls=row['calls'],
                    nodes=row['nodes'],allocations=allocations,frees=frees,kernels=types.get('0',0),
                    types=json.dumps(types,sort_keys=True),setup_ms=row['capture_instantiate_host_ms'],first_launch_host_ms=row['first_launch_host_ms']))
    timing=[];lookup={}
    for row in measured['samples']:
        require(row['calls']==128,'Wrong work count')
        require(all(np.isfinite(row[k]) and row[k]>=0 for k in ['host_ms','submit_ms','event_ms','select_ms','reset_ms','verify_export_ms']), 'Invalid timing')
        require(row['event_ms']>0 and row['host_ms']>=row['submit_ms'],'Timing windows inconsistent')
        v=check(row['name'],row['method'],row['output'],'measured',execution=row['execution'],workspace=row['workspace'],round=row['round'],group=row['group'])
        require(rel(v,warm[('measure',row['name'],row['method'],'host128','default')])<=1e-3,'Measured action differs')
        key=tuple(row[k] for k in ['name','round','group','method','execution','workspace']);require(key not in lookup,'Duplicate measured key');lookup[key]=row
        timing.append(dict(**row,event_us_per_call=row['event_ms']*1000/128,host_us_per_call=row['host_ms']*1000/128))
    pairs=[];aggregate=[];costs=[]
    for name,method,execution in dict.fromkeys((r['name'],r['method'],r['execution']) for r in timing):
        ratios=[];all_arm={}
        for rnd in range(1,4):
            for group in range(1,7):
                default=lookup[(name,rnd,group,method,execution,'default')];user=lookup[(name,rnd,group,method,execution,'user4MiB')]
                require(default['order']==user['order'] and {default['position'],user['position']}=={1,2},'Pair not adjacent')
                diff=rel(np.fromfile(user['output'],dtype='<f8'),np.fromfile(default['output'],dtype='<f8'));require(diff<=1e-3,'Independent paired workspace action differs')
                pair=dict(name=name,method=method,execution=execution,round=rnd,group=group,
                    event_ratio=user['event_ms']/default['event_ms'],host_ratio=user['host_ms']/default['host_ms'],
                    submit_ratio=user['submit_ms']/default['submit_ms'],output_relative=diff,user_position=user['position'])
                pairs.append(pair);ratios.append(pair)
        for workspace in manifest['workspaces']:
            rows=[r for r in timing if (r['name'],r['method'],r['execution'],r['workspace'])==(name,method,execution,workspace)]
            cost=dict(name=name,method=method,execution=execution,workspace=workspace,n=len(rows),
                **{k+'_median':float(np.median([r[k] for r in rows])) for k in ['event_us_per_call','host_us_per_call','submit_ms','select_ms','reset_ms','verify_export_ms']})
            costs.append(cost);all_arm[workspace]=cost
        rounds=[float(np.median([r['event_ratio'] for r in ratios if r['round']==rnd])) for rnd in range(1,4)]
        aggregate.append(dict(name=name,method=method,execution=execution,n=len(ratios),
            event_ratio_median=float(np.median([r['event_ratio'] for r in ratios])),host_ratio_median=float(np.median([r['host_ratio'] for r in ratios])),
            event_ratio_min=min(r['event_ratio'] for r in ratios),event_ratio_max=max(r['event_ratio'] for r in ratios),
            independent_event_medians_ratio=all_arm['user4MiB']['event_us_per_call_median']/all_arm['default']['event_us_per_call_median'],
            round1=rounds[0],round2=rounds[1],round3=rounds[2],improving_rounds=sum(r<1 for r in rounds)))
    gate=[r for r in aggregate if r['method']=='fused' and r['execution']=='graph1x128']
    controls=[r for r in aggregate if r['method']=='MAS' and r['execution']=='graph1x128']
    require(len(gate)==len(controls)==4,'Missing predefined gate/control')
    signal=all(r['event_ratio_median']<=.95 and r['improving_rounds']>=2 and r['independent_event_medians_ratio']<=1 for r in gate)
    control=all(.95<=r['event_ratio_median']<=1.05 for r in controls)
    csv_new(tables/'output_CPU.csv',audit);csv_new(tables/'timings.csv',timing);csv_new(tables/'pairs.csv',pairs)
    csv_new(tables/'aggregate.csv',aggregate);csv_new(tables/'costs.csv',costs);csv_new(tables/'graphs.csv',graphs)
    require(len(audit)==1476 and len(pairs)==648,'Wrong full-output/pair count')
    write_new(folder/'summary.json',dict(implementation_checks_passed=True,node_gate_passed=True,independently_checked_outputs=len(audit),
        measured_outputs=1296,contracts=108,warm=72,maximum_CPU_relative=max(r['CPU_relative'] for r in audit),
        maximum_paired_output_relative=max(r['output_relative'] for r in pairs),construction=construction,gate=gate,controls=controls,
        finite_cost_signal=signal,negative_control_passed=control,finite_performance_gate_passed=signal and control,
        aggregate=aggregate,costs=costs,graphs=graphs,performance_certified=False,production_solver_modified=False,
        inputs=tree(RAW/'contracts')+tree(RAW/'measure')))
    print(json.dumps(dict(numerics=True,outputs=1476,nodes=True,cost_signal=signal,negative_control=control,gate=gate)),flush=True)
if __name__=='__main__':main()
