"""Independent full-output and matched-pair audit of native full-apply replay."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_full_apply_graph import RAW,REPORT,BUILD,read
from local_identity import record,verify,tree
from linux_runner import require,write_new
from m3_numeric import FrozenSystem,rel
import numpy as np
from scipy.linalg import cho_solve

def csv_new(path,rows):
    require(rows and not path.exists(),'Empty/existing table');keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def main():
    folder=RAW/'analysis';require(not folder.exists(),'Analysis already exists');folder.mkdir()
    tables=REPORT/'tables';tables.mkdir()
    manifest=read(RAW/'input_manifest.json');packet=read(RAW/'runtime_seal.json');identity=read(BUILD/'identity.json')
    verify(manifest['original_inputs']+manifest['generated_inputs']+identity['sources']+[packet['identity'],packet['runner'],packet['manifest']]+packet['compiled']+packet['helpers'])
    for mode in ['contracts','measure']:require(read(RAW/mode/'result.json')['passed'],'Owned run did not pass')
    contract=read(RAW/'contracts/contracts.json');measured=read(RAW/'measure/measurements.json')
    require(len(contract['cases'])==6 and len(measured['cases'])==4 and len(measured['samples'])==648,'Incomplete result')
    expected=[]
    for case in manifest['cases']:
        if case['probe']=='zero':continue
        for rnd in range(3):
            for group in range(6):
                for order in range(9):
                    code=(order+group*5+rnd*3)%9
                    expected.append((case['name'],rnd+1,group+1,order+1,manifest['methods'][code//3],manifest['executions'][code%3]))
    observed=[tuple(r[k] for k in ['name','round','group','order','method','execution']) for r in measured['samples']]
    require(expected==observed,'Measured ordering differs from protocol')
    systems={};reference={};construction=[]
    for case in manifest['cases']:
        key=case['scene']
        if key not in systems:systems[key]=FrozenSystem(case['prefix'])
        s=systems[key];n=case['n'];Z=np.fromfile(case['Z'],dtype='<f8').reshape(n,8,order='F')
        W=np.fromfile(case['W'],dtype='<f8').reshape(n,8,order='F');L=np.fromfile(case['L'],dtype='<f8').reshape(8,8,order='F')
        r=np.fromfile(case['files']['r'],dtype='<f8');base=s.full_apply(r);a=cho_solve((L,True),Z.T@r)
        u=s.full_apply(r-W@a);value=u+Z@(a-cho_solve((L,True),W.T@u))
        require(rel(W,s.full_A@Z)<1e-9,'W differs from original CSR')
        for field,array in [('MAS_expected',base),('balanced_expected',value)]:
            saved=np.fromfile(case['files'][field],dtype='<f8');require(rel(saved,array)<=1e-12,'Saved expected differs from independent reconstruction')
        reference[case['name']]=dict(MAS=base,reference=value,fused=value)
        construction.append(dict(name=case['name'],n=n,CSR_W_relative=rel(W,s.full_A@Z),
            CPU_reference='stored FP32 local coefficients symmetrized, applied in FP64',rhs_norm=float(np.linalg.norm(r))))
    audits=[];graphs=[];profiles=[];warm_outputs={}
    def check(name,method,path,kind,**extra):
        v=np.fromfile(path,dtype='<f8');target=reference[name][method]
        require(v.shape==target.shape and np.isfinite(v).all(),'Output missing/nonfinite')
        error=rel(v,target);require(error<=1e-3,'Independent CPU action failed')
        audits.append(dict(name=name,kind=kind,method=method,CPU_relative=error,output=path,**extra));return v
    for phase,data in [('contracts',contract),('measure',measured)]:
        for case in data['cases']:
            require(case['restoration_passed'] and case['post_graph_exception_restored'],'State restoration failed')
            name=case['name'];require(len(case['warm'])==9 and len(case['graphs'])==6,'Missing warm or graph')
            for row in case['warm']:
                v=check(name,row['method'],row['output'],phase+'_warm',execution=row['execution'])
                warm_outputs[(phase,name,row['method'],row['execution'])]=v
                if row['execution']!='host16':
                    h=warm_outputs[(phase,name,row['method'],'host16')]
                    require(rel(v,h)<=1e-3,'Independent Host/Graph calibration failed')
                if name.endswith('_rhs'):
                    source=next(c for c in manifest['cases'] if c['name']==name)
                    old=np.fromfile(source['historical_b'][row['method']],dtype='<f8');require(rel(v,old)<=1e-3,'Historical native calibration failed')
            for row in case['graphs']:
                require(row['nodes']>0 and sum(row['types_cuda_enum'].values())==row['nodes'],'Invalid graph evidence')
                graphs.append(dict(phase=phase,name=name,method=row['method'],calls=row['calls'],nodes=row['nodes'],
                    types_cuda_enum=json.dumps(row['types_cuda_enum'],sort_keys=True),
                    setup_ms=row['capture_instantiate_host_ms'],first_launch_host_ms=row['first_launch_host_ms']))
            for profile in case['stage_profiles']:
                require(len(profile['stages'])==5,'Missing balanced stage')
                profiles.extend(dict(name=name,method=profile['method'],**stage) for stage in profile['stages'])
            if phase=='measure':
                p=case['instrumented_MAS_internal'];require(len(p['stages'])==6,'Missing MAS stage')
                check(name,'MAS',p['output'],'instrumented_MAS')
                profiles.extend(dict(name=name,method='MAS_internal',**stage) for stage in p['stages'])
    timing=[];lookup={}
    for row in measured['samples']:
        require(row['calls']==16,'Wrong timed work')
        require(all(np.isfinite(row[k]) and row[k]>=0 for k in ['host_ms','submit_ms','event_ms','reset_ms','verify_export_ms']), 'Invalid timing')
        require(row['event_ms']>0 and row['host_ms']>=row['submit_ms'],'Timing windows inconsistent')
        v=check(row['name'],row['method'],row['output'],'measured',execution=row['execution'],round=row['round'],group=row['group'])
        require(rel(v,warm_outputs[('measure',row['name'],row['method'],'host16')])<=1e-3,'Measured action differs')
        lookup[(row['name'],row['round'],row['group'],row['method'],row['execution'])]=row
        timing.append(dict(**row,event_us_per_call=row['event_ms']*1000/16,host_us_per_call=row['host_ms']*1000/16))
    pairs=[]
    for name in [c['name'] for c in manifest['cases'] if c['probe']!='zero']:
        for rnd in range(1,4):
            for group in range(1,7):
                def pair(label,numerator,denominator):
                    a=lookup[(name,rnd,group,*numerator)];b=lookup[(name,rnd,group,*denominator)]
                    pairs.append(dict(name=name,round=rnd,group=group,comparison=label,event_ratio=a['event_ms']/b['event_ms'],
                        host_ratio=a['host_ms']/b['host_ms'],submit_ratio=a['submit_ms']/b['submit_ms']))
                for execution in manifest['executions']:
                    pair('fused/reference '+execution,('fused',execution),('reference',execution))
                    pair('reference/MAS '+execution,('reference',execution),('MAS',execution))
                    pair('fused/MAS '+execution,('fused',execution),('MAS',execution))
                for method in manifest['methods']:
                    for execution in ['graph16','graph1x16']:pair(execution+'/host16 '+method,(method,execution),(method,'host16'))
                    pair('graph16/graph1x16 '+method,(method,'graph16'),(method,'graph1x16'))
    aggregate=[]
    for name,comparison in dict.fromkeys((r['name'],r['comparison']) for r in pairs):
        rows=[r for r in pairs if r['name']==name and r['comparison']==comparison];rounds=[]
        for rnd in range(1,4):rounds.append(float(np.median([r['event_ratio'] for r in rows if r['round']==rnd])))
        aggregate.append(dict(name=name,comparison=comparison,n=len(rows),event_ratio_median=float(np.median([r['event_ratio'] for r in rows])),
            host_ratio_median=float(np.median([r['host_ratio'] for r in rows])),submit_ratio_median=float(np.median([r['submit_ratio'] for r in rows])),
            event_ratio_min=min(r['event_ratio'] for r in rows),event_ratio_max=max(r['event_ratio'] for r in rows),
            round1=rounds[0],round2=rounds[1],round3=rounds[2],improving_rounds=sum(v<1 for v in rounds)))
    gate=[r for r in aggregate if r['comparison']=='fused/reference graph16']
    require(len(gate)==4,'Missing predefined gate')
    passed=all(r['event_ratio_median']<=.95 and r['improving_rounds']>=2 for r in gate)
    costs=[]
    for name,method,execution in dict.fromkeys((r['name'],r['method'],r['execution']) for r in timing):
        rows=[r for r in timing if (r['name'],r['method'],r['execution'])==(name,method,execution)]
        costs.append(dict(name=name,method=method,execution=execution,n=len(rows),
            **{k+'_median':float(np.median([r[k] for r in rows])) for k in ['event_us_per_call','host_us_per_call','submit_ms','reset_ms','verify_export_ms']}))
    csv_new(tables/'output_CPU.csv',audits);csv_new(tables/'timings.csv',timing);csv_new(tables/'pairs.csv',pairs)
    csv_new(tables/'aggregate.csv',aggregate);csv_new(tables/'costs.csv',costs);csv_new(tables/'graphs.csv',graphs);csv_new(tables/'instrumented_stages.csv',profiles)
    write_new(folder/'summary.json',dict(implementation_checks_passed=True,contracts_warm_outputs=54,measure_warm_outputs=36,
        measured_outputs=648,independently_checked_outputs=len(audits),maximum_CPU_relative=max(r['CPU_relative'] for r in audits),
        construction=construction,gate=gate,finite_performance_gate_passed=passed,aggregate=aggregate,costs=costs,
        performance_certified=False,production_solver_modified=False,inputs=tree(RAW/'contracts')+tree(RAW/'measure')))
    print(json.dumps(dict(numerics=True,measured=648,checked_outputs=len(audits),finite_performance_gate=passed,gate=gate)),flush=True)
if __name__=='__main__':main()
