"""Independent CPU audit and paired analysis of the fixed-input projection probe."""
import json
from pathlib import Path
import re
import statistics
from decimal import Decimal,localcontext
from m3_numeric import ROOT,need,rel
import numpy as np
from local_identity import record,verify,tree
from linux_runner import write_new
from analyze_native_balanced import csv_new
from prepare_projection_probe import RAW,REPORT,BUILD,read

METHODS=('input_reference','input_fused','output_reference','output_naive','output_shared')
ORDERS=((2,3,4),(4,3,2),(3,2,4),(4,2,3),(2,4,3),(3,4,2))
def fixture(n):
    i=np.arange(n,dtype=float);j=np.arange(8,dtype=float)
    return dict(r=np.sin(.17*i)+.3,u=np.cos(.12*i)-.2,a=.13*(j-3),c=.07*(4-j),
        W=np.cos((.013*(i+1))[:,None]*(j+1)),Z=np.sin((.021*(i+1))[:,None]*(j+2)))
def expectations(data):return dict(input=data['r']-data['W']@data['a'],output=data['u']+data['Z']@(data['a']-data['c']))
def warm_audit(rows,data_by_case,tolerance):
    audited=[];values={};worst={name:np.zeros(len(d['r'])) for name,d in data_by_case.items()}
    for row in rows:
        case=row['case'];n=len(data_by_case[case]['r']);need(row['n']==n,'Size mismatch')
        method=row['method'];need(method in METHODS,'Unknown method');op='input' if method.startswith('input') else 'output'
        expected=expectations(data_by_case[case])[op];v=np.fromfile(row['output_file'],dtype='<f8').reshape(n,32,order='F')
        need(np.isfinite(v).all(),'Warm output nonfinite');errs=np.linalg.norm(v-expected[:,None],axis=0)/max(np.linalg.norm(expected),1e-300)
        error=float(errs.max());need(error<=tolerance and row['max_bank_relative_l2']<=tolerance
            and row['all_32_outputs_checked'],'Warm CPU formula differs')
        key=(case,method,row['graph']);need(key not in values,'Duplicate warm variant');values[key]=v
        if row['graph']:
            need(row['nodes']>=32,'Graph incomplete')
            if method in ('input_fused','output_naive','output_shared'):need(row['nodes']==32,'Unexpected fused graph')
        worst[case]=np.maximum(worst[case],np.max(np.abs(v-expected[:,None]),axis=1))
        audited.append(dict(case=case,method=method,graph=row['graph'],n=n,CPU_max_bank_relative=error,
            native_max_bank_relative=row['max_bank_relative_l2'],nodes=row.get('nodes'),output_file=row['output_file']))
    for name in data_by_case:
        for method in METHODS:
            need((name,method,False) in values and (name,method,True) in values,'Missing warm variant')
    return audited,values,worst

def main():
    for action in ('contracts','measure'):
        result=read(RAW/action/'result.json');need(result['passed'] and result['cleanup_owned_job_empty'],'Native run incomplete')
        verify([result['source'],result['binary']])
    packet=read(RAW/'runtime_seal.json');verify([packet[k] for k in ('identity','manifest','protocol','runner','parent')]+packet['compiled'])
    manifest=read(RAW/'input_manifest.json');verify(manifest['generated_inputs']+manifest['original_inputs']+[manifest['source'],manifest['protocol']])
    out=RAW/'analysis';need(not out.exists(),'Fresh analysis required');out.mkdir()
    tables=REPORT/'tables';need(not tables.exists(),'Fresh tables required');tables.mkdir()
    inputs=tree(RAW);contracts=read(RAW/'contracts/contracts.json');measured=read(RAW/'measure/measurements.json')
    need(contracts['passed'] and len(contracts['warm'])==80 and measured['passed']
        and measured['read_only_inputs_unchanged'] and measured['calls_per_batch']==32,'Incomplete native output')
    contract_data={f'contract_{n}':fixture(n) for n in (1,31,32,33,255,256,257,1025)}
    contract_rows,_,_=warm_audit(contracts['warm'],contract_data,2e-12)
    cases=manifest['cases'];names=[c['name'] for c in cases]
    need(names==['bunny_rhs','bunny_residual','mixed_rhs','mixed_residual'],'Case protocol changed')
    data={}
    for case in cases:
        n=case['n'];values={}
        for k in ('Z','W','r','u','a','c'):
            value=np.fromfile(case['files'][k],dtype='<f8')
            values[k]=value.reshape(n,8,order='F') if k in ('Z','W') else value
            need(np.isfinite(value).all() and np.isfinite(np.linalg.norm(value)),'Invalid input values/norm')
        data[case['name']]=values
        for k,v in expectations(values).items():
            saved=np.fromfile(case['files'][k+'_expected'],dtype='<f8')
            need(rel(saved,v)<=1e-14,'Prepared expected vector differs')
    warm_rows,warm,worst=warm_audit(measured['warm'],data,1e-10)
    expected_order=[]
    for rnd in range(3):
        ids=list(range(4))
        if rnd==1:ids.reverse()
        if rnd==2:ids=ids[2:]+ids[:2]
        for i in ids:
            for rep in range(6):
                for group in range(2):
                    for mode_order in range(2):
                        graph=bool((rnd+mode_order)%2)
                        methods=ORDERS[rep] if group else ((rep)%2,(rep+1)%2)
                        for order,method in enumerate(methods):
                            expected_order.append((names[i],rnd+1,rep+1,group,mode_order+1,order+1,graph,METHODS[method]))
    samples=measured['samples'];need(len(samples)==720,'Wrong sample count')
    exports=[];timings=[];lookup={}
    for row,key in zip(samples,expected_order):
        got=tuple(row[k] for k in ('case','round','repeat','group','mode_order','order','graph','method'))
        need(got==key and row['calls']==32 and row['all_32_outputs_checked'],'Sample order/call count mismatch')
        need(0<=row['max_bank_relative_l2']<=1e-10,'Native sample formula mismatch')
        need(all(np.isfinite(row[k]) and row[k]>=0 for k in
            ('host_ms','event_ms','reset_host_ms','validation_host_ms','export_host_ms')) and min(row['host_ms'],row['event_ms'])>0,
            'Invalid sample timing')
        expected=expectations(data[row['case']])['input' if row['group']==0 else 'output']
        need(('first_file' in row)==(row['repeat']==1) and ('last_file' in row)==(row['repeat']==1),'Export policy differs')
        if row['repeat']==1:
            for label in ('first','last'):
                v=np.fromfile(row[label+'_file'],dtype='<f8');need(v.shape==expected.shape and np.isfinite(v).all(),'Bad saved output')
                error=rel(v,expected);need(error<=1e-10,'Saved output independent CPU check failed')
                exports.append(dict(case=row['case'],round=row['round'],method=row['method'],graph=row['graph'],bank=label,relative_l2=error))
        compact={k:v for k,v in row.items() if not k.endswith('_file')}
        compact.update(host_us_per_call=row['host_ms']*1000/32,event_us_per_call=row['event_ms']*1000/32);timings.append(compact)
        lk=(row['case'],row['round'],row['repeat'],row['graph'],row['method']);need(lk not in lookup,'Duplicate sample');lookup[lk]=compact
    numeric=[];equivalence=[]
    for name,d in data.items():
        selected=set((0,1,7,31,32,255,256,len(d['r'])-1))
        for i in np.argsort(worst[name])[::-1]:
            if len(selected)>=33:break
            selected.add(int(i))
        with localcontext() as ctx:
            ctx.prec=50
            dec=lambda x:Decimal.from_float(float(x))
            for i in sorted(selected):
                for op in ('input','output'):
                    base=dec(d['r' if op=='input' else 'u'][i]);terms=[]
                    for j in range(8):
                        coefficient=dec(d['a'][j]) if op=='input' else dec(d['a'][j])-dec(d['c'][j])
                        terms.append(dec(d['W' if op=='input' else 'Z'][i,j])*coefficient)
                    exact=base-sum(terms) if op=='input' else base+sum(terms)
                    scale=max(abs(base)+sum(abs(t) for t in terms),Decimal('1e-300'))
                    for method in METHODS[:2] if op=='input' else METHODS[2:]:
                        for graph in (False,True):
                            value=warm[(name,method,graph)][i,0];scaled=float(abs(dec(value)-exact)/scale)
                            need(np.isfinite(scaled) and scaled<=1e-12,'Decimal50 selected-row formula failed')
                            numeric.append(dict(case=name,row=i,method=method,graph=graph,Decimal50_scaled_absolute_error=scaled))
        for method in METHODS:
            a,b=warm[(name,method,False)],warm[(name,method,True)]
            equivalence.append(dict(case=name,comparison=method+'_host_graph',bitwise_equal=bool(np.array_equal(a.view('<u8'),b.view('<u8'))),relative=rel(b,a)))
        for graph in (False,True):
            a,b=warm[(name,'output_naive',graph)],warm[(name,'output_shared',graph)]
            equivalence.append(dict(case=name,comparison='naive_shared',graph=graph,
                bitwise_equal=bool(np.array_equal(a.view('<u8'),b.view('<u8'))),relative=rel(b,a)))
    paired=[];mode_pairs=[]
    comparisons=(('input_fused','input_reference'),('output_naive','output_reference'),
                 ('output_shared','output_reference'),('output_shared','output_naive'))
    for name in names:
        for rnd in (1,2,3):
            for rep in range(1,7):
                for graph in (False,True):
                    for candidate,reference in comparisons:
                        c,r=(lookup[(name,rnd,rep,graph,m)] for m in (candidate,reference))
                        paired.append(dict(case=name,round=rnd,repeat=rep,graph=graph,comparison=candidate+'/'+reference,
                            event_ratio=c['event_ms']/r['event_ms'],host_ratio=c['host_ms']/r['host_ms'],
                            candidate_event_us=c['event_us_per_call'],reference_event_us=r['event_us_per_call']))
                for method in METHODS:
                    h,g=(lookup[(name,rnd,rep,m,method)] for m in (False,True))
                    mode_pairs.append(dict(case=name,round=rnd,repeat=rep,method=method,
                        event_graph_over_host=g['event_ms']/h['event_ms'],host_graph_over_host=g['host_ms']/h['host_ms']))
    aggregates=[]
    for name in names:
        for graph in (False,True):
            for candidate,reference in comparisons:
                comparison=candidate+'/'+reference;rows=[r for r in paired if r['case']==name and r['graph']==graph and r['comparison']==comparison]
                vals=[r['event_ratio'] for r in rows];rounds=[statistics.median(r['event_ratio'] for r in rows if r['round']==i) for i in (1,2,3)]
                aggregates.append(dict(case=name,graph=graph,comparison=comparison,count=len(rows),event_ratio_median=statistics.median(vals),
                    host_ratio_median=statistics.median(r['host_ratio'] for r in rows),event_ratio_min=min(vals),event_ratio_max=max(vals),
                    event_ratio_rounds=rounds,improving_rounds=sum(v<1 for v in rounds),
                    candidate_event_us_median=statistics.median(r['candidate_event_us'] for r in rows),
                    reference_event_us_median=statistics.median(r['reference_event_us'] for r in rows)))
    primary=[r for r in aggregates if r['case'].startswith('mixed_') and r['graph'] and r['comparison']=='output_shared/output_naive']
    gate=all(r['event_ratio_median']<=.95 and r['improving_rounds']>=2 for r in primary)
    resources=[];symbol=None
    for line in (BUILD/'build.log').read_text(encoding='utf-8-sig',errors='replace').splitlines():
        match=re.search("Compiling entry function '([^']+)'",line)
        if match:symbol=match.group(1)
        match=re.search(r'Used (\d+) registers, used (\d+) barriers',line)
        if match:
            shared=re.search(r'(\d+) bytes smem',line)
            resources.append(dict(symbol=symbol,registers=int(match[1]),barriers=int(match[2]),shared_bytes=int(shared[1]) if shared else 0))
    for name,rows in (('contracts_CPU',contract_rows),('warm_CPU',warm_rows),('timings',timings),('saved_outputs_CPU',exports),
                      ('Decimal50_selected_rows',numeric),('equivalence',equivalence),('paired_ratios',paired),
                      ('host_graph_pairs',mode_pairs),('aggregate_ratios',aggregates),('kernel_resources',resources)):
        csv_new(tables/(name+'.csv'),rows)
    verify(inputs)
    write_new(out/'summary.json',dict(implementation_checks_passed=True,source=record(Path(__file__)),inputs=inputs,
        contract_checks=len(contract_rows),sample_count=len(samples),saved_output_checks=len(exports),Decimal50_checks=len(numeric),
        warm=warm_rows,equivalence=equivalence,aggregates=aggregates,resources=resources,primary=primary,
        shared_prototype_gate_passed=gate,production_solver_modified=False,performance_certified=False))
    print(json.dumps(dict(implementation_checks_passed=True,contract_checks=len(contract_rows),sample_count=len(samples),
        shared_prototype_gate_passed=gate,primary=primary)),flush=True)

if __name__=='__main__':main()
