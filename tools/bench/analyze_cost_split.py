"""Independent action checks and predeclared A/A stability thresholds."""
import argparse
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_cost_split import RAW,REPORT,BUILD,read
from local_identity import verify,tree
from linux_runner import require,write_new
from m3_numeric import FrozenSystem,rel
import numpy as np
from scipy.linalg import cho_solve

def csv_new(path,rows):
    require(rows and not path.exists(),'Empty/existing table');path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def audit_outputs(action):
    folder=RAW/action;require(read(folder/'result.json')['passed'],'Owned run did not pass')
    m=read(RAW/'input_manifest.json');identity=read(BUILD/'identity.json');packet=read(RAW/'runtime_seal.json')
    verify(identity['sources']+identity['build_evidence']+identity['runtime']+[identity['exe'],packet['identity'],packet['manifest'],packet['protocol'],packet['runner']]+packet['compiled']+packet['helpers']+m['original_inputs'])
    data=read(folder/'data.json');require(data['passed'] and data['action']==action,'Wrong action result')
    reference={};systems={};cases={c['name']:c for c in m['cases']};construction=[]
    for c in data['cases']:
        item=cases[c['name']]
        if item['scene'] not in systems:systems[item['scene']]=FrozenSystem(item['prefix'])
        s=systems[item['scene']];n=item['n'];Z=np.fromfile(item['Z'],dtype='<f8').reshape(n,8,order='F')
        W=np.fromfile(item['W'],dtype='<f8').reshape(n,8,order='F');L=np.fromfile(item['L'],dtype='<f8').reshape(8,8,order='F')
        r=np.fromfile(item['files']['r'],dtype='<f8');base=s.full_apply(r);a=cho_solve((L,True),Z.T@r)
        u=s.full_apply(r-W@a);value=u+Z@(a-cho_solve((L,True),W.T@u))
        require(rel(W,s.full_A@Z)<1e-9 and rel(L@L.T,.5*(Z.T@W+W.T@Z))<1e-9,'Original CSR/coarse relation differs')
        require(rel(base,np.fromfile(item['files']['MAS_expected'],dtype='<f8'))<=1e-12
            and rel(value,np.fromfile(item['files']['balanced_expected'],dtype='<f8'))<=1e-12,'Parent reference differs')
        reference[c['name']]=dict(MAS=base,reference=value,fused=value)
        construction.append(dict(name=c['name'],n=n,CSR_W_relative=rel(W,s.full_A@Z)))
    rows=[];warm={}
    def check(name,method,path,kind,**extra):
        v=np.fromfile(path,dtype='<f8');expected=reference[name][method]
        require(v.shape==expected.shape and np.isfinite(v).all(),'Missing/nonfinite vector');error=rel(v,expected)
        require(error<=1e-3,'Independent CPU action failed')
        if name.endswith('_zero'):require(np.count_nonzero(v)==0,'Zero output not zero')
        rows.append(dict(name=name,method=method,kind=kind,CPU_relative=error,output=path,**extra));return v
    for case in data['cases']:
        name=case['name'];require(case['restored'] and case['guards_passed'] and len(case['warm'])==len(case['graphs'])==6,'Missing contract/state')
        for g in case['graphs']:
            require(g['types_cuda_enum'].get('10',0)==g['types_cuda_enum'].get('11',0)==0,'Memory nodes remain')
            require(sum(g['types_cuda_enum'].values())==g['nodes'],'Graph node count differs')
        for w in case['warm']:
            v=check(name,w['method'],w['output'],'warm',capture_calls=w['capture_calls'])
            key=(name,w['method']);
            if key in warm:require(rel(v,warm[key])<=1e-3,'Graph grain action differs')
            else:warm[key]=v
            if name.endswith('_rhs'):
                historic=np.fromfile(cases[name]['historical_b'][w['method']],dtype='<f8');require(rel(v,historic)<=1e-3,'Historical action differs')
    for row in data['samples']:
        require(all(np.isfinite(row[k]) and row[k]>=0 for k in ['host_ms','submit_ms','event_ms','reset_ms','verify_export_ms']), 'Invalid timing')
        require(row['event_ms']>0 and row['host_ms']>=row['submit_ms'],'Invalid event/window')
        v=check(row['name'],row['method'],row['output'],'measured',tag=row['tag'])
        require(rel(v,warm[(row['name'],row['method'])])<=1e-3,'Measured action differs')
    return data,rows,construction

def main(action):
    folder=RAW/action;require(not (folder/'audit.json').exists(),'Audit exists')
    data,outputs,construction=audit_outputs(action);tables=REPORT/'tables'
    csv_new(tables/(action+'_CPU.csv'),outputs)
    if action=='contracts':
        require(len(data['cases'])==6 and len(outputs)==36 and not data['samples'],'Contract count differs')
        write_new(folder/'audit.json',dict(passed=True,checked_outputs=len(outputs),maximum_CPU_relative=max(r['CPU_relative'] for r in outputs),
            construction=construction,inputs=tree(folder)))
        print(dict(contracts=True,outputs=36));return
    require(action=='stability' and len(data['cases'])==2 and len(data['samples'])==576 and len(outputs)==588,'Stability count differs')
    expected=[]
    for name in ['bunny_rhs','mixed_rhs']:
        for rnd in range(3):
            for mo in range(2):
                for lo in range(3):
                    mode=(mo+rnd)%2;calls=[128,512,2048][(lo+rnd)%3]
                    for q in range(4):
                        for p in range(4):
                            label=int(p in [0,3])^((rnd+q)%2)
                            expected.append((name,rnd+1,q+1,p+1,16 if mode else 1,calls,'B' if label else 'A'))
    require(expected==[tuple(r[k] for k in ['name','round','quartet','position','capture_calls','calls','label']) for r in data['samples']],'A/A order differs')
    require(all(r['method']=='MAS' for r in data['samples']),'A/A did not use MAS')
    groups={}
    for row in data['samples']:
        key=tuple(row[k] for k in ['name','capture_calls','calls','round','quartet']);groups.setdefault(key,[]).append(row)
    quartets=[]
    for key,rows in groups.items():
        require(len(rows)==4 and [r['position'] for r in rows]==[1,2,3,4] and sum(r['label']=='A' for r in rows)==2,'Incomplete ABBA')
        a=[r for r in rows if r['label']=='A'];b=[r for r in rows if r['label']=='B']
        quartets.append(dict(zip(['name','capture_calls','calls','round','quartet'],key),
            event_ratio=float(np.mean([r['event_ms'] for r in b])/np.mean([r['event_ms'] for r in a])),
            host_ratio=float(np.mean([r['host_ms'] for r in b])/np.mean([r['host_ms'] for r in a]))))
    summary=[]
    for name,grain,calls in dict.fromkeys((r['name'],r['capture_calls'],r['calls']) for r in quartets):
        q=[r for r in quartets if (r['name'],r['capture_calls'],r['calls'])==(name,grain,calls)];ratio=np.array([r['event_ratio'] for r in q])
        require(len(q)==12,'Missing quartets');rounds=[float(np.median([r['event_ratio'] for r in q if r['round']==i])) for i in range(1,4)]
        raw=[r for r in data['samples'] if (r['name'],r['capture_calls'],r['calls'])==(name,grain,calls)]
        arm={}
        for label in ['A','B']:
            values=np.array([r['event_ms'] for r in raw if r['label']==label]);median=float(np.median(values))
            arm[label]=float(np.median(np.abs(values-median))/median)
        med=float(np.median(ratio));lo,hi=map(float,np.quantile(ratio,[.1,.9]))
        passed=abs(med-1)<=.02 and all(abs(v-1)<=.05 for v in rounds) and lo>=.90 and hi<=1.10 and max(arm.values())<=.05
        summary.append(dict(name=name,capture_calls=grain,calls=calls,quartets=12,event_ratio_median=med,
            round1=rounds[0],round2=rounds[1],round3=rounds[2],p10=lo,p90=hi,A_relative_MAD=arm['A'],B_relative_MAD=arm['B'],passed=passed))
    selected=None
    for calls in [128,512,2048]:
        cells=[r for r in summary if r['calls']==calls];require(len(cells)==4,'Missing global selection cells')
        if selected is None and all(r['passed'] for r in cells):selected=calls
    csv_new(tables/'stability_windows.csv',data['samples']);csv_new(tables/'stability_quartets.csv',quartets);csv_new(tables/'stability_summary.csv',summary)
    write_new(folder/'audit.json',dict(passed=True,selected_calls=selected,stable=selected is not None,summary=summary,
        checked_outputs=len(outputs),maximum_CPU_relative=max(r['CPU_relative'] for r in outputs),construction=construction,inputs=tree(folder)))
    print(dict(numerics=True,outputs=588,windows=576,selected_calls=selected,summary=summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['contracts','stability']);main(p.parse_args().action)
