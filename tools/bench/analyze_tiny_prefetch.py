"""Independent FP64 action audit, byte equality, and predeclared paired stats."""
import argparse
import csv
from decimal import Decimal,localcontext
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_tiny_prefetch import RAW,REPORT,BUILD,read
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

def verify_run(action):
    folder=RAW/action;require(read(folder/'result.json')['passed'],'Owned run failed')
    m=read(RAW/'input_manifest.json');identity=read(BUILD/'identity.json');packet=read(RAW/'runtime_seal.json')
    verify(identity['sources']+identity['build_evidence']+identity['runtime']+[identity['exe'],packet['identity'],packet['manifest'],packet['protocol'],packet['amendment'],packet['inspector_amendment'],packet['runner'],packet['production_before']]+packet['compiled']+packet['helpers']+m['original_inputs']+m['generated_inputs'])
    verify(read(RAW/'production_before.json'))
    data=read(folder/'data.json');require(data['passed'] and data['action']==action,'Wrong action')
    return m,data

def check_graphs(case,tiny):
    require(case['guards_passed'] and (tiny or case['restored']),'Missing guard/state')
    require(len(case['graphs'])==(4 if tiny else 6),'Wrong graph count')
    for g in case['graphs']:
        count=g['calls'];require(count in [1,16],'Wrong graph grain')
        method=g['method'];types=g['types_cuda_enum'];require(types.get('10',0)==types.get('11',0)==0,'Graph alloc/free')
        kernels=(1 if tiny else 4 if method=='MAS' else 12)*count
        nodes=(2 if tiny else 7 if method=='MAS' else 15)*count
        require(g['nodes']==nodes and sum(types.values())==nodes and types.get('0',0)==kernels,'Graph shape differs')
        require(types.get('1',0)==count and types.get('2',0)==(0 if tiny else 2*count),'Memory node count differs')
        launch_count=(1 if tiny else 0 if method=='MAS' else 2)*count
        if not tiny:require(g['opaque_library_kernels']==(0 if method=='MAS' else 4*count),'Wrong opaque library kernels')
        require(len(g['tiny_launches'])==launch_count,'Wrong tiny kernel count')
        for k in g['tiny_launches']:
            require(k['kernel']==method and k['grid']==1 and k['block']==(32 if method=='prefetch' else 1)
                and k['static_shared_bytes']==(512 if method=='prefetch' else 0) and k['local_bytes']==0,'Wrong tiny attributes')

def tiny_audit(m,data):
    cases={c['name']:c for c in m['tiny_cases']};rows=[];baseline={}
    def check(name,method,path,kind,**extra):
        c=cases[name];L=np.fromfile(c['L'],dtype='<f8').reshape(8,8,order='F');b=np.fromfile(c['b'],dtype='<f8')
        lower=np.tril(L);require(np.isfinite(lower).all() and np.all(np.diag(lower)>0),'Invalid factor')
        # Windows numpy.longdouble has only 52 mantissa bits on this machine.
        # Decimal.from_float retains the exact input doubles, with 80 digit arithmetic.
        with localcontext() as context:
            context.prec=80;x=[Decimal.from_float(float(v)) for v in b]
            dl=[[Decimal.from_float(float(v)) for v in row] for row in lower]
            for i in range(8):
                t=x[i]
                for j in range(i):t-=dl[i][j]*x[j]
                x[i]=t/dl[i][i]
            for i in range(7,-1,-1):
                t=x[i]
                for j in range(i+1,8):t-=dl[j][i]*x[j]
                x[i]=t/dl[i][i]
            expected=np.array([float(v) for v in x],dtype=np.float64)
        scipy=cho_solve((lower,True),b)
        require(rel(scipy,np.array(c['expected']))<=1e-14 and rel(expected,scipy)<=1e-9,'CPU references differ')
        v=np.fromfile(path,dtype='<f8');require(v.shape==(8,) and np.isfinite(v).all(),'Invalid tiny output')
        error=rel(v,expected);require(error<=1e-9,'Tiny CPU action failed')
        E=lower@lower.T
        denom=np.linalg.norm(E,ord=np.inf)*np.linalg.norm(v,ord=np.inf)+np.linalg.norm(b,ord=np.inf)
        backward=float(np.max(np.abs(E@v-b)))/denom if denom else 0.
        require(backward<=5e-13,'Tiny backward error failed')
        if not np.count_nonzero(b):require(np.count_nonzero(v)==0,'Zero tiny output not zero')
        raw=Path(path).read_bytes()
        if name in baseline:require(raw==baseline[name],'Tiny original/prefetch/Graph bytes differ')
        else:require(method=='original','Original must establish byte baseline');baseline[name]=raw
        rows.append(dict(name=name,method=method,kind=kind,CPU_relative=error,backward_error=backward,output=path,**extra))
    for c in data['cases']:
        check_graphs(c,True);require(len(c['warm'])==6,'Wrong tiny warm count')
        for w in c['warm']:check(c['name'],w['method'],w['output'],'warm',execution=w['execution'])
    for r in data['samples']:check(r['name'],r['method'],r['output'],'measured',tag=r['tag'])
    return rows

def full_audit(m,data):
    items={c['name']:c for c in m['cases']};reference={};systems={};rows=[];original={}
    for c in data['cases']:
        item=items[c['name']]
        if item['scene'] not in systems:systems[item['scene']]=FrozenSystem(item['prefix'])
        s=systems[item['scene']];n=item['n'];Z=np.fromfile(item['Z'],dtype='<f8').reshape(n,8,order='F')
        W=np.fromfile(item['W'],dtype='<f8').reshape(n,8,order='F');L=np.fromfile(item['L'],dtype='<f8').reshape(8,8,order='F')
        r=np.fromfile(item['files']['r'],dtype='<f8');base=s.full_apply(r);a=cho_solve((L,True),Z.T@r)
        u=s.full_apply(r-W@a);value=u+Z@(a-cho_solve((L,True),W.T@u))
        require(rel(W,s.full_A@Z)<1e-9 and rel(L@L.T,.5*(Z.T@W+W.T@Z))<1e-9,'CSR/coarse relation differs')
        require(rel(base,np.fromfile(item['files']['MAS_expected'],dtype='<f8'))<=1e-12
            and rel(value,np.fromfile(item['files']['balanced_expected'],dtype='<f8'))<=1e-12,'Parent CPU reference differs')
        reference[c['name']]=dict(MAS=base,original=value,prefetch=value)
    def check(name,method,path,kind,**extra):
        v=np.fromfile(path,dtype='<f8');expected=reference[name][method]
        require(v.shape==expected.shape and np.isfinite(v).all(),'Invalid full output');error=rel(v,expected)
        require(error<=1e-3,'Independent full CPU action failed')
        if name.endswith('_zero'):require(np.count_nonzero(v)==0,'Zero full output not zero')
        raw=Path(path).read_bytes();key=(name,method if method=='MAS' else 'balanced')
        if key in original:
            difference=rel(v,np.frombuffer(original[key],dtype='<f8'));require(difference<=1e-5,'Full original/prefetch/Graph action differs')
        else:require(method!='prefetch','Original must establish full bytes');original[key]=raw
        if name.endswith('_rhs'):
            historical=np.fromfile(items[name]['historical_b']['MAS' if method=='MAS' else 'fused'],dtype='<f8')
            require(rel(v,historical)<=1e-3,'Historical native action differs')
        rows.append(dict(name=name,method=method,kind=kind,CPU_relative=error,
            original_relative=rel(v,np.frombuffer(original[key],dtype='<f8')),original_bitwise_equal=raw==original[key],output=path,**extra))
    for c in data['cases']:
        check_graphs(c,False);require(len(c['warm'])==6,'Wrong full warm count')
        for w in c['warm']:check(c['name'],w['method'],w['output'],'warm',capture_calls=w['capture_calls'])
    for r in data['samples']:check(r['name'],r['method'],r['output'],'measured',tag=r['tag'])
    return rows

def paired_stats(data,tiny):
    expected=[]
    for name in (['bunny_first','mixed_first'] if tiny else ['bunny_rhs','mixed_rhs']):
        for rnd in range(3):
            for mo in range(2):
                for q in range(4):
                    for co in range(1 if tiny else 2):
                        for pos in range(4):
                            comp=1 if tiny else (co+q+rnd)%2;label=int(pos in [0,3])^((rnd+q)%2)
                            method=('prefetch' if label else 'original') if comp else 'MAS'
                            expected.append((name,rnd+1,q+1,pos+1,16 if (mo+rnd)%2 else 1,2048,'AB' if comp else 'AA','B' if label else 'A',method))
    samples=data['samples']
    require(expected==[tuple(r[k] for k in ['name','round','quartet','position','capture_calls','calls','comparison','label','method']) for r in samples],'Window order/work differs')
    for r in samples:
        require(all(np.isfinite(r[k]) and r[k]>=0 for k in ['host_ms','submit_ms','event_ms','reset_ms','verify_export_ms'])
            and r['event_ms']>0 and r['host_ms']>=r['submit_ms'],'Invalid timing')
    groups={}
    for r in samples:
        key=tuple(r[k] for k in ['name','capture_calls','comparison','round','quartet']);groups.setdefault(key,[]).append(r)
    quartets=[]
    for key,rows in groups.items():
        require(len(rows)==4 and [r['position'] for r in rows]==[1,2,3,4] and sum(r['label']=='A' for r in rows)==2,'Incomplete quartet')
        a=[r for r in rows if r['label']=='A'];b=[r for r in rows if r['label']=='B']
        quartets.append(dict(zip(['name','capture_calls','comparison','round','quartet'],key),
            event_ratio=float(np.mean([r['event_ms'] for r in b])/np.mean([r['event_ms'] for r in a])),
            host_ratio=float(np.mean([r['host_ms'] for r in b])/np.mean([r['host_ms'] for r in a]))))
    summary=[]
    for name,grain,comparison in dict.fromkeys((r['name'],r['capture_calls'],r['comparison']) for r in quartets):
        q=[r for r in quartets if (r['name'],r['capture_calls'],r['comparison'])==(name,grain,comparison)]
        require(len(q)==12,'Missing quartets');ratios=np.array([r['event_ratio'] for r in q]);med=float(np.median(ratios))
        rounds=[float(np.median([r['event_ratio'] for r in q if r['round']==i])) for i in range(1,4)]
        lo,hi=map(float,np.quantile(ratios,[.1,.9]));raw=[r for r in samples if (r['name'],r['capture_calls'],r['comparison'])==(name,grain,comparison)]
        arms={}
        for label in ['A','B']:
            times=np.array([r['event_ms'] for r in raw if r['label']==label]);mid=float(np.median(times))
            arms[label+'_relative_MAD']=float(np.median(np.abs(times-mid))/mid);arms[label+'_median_us']=mid/2048*1000
        aa=abs(med-1)<=.02 and all(abs(x-1)<=.05 for x in rounds) and lo>=.9 and hi<=1.1 and max(arms['A_relative_MAD'],arms['B_relative_MAD'])<=.05
        signal=med<.98 and all(x<1 for x in rounds)
        summary.append(dict(name=name,capture_calls=grain,comparison=comparison,quartets=12,event_ratio_median=med,
            speedup=1/med,round1=rounds[0],round2=rounds[1],round3=rounds[2],p10=lo,p90=hi,
            host_ratio_median=float(np.median([r['host_ratio'] for r in q])),**arms,AA_passed=aa if comparison=='AA' else None,
            improvement_signal=signal if comparison=='AB' else None))
    return quartets,summary

def main(action):
    folder=RAW/action;require(not (folder/'audit.json').exists(),'Audit already exists');m,data=verify_run(action)
    tiny=action.startswith('tiny');outputs=tiny_audit(m,data) if tiny else full_audit(m,data)
    expected=dict(tiny=420,contracts=36,tiny_measure=204,measure=396)[action];require(len(outputs)==expected,'Wrong output count')
    csv_new(REPORT/'tables'/(action+'_CPU.csv'),outputs)
    audit=dict(passed=True,checked_outputs=len(outputs),maximum_CPU_relative=max(r['CPU_relative'] for r in outputs),
        bitwise_equal=True if tiny else all(r['original_bitwise_equal'] for r in outputs))
    if not tiny:audit['maximum_original_relative']=max(r['original_relative'] for r in outputs)
    if action in ['tiny_measure','measure']:
        quartets,summary=paired_stats(data,tiny)
        csv_new(REPORT/'tables'/(action+'_windows.csv'),data['samples']);csv_new(REPORT/'tables'/(action+'_quartets.csv'),quartets)
        csv_new(REPORT/'tables'/(action+'_summary.csv'),summary);audit.update(windows=len(data['samples']),summary=summary)
        if not tiny:
            controls=[r for r in summary if r['comparison']=='AA'];require(len(controls)==4,'Missing controls')
            audit['all_controls_passed']=all(r['AA_passed'] for r in controls)
            audit['all_AB_signal']=all(r['improvement_signal'] for r in summary if r['comparison']=='AB')
    audit['inputs']=tree(folder);write_new(folder/'audit.json',audit)
    print({k:v for k,v in audit.items() if k not in ['inputs','summary']},flush=True)
    if 'summary' in audit:print(audit['summary'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tiny','contracts','tiny_measure','measure']);main(p.parse_args().action)
