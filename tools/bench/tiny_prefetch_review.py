"""Strict v2 metadata and independent re-audit; historical files stay immutable."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_tiny_prefetch import RAW as OLD_RAW,REPORT as OLD_REPORT,BUILD,PARENT,read,edited
from local_identity import record,verify,tree
from linux_runner import require,write_new
import analyze_tiny_prefetch as v1
RAW=ROOT/'runs/innovation_p5_tiny_review_20261017'
REPORT=ROOT/'reports/innovation_p5_tiny_review_20261017'
ACTIONS=['tiny','contracts','tiny_measure','measure']

def exact(values,expected,why):
    require(values==expected,why)

def finite(value,why,upper=None):
    require(type(value) in [int,float] and math.isfinite(value) and value>=0 and (upper is None or value<=upper),why)

def strict_metadata(manifest,data,folder,action):
    """One-to-one cases, launch matrix, warm matrix, and output ownership."""
    exact(data['action'],action,'Action differs');require(data['passed'] is True,'Native action failed')
    tiny=action.startswith('tiny');measure=action.endswith('measure') or action=='measure'
    expected_items=manifest['tiny_cases'] if tiny else manifest['cases']
    if measure:expected_items=[c for c in expected_items if c['name'] in (['bunny_first','mixed_first'] if tiny else ['bunny_rhs','mixed_rhs'])]
    names=[c['name'] for c in expected_items];require(len(names)==len(set(names)),'Manifest duplicate cases')
    exact([c['name'] for c in data['cases']],names,'Case coverage/order differs')
    methods=['original','prefetch'] if tiny else ['MAS','original','prefetch']
    graph_keys=[(method,grain) for method in methods for grain in [1,16]]
    warm_keys=[(method,execution) for method in methods for execution in (['host','graph1','graph16'] if tiny else [1,16])]
    seen=set();folder=Path(folder).resolve()
    def output(path,filename):
        p=Path(path);require(p.is_absolute() and p.is_file() and not p.is_symlink(),'Invalid output file')
        q=p.resolve();exact(q,folder/filename,'Output owner/name differs')
        require(q not in seen,'Output reused');seen.add(q)
    for case,item in zip(data['cases'],expected_items):
        require(case['guards_passed'] is True and (tiny or case['restored'] is True),'Guard/state failed')
        if not tiny:exact(case['probe'],item['probe'],'Probe differs')
        exact([(g['method'],g['calls']) for g in case['graphs']],graph_keys,'Graph method/grain coverage differs')
        v1.check_graphs(case,tiny)
        exact([(w['method'],w['execution'] if tiny else w['capture_calls']) for w in case['warm']],warm_keys,'Warm method/execution coverage differs')
        for w in case['warm']:
            method=w['method'];name=case['name']
            if tiny:
                index=methods.index(method);ex=w['execution']
                filename=name+('_host_' if ex=='host' else '_g'+ex.removeprefix('graph')+'_')+str(index)+'.bin'
            else:
                filename=name+'_warm_'+method+'_g'+str(w['capture_calls'])+'.bin'
                for key,limit in [('CPU_relative',1e-3),('host_relative',1e-5),('original_relative',1e-5)]:finite(w[key],key+' invalid',limit)
                require(type(w['original_bitwise_equal']) is bool,'Invalid byte equality metadata')
            output(w['output'],filename)
    for row in data['samples']:
        for k in ['host_ms','submit_ms','event_ms','reset_ms','verify_export_ms']:finite(row[k],'Invalid '+k)
        require(row['event_ms']>0 and row['host_ms']>=row['submit_ms'],'Invalid timed window')
        for k in ['capture_calls','calls','round','quartet','position']:require(type(row[k]) is int,'Noninteger work field')
        require(row['capture_calls'] in [1,16] and row['calls']==2048,'Wrong work count')
        require(row['name'] in names and row['method'] in methods,'Unknown sample owner/method')
        require(row['comparison']=='AB' if tiny else row['comparison'] in ['AA','AB'],'Wrong comparison')
        tag=row['name']+'_g'+str(row['capture_calls'])+'_r'+str(row['round'])+'_q'+str(row['quartet'])
        tag+=('_p'+str(row['position'])) if tiny else '_c'+str(int(row['comparison']=='AB'))+'_p'+str(row['position'])
        if not tiny:
            tag='pair_'+tag
            finite(row['CPU_relative'],'Sample CPU metadata invalid',1e-3);finite(row['original_relative'],'Sample original metadata invalid',1e-5)
            require(type(row['original_bitwise_equal']) is bool,'Sample byte metadata invalid')
        exact(row['tag'],tag,'Sample tag differs');output(row['output'],tag+'.bin')
    expected_samples=dict(tiny=0,contracts=0,tiny_measure=192,measure=384)[action]
    exact(len(data['samples']),expected_samples,'Wrong sample count')
    if measure:v1.paired_stats(data,tiny) # exact independent native-order reconstruction
    exact({p.resolve() for p in folder.glob('*.bin')},seen,'Missing or orphan output vectors')
    return dict(cases=len(names),outputs=len(seen),windows=len(data['samples']))

def independent_statistics(data):
    """Standard-library calculation, independent of NumPy implementation in v1."""
    groups={}
    for r in data['samples']:
        key=tuple(r[k] for k in ['name','capture_calls','comparison','round','quartet']);groups.setdefault(key,[]).append(r)
    ratios={}
    for key,rows in groups.items():
        a=[float(r['event_ms']) for r in rows if r['label']=='A'];b=[float(r['event_ms']) for r in rows if r['label']=='B']
        require(len(a)==len(b)==2,'Unpaired group');ratios[key]=st.fmean(b)/st.fmean(a)
    def q(values,p):
        s=sorted(values);index=(len(s)-1)*p;lo=math.floor(index);hi=math.ceil(index)
        return s[lo]+(s[hi]-s[lo])*(index-lo)
    result=[]
    for cell in dict.fromkeys(key[:3] for key in ratios):
        values=[v for k,v in ratios.items() if k[:3]==cell];require(len(values)==12,'Missing independent groups')
        rounds=[st.median([v for k,v in ratios.items() if k[:3]==cell and k[3]==i]) for i in [1,2,3]]
        rows=[r for r in data['samples'] if tuple(r[k] for k in ['name','capture_calls','comparison'])==cell]
        mad={}
        for label in ['A','B']:
            a=[float(r['event_ms']) for r in rows if r['label']==label];mid=st.median(a)
            mad[label]=st.median([abs(x-mid) for x in a])/mid
        med=st.median(values);lo=q(values,.1);hi=q(values,.9)
        result.append(dict(name=cell[0],capture_calls=cell[1],comparison=cell[2],event_ratio_median=med,speedup=1/med,
            round1=rounds[0],round2=rounds[1],round3=rounds[2],p10=lo,p90=hi,A_relative_MAD=mad['A'],B_relative_MAD=mad['B'],
            AA_passed=(abs(med-1)<=.02 and all(abs(x-1)<=.05 for x in rounds) and lo>=.9 and hi<=1.1 and max(mad.values())<=.05) if cell[2]=='AA' else None,
            improvement_signal=(med<.98 and all(x<1 for x in rounds)) if cell[2]=='AB' else None))
    return result

def mutations(manifest,data,folder):
    """Only in-memory metadata changes. Historical files are never edited."""
    tiny=data['action'].startswith('tiny');rows=[]
    foreign_action='contracts' if tiny else 'tiny'
    foreign_file=read(OLD_RAW/foreign_action/'data.json')['cases'][0]['warm'][0]['output']
    scenarios={
        'duplicate_graph':lambda d:d['cases'][0].__setitem__('graphs',[copy.deepcopy(d['cases'][0]['graphs'][0]) for _ in d['cases'][0]['graphs']]),
        'duplicate_warm':lambda d:d['cases'][0].__setitem__('warm',[copy.deepcopy(d['cases'][0]['warm'][0]) for _ in d['cases'][0]['warm']]),
        'duplicate_case':lambda d:d['cases'].__setitem__(-1,copy.deepcopy(d['cases'][0])),
        'wrong_case_order':lambda d:d['cases'].reverse(),
        'swapped_graph_grain':lambda d:d['cases'][0]['graphs'][0].__setitem__('calls',16),
        'wrong_block':lambda d:next(g for g in d['cases'][0]['graphs'] if g['method']=='prefetch')['tiny_launches'][0].__setitem__('block',1),
        'wrong_shared_bytes':lambda d:next(g for g in d['cases'][0]['graphs'] if g['method']=='prefetch')['tiny_launches'][0].__setitem__('static_shared_bytes',0),
        'reuse_output':lambda d:d['cases'][0]['warm'][1].__setitem__('output',d['cases'][0]['warm'][0]['output']),
        'missing_output':lambda d:d['cases'][0]['warm'][0].__setitem__('output',str(folder/'missing.bin')),
        'foreign_output':lambda d:d['cases'][0]['warm'][0].__setitem__('output',foreign_file),
        'guard_false':lambda d:d['cases'][0].__setitem__('guards_passed',False),
        'unknown_warm_method':lambda d:d['cases'][0]['warm'][0].__setitem__('method','unknown'),
    }
    if not tiny:
        scenarios['nan_CPU_metadata']=lambda d:d['cases'][0]['warm'][0].__setitem__('CPU_relative',float('nan'))
        scenarios['nonboolean_bit_metadata']=lambda d:d['cases'][0]['warm'][0].__setitem__('original_bitwise_equal',1)
    for name,mutate in scenarios.items():
        bad=copy.deepcopy(data);mutate(bad);rejected=False;reason=''
        try:strict_metadata(manifest,bad,folder,data['action'])
        except (ValueError,KeyError) as e:rejected=True;reason=str(e)
        require(rejected,'V2 accepted corrupt metadata: '+name)
        rows.append(dict(action=data['action'],test=name,rejected=True,reason=reason))
    return rows

def history():
    require(not (RAW/'history.json').exists(),'No implicit overwrite');RAW.mkdir(exist_ok=True)
    verify(read(OLD_REPORT/'DELIVERY_SEAL.json')['files']);verify(read(PARENT)['files'])
    manifest=read(OLD_RAW/'input_manifest.json');counts=[];negative=[];all_outputs=[];stats=[]
    # Explicitly reproduce the two known v1 false accepts before applying v2.
    original=read(OLD_RAW/'tiny/data.json');bad=copy.deepcopy(original['cases'][0]);bad['graphs']=[copy.deepcopy(bad['graphs'][0]) for _ in bad['graphs']]
    v1.check_graphs(bad,True)
    bad=copy.deepcopy(original);bad['cases'][0]['warm']=[copy.deepcopy(bad['cases'][0]['warm'][0]) for _ in range(6)]
    require(len(v1.tiny_audit(manifest,bad))==420,'V1 reproduction changed')
    for action in ACTIONS:
        m,data=v1.verify_run(action);counts.append(dict(action=action,**strict_metadata(m,data,OLD_RAW/action,action)))
        outputs=v1.tiny_audit(m,data) if action.startswith('tiny') else v1.full_audit(m,data)
        v1.csv_new(REPORT/'tables'/(action+'_CPU_v2.csv'),outputs);all_outputs+=outputs
        if action in ['tiny','contracts']:negative+=mutations(m,data,OLD_RAW/action)
        if data['samples']:
            computed=independent_statistics(data);legacy=read(OLD_RAW/action/'audit.json')['summary']
            exact(len(computed),len(legacy),'Statistics cells differ')
            for new,old in zip(computed,legacy):
                for key,value in new.items():
                    if type(value) is float:require(math.isclose(value,old[key],rel_tol=2e-14,abs_tol=2e-14),'Statistics changed: '+key)
                    else:exact(value,old[key],'Statistics field changed: '+key)
            stats+=computed
    exact(sum(c['outputs'] for c in counts),1056,'Wrong historical total');exact(len(all_outputs),1056,'Missing numeric output')
    v1.csv_new(REPORT/'tables/independent_statistics.csv',stats);v1.csv_new(REPORT/'tables/negative_tests.csv',negative)
    write_new(RAW/'history.json',dict(passed=True,counts=counts,negative_tests=len(negative),v1_false_accepts_reproduced=2,
        maximum_CPU_relative=max(r['CPU_relative'] for r in all_outputs),outputs=1056,windows=576,statistics_unchanged=True,
        numerical_kernel_issue_found=False,source=record(Path(__file__)),parent=record(OLD_REPORT/'DELIVERY_SEAL.json')))
    print(dict(history_passed=True,outputs=1056,negative_tests=len(negative),statistics_unchanged=True),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['history']);a=p.parse_args();history()
