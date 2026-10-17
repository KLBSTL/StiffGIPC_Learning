"""Compile-resource gate and strict three-arm independent numeric/timing audit."""
import argparse
import collections
import copy
import importlib.util
import math
from pathlib import Path
import re
import statistics as st
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_tiny_cache import RAW,REPORT,BUILD,PARENT,read
from local_identity import record,verify,tree
from linux_runner import require,write_new

def implementation():
    spec=importlib.util.spec_from_file_location('tiny_cache_validation',BUILD/'overlay/validation_impl.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def resources():
    folder=RAW/'compiler';require(not folder.exists(),'No implicit overwrite');folder.mkdir()
    identity=read(BUILD/'identity.json');verify(identity['sources']+[identity['exe']])
    tool=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/cuobjdump.exe');commands=[]
    def run(args,name):
        cmd=[str(tool)]+args+[identity['exe']['path']];commands.append(cmd)
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=30,check=True);require(not result.stderr.strip(),'Unexpected cuobjdump stderr')
        (folder/name).write_text(result.stdout,encoding='utf-8');return result.stdout
    symbols=run(['--dump-elf-symbols'],'symbols.txt')
    names=re.findall(r'^STT_FUNC\s+STB_LOCAL\s+STO_ENTRY\s+(\S*balanced_tiny_solve\S*)$',symbols,re.M)
    require(len(names)==3 and len(set(names))==3,'Missing three actual static symbols')
    args=['--function',','.join(names)]
    usage=run(['--dump-resource-usage']+args,'resources.txt');sass=run(['--dump-sass']+args,'sass.txt')
    rows=[]
    for symbol,body in re.findall(r' Function (\S+):\s*([^\n]+)',usage):
        if symbol not in names:continue
        method='cached' if 'cached' in symbol else 'prefetch' if 'prefetch' in symbol else 'original'
        fields={k:int(v) for k,v in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)',body)}
        require(set(fields)=={'REG','STACK','SHARED','LOCAL'},'Missing resource schema')
        require(fields['LOCAL']==fields['STACK']==0 and fields['SHARED']==(0 if method=='original' else 512),'Register/cache spilled or unexpected shared memory')
        sections=[body for name,body in re.findall(r'Function : (\S+)\s*(.*?)(?=Function :|\Z)',sass,re.S) if name==symbol]
        require(len(sections)==1,'Missing SASS function')
        counts=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P[0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b',sections[0]))
        require(not any(k.startswith(('LDL','STL')) for k in counts),'Local spill instruction found')
        rows.append(dict(method=method,symbol=symbol,resources=fields,static_instruction_sites=dict(counts)))
    require({r['method'] for r in rows}=={'original','prefetch','cached'},'Incomplete resource matrix')
    write_new(folder/'audit.json',dict(passed=True,rows=rows,tool=record(tool),exe=identity['exe'],commands=commands,source=record(Path(__file__)),files=tree(folder)))
    print([dict(method=r['method'],**r['resources']) for r in rows],flush=True)

def verify_identity():
    packet=read(RAW/'runtime_seal.json');identity=read(BUILD/'identity.json');manifest=read(RAW/'input_manifest.json')
    verify([packet[k] for k in ['identity','manifest','protocol','runner','analyzer','resources']]+packet['helpers']+packet['compiled'])
    verify(identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe']]+manifest['original_inputs']+manifest['generated_inputs'])
    verify(read(RAW/'production_before.json'));return packet,identity,manifest

def schedule(data):
    tiny=data['action']=='tiny_measure';expected=[];names=['bunny_first','mixed_first'] if tiny else ['bunny_rhs','mixed_rhs']
    for name in names:
        for rnd in range(3):
            for mo in range(2):
                for q in range(4):
                    for co in range(3):
                        comparison=(co+q+rnd)%3;kind=['AA','vs_original','vs_prefetch'][comparison]
                        for pos in range(4):
                            label=int(pos in [0,3])^((rnd+q)%2)
                            method=('original' if tiny else 'MAS') if comparison==0 else 'cached' if label else 'original' if comparison==1 else 'prefetch'
                            expected.append((name,rnd+1,q+1,pos+1,16 if (mo+rnd)%2 else 1,2048,kind,'B' if label else 'A',method))
    actual=[tuple(r[k] for k in ['name','round','quartet','position','capture_calls','calls','comparison','label','method']) for r in data['samples']]
    require(actual==expected and len(actual)==576,'Measured sequence/work differs')

def audit(stage):
    packet,identity,manifest=verify_identity();folder=RAW/stage;run=read(folder/'result.json');require(run['passed'],'Owned stage failed')
    data=read(folder/'data.json');action='tiny' if stage.startswith('sanitizer_') else stage
    mod=implementation();counts=mod.strict_metadata(manifest,data,folder,action)
    outputs=mod.tiny_audit(manifest,data) if action.startswith('tiny') else mod.full_audit(manifest,data)
    expected=dict(tiny=630,contracts=48,tiny_measure=594,measure=592)[action];require(len(outputs)==expected,'Wrong independently checked output count')
    mod.csv_new(REPORT/'tables'/(stage+'_CPU.csv'),outputs)
    result=dict(passed=True,outputs=len(outputs),maximum_CPU_relative=max(r['CPU_relative'] for r in outputs),metadata=counts,
        tiny_bitwise_equal=True if action.startswith('tiny') else None)
    if action in ['tiny','contracts']:
        negatives=[]
        def test(name,mutate):
            bad=copy.deepcopy(data);mutate(bad);rejected=False
            try:mod.strict_metadata(manifest,bad,folder,action)
            except (ValueError,KeyError):rejected=True
            require(rejected,'Corrupt cached metadata accepted: '+name);negatives.append(name)
        test('duplicate_graph',lambda d:d['cases'][0].__setitem__('graphs',[d['cases'][0]['graphs'][0]]*len(d['cases'][0]['graphs'])))
        test('duplicate_warm',lambda d:d['cases'][0].__setitem__('warm',[d['cases'][0]['warm'][0]]*len(d['cases'][0]['warm'])))
        test('cached_block',lambda d:next(g for g in d['cases'][0]['graphs'] if g['method']=='cached')['tiny_launches'][0].__setitem__('block',1))
        test('cached_spill',lambda d:next(g for g in d['cases'][0]['graphs'] if g['method']=='cached')['tiny_launches'][0].__setitem__('local_bytes',8))
        test('cached_alias_output',lambda d:next(w for w in d['cases'][0]['warm'] if w['method']=='cached').__setitem__('output',d['cases'][0]['warm'][0]['output']))
        test('cached_as_prefetch',lambda d:next(w for w in d['cases'][0]['warm'] if w['method']=='cached').__setitem__('method','prefetch'))
        result['negative_tests_rejected']=negatives
    if action in ['tiny_measure','measure']:
        schedule(data);summary=mod.independent_statistics(data)
        require(len(summary)==12,'Missing three comparisons per scene/grain')
        for r in summary:
            raw=[x for x in data['samples'] if (x['name'],x['capture_calls'],x['comparison'])==(r['name'],r['capture_calls'],r['comparison'])]
            for label in ['A','B']:r[label+'_median_us']=st.median([x['event_ms'] for x in raw if x['label']==label])/2048*1000
        mod.csv_new(REPORT/'tables'/(stage+'_windows.csv'),data['samples']);mod.csv_new(REPORT/'tables'/(stage+'_summary.csv'),summary)
        result.update(summary=summary,AA_passed=all(r['AA_passed'] for r in summary if r['comparison']=='AA'),
            all_AB_signal=all(r['improvement_signal'] for r in summary if r['comparison']!='AA'))
    write_new(folder/'audit.json',dict(result,files=tree(folder)))
    print({k:v for k,v in result.items() if k not in ['summary','negative_tests_rejected']},flush=True)
    if 'summary' in result:print([dict(name=r['name'],grain=r['capture_calls'],comparison=r['comparison'],ratio=r['event_ratio_median'],speedup=r['speedup'],AA=r['AA_passed']) for r in result['summary']],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['resources','audit']);p.add_argument('stage',nargs='?');a=p.parse_args()
    resources() if a.action=='resources' else audit(a.stage)
