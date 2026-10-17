"""Independent original-CSR quality, exact coverage and paired PCG cost audit."""
import argparse
import collections
import copy
import json
import math
from pathlib import Path
import re
import statistics as st
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_cache_pcg import RAW,REPORT,BUILD,PROJECT,PARENT,FROZEN,read,edit
from m3_numeric import FrozenSystem,rel,residual_metrics
import numpy as np
from local_identity import record,verify,tree
from linux_runner import require,write_new
from analyze_quality_screen import schedule
from analyze_native_balanced import csv_new

def resources():
    folder=RAW/'compiler';require(not folder.exists(),'Fresh resource evidence required');folder.mkdir()
    identity=read(BUILD/'identity.json');verify(identity['sources']+[identity['exe']]);commands=[]
    tool=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/cuobjdump.exe')
    def run(args,name):
        cmd=[str(tool)]+args+[identity['exe']['path']];commands.append(cmd)
        r=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=30)
        require(not r.stderr.strip(),'Unexpected cuobjdump error');(folder/name).write_text(r.stdout,encoding='utf-8');return r.stdout
    symbols=run(['--dump-elf-symbols'],'symbols.txt')
    names=re.findall(r'^STT_FUNC\s+STB_LOCAL\s+STO_ENTRY\s+(\S*balanced_tiny_solve\S*)$',symbols,re.M)
    # The graph inspector takes the prefetch function address too, retaining
    # that entry point even though no PCG arm launches it.
    require(len(names)==len(set(names))==3,'Expected three retained tiny entry points')
    usage=run(['--dump-resource-usage','--function',','.join(names)],'resources.txt')
    sass=run(['--dump-sass','--function',','.join(names)],'sass.txt');rows=[]
    for symbol,body in re.findall(r' Function (\S+):\s*([^\n]+)',usage):
        if symbol not in names:continue
        method='cached' if 'cached' in symbol else 'prefetch' if 'prefetch' in symbol else 'original'
        fields={k:int(v) for k,v in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)',body)}
        require(set(fields)=={'REG','STACK','SHARED','LOCAL'} and fields['LOCAL']==fields['STACK']==0,'Missing/spilled resource')
        section=[b for n,b in re.findall(r'Function : (\S+)\s*(.*?)(?=Function :|\Z)',sass,re.S) if n==symbol]
        require(len(section)==1 and not re.search(r'\b(?:LDL|STL)\b',section[0]),'Missing SASS/local traffic')
        rows.append(dict(method=method,resources=fields,symbol=symbol))
    require({r['method'] for r in rows}=={'original','prefetch','cached'},'Resource coverage')
    write_new(folder/'audit.json',dict(passed=True,rows=rows,tool=record(tool),commands=commands,files=tree(folder)))
    print(rows,flush=True)

def verify_identity():
    packet=read(RAW/'runtime_seal.json');identity=read(BUILD/'identity.json');m=read(RAW/'input_manifest.json')
    verify(packet['files']+identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe']]
        +m['original_inputs']+m['generated_inputs']+m['pcg_inputs']+read(RAW/'production_before.json'))
    return packet,identity,m

def expected_rows(name):
    expected=[]
    for rnd in range(3):
        for mo in range(2):
            for q in range(4):
                for co in range(2):
                    for pos in range(4):
                        mode=(mo+rnd)%2;cmp=(co+q+rnd)%2;label=int(pos in [0,3])^((rnd+q)%2)
                        tag=f'{name}_r{rnd+1}_e{mode}_q{q+1}_c{cmp}_p{pos+1}'
                        expected.append((rnd+1,q+1,pos+1,'graph_apply1' if mode else 'host','AB' if cmp else 'AA',
                            'B' if label else 'A','cached' if cmp and label else 'original',tag))
    return expected

def metadata(data,manifest,folder,action):
    require(data['passed'] and data['action']==action and data['production_state_exists'] is False,'Wrong stage/scope')
    require([c['name'] for c in data['cases']]==[c['name'] for c in manifest['cases']],'Exact case sequence differs')
    require(data['expected_measured_solves']==(384 if action=='measure' else 0),'Wrong expected solve count')
    paths=[]
    def output(path,name):
        p=Path(path).resolve();require(p==folder.resolve()/name and p.is_file() and p not in paths,'Output ownership/alias');paths.append(p)
    resources={r['method']:r['resources'] for r in read(RAW/'compiler/audit.json')['rows']}
    for c,item in zip(data['cases'],manifest['cases']):
        name=c['name'];require(c['n']==item['n'] and c['readonly_and_scratch_unchanged'],'State/dimension changed')
        require([g['method'] for g in c['graphs']]==['original','cached','MAS'],'Graph method coverage')
        for g in c['graphs']:
            method=g['method'];mas=method=='MAS';info=g['info']
            require(g['opaque']==(0 if mas else 4) and len(g['tiny'])==(0 if mas else 2),'Graph tiny/opaque count')
            require(info['nodes']==(7 if mas else 15) and info['types_cuda_enum']=={'0':4 if mas else 12,'1':1,'2':2},'Graph node topology')
            for k in g['tiny']:
                r=resources[method];require(k['kernel']==method and k['grid']==1 and k['block']==(1 if method=='original' else 32)
                    and k['static_shared_bytes']==r['SHARED'] and k['local_bytes']==r['LOCAL']==0
                    and k['registers_per_thread']==r['REG'],'Wrong actual captured kernel/resource')
        expected=[('original','host'),('cached','host'),('MAS','host'),('original','graph_apply1'),('cached','graph_apply1'),('MAS','graph_apply1')]
        require([(r['method'],r['execution']) for r in c['warm']]==expected,'Warm matrix coverage')
        for i,r in enumerate(c['warm']):
            tag=f'{name}_warm_m{i%3}_e{i//3}'
            for key,suffix in [('solution_file','_x.bin'),('residual_file','_residual.bin'),('MAS_residual_file','_MAS_residual.bin')]:output(r[key],tag+suffix)
        require([(r['method'],r['execution']) for r in c['zero']]==expected,'Zero matrix coverage')
        for i,r in enumerate(c['zero']):
            require(r['algebra_only'] is True,'Zero scope changed');output(r['output_file'],f'{name}_zero_m{i%3}_e{i//3}.bin')
        output(c['native_A_b_file'],name+'_native_A_b.bin');output(c['native_MAS_b_file'],name+'_native_MAS_b.bin')
        expected=expected_rows(name) if action=='measure' else []
        actual=[tuple(r[k] for k in ['round','quartet','position','execution','comparison','label','method']) for r in c['runs']]
        require(actual==[r[:-1] for r in expected],'Exact ABBA/BAAB coverage differs')
        for row,e in zip(c['runs'],expected):
            for key,suffix in [('solution_file','_x.bin'),('residual_file','_residual.bin'),('MAS_residual_file','_MAS_residual.bin')]:output(row[key],e[-1]+suffix)
    require(set(paths)==set(folder.glob('*.bin')),'Unowned/missing binary output');return len(paths)

def stats(data,systems):
    quartets=[];cells=[];differences=[]
    for c in data['cases']:
        A=systems[c['name']].full_A
        groups=collections.defaultdict(list)
        for r in c['runs']:groups[(r['execution'],r['comparison'],r['round'],r['quartet'])].append(r)
        for (execution,cmp,rnd,q),rows in groups.items():
            a=[r for r in rows if r['label']=='A'];b=[r for r in rows if r['label']=='B'];require(len(a)==len(b)==2,'Quartet labels')
            def average(bank,key):return st.mean(r[key] for r in bank)
            ai=st.median(r['result']['iterations'] for r in a)
            budget=cmp=='AA' or all(r['result']['iterations']<=1.1*ai for r in b)
            xref=np.fromfile(a[0]['solution_file'],dtype='<f8')
            for r in b:
                delta=np.fromfile(r['solution_file'],dtype='<f8')-xref
                energy=float(delta@(A@delta));denom=float(xref@(A@xref))
                require(energy>=0 and denom>0,'Invalid A-energy comparison')
                differences.append(dict(scene=c['name'],execution=execution,comparison=cmp,round=rnd,quartet=q,
                    relative_solution_difference=float(np.linalg.norm(delta)/max(np.linalg.norm(xref),1e-300)),
                    relative_A_energy_difference=math.sqrt(energy/denom)))
            quartets.append(dict(scene=c['name'],execution=execution,comparison=cmp,round=rnd,quartet=q,
                host_ratio=average(b,'solve_host_ms')/average(a,'solve_host_ms'),event_ratio=average(b,'solve_event_ms')/average(a,'solve_event_ms'),
                A_ms=average(a,'solve_host_ms'),B_ms=average(b,'solve_host_ms'),A_iterations=ai,
                B_iterations=st.median(r['result']['iterations'] for r in b),iteration_budget_passed=budget))
    groups=collections.defaultdict(list)
    for r in quartets:groups[(r['scene'],r['execution'],r['comparison'])].append(r)
    def percentile(v,p):
        values=sorted(v);index=(len(values)-1)*p;low=int(index);high=math.ceil(index)
        return values[low]+(values[high]-values[low])*(index-low)
    for key,rows in sorted(groups.items()):
        ratios=[r['host_ratio'] for r in rows];m=st.median(ratios);mad=st.median(abs(x-m) for x in ratios)
        rounds=[st.median(r['host_ratio'] for r in rows if r['round']==i) for i in [1,2,3]]
        aa=abs(m-1)<=.02 and all(abs(v-1)<=.05 for v in rounds) and percentile(ratios,.1)>=.9 and percentile(ratios,.9)<=1.1 and mad<=.05
        cells.append(dict(scene=key[0],execution=key[1],comparison=key[2],ratio=m,speedup=1/m,rounds=rounds,MAD=mad,
            p10=percentile(ratios,.1),p90=percentile(ratios,.9),A_ms=st.median(r['A_ms'] for r in rows),B_ms=st.median(r['B_ms'] for r in rows),
            A_iterations=st.median(r['A_iterations'] for r in rows),B_iterations=st.median(r['B_iterations'] for r in rows),
            iteration_budget_passed=all(r['iteration_budget_passed'] for r in rows),signal_passed=aa if key[2]=='AA' else m<.98 and all(v<1 for v in rounds)))
    return quartets,cells,differences

def audit(stage):
    _,_,manifest=verify_identity();folder=RAW/stage;job=read(folder/'result.json')
    require(job['passed'] and job['cleanup_owned_job_empty'],'Owned stage did not complete')
    data=read(folder/'data.json');action='contracts' if stage=='memcheck' else stage
    files=metadata(data,manifest,folder,action);rows=[];calibrations=[];curves=[];systems={}
    for c,item in zip(data['cases'],manifest['cases']):
        system=FrozenSystem(item['prefix']);n=item['n'];nb=np.linalg.norm(system.full_b);norms=(np.linalg.norm(system.b),float(system.b@system.apply(system.b)))
        systems[c['name']]=system
        def load(path):
            value=np.fromfile(path,dtype='<f8');require(value.shape==(n,) and np.isfinite(value).all(),'Invalid vector shape/finiteness');return value
        A_b=system.full_A@system.full_b;M_b=system.full_apply(system.full_b)
        ae=rel(load(c['native_A_b_file']),A_b);me=rel(load(c['native_MAS_b_file']),M_b)
        require(ae<=1e-9 and me<=1e-3,'Native operator calibration');require(abs(c['rhs_norm2']/nb**2-1)<1e-12,'RHS norm metadata')
        require(abs(c['common_bMb']/float(system.full_b@load(c['native_MAS_b_file']))-1)<1e-12,'Native RHS quadratic metadata')
        calibrations.append(dict(scene=c['name'],A_b_relative=ae,MAS_b_CPU_relative=me))
        for z in c['zero']:require(np.count_nonzero(load(z['output_file']))==0,'Nonzero zero action')
        for index,r in enumerate(c['warm']+c['runs']):
            result=r['result'];schedule(dict(result=result,screened=True),dict(max_iter=2001))
            require(result['augmented']==result['fused_apply']==(r['method']!='MAS'),'Wrong actual PCG method')
            for point in result['trace']:
                require(abs(point['alpha']/(point['pre_update_rho']/point['p_Ap'])-1)<1e-12,'Invalid alpha recurrence')
            x=load(r['solution_file']);res=load(r['residual_file']);mr=load(r['MAS_residual_file']);explicit=system.full_b-system.full_A@x
            metrics,_=residual_metrics(system.A,system.b,x[system.active],system.apply,norms)
            require(max(metrics.values())<=.01,'Independent CPU quality failed')
            require(np.linalg.norm(res-explicit)/nb<=1e-8,'Terminal residual vector differs from original CSR')
            require(rel(mr,system.full_apply(res))<=1e-3,'Terminal native MAS action differs')
            l2=np.linalg.norm(explicit)/nb;mas=math.sqrt(float(res@mr)/c['common_bMb'])
            require(max(abs(l2-r['terminal_l2']),abs(l2-result['true_relative_l2']),abs(mas-r['terminal_MAS']))<=1e-8,'Terminal quality metadata')
            for key in ['solve_host_ms','solve_event_ms','reset_host_ms','export_audit_host_ms']:
                require(math.isfinite(r[key]) and r[key]>0,'Nonpositive/nonfinite time')
            remainder=r['solve_host_ms']-result['quality_host_ms']-result['screen_host_ms'];require(remainder>0,'Invalid inclusive cost accounting')
            row=dict(scene=c['name'],phase='warm' if index<6 else 'measure',method=r['method'],execution=r['execution'],
                **{k:r[k] for k in ['solve_host_ms','solve_event_ms','reset_host_ms','export_audit_host_ms']},
                **{k:result[k] for k in ['iterations','quality_checks','quality_host_ms','screen_checks','screen_host_ms','preconditioner_calls']},
                remaining_host_ms=remainder,**metrics,terminal_l2=r['terminal_l2'],terminal_MAS=r['terminal_MAS'],solution_file=r['solution_file'])
            for k in ['round','quartet','position','comparison','label']:
                if k in r:row[k]=r[k]
            rows.append(row)
            for point in result['trace']:curves.append(dict(scene=c['name'],solve_index=index,method=r['method'],execution=r['execution'],**point))
    negative=[]
    mutations=[('duplicate_graph',lambda d:d['cases'][0]['graphs'].__setitem__(1,copy.deepcopy(d['cases'][0]['graphs'][0]))),
        ('duplicate_warm',lambda d:d['cases'][0]['warm'].__setitem__(1,copy.deepcopy(d['cases'][0]['warm'][0]))),
        ('output_alias',lambda d:d['cases'][0]['warm'][1].__setitem__('solution_file',d['cases'][0]['warm'][0]['solution_file'])),
        ('wrong_kernel',lambda d:d['cases'][0]['graphs'][1]['tiny'][0].__setitem__('kernel','original'))]
    if action=='measure':mutations.append(('wrong_AB_label',lambda d:d['cases'][0]['runs'][0].__setitem__('label','invalid')))
    for name,mutate in mutations:
        bad=copy.deepcopy(data);mutate(bad)
        try:metadata(bad,manifest,folder,action)
        except (ValueError,AssertionError,RuntimeError):negative.append(name)
        else:raise ValueError('Accepted negative contract '+name)
    quartets,cells,diff=stats(data,systems)
    tables=REPORT/'tables';tables.mkdir(exist_ok=True)
    csv_new(tables/(stage+'_solves.csv'),rows);csv_new(tables/(stage+'_curves.csv'),curves)
    if quartets:csv_new(tables/(stage+'_quartets.csv'),quartets);csv_new(tables/(stage+'_solution_differences.csv'),diff)
    write_new(folder/'audit.json',dict(passed=True,solves=len(rows),vector_files=files,calibrations=calibrations,cells=cells,negative_rejected=negative,
        max_CPU_l2=max(r['relative_l2'] for r in rows),max_CPU_MAS=max(r['common_MAS_relative'] for r in rows),
        maximum_pair_relative_solution_difference=max((r['relative_solution_difference'] for r in diff),default=0),
        maximum_pair_relative_A_energy_difference=max((r['relative_A_energy_difference'] for r in diff),default=0),
        iteration_budget_passed=all(c['iteration_budget_passed'] for c in cells),source=record(Path(__file__))))
    print(dict(passed=True,stage=stage,solves=len(rows),vector_files=files,cells=cells),flush=True)

def finish():
    _,identity,_=verify_identity();verify(read(PARENT)['files'])
    generation=read(BUILD/'generation.json')
    for g in generation['files']:
        verify([g['source'],g['target']]);source=Path(g['source']['path']).read_text(encoding='utf-8')
        if 'boundary' in g:
            expected=edit(source.split(g['boundary'])[0],g['edits'])+Path(g['continuation']['path']).read_text(encoding='utf-8');verify([g['continuation']])
        elif 'begin' in g:expected=edit(source[source.index(g['begin']):source.index(g['end'])],g.get('edits',[]))
        else:expected=source
        require(Path(g['target']['path']).read_text(encoding='utf-8')==expected,'Generated native source differs')
    audits={s:read(RAW/s/'audit.json') for s in ['contracts','memcheck','measure']}
    require(all(a['passed'] for a in audits.values()),'Incomplete audits')
    a=audits['measure'];aa=all(c['signal_passed'] for c in a['cells'] if c['comparison']=='AA')
    ab=all(c['signal_passed'] for c in a['cells'] if c['comparison']=='AB');eligible=aa and ab and a['iteration_budget_passed']
    decision=dict(numerical_passed=True,AA_passed=aa,full_PCG_signal=ab,iteration_budget_passed=a['iteration_budget_passed'],
        next_integration_eligible=eligible,performance_certified=False,production_modified=False,
        next_step='有条件进入真实推进的短程接入验证；先计入粗空间构建与刷新成本。' if eligible else '保留局部核优化；先定位完整PCG成本或迭代差异，不直接接入100帧。')
    lines=['# 缓存 8D 核：完整固定系统 PCG 对照','',
        '同一封存 A/b、原生 block COO SpMV、CUB 归约、ABD/MAS 与 PCG 递推。未修改生产代码。当前电源模式沿用野兽模式。',
        'Graph1 仅重放一次预条件器 apply；PCG 其余控制仍在 Host。solve 包含原有逐次 trace 和显式质量检查，属于诊断耗时。','',
        '| 场景 | 执行 | 原核 ms | 缓存核 ms | 配对加速比 | 原/缓存迭代 | 三轮 B/A | A/A |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for c in a['cells']:
        if c['comparison']!='AB':continue
        control=next(v for v in a['cells'] if v['scene']==c['scene'] and v['execution']==c['execution'] and v['comparison']=='AA')
        rounds='/'.join(f'{v:.4f}' for v in c['rounds'])
        lines.append(f'| {c["scene"]} | {c["execution"]} | {c["A_ms"]:.4f} | {c["B_ms"]:.4f} | {c["speedup"]:.4f}x | {c["A_iterations"]:g}/{c["B_iterations"]:g} | {rounds} | {"通过" if control["signal_passed"] else "未通过"} |')
    lines+=['','ms 列为各 quartet 内两次求解均值的中位数；加速比为配对 B/A 中位数的倒数，两者不能直接互除替代。',
        f'独立质量：最大原 A 相对 L2={a["max_CPU_l2"]:.9g}，最大 CPU MAS 加权残差={a["max_CPU_MAS"]:.9g}；全部 <=.01。',
        f'配对最终解最大相对差={a["maximum_pair_relative_solution_difference"]:.9g}，该差值不等于真实解误差。',
        '完整解、显式残差、原生 MAS 残差向量保存在 runs；每次迭代 rho、pAp、alpha、递推/真残差、检查标志见 tables/measure_curves.csv。',
        '每次 solve 的 Host/Event、quality/screen、余项、重置和导出耗时及调用次数见 tables/measure_solves.csv；余项是相减记账，不是隔离测得的纯核时间。',
        'memcheck 属于独立仪器化合同，未混入性能统计。零输入只验证代数动作，不声称测试了零 RHS PCG 返回路径。','',
        '## 有限决策','',json.dumps(decision,ensure_ascii=False,indent=2),
        '', '这是两个固定检查点的诊断，不能外推真实100帧、总模拟加速比或2倍目标。']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');write_new(REPORT/'DECISION.json',decision)
    # Seal analysis scripts, generated code, records and every actual output.
    files=tree(RAW)+tree(REPORT)+tree(PROJECT)+tree(BUILD/'overlay')+tree(BUILD/'project')+[record(BUILD/'identity.json'),record(BUILD/'generation.json'),record(Path(__file__)),record(ROOT/'tools/local/prepare_cache_pcg.py'),record(ROOT/'tools/local/run_cache_pcg.py')]
    verify(files);write_new(REPORT/'DELIVERY_SEAL.json',dict(files=files,exe=identity['exe'],decision=decision));print(dict(files=len(files),**decision),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['resources','audit','finish']);p.add_argument('stage',nargs='?');args=p.parse_args()
    resources() if args.action=='resources' else audit(args.stage) if args.action=='audit' else finish()
