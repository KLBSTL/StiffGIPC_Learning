"""Independent four-arm coverage, policy events and original-CSR verification."""
import argparse
import ast
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
from prepare_weighted_gpu import RAW,REPORT,BUILD,PROJECT,PARENTS,read,edit,function
from local_identity import record,verify,tree
from linux_runner import require,write_new
from m3_numeric import FrozenSystem,rel,residual_metrics
import numpy as np
from analyze_native_balanced import csv_new
COMPARISONS=['AA_legacy','AA_cooldown','legacy_net','cooldown_net','MAS_policy','cached_policy']
ARMS=[(0,0),(2,2),(0,1),(2,3),(0,2),(1,3)]

def verify_identity():
    p=read(RAW/'runtime_seal.json');identity=read(BUILD/'identity.json');m=read(RAW/'input_manifest.json')
    verify(p['files']+identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe']]
        +m['original_inputs']+m['generated_inputs']+m['pcg_inputs']+read(RAW/'production_before.json'))
    return p,identity,m

def arm_identity(arm):return ('cached' if arm%2 else 'MAS','cooldown2' if arm>=2 else 'legacy')

def expected_rows(name):
    rows=[]
    for rnd in range(3):
        for mo in range(2):
            for q in range(4):
                for co in range(6):
                    for pos in range(4):
                        mode=(mo+rnd)%2;cmp=(co+q+rnd)%6;label=int(pos in [0,3])^((rnd+q)%2);arm=ARMS[cmp][label]
                        method,policy=arm_identity(arm)
                        rows.append((rnd+1,q+1,pos+1,'graph_apply1' if mode else 'host',COMPARISONS[cmp],
                            'B' if label else 'A',method,policy,arm,f'{name}_r{rnd+1}_e{mode}_q{q+1}_c{cmp}_p{pos+1}'))
    return rows

def metadata(data,manifest,folder,action):
    require(data['passed'] and data['action']==action and data['production_state_exists'] is False,'Wrong action/scope')
    require([c['name'] for c in data['cases']]==[c['name'] for c in manifest['cases']],'Exact case order')
    require(data['expected_measured_solves']==(1152 if action=='measure' else 0),'Expected solve count')
    paths=[]
    def output(path,filename):
        p=Path(path).resolve();require(p==folder.resolve()/filename and p.is_file() and p not in paths,'Output ownership/alias');paths.append(p)
    resources={r['method']:r['resources'] for r in read(RAW/'compiler/audit.json')['rows']}
    def identity(r,arm,execution):
        method,policy=arm_identity(arm)
        require((r['arm'],r['method'],r['policy'],r['execution'])==(arm,method,policy,execution),'Requested/actual arm')
        require(r['result']['policy']==policy and r['result']['augmented']==r['result']['fused_apply']==(method=='cached'),'Wrong actual method/policy')
    for c,item in zip(data['cases'],manifest['cases']):
        name=c['name'];require(c['n']==item['n'] and c['readonly_and_scratch_unchanged'],'State/dimensions')
        require([g['method'] for g in c['graphs']]==['original','cached','MAS'],'Graph method coverage')
        for g in c['graphs']:
            method=g['method'];mas=method=='MAS';info=g['info']
            require(g['opaque']==(0 if mas else 4) and len(g['tiny'])==(0 if mas else 2),'Tiny/opaque coverage')
            require(info['nodes']==(7 if mas else 15) and info['types_cuda_enum']=={'0':4 if mas else 12,'1':1,'2':2},'Graph topology')
            for k in g['tiny']:
                r=resources[method];require(k['kernel']==method and k['grid']==1 and k['block']==(1 if method=='original' else 32)
                    and k['static_shared_bytes']==r['SHARED'] and k['local_bytes']==r['LOCAL']==0 and k['registers_per_thread']==r['REG'],'Actual kernel attributes')
        require(len(c['warm'])==8,'Warm four-arm matrix')
        for i,r in enumerate(c['warm']):
            identity(r,i%4,'graph_apply1' if i//4 else 'host');tag=f'{name}_warm_a{i%4}_e{i//4}'
            require(r['result']['iteration_limit']==2000,'Normal iteration limit')
            for key,suffix in [('solution_file','_x.bin'),('residual_file','_residual.bin'),('MAS_residual_file','_MAS_residual.bin')]:output(r[key],tag+suffix)
        expected_zero=[(method,execution) for execution in ['host','graph_apply1'] for method in ['original','cached','MAS']]
        require([(r['method'],r['execution']) for r in c['zero']]==expected_zero,'Zero algebra matrix')
        for i,r in enumerate(c['zero']):
            require(r['algebra_only'] is True,'Wrong zero scope');output(r['output_file'],f'{name}_zero_m{i%3}_e{i//3}.bin')
        output(c['native_A_b_file'],name+'_native_A_b.bin');output(c['native_MAS_b_file'],name+'_native_MAS_b.bin')
        require(len(c['budget_checks'])==(8 if action=='contracts' else 0),'Budget failure matrix')
        for i,r in enumerate(c['budget_checks']):
            identity(r,i%4,'graph_apply1' if i//4 else 'host')
            require(r['state_restored'] and r['result']['iteration_limit']==2 and not r['result']['converged'],'Budget rejection/restoration')
            output(r['solution_file'],f'{name}_budget_a{i%4}_e{i//4}_x.bin')
        expected=expected_rows(name) if action=='measure' else []
        actual=[tuple(r[k] for k in ['round','quartet','position','execution','comparison','label','method','policy','arm']) for r in c['runs']]
        require(actual==[r[:-1] for r in expected],'Exact quartet sequence/method/policy')
        for r,e in zip(c['runs'],expected):
            identity(r,e[-2],e[3]);require(r['result']['iteration_limit']==2000,'Measured iteration limit')
            for key,suffix in [('solution_file','_x.bin'),('residual_file','_residual.bin'),('MAS_residual_file','_MAS_residual.bin')]:output(r[key],e[-1]+suffix)
    require(set(paths)==set(folder.glob('*.bin')),'Unowned/missing output');return len(paths)

def schedule(row,study,allow_failure=False):
    r=row['result'];trace=r['trace'];events=r['quality_events'];cool=r['policy']=='cooldown2';limit=r['iteration_limit'];bb=r['rhs_norm2']
    require(r['policy'] in ['legacy','cooldown2'] and r['screened'] and r['residual_restarts']==0,'Policy/restart')
    require(math.isfinite(bb) and bb>0 and len(trace)==r['iterations'] and 0<r['iterations']<=limit,'Trace/limit')
    active=False;state=False;last=0;event_index=0;screens=0;queries=0;zero=bool(r.get('zero_rho_true_quality_checked',False))
    for i,p in enumerate(trace,1):
        require(p['iteration']==i and p['pre_update_rho']>0 and p['p_Ap']>0 and p['alpha']>0
            and all(math.isfinite(p[k]) for k in ['pre_update_rho','p_Ap','alpha']),'Recurrence scalars/order')
        require(abs(p['alpha']/(p['pre_update_rho']/p['p_Ap'])-1)<1e-12,'Alpha recurrence')
        require(p['policy_cooldown_before']==state,'Cooldown initial state/reset')
        active=active or p['pre_update_rho']<=1e-4*r['rho_initial'];query=active or i==limit
        due=False;force=False;norm_needed=query and not state
        require(p['recursive_norm_evaluated']==norm_needed,'Unexpected/missing norm reduction')
        if norm_needed:
            rr=p['recursive_norm_rr'];require(rr is not None and math.isfinite(rr) and rr>=0,'Raw norm')
            require(p['recursive_relative_l2'] is not None and abs(p['recursive_relative_l2']-math.sqrt(rr/bb))<1e-14,'Norm metadata')
            screens+=1
        else:require(p['recursive_norm_rr'] is None and p['recursive_relative_l2'] is None,'Fabricated/unused norm')
        if query:
            queries+=1;force=last==0 or i==limit or i-last>=(2 if state else 8)
            due=force or (not state and p['recursive_norm_rr']<=.012*.012*bb)
        override=zero and i==len(trace)
        require(p['schedule_due']==due and p['quality_checked']==(due or override),'Explicit schedule/quality')
        require(p['screened_out']==(query and not due and not override) and p['forced_quality_check']==(force or override),'Skip/forced flags')
        current=[]
        while event_index<len(events) and events[event_index]['iteration']==i:
            e=events[event_index];event_index+=1;current.append(e)
        count=int(due)+int(override);require(len(current)==count,'Explicit event multiplicity')
        for index,e in enumerate(current):
            expected_override=override and (not due or index==1)
            require(e['zero_override']==expected_override and e['repeat']==(last==i),'Override/repeat provenance')
            require(e['cooldown_before']==state,'Event state before')
            l2=e['true_relative_l2'];weighted=e['common_native_MAS_relative']
            require(all(math.isfinite(v) and v>=0 for v in [l2,weighted]),'Nonfinite true quality')
            accepted=l2<=.0095 and weighted<=.0095
            require(e['accepted']==accepted and (not accepted or (i==len(trace) and index==len(current)-1)),'Premature/false acceptance')
            state=cool and l2<=.0095 and weighted>.0095;last=i
            require(e['cooldown_after']==state,'Event transition')
        require(p['policy_cooldown_after']==state,'Final per-update cooldown')
        if current:
            require(p['true_relative_l2']==current[-1]['true_relative_l2'] and p['common_native_MAS_relative']==current[-1]['common_native_MAS_relative'],'Event/trace values')
        else:require(p['true_relative_l2'] is None and p['common_native_MAS_relative'] is None,'Unchecked true quality')
    require(event_index==len(events)==r['quality_checks'] and screens==r['screen_checks'] and queries==r['schedule_queries'],'Operation/event counts')
    require(r['last_quality_iteration']==last and r['preconditioner_calls']==r['iterations']+int(zero)+int(not r['converged']),'Preconditioner count/last')
    if allow_failure:
        require(not r['converged'] and r['iterations']==limit and trace[-1]['forced_quality_check'] and not events[-1]['accepted'],'Budget failure must reject')
    else:
        require(r['converged'] and events[-1]['accepted'] and max(r['true_relative_l2'],r['common_native_MAS_relative'])<=.0095,'Final acceptance')
    return dict(schedule_passed=True,recursive_norm_checks=screens,schedule_queries=queries)

_VALIDATION=None
def validation():
    global _VALIDATION
    if _VALIDATION is not None:return _VALIDATION
    source=ROOT/'tools/bench/cache_pcg_experiment.py';definitions={name:function(source,name) for name in ['resources','stats','audit']};changes={}
    changes['resources']=[]
    changes['stats']=[("cmp=='AA'","cmp.startswith('AA')",1),("key[2]=='AA'","key[2].startswith('AA')",1)]
    changes['audit']=[("index<6","index<8",1),
        ("row=dict(scene=c['name'],phase=","row=dict(scene=c['name'],policy=r['policy'],arm=r['arm'],phase=",1),
        ("solve_index=index,method=r['method']","solve_index=index,policy=r['policy'],arm=r['arm'],method=r['method']",1)]
    ns=dict(globals());ns.update(__file__=str(Path(__file__)),schedule=schedule,metadata=metadata,verify_identity=verify_identity)
    for name in ['resources','stats','audit']:
        code=edit(definitions[name],changes[name]);exec(compile(code,str(source)+'#'+name,'exec'),ns)
    _VALIDATION=ns;return ns

def profile_checks(data,enabled):
    rows=[]
    for c in data['cases']:
        for r in c['warm']+c['runs']:
            p=r['profile'];require(p['enabled']==enabled,'Wrong profiler activation')
            if not enabled:require(p['stages']=={},'Profiler contaminated primary timing');continue
            s=r['result'];n=s['iterations'];q=s['quality_checks'];norm=s['screen_checks'];zero=int(s.get('zero_rho_true_quality_checked',False))
            counts=dict(recurrence_apply=n+zero,SpMV=n+q,dot_with_host_readback=2*n+zero+norm+2*q,
                quality_native_MAS=q,vector_update_x_r=n,vector_update_direction=n-1,copy_rhs=1,copy_initial_direction=1,explicit_residual_vector=q)
            require(set(p['stages'])==set(counts),'Profile category coverage')
            for name,count in counts.items():require(p['stages'][name]['calls']==count and math.isfinite(p['stages'][name]['host_ms']) and p['stages'][name]['host_ms']>=0,'Profile operation count/time')
            require(sum(v['host_ms'] for v in p['stages'].values())<=r['solve_host_ms'],'Profile double counting')
            rows.append(dict(scene=c['name'],method=r['method'],policy=r['policy'],execution=r['execution'],iterations=n,
                **{name+'_ms':v['host_ms'] for name,v in p['stages'].items()},**{name+'_calls':v['calls'] for name,v in p['stages'].items()}))
    require(len(rows)==(16 if enabled else 0),'Profile four-arm coverage');return rows

def audit(stage):
    ns=validation();ns['audit'](stage);_,_,manifest=verify_identity();folder=RAW/stage;data=read(folder/'data.json');budgets=[]
    for c,item in zip(data['cases'],manifest['cases']):
        if not c['budget_checks']:continue
        system=FrozenSystem(item['prefix'])
        for r in c['budget_checks']:
            schedule(r,{},True);x=np.fromfile(r['solution_file'],dtype='<f8');require(x.shape==system.full_b.shape and np.isfinite(x).all(),'Budget x')
            metrics,_=residual_metrics(system.A,system.b,x[system.active],system.apply);e=r['result']['quality_events'][-1]
            require(abs(metrics['relative_l2']-e['true_relative_l2'])<=1e-8 and abs(metrics['common_MAS_relative']/e['common_native_MAS_relative']-1)<=1e-3,'Budget independent quality')
            budgets.append(dict(scene=c['name'],arm=r['arm'],policy=r['policy'],execution=r['execution'],**metrics))
    details=profile_checks(data,stage=='profile');negative=[]
    for label,mutation in [
        ('false_cooldown',lambda r:r['result']['trace'][0].__setitem__('policy_cooldown_before',True)),
        ('missing_event',lambda r:r['result']['quality_events'].pop()),
        ('wrong_norm_flag',lambda r:r['result']['trace'][0].__setitem__('recursive_norm_evaluated',True)),
        ('false_acceptance',lambda r:r['result']['quality_events'][0].__setitem__('accepted',not r['result']['quality_events'][0]['accepted']))]:
        bad=copy.deepcopy(data['cases'][0]['warm'][0]);mutation(bad)
        try:schedule(bad,{})
        except ValueError:negative.append(label)
        else:raise ValueError('Accepted bad policy trace '+label)
    if budgets:csv_new(REPORT/'tables'/(stage+'_budget_CPU.csv'),budgets)
    if details:csv_new(REPORT/'tables/exclusive_profile.csv',details)
    write_new(folder/'policy_audit.json',dict(passed=True,budget_rejections=len(budgets),profiled_solves=len(details),negative_rejected=negative,source=record(Path(__file__))))
    print(dict(policy_audit_passed=True,stage=stage,budget_rejections=len(budgets),profiled_solves=len(details)),flush=True)

def finish():
    packet,identity,m=verify_identity()
    for p in PARENTS:verify(read(p)['files'])
    for g in read(BUILD/'generation.json')['files']:
        verify([g['source'],g['target']]);require(Path(g['target']['path']).read_text(encoding='utf-8')==edit(Path(g['source']['path']).read_text(encoding='utf-8'),g['edits']),'Generated source changed')
    jobs=[];total=0;vectors=0;rejected=0
    for stage in ['contracts','memcheck','profile','measure']:
        f=RAW/stage;j=read(f/'result.json');a=read(f/'audit.json');p=read(f/'policy_audit.json');d=read(f/'data.json')
        require(j['passed'] and j['cleanup_owned_job_empty'] and a['passed'] and p['passed'] and 0<j['wall_seconds']<=240,'Incomplete/invalid stage')
        for sample in j['samples']:
            require(sample['power_guid']==packet['power_guid'] and sample['gpu']['uuid']==j['gpu_before']['uuid'],'Power/GPU drift')
            require(sample['gpu']['memory.free']>=768 and j['gpu_before']['memory.free']-sample['gpu']['memory.free']<=j['memory_budget_mib'] and not sample['processes']['blocking_foreign_pids'],'Resource violation')
        count=metadata(d,m,f,'contracts' if stage=='memcheck' else stage);require(count==a['vector_files'],'Vector count changed')
        profile_checks(d,stage=='profile');total+=a['solves'];vectors+=count;rejected+=p['budget_rejections'];jobs.append(dict(stage=stage,wall_seconds=j['wall_seconds'],exit_code=j['exit_code']))
    d=read(RAW/'measure/data.json');systems={r['name']:FrozenSystem(r['prefix']) for r in m['cases']};_,cells,_=validation()['stats'](d,systems)
    a=read(RAW/'measure/audit.json');require(cells==a['cells'],'Statistics reconstruction')
    aa=all(c['signal_passed'] for c in cells if c['comparison'].startswith('AA'))
    net=all(c['signal_passed'] for c in cells if c['comparison']=='cooldown_net')
    stable=all(c['ratio']<=1.02 for c in cells if c['comparison']=='cached_policy')
    mixed_gain=all(c['signal_passed'] for c in cells if c['scene']=='mixed_rhs' and c['comparison']=='cached_policy')
    decision=dict(numerical_passed=True,AA_passed=aa,cooldown_net_signal_all_cells=net,cached_nonregression=stable,
        mixed_scheduling_gain=mixed_gain,iteration_budget_passed=a['iteration_budget_passed'],
        next_stage_eligible=aa and net and stable and mixed_gain and a['iteration_budget_passed'],
        production_modified=False,performance_certified=False,coarse_setup_included=False,normal_solves=total,budget_rejections=rejected,vector_files=vectors)
    lines=['# 条件两步冷却：GPU四臂对照','',
        '同一固定A/b/M/Z/W/L。MAS、缓存balanced各测legacy与cooldown2；最终原A真L2和原MAS加权验收不变。Graph只重放apply。',
        '1152次正式配对，三轮、ABBA/BAAB、无删尾；独立同步profile不混入加速比。计时包含原trace及新增显式检查事件记录，属于诊断。','',
        '| 场景 | 执行 | 比较(A→B) | A ms | B ms | 配对加速比 | A/B迭代 | 三轮B/A |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for c in cells:
        if c['comparison'].startswith('AA'):continue
        rounds='/'.join(f'{v:.4f}' for v in c['rounds'])
        lines.append(f'| {c["scene"]} | {c["execution"]} | {c["comparison"]} | {c["A_ms"]:.4f} | {c["B_ms"]:.4f} | {c["speedup"]:.4f}x | {c["A_iterations"]:g}/{c["B_iterations"]:g} | {rounds} |')
    lines+=['','legacy_net/cooldown_net=对应同调度MAS→cached；MAS_policy/cached_policy=同方法legacy→cooldown2。ms为quartet两次均值的中位数；配对比是各quartet B/A中位数的倒数，不能直接相除替代。',
        f'全部正常解CPU质量通过，最大L2={a["max_CPU_l2"]:.9g}，最大MAS加权={a["max_CPU_MAS"]:.9g}。每次检查事件和每步调度状态逐项审核。',
        f'合计{total}次正常求解、{rejected}次budget=2预期拒绝、{vectors}个向量文件。预算失败的终态x另有CPU残差核验，之后正常求解证明状态重新初始化。',
        '零输入仅是原代数动作合同，未伪称zero-RHS完整PCG返回已测。zero-rho重复路径的状态机CPU合同与事件审核保留。',
        'memcheck保留32条已知metadata查询API报告及原退出99；严格解析无其他内存/API finding，不称工具零错误通过。',
        '生产代码和停止规则未改。固定空间的构造、刷新和真实推进尚未计入，不能外推100帧或2倍。','',
        '## 决策','', '```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','',
        '复核命令：`E:/Anaconda/envs/DL/python.exe -X utf8 -B tools/bench/weighted_gpu_experiment.py finish`。']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');write_new(REPORT/'DECISION.json',decision)
    sources=REPORT/'sources';sources.mkdir();source_files=[Path(__file__),ROOT/'tools/local/prepare_weighted_gpu.py',ROOT/'tools/local/run_weighted_gpu.py']
    for p in source_files:(sources/p.name).write_bytes(p.read_bytes())
    write_new(REPORT/'SOURCE_CLOSURE.json',dict(sources=[record(p) for p in source_files],jobs=jobs))
    files=tree(RAW)+tree(REPORT)+tree(PROJECT)+tree(BUILD/'overlay')+tree(BUILD/'project')+[record(BUILD/'identity.json'),record(BUILD/'generation.json')]+[record(p) for p in source_files]
    verify(files);write_new(REPORT/'DELIVERY_SEAL.json',dict(files=files,exe=identity['exe'],decision=decision));verify(read(REPORT/'DELIVERY_SEAL.json')['files'])
    print(dict(delivery_rehashed=True,files=len(files),**decision),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['resources','audit','finish']);p.add_argument('stage',nargs='?');args=p.parse_args()
    validation()['resources']() if args.action=='resources' else audit(args.stage) if args.action=='audit' else finish()
