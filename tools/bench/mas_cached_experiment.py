"""Reuse sealed PCG quality/coverage audits; change only the declared paired arms."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_mas_cached import RAW,REPORT,BUILD,PROJECT,PARENT,read,edit
from local_identity import record,verify,tree
from linux_runner import require,write_new
from analyze_native_balanced import csv_new
import cache_pcg_experiment as previous

# A private instance of the old auditor, bound to this experiment only. No
# previous files are edited, and its original functions remain hash-sealed.
previous.RAW=RAW;previous.REPORT=REPORT;previous.BUILD=BUILD
def expected_rows(name):
    rows=[]
    for rnd in range(3):
        for mo in range(2):
            for q in range(4):
                for co in range(2):
                    for pos in range(4):
                        mode=(mo+rnd)%2;cmp=(co+q+rnd)%2;label=int(pos in [0,3])^((rnd+q)%2)
                        tag=f'{name}_r{rnd+1}_e{mode}_q{q+1}_c{cmp}_p{pos+1}'
                        rows.append((rnd+1,q+1,pos+1,'graph_apply1' if mode else 'host','AB' if cmp else 'AA',
                            'B' if label else 'A','cached' if cmp and label else 'MAS',tag))
    return rows
previous.expected_rows=expected_rows
verify_identity=previous.verify_identity

def profile_checks(data,enabled):
    details=[]
    for c in data['cases']:
        for r in c['warm']+c['runs']:
            profile=r['profile'];require(profile['enabled']==enabled,'Wrong profile activation')
            stages=profile['stages']
            if not enabled:require(stages=={},'Profiling contaminated timing/contracts');continue
            result=r['result'];n=result['iterations'];q=result['quality_checks'];s=result['screen_checks']
            zero=int(result.get('zero_rho_true_quality_checked',False))
            expected=dict(recurrence_apply=n+zero,SpMV=n+q,dot_with_host_readback=2*n+zero+s+2*q,
                quality_native_MAS=q,vector_update_x_r=n,vector_update_direction=n-1,
                copy_rhs=1,copy_initial_direction=1,explicit_residual_vector=q)
            require(set(stages)==set(expected),'Missing/extra exclusive profile category')
            for name,calls in expected.items():
                row=stages[name];require(row['calls']==calls and math.isfinite(row['host_ms']) and row['host_ms']>=0,'Profile call count/time')
            total=sum(v['host_ms'] for v in stages.values());remainder=r['solve_host_ms']-total
            require(remainder>=0,'Profile double counted inclusive stages')
            details.append(dict(scene=c['name'],method=r['method'],execution=r['execution'],iterations=n,
                solve_host_ms=r['solve_host_ms'],classified_host_ms=total,remaining_host_ms=remainder,
                **{name+'_ms':row['host_ms'] for name,row in stages.items()},
                **{name+'_calls':row['calls'] for name,row in stages.items()}))
    require(len(details)==(12 if enabled else 0),'Profile solve coverage');return details

def audit(stage):
    previous.audit(stage)
    folder=RAW/stage;data=read(folder/'data.json');details=profile_checks(data,stage=='profile');negative=[]
    bad=copy.deepcopy(data)
    if stage=='profile':
        bad['cases'][0]['warm'][0]['profile']['stages']['recurrence_apply']['calls']+=1
    else:bad['cases'][0]['warm'][0]['profile']['enabled']=True
    try:profile_checks(bad,stage=='profile')
    except ValueError:negative.append('profile_count_or_activation')
    else:raise ValueError('Accepted wrong profile contract')
    if stage=='profile':
        bad=copy.deepcopy(data);bad['cases'][0]['warm'][0]['profile']['stages']['quality_native_MAS']['calls']+=1
        try:profile_checks(bad,True)
        except ValueError:negative.append('nested_MAS_double_count')
        else:raise ValueError('Accepted nested MAS double counting')
        csv_new(REPORT/'tables/exclusive_profile.csv',details)
    write_new(folder/'profile_audit.json',dict(passed=True,enabled=stage=='profile',solves=len(details),negative_rejected=negative,source=record(Path(__file__))))
    print(dict(profile_contract_passed=True,stage=stage,profiled_solves=len(details)),flush=True)

def finish():
    packet,identity,manifest=verify_identity();verify(read(PARENT)['files'])
    for row in read(BUILD/'generation.json')['files']:
        verify([row['source'],row['target']]);actual=Path(row['target']['path']).read_text(encoding='utf-8')
        require(actual==edit(Path(row['source']['path']).read_text(encoding='utf-8'),row['edits']),'Generated source reconstruction failed')
    jobs=[];solves=0;vectors=0
    for stage in ['contracts','memcheck','profile','measure']:
        folder=RAW/stage;job=read(folder/'result.json');a=read(folder/'audit.json');extra=read(folder/'profile_audit.json');data=read(folder/'data.json')
        require(job['passed'] and job['cleanup_owned_job_empty'] and a['passed'] and extra['passed'],'Incomplete stage')
        require(0<job['wall_seconds']<=240 and job['exit_code'] in ([0,99] if stage=='memcheck' else [0]),'Exit/deadline')
        for sample in job['samples']:
            require(sample['power_guid']==packet['power_guid'] and sample['gpu']['uuid']==job['gpu_before']['uuid'],'Power/UUID drift')
            require(sample['gpu']['memory.free']>=768 and job['gpu_before']['memory.free']-sample['gpu']['memory.free']<=job['memory_budget_mib'],'Memory budget')
            require(not sample['processes']['blocking_foreign_pids'],'Foreign compute')
        count=previous.metadata(data,manifest,folder,'contracts' if stage=='memcheck' else stage);require(count==a['vector_files'],'Output coverage')
        profile_checks(data,stage=='profile');solves+=a['solves'];vectors+=count
        jobs.append(dict(stage=stage,wall_seconds=job['wall_seconds'],exit_code=job['exit_code'],instrumented=job['instrumented']))
    from m3_numeric import FrozenSystem
    data=read(RAW/'measure/data.json');systems={r['name']:FrozenSystem(r['prefix']) for r in manifest['cases']}
    _,cells,_=previous.stats(data,systems);a=read(RAW/'measure/audit.json');require(cells==a['cells'],'Statistics reconstruction differs')
    aa=all(c['signal_passed'] for c in cells if c['comparison']=='AA');ab=all(c['signal_passed'] for c in cells if c['comparison']=='AB')
    decision=dict(numerical_passed=True,AA_passed=aa,direct_MAS_signal_all_cells=ab,iteration_budget_passed=a['iteration_budget_passed'],
        next_integration_eligible=aa and ab and a['iteration_budget_passed'],production_modified=False,performance_certified=False,
        frozen_space_setup_included=False,solves=solves,vector_files=vectors)
    lines=['# 缓存 balanced 与原 MAS：完整固定系统 PCG 直接对照','',
        '同一封存 A/b、原生 ABD/MAS、同一质量验收。A=原 MAS，B=缓存 balanced。Graph 仅重放预条件器 apply。野兽模式保持一致。',
        '正式加速比来自384次无逐操作同步的配对求解，包含原trace和显式质量检查；另有独立profile运行，不混入加速比。','',
        '| 场景 | 执行 | MAS ms | 缓存balanced ms | 配对加速比 | MAS/缓存迭代 | 三轮B/A | A/A | 判定 |',
        '|---|---|---:|---:|---:|---:|---|---|---|']
    for c in cells:
        if c['comparison']!='AB':continue
        control=next(v for v in cells if v['scene']==c['scene'] and v['execution']==c['execution'] and v['comparison']=='AA')
        accepted=control['signal_passed'] and c['signal_passed'] and c['iteration_budget_passed']
        reason='通过本轮门' if accepted else 'A/A未通过' if not control['signal_passed'] else '未达稳定2%收益门'
        rounds='/'.join(f'{v:.4f}' for v in c['rounds'])
        lines.append(f'| {c["scene"]} | {c["execution"]} | {c["A_ms"]:.4f} | {c["B_ms"]:.4f} | {c["speedup"]:.4f}x | {c["A_iterations"]:g}/{c["B_iterations"]:g} | {rounds} | {"通过" if control["signal_passed"] else "未通过"} | {reason} |')
    lines+=['','ms为quartet中两次均值的中位数；配对比是B/A中位数的倒数，不能直接把表中ms相除替代。',
        f'最大独立原CSR相对L2={a["max_CPU_l2"]:.9g}，最大CPU MAS加权残差={a["max_CPU_MAS"]:.9g}；均<=.01。',
        f'总计{solves}次求解、{vectors}个向量核验；每次解、r/Mr、迭代标量及质量调度均保存。','',
        '## 独立同步成本诊断（非性能计时）','',
        '| 场景 | 执行 | 方法 | 迭代 | solve ms | apply ms | SpMV ms | dot/回读 ms | 质量MAS ms | 向量更新 ms | 其余 ms |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    prof=read(RAW/'profile/data.json');costs=profile_checks(prof,True)
    for r in costs:
        vectors_ms=r['vector_update_x_r_ms']+r['vector_update_direction_ms']
        other=r['solve_host_ms']-sum(r[k] for k in ['recurrence_apply_ms','SpMV_ms','dot_with_host_readback_ms','quality_native_MAS_ms'])-vectors_ms
        lines.append(f'| {r["scene"]} | {r["execution"]} | {r["method"]} | {r["iterations"]} | {r["solve_host_ms"]:.4f} | {r["recurrence_apply_ms"]:.4f} | {r["SpMV_ms"]:.4f} | {r["dot_with_host_readback_ms"]:.4f} | {r["quality_native_MAS_ms"]:.4f} | {vectors_ms:.4f} | {other:.4f} |')
    lines+=['','apply包含整个预条件动作；balanced内部MAS按嵌套深度排除重复记账。质量MAS仅记显式检查中的原MAS。其余含初始化copy、显式残差核、Host控制与记录；完整细分见tables/exclusive_profile.csv。',
        '逐操作同步改变调度，这些是诊断中的Host耗时，不能当纯核时间，也不能直接与正式measure相减。每类别调用数已独立按迭代/质量检查/筛选次数核对。',
        'quality/screen旧计时是包含关系，不能再与分类总和相加。关闭profile的全部正式/合同运行均验证空计数。','',
        '## 证据边界与下一步','',
        '固定空间的Z/W/L从历史检查点加载，粗空间构建、存储、刷新成本尚未包含；因此即使本轮加速，也只是可复用粗空间下的求解收益。',
        'memcheck完整日志和原始退出码保留：若为99，只允许预先声明的32条metadata查询API错误，严格核验没有其他内存/API finding；不是工具零错误通过。',
        '未修改生产推进，未运行100帧，未证明整场景2倍。',
        '若Bunny通过而Mixed未通过，应先解释Mixed每次balanced apply的额外成本与迭代收益为何不匹配，按场景决定是否启用；不得用Bunny的通过替代Mixed的证据。','',
        '完整配置与判据：[PLAN.md](PLAN.md)。逐行实现审查：[SOURCE_REVIEW.md](SOURCE_REVIEW.md)。','',
        '验证命令：`E:/Anaconda/envs/DL/python.exe -X utf8 -B tools/bench/mas_cached_experiment.py finish`。','',
        '```json',json.dumps(decision,ensure_ascii=False,indent=2),'```']
    write_new(REPORT/'DECISION.json',decision);(REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    source_files={Path(__file__),ROOT/'tools/local/prepare_mas_cached.py',ROOT/'tools/local/run_mas_cached.py',ROOT/'tools/bench/cache_pcg_experiment.py',ROOT/'tools/local/recover_cache_pcg_memcheck.py'}
    sources=REPORT/'sources';sources.mkdir();copies=[]
    for path in sorted(source_files):
        target=sources/path.name;target.write_bytes(path.read_bytes());copies.append(dict(source=record(path),copy=record(target)))
    write_new(REPORT/'SOURCE_CLOSURE.json',dict(files=copies,jobs=jobs))
    files=tree(RAW)+tree(REPORT)+tree(PROJECT)+tree(BUILD/'overlay')+tree(BUILD/'project')+[record(BUILD/'identity.json'),record(BUILD/'generation.json')]+[record(p) for p in sorted(source_files)]
    verify(files);write_new(REPORT/'DELIVERY_SEAL.json',dict(files=files,exe=identity['exe'],decision=decision));verify(read(REPORT/'DELIVERY_SEAL.json')['files'])
    print(dict(delivery_rehashed=True,files=len(files),**decision),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['resources','audit','finish']);p.add_argument('stage',nargs='?');args=p.parse_args()
    previous.resources() if args.action=='resources' else audit(args.stage) if args.action=='audit' else finish()
