"""Final independent delivery checks, inclusive cost tables and source copies."""
import csv
import json
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_cache_pcg import RAW,REPORT,BUILD,PROJECT,PARENT,read
import cache_pcg_experiment as main_audit
from recover_cache_pcg_memcheck import parse
from local_identity import record,verify,tree
from linux_runner import require,write_new
from m3_numeric import FrozenSystem
from analyze_native_balanced import csv_new

def main():
    # Preserve the original runtime/analysis seal. The sole later source edit
    # is a missing import used only by the report formatter. Verify that exact
    # byte delta against the pre-repair source copy before adapting its record.
    original_source=REPORT/'report_finalize_attempt1/sources/cache_pcg_experiment.py'
    current_source=ROOT/'tools/bench/cache_pcg_experiment.py'
    old=original_source.read_bytes();new=current_source.read_bytes()
    require(old.count(b'import copy\n')==1 and new==old.replace(b'import copy\n',b'import copy\nimport json\n',1),'Unexpected post-measure source change')
    original_read=main_audit.read;packet=original_read(RAW/'runtime_seal.json')
    originals=[r for r in packet['files'] if Path(r['path']).resolve()==current_source.resolve()]
    archived=record(original_source)
    require(len(originals)==1 and originals[0]['sha256']==archived['sha256'] and originals[0]['bytes']==archived['bytes'],'Original analysis identity differs')
    repair=dict(scope='Report-only missing json import; numeric/timing code unchanged',old_runtime_record=originals[0],archived=archived,current=record(current_source),
        old_bytes='import copy\\n',new_bytes='import copy\\nimport json\\n')
    def report_read(path):
        value=original_read(path)
        if Path(path).resolve()==(RAW/'runtime_seal.json').resolve():
            value=dict(value,files=[repair['current'] if r==originals[0] else r for r in value['files']])
        return value
    main_audit.read=report_read
    write_new(REPORT/'REPORT_IMPORT_REPAIR.json',repair)
    _,_,manifest=main_audit.verify_identity();require(not (REPORT/'DELIVERY_SEAL.json').exists(),'Already delivered')
    compat=read(RAW/'memcheck/compatibility.json');verify([compat[k] for k in ['source','failed_ledger','tool_log','tool']])
    require(compat['passed'] and compat['known_inspection_api_reports']==32 and compat['memory_findings']==0
        and not compat['API_errors_suppressed'],'Compatibility scope changed')
    require(parse(Path(compat['tool_log']['path']).read_text(encoding='utf-8'),32)['memory_findings']==0,'Unexpected sanitizer finding')
    all_jobs=[];vectors=0
    for stage in ['contracts','memcheck','measure']:
        folder=RAW/stage;job=read(folder/'result.json');data=read(folder/'data.json')
        require(job['cleanup_owned_job_empty'] and 0<job['wall_seconds']<=240,'Owned job/deadline failure')
        require(job['exit_code']==(99 if stage=='memcheck' else 0) and job['passed']==(stage!='memcheck'),'Wrong original exit/ledger')
        before=job['gpu_before'];require(job['samples'],'No resource samples')
        for sample in job['samples']:
            require(sample['power_guid']==job['power_guid'] and sample['gpu']['uuid']==before['uuid'],'Power/UUID drift')
            require(sample['gpu']['memory.free']>=768 and before['memory.free']-sample['gpu']['memory.free']<=job['memory_budget_mib'],'Resource budget')
            require(not sample['processes']['blocking_foreign_pids'],'Foreign compute')
        count=main_audit.metadata(data,manifest,folder,'contracts' if stage=='memcheck' else stage);vectors+=count
        require(read(folder/'audit.json')['vector_files']==count,'Output audit count drift')
        all_jobs.append(dict(stage=stage,exit_code=job['exit_code'],original_passed=job['passed'],wall_seconds=job['wall_seconds'],vectors=count))
    data=read(RAW/'measure/data.json');systems={c['name']:FrozenSystem(c['prefix']) for c in manifest['cases']}
    _,cells,_=main_audit.stats(data,systems);require(cells==read(RAW/'measure/audit.json')['cells'],'Paired statistics do not reproduce')
    costs=[];contexts=[];normalized=[]
    for c in data['cases']:
        for execution in ['host','graph_apply1']:
            for method in ['original','cached']:
                bank=[r for r in c['runs'] if r['comparison']=='AB' and r['execution']==execution and r['method']==method]
                require(len(bank)==24,'Cost arm coverage')
                values={key:st.mean(r[key] for r in bank) for key in ['solve_host_ms','solve_event_ms','reset_host_ms','export_audit_host_ms']}
                for key in ['iterations','quality_checks','quality_host_ms','screen_checks','screen_host_ms','preconditioner_calls']:
                    values[key]=st.mean(r['result'][key] for r in bank)
                values['remaining_host_ms']=values['solve_host_ms']-values['quality_host_ms']-values['screen_host_ms']
                costs.append(dict(scene=c['name'],execution=execution,method=method,**values))
            ratios=[];event=[]
            for rnd in [1,2,3]:
                for q in [1,2,3,4]:
                    bank=[r for r in c['runs'] if r['comparison']=='AB' and r['execution']==execution and r['round']==rnd and r['quartet']==q]
                    a=[r for r in bank if r['label']=='A'];b=[r for r in bank if r['label']=='B']
                    ratios.append(st.mean(r['solve_host_ms']/r['result']['iterations'] for r in b)/st.mean(r['solve_host_ms']/r['result']['iterations'] for r in a))
                    event.append(st.mean(r['solve_event_ms'] for r in b)/st.mean(r['solve_event_ms'] for r in a))
            normalized.append(dict(scene=c['name'],execution=execution,per_iteration_speedup=1/st.median(ratios),event_speedup=1/st.median(event)))
            for method in ['MAS','original','cached']:
                r=next(r for r in c['warm'] if r['method']==method and r['execution']==execution)
                contexts.append(dict(scene=c['name'],execution=execution,method=method,host_ms=r['solve_host_ms'],iterations=r['result']['iterations'],single_warm_context=True))
    csv_new(REPORT/'tables/inclusive_cost_means.csv',costs);csv_new(REPORT/'tables/single_warm_context.csv',contexts)
    csv_new(REPORT/'tables/normalized_pairs.csv',normalized)
    lines=['# 结果解释与成本记账','',
        '缓存核的局部 apply 收益没有等比例传递到完整 PCG。本轮384次正式求解、12次warm、12次数值合同和12次仪器化合同共420次非零 RHS 求解；另有36个零输入预条件动作输出。共1308个向量文件逐个核验。',
        '所有求解通过独立原 CSR/CPU MAS 质量验收及逐次递推/停止调度审核，没有超过110%的配对迭代预算。',
        'Bunny Host 观察加速1.0178x，低于预设2%门；Mixed Graph虽然观察1.0325x，但第三轮A/A中位数B/A=1.06729，超过5%门。两个限制必须保留，不能选择性只报通过的格或修改判据。',
        'Bunny Graph观察1.0372x、Mixed Host观察1.0376x，两格通过本轮预设控制与信号门；仍仅代表固定系统诊断，不是模拟场景认证。','',
        '## 相同配对样本的加法成本（算术均值，ms）','',
        '| 场景 | 执行 | 核 | solve | 显式质量 | 递推筛选 | 其余 | 平均迭代 | 平均完整检查 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for c in costs:
        lines.append(f'| {c["scene"]} | {c["execution"]} | {c["method"]} | {c["solve_host_ms"]:.4f} | {c["quality_host_ms"]:.4f} | {c["screen_host_ms"]:.4f} | {c["remaining_host_ms"]:.4f} | {c["iterations"]:.2f} | {c["quality_checks"]:.2f} |')
    lines+=['','三项之和等于同一组 solve 均值。其余项包含所有预条件动作、SpMV、点积及回读、向量更新、Host控制和trace；本轮没有把它进一步假拆成单核时间。Event 时长也包含同一流上的Host供给间隙，不能解释成纯GPU计算时间。',
        '重置与导出审核在solve外，均已逐次记录；本轮测量不是无诊断记录的生产运行。','',
        '## 迭代数影响与 Event 交叉检查','',
        '| 场景 | 执行 | 按每次求解的 ms/迭代归一后的配对比 | Event配对加速比 |',
        '|---|---|---:|---:|']
    for c in normalized:lines.append(f'| {c["scene"]} | {c["execution"]} | {c["per_iteration_speedup"]:.4f}x | {c["event_speedup"]:.4f}x |')
    lines+=['','归一化只用于观察迭代数变化，不能隔离tiny核因果成本；每步是否执行完整质量检查也会改变成本。',
        '最终解差异及A能量差保存在measure_solution_differences.csv；两版不要求浮点非确定MAS下逐位相同。',
        '## 原 MAS 上下文（每格仅一次warm，不作正式配对结论）','',
        '| 场景 | 执行 | MAS ms/迭代 | 原balanced ms/迭代 | 缓存balanced ms/迭代 |',
        '|---|---|---|---|---|']
    for name in ['bunny_rhs','mixed_rhs']:
        for execution in ['host','graph_apply1']:
            bank=[next(r for r in contexts if r['scene']==name and r['execution']==execution and r['method']==m) for m in ['MAS','original','cached']]
            cells_text=' | '.join(f'{r["host_ms"]:.4f}/{r["iterations"]}' for r in bank)
            lines.append(f'| {name} | {execution} | {cells_text} |')
    lines+=['','这个上下文也不能当成“StiffGIPC整场景Base”结果：它只是同一固定A/b的MAS求解。尤其Mixed仍需正式配对证明balanced相对MAS的总收益，不能把相对旧balanced的局部收益当成超越Base。',
        '## 工具与交付限制','',
        'memcheck原始退出99与32条metadata API错误完整保留；严格核验没有内存finding，不是工具零错误通过。见COMPATIBILITY_AMENDMENT.md。',
        '未修改生产推进，未运行100帧，未宣称2倍。当前证据支持保留缓存核作为私有局部优化；下一步应先对缓存balanced与原MAS做直接配对并分解剩余PCG成本，再决定是否接入真实推进。']
    (REPORT/'DETAILS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=[ROOT/'tools/local'/n for n in ['prepare_cache_pcg.py','run_cache_pcg.py','recover_cache_pcg_memcheck.py']]
    files+=[ROOT/'tools/bench/cache_pcg_experiment.py',Path(__file__)];files+=list(PROJECT.glob('*'))
    copies=REPORT/'sources';copies.mkdir();source_rows=[]
    for path in files:
        dest=copies/path.name;require(not dest.exists(),'Source copy exists');dest.write_bytes(path.read_bytes())
        source_rows.append(dict(source=record(path),copy=record(dest)))
    write_new(REPORT/'SOURCE_CLOSURE.json',dict(files=source_rows,jobs=all_jobs,vector_files=vectors,recomputed_statistics_equal=True,
        memcheck_original_exit=99,expected_metadata_API_reports=32,unexpected_memory_findings=0))
    main_audit.finish()
    verify(read(REPORT/'DELIVERY_SEAL.json')['files']);print(dict(delivery_rehashed=True,solves=420,vector_files=vectors),flush=True)
if __name__=='__main__':main()
