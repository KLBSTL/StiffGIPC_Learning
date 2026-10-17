"""Explain audited four-arm results without modifying measurements or gates."""
import json
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_weighted_gpu import RAW,REPORT,read
from weighted_gpu_experiment import verify_identity
from local_identity import record
from linux_runner import require,write_new
from analyze_native_balanced import csv_new

def main():
    verify_identity();audit=read(RAW/'measure/audit.json');require(audit['passed'] and read(RAW/'measure/policy_audit.json')['passed'],'Full audit required')
    data=read(RAW/'measure/data.json');rows=[]
    for c in data['cases']:
        for execution in ['host','graph_apply1']:
            for method in ['MAS','cached']:
                for policy in ['legacy','cooldown2']:
                    comparison='MAS_policy' if method=='MAS' else 'cached_policy'
                    bank=[r for r in c['runs'] if r['execution']==execution and r['method']==method and r['policy']==policy and r['comparison']==comparison]
                    require(len(bank)==24,'Policy-paired sample coverage')
                    def mean(key):return st.mean(r['result'][key] for r in bank)
                    row=dict(scene=c['name'],execution=execution,method=method,policy=policy,samples=len(bank),
                        iterations_mean=mean('iterations'),quality_checks_mean=mean('quality_checks'),norm_checks_mean=mean('screen_checks'),
                        queries_mean=mean('schedule_queries'),solve_host_mean_ms=st.mean(r['solve_host_ms'] for r in bank),
                        quality_host_mean_ms=mean('quality_host_ms'),screen_host_mean_ms=mean('screen_host_ms'),
                        iterations_min=min(r['result']['iterations'] for r in bank),iterations_max=max(r['result']['iterations'] for r in bank),
                        checks_min=min(r['result']['quality_checks'] for r in bank),checks_max=max(r['result']['quality_checks'] for r in bank))
                    row['SpMV_calls_mean']=row['iterations_mean']+row['quality_checks_mean']
                    row['dot_calls_mean']=st.mean(2*r['result']['iterations']+int(r['result'].get('zero_rho_true_quality_checked',False))+r['result']['screen_checks']+2*r['result']['quality_checks'] for r in bank)
                    rows.append(row)
    csv_new(REPORT/'tables/policy_paired_means.csv',rows)
    cells=audit['cells'];lines=['# 四臂结果解释','',
        '主净收益比较使用两方法各自相同调度，调度收益则比较同一方法的新旧调度。以下数值来自本轮同一二进制，不能直接与上一轮不同诊断记录开销的数值相减。','',
        '| 场景 | 执行 | 旧调度cached/MAS配对加速 | 新调度cached/MAS配对加速 | cached新/旧调度配对加速 | 新调度A/A |',
        '|---|---|---:|---:|---:|---|']
    def cell(scene,execution,comparison):return next(r for r in cells if r['scene']==scene and r['execution']==execution and r['comparison']==comparison)
    for name in ['bunny_rhs','mixed_rhs']:
        for execution in ['host','graph_apply1']:
            old,new,policy,control=[cell(name,execution,k) for k in ['legacy_net','cooldown_net','cached_policy','AA_cooldown']]
            lines.append(f'| {name} | {execution} | {old["speedup"]:.4f}x | {new["speedup"]:.4f}x | {policy["speedup"]:.4f}x | {"通过" if control["signal_passed"] else "未通过"} |')
    lines+=['','这里“加速”都是A/B配对B/A中位数倒数，不是把两组独立均值或中位时间直接相除。A/A未通过的格不认证稳定净收益，其他门槛详见PLAN与完整REPORT。','',
        '## 同方法调度比较中的实际工作量','',
        '| 场景 | 执行 | 方法 | 调度 | 平均迭代 | 完整检查 | rr归约 | 完整检查ms | 筛选ms |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["scene"]} | {r["execution"]} | {r["method"]} | {r["policy"]} | {r["iterations_mean"]:.3f} | {r["quality_checks_mean"]:.3f} | {r["norm_checks_mean"]:.3f} | {r["quality_host_mean_ms"]:.4f} | {r["screen_host_mean_ms"]:.4f} |')
    bunny_host=cell('bunny_rhs','host','cached_policy');bunny_control=cell('bunny_rhs','graph_apply1','AA_cooldown')
    lines+=['','rr归约次数是实际执行计数；调度查询仍可能每步发生。完整检查仍含原A SpMV、原MAS和两个dot及Host回读，减少检查不是放宽质量。迭代增加、额外记录和全部显式检查都已计入solve。',
        '被旧轨迹删失的候选末尾已通过本轮真实求解补出，不能用旧终点强制接受。状态机在每次求解重置；预算失败后继续正常解、逐步状态/事件和CPU残差核验均通过。',
        '', '## 尚未通过的整体验收', '',
        f'Bunny Host的cached_policy中位耗时B/A={bunny_host["ratio"]:.8f}，即增加{100*(bunny_host["ratio"]-1):.3f}%，超过预设2%不退化门。三轮B/A为'+ '/'.join(f'{v:.6f}' for v in bunny_host['rounds'])+'。不能凭Mixed收益忽略此格。',
        'Bunny Graph的AA_cooldown三轮B/A为'+ '/'.join(f'{v:.6f}' for v in bunny_control['rounds'])+
            '；第三轮超出5%控制门，虽然总中位偏差在2%以内，仍判该控制未通过。保留全部原始样本，不改门或自动追加样本。',
        '因此本轮可以确认Mixed局部诊断调度有价值，但不判四格整体通过。Bunny的代价是否来自新增Host判定、迭代轨迹差或计时噪声尚未隔离，不能直接归因；需要下一轮预先固定的小范围分析/对照。',
        '## 后续边界','',
        '本轮是固定空间的诊断求解。即使新调度使Mixed超过MAS，仍不能证明包含历史采集、Z构造、W/E刷新、Graph重建和回退的净收益，更不能证明真实帧2倍。',
        '下一关先解释Bunny Host不退化失败和Graph控制波动；然后再按路线检查Mixed剩余单次apply成本与setup预算，满足净收益才做默认关闭的真实推进短段接入。','',
        'memcheck只有已知32条metadata查询API报告，原退出99保留，无其他内存/API finding；不能称工具零错误通过。']
    (REPORT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_new(REPORT/'FINDINGS.json',dict(paired_arm_means=rows,cells=cells,source=record(Path(__file__)),new_performance_runs=0))
    source=REPORT/'analysis_sources';source.mkdir();dest=source/Path(__file__).name;dest.write_bytes(Path(__file__).read_bytes())
    write_new(source/'identity.json',dict(source=record(Path(__file__)),copy=record(dest)));verify_identity()
    print(dict(paired_arm_mean_rows=len(rows),new_runs=0),flush=True)
if __name__=='__main__':main()
