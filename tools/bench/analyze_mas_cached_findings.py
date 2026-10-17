"""Post-run explanation only; no new timing, threshold or acceptance changes."""
import json
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_mas_cached import RAW,REPORT,read
from mas_cached_experiment import verify_identity,profile_checks
from local_identity import record,verify
from linux_runner import require,write_new
from analyze_native_balanced import csv_new

def main():
    verify_identity();require(read(RAW/'measure/audit.json')['passed'],'Numerical audit required')
    data=read(RAW/'measure/data.json');profile=read(RAW/'profile/data.json');profile_checks(profile,True)
    rows=[];examples=[];contrasts=[]
    for c in data['cases']:
        for execution in ['host','graph_apply1']:
            for method in ['MAS','cached']:
                bank=[r for r in c['runs'] if r['comparison']=='AB' and r['execution']==execution and r['method']==method]
                require(len(bank)==24,'Matched arm coverage')
                def mean(key):return st.mean(r['result'][key] for r in bank)
                row=dict(scene=c['name'],execution=execution,method=method,samples=len(bank),
                    mean_iterations=mean('iterations'),mean_quality_checks=mean('quality_checks'),mean_screens=mean('screen_checks'),
                    mean_solve_host_ms=st.mean(r['solve_host_ms'] for r in bank),mean_quality_host_ms=mean('quality_host_ms'),mean_screen_host_ms=mean('screen_host_ms'))
                row['mean_remaining_host_ms']=row['mean_solve_host_ms']-row['mean_quality_host_ms']-row['mean_screen_host_ms']
                row['mean_SpMV_calls']=row['mean_iterations']+row['mean_quality_checks']
                row['mean_dot_calls']=st.mean(2*r['result']['iterations']+int(r['result'].get('zero_rho_true_quality_checked',False))+r['result']['screen_checks']+2*r['result']['quality_checks'] for r in bank)
                first=[next(p for p in r['result']['trace'] if p['quality_checked']) for r in bank]
                row['first_quality_iteration_min']=min(p['iteration'] for p in first);row['first_quality_iteration_max']=max(p['iteration'] for p in first)
                if c['name']=='mixed_rhs' and method=='cached':
                    require(all(p['true_relative_l2']<.0095 and p['common_native_MAS_relative']>.0095 for p in first),'First-check mismatch was not consistent')
                    row['first_L2_pass_but_MAS_fail_count']=len(first)
                    require(all(all(p['quality_checked'] for p in r['result']['trace'] if p['iteration']>=f['iteration']) for r,f in zip(bank,first)),
                        'Quality checks were not every update after activation')
                rows.append(row)
            pc=next(v for v in profile['cases'] if v['name']==c['name'])
            original=next(r for r in pc['warm'] if r['method']=='MAS' and r['execution']==execution)
            cached=next(r for r in pc['warm'] if r['method']=='cached' and r['execution']==execution)
            def apply_per_call(r):
                v=r['profile']['stages']['recurrence_apply'];return v['host_ms']/v['calls']
            contrast=dict(scene=c['name'],execution=execution,profile_only=True,
                MAS_apply_us=1000*apply_per_call(original),cached_apply_us=1000*apply_per_call(cached),
                per_apply_cost_ratio=apply_per_call(cached)/apply_per_call(original),
                MAS_iterations=original['result']['iterations'],cached_iterations=cached['result']['iterations'],
                apply_total_delta_ms=cached['profile']['stages']['recurrence_apply']['host_ms']-original['profile']['stages']['recurrence_apply']['host_ms'])
            contrasts.append(contrast)
            if c['name']=='mixed_rhs':
                for r in [original,cached]:
                    first=next(p for p in r['result']['trace'] if p['quality_checked'])
                    examples.append(dict(execution=execution,method=r['method'],iterations=r['result']['iterations'],checks=r['result']['quality_checks'],
                        first=first,terminal=dict(l2=r['terminal_l2'],MAS=r['terminal_MAS'])))
    csv_new(REPORT/'tables/formal_quality_cost_means.csv',rows);csv_new(REPORT/'tables/profile_apply_contrast.csv',contrasts)
    write_new(REPORT/'FINDINGS.json',dict(formal_arm_means=rows,profile_apply_contrasts=contrasts,mixed_profile_examples=examples,
        source=record(Path(__file__)),interpretation='Post-run explanatory analysis; does not change original protocol or gate.'))
    lines=['# 本轮确定的结论与下一步','',
        '直接MAS对照已回答净收益问题：Bunny两模式有收益；Mixed两模式都没有收益。所有A/A格通过，不再把Mixed退化归因于上一轮的计时波动。详细配对数值见REPORT.md。','',
        '## Mixed的两个实际成本来源','',
        '| 执行 | 原MAS每次apply (us) | 缓存balanced每次apply (us) | 单次成本倍数 | 全部apply额外耗时 (ms) |',
        '|---|---:|---:|---:|---:|']
    for r in contrasts:
        if r['scene']=='mixed_rhs':lines.append(f'| {r["execution"]} | {r["MAS_apply_us"]:.3f} | {r["cached_apply_us"]:.3f} | {r["per_apply_cost_ratio"]:.3f} | {r["apply_total_delta_ms"]:+.4f} |')
    lines+=['','上表来自独立同步诊断，各格只有一次，包含同步/提交开销；不能作为纯核时间或无仪器化加速比。但实际操作计数和边界已核对，能定位本轮apply成本膨胀。Mixed迭代从约125降到79，仍不足以抵消单次apply约2.25–2.35倍的成本。','',
        '第二个来源是固定系统诊断的显式质量检查：缓存balanced较早触发rho候选，但原MAS加权残差还未达标，而L2已经达标。现有screen看到L2小于.012后，每一步都执行昂贵完整检查。此时仍必须等MAS加权残差通过，不能直接提前接受。','',
        '| 执行 | 方法 | 正式平均迭代 | 平均完整检查 | 平均完整检查耗时(ms) | 首次检查迭代范围 |',
        '|---|---|---:|---:|---:|---|']
    for r in rows:
        if r['scene']=='mixed_rhs':
            lines.append(f'| {r["execution"]} | {r["method"]} | {r["mean_iterations"]:.3f} | {r["mean_quality_checks"]:.3f} | {r["mean_quality_host_ms"]:.4f} | {r["first_quality_iteration_min"]}–{r["first_quality_iteration_max"]} |')
    lines+=['','正式两模式各24次cached样本，首次检查全部呈现“L2已通过、原MAS加权未通过”，之后直至接受每步都有完整检查。这是全部48次样本核实的调度现象，不是单次profile的偶然结果。',
        '单次profile的例子：第57步L2约.0059/.0066，已经低于.0095；原MAS加权约.0197/.0191，仍高于.0095。因此不能在第57步接受，但当前实现一直检查到第79步，共23次。MAS的对应profile只检查1或3次。',
        '每次完整检查增加原A SpMV、原MAS动作和两个dot/Host回读。于是迭代数减少并不等于所有操作同比减少；实际调用数和正式均值已保存。',
        '质量检查开销属于这套固定系统共同质量验收；不能直接解释成生产100帧代码的相同开销，也不能据此放宽精度。','',
        '## 有限下一步','',
        '保留当前缓存核私有实现，暂不整体接管真实推进。下一轮优先在同一固定A/b上比较显式质量检查调度：保留两项最终显式验收不变，测试减少重复检查与额外迭代之间的折中，并计入全部成本。不得只用递推L2或balanced rho代替原MAS加权验收。',
        '若调度优化仍不足以让Mixed超过原MAS，再细分其两次转置投影/8D求解/原MAS/向量投影成本，选择占比最大的环节做小原型。Bunny只具备固定空间求解收益，接入前仍需计入粗空间构建与刷新成本。',
        '这轮没有改停止规则、阈值、材料、精度或生产推进。']
    (REPORT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    folder=REPORT/'analysis_sources';folder.mkdir();dest=folder/Path(__file__).name;dest.write_bytes(Path(__file__).read_bytes())
    write_new(folder/'identity.json',dict(source=record(Path(__file__)),copy=record(dest)));verify_identity()
    print(dict(formal_arm_rows=len(rows),mixed_first_check_mismatch_samples=48,findings_passed=True),flush=True)
if __name__=='__main__':main()
