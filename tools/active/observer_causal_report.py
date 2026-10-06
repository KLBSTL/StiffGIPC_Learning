"""Readable synthesis of observation controls and fixed-operator evidence."""
import statistics
from pathlib import Path
import numpy as np
from config import ROOT,read,sha
from ipc_benchmark import write
from observer_causal import TAG

def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.9))
    labels=['Raw OFF','Observed OFF','Observed ON'];colors=['#747474','#487bbb','#db9745'];x=np.arange(3)
    for axis,field,label in [(axes[0],'max_stretch','Whole-segment maximum stretch'),(axes[1],'p99_stretch','Maximum per-frame p99 stretch')]:
        for i,(arm,c) in enumerate(zip(data['summary'],colors)):
            values=data['summary'][arm][field];axis.scatter(i+np.linspace(-.08,.08,5),values,color=c,s=35)
            axis.plot([i-.16,i+.16],[np.median(values)]*2,color=c,lw=2)
        bound=data['summary']['raw_off']['bounds'][field];axis.axhline(bound,color='#b44343',ls='--',lw=1,label='Old frozen bound')
        axis.set_xticks(x,labels,rotation=12);axis.set_ylabel(label);axis.ticklabel_format(style='plain',axis='y',useOffset=False)
        axis.legend(loc='best',fontsize=8)
    cases=['within_raw_off','within_observed_off','within_observed_on','binary_identity','velocity_observation']
    for i,k in enumerate(cases):
        y=[p['max_frame_cloth_rms_m']*100 for p in data['pairs'][k]]
        axes[2].scatter(i+np.linspace(-.1,.1,len(y)),y,color='#467b8b',s=20)
    axes[2].set_xticks(np.arange(5),['Raw repeats','OFF repeats','ON repeats','Raw vs OFF','OFF vs ON'],rotation=24)
    axes[2].set_ylabel('Maximum frame cloth RMS difference (cm)')
    for a in axes:a.spines[['top','right']].set_visible(False)
    fig.suptitle('Fixed bunny cloth: observation controls (5 runs per arm)')
    fig.text(.5,.01,'No quality threshold was widened. Shared desktop diagnosis; pairwise comparisons are correlated.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.94]);folder=ROOT/'reports/active/figures'
    for extension in ['png','pdf']:fig.savefig(folder/f'{TAG}.{extension}',dpi=170)
    plt.close(fig)

def main():
    folder=ROOT/'reports/active';data=read(folder/f'{TAG}_analysis.json')
    probe=read(folder/'ipc_observer_fixed_probe_20261005_analysis.json');plot(data)
    names={'raw_off':'原版 / 速度关','observed_off':'同观测程序 / 速度关','observed_on':'同观测程序 / 速度开'}
    lines=['# 固定兔子重复性与观测影响：下一步实施结果（2026-10-05）','',
        '**本轮已确认一个具体数值机制：同一进程、同一份已冻结 A/b/M 上，legacy MAS 的作用输出不是逐位重复。** f2 的相对差异为6.17e-8–1.25e-7，f25为3.77e-8–4.70e-8；同系统SpMV差异约1e-16/1e-17。原方向逐位恢复，CPU准确参考通过。它是微差来源的直接证据，尚不是“所有长窗分岔均由MAS造成”的证明。',
        '', '15次三臂交错对照未隔离出稳定的程序身份或速度导出效应。原版关闭速度导出自身也分岔；三个臂都可能超出旧材料范围。旧范围与候选失败均保留，不重标定、不追认质量通过。',
        '', '## 并行分工与执行边界','',
        '本轮按用户要求启用子代理：应用观测代码审查、历史逐帧前缀审计、质量协议复核并行；主代理编排GPU串行实验、固定系统探针和综合分析。子代理没有修改构建输入或运行GPU。',
        '', '总预算为15次固定兔子连续100帧诊断，加f2/f25两个受保护固定系统诊断。均从零推进、dt=.01、单次120秒上限。没有追加重复直到通过，没有写新求解kernel。两个生产程序及活动程序SHA、材料、停止阈值保持不变。AutoDL没有连接。',
        '', '## 1. 原质量标定的混杂已拆开','',
        '旧标定来自观测程序＋每帧速度导出，正式原版来自另一程序＋速度关。因此旧比较同时变了程序身份与运行路径。应用差分只新增9行D2H＋写盘；它没有GPU回写，每帧求解后原本已有cudaDeviceSynchronize，新增导出不修补缺失屏障。',
        '', '新对照先冻结协议，再运行：原版速度关(A)、同观测程序速度关(B)、同观测程序速度开(C)，三臂各5次。A/B隔离程序/构建身份，B/C隔离速度导出整条路径；其他参数、逐帧位置导出、输入与停止一致。不能进一步把B/C变化归到D2H、I/O或某个kernel。',
        '', '| 臂 | 最大拉伸5次范围 | max逐帧p99范围 | PCG中位 | 旧协议通过 | 速度文件数 |','|---|---|---|---:|---:|---:|']
    for arm,r in data['summary'].items():
        lines.append(f'| {names[arm]} | {min(r["max_stretch"]):.9f}–{max(r["max_stretch"]):.9f} | {min(r["p99_stretch"]):.9f}–{max(r["p99_stretch"]):.9f} | {statistics.median(r["pcg"]):.0f} | {r["material_passed"]}/5 | {r["velocity_frames"][0]} |')
    lines+=['', '原版两次失败来自p99，观测速度关一次来自最大拉伸，观测速度开一次来自p99。开启速度导出不保证进入旧范围；关闭也不保证失败。因此不支持用一个开关直接解释旧5次观测标定为什么恰好通过。',
        '', '| 单因素对照 | 平均PCG变化 | 平均最大拉伸变化 | 平均p99变化 |','|---|---:|---:|---:|']
    for k,label in [('binary_identity','A→B：程序身份'),('velocity_observation','B→C：速度导出')]:
        e=data['effects'][k];lines.append(f'| {label} | {e["pcg"]["mean_b_minus_a"]:+.1f} | {e["max_stretch"]["mean_b_minus_a"]:+.3e} | {e["p99_stretch"]["mean_b_minus_a"]:+.3e} |')
    lines+=['', '配对变化方向混合，材料及工作量变化与组内重复波动重叠。分析保留5对符号翻转枚举，仅为描述性诊断；固定平衡顺序不是随机试验，样本也不足以证明等价。不能用“没有显著差异”宣布观测中性。',
        '', '## 2. 微差在哪个阶段放大','', '| 比较 | 最大帧位置RMS差范围 / m | 首次PCG签名差 | 首次位置RMS>1e-3m |','|---|---|---|---|']
    for k,label in [('within_raw_off','原版内部'),('within_observed_off','观测OFF内部'),('within_observed_on','观测ON内部'),('binary_identity','A/B同轮'),('velocity_observation','B/C同轮')]:
        pairs=data['pairs'][k];v=[p['max_frame_cloth_rms_m'] for p in pairs]
        def first_range(s):
            z=[p['first_divergence'][s] for p in pairs if s in p['first_divergence']]
            return f'{min(z)}–{max(z)}' if z else '未观察'
        lines.append(f'| {label} | {min(v):.6f}–{max(v):.6f} | {first_range("pcg_signature")} | {first_range("0.001")} |')
    lines+=['', '本轮同臂及跨臂的接受alpha微差都从f2开始，布料位置RMS首次超过1e-8m均在f25。PCG整数签名分支集中f25–27；接触候选数量差在f29–33，方向数随后分歧。这是可定位的时间顺序，不能直接等同具体kernel因果。',
        '', '旧19轨迹审计同样发现：f2步长微差→f25–27 PCG分支→f29–32候选数量差→f44峰值/f49 p99越界。距离微差早于数量变化，不把它叫活动集已改变。旧各Graph/base退出种类几乎一致，combined_host r2仅f73有例外；本轮少量重复的退出类型也可在后期变化。不能把所有分岔归给新门控。',
        '', '![观测对照与组内分岔]('+(folder/'figures'/f'{TAG}.png').as_posix()+')','',
        '## 3. 同A/b/M探针：MAS作用是直接证据','',
        '复用当前活动程序的fixed_system_study及mas=legacy支路，选f2和首次分支f25的第一个方向，在同进程保持已准备矩阵、RHS和预条件器数据不变，重复作用与host/Graph求解，再恢复原方向。全系统输入身份未变，两次primary_restored_bitwise=true，预条件器完整导出。不是重加载检查点，也不比较不同状态。这里不是直接在raw原版exe内重放，也没有宣称两个程序逐位等价；算子身份比较允许正常R/Z scratch更新。','',
        '| 帧 | DOF | SpMV重复相对差 | MAS重复相对差 | 默认CG方向重复相对差 |','|---|---:|---|---|---|']
    for s in probe['systems']:
        a=[v['relative'] for v in s['operator_repeats']['spmv']];m=[v['relative'] for v in s['operator_repeats']['preconditioner']]
        c=[v['repeat_difference']['relative'] for v in s['replayed_solves'] if v['rho_tolerance']==1e-4 and v['repeat']>1]
        lines.append(f'| {s["frame"]} | {s["dofs"]} | {min(a):.3e}–{max(a):.3e} | {min(m):.3e}–{max(m):.3e} | {min(c):.3e}–{max(c):.3e} |')
    lines+=['', '这些是整个legacy预条件器作用的重复差，不是限制/延拓子kernel的独立归因。源代码中布料/弯曲梯度和SpMV有共享浮点原子累加，MAS层级还有FP32聚合，均提供微差机制；同状态完整M重放将候选焦点缩小到了预条件器应用，但尚未隔离其内部阶段。',
        '', '## 4. 真残差核查：不混淆rho阈值','', '| 帧 | CPU准确参考真相对残差 | 默认host CG真残差 | 默认Graph CG真残差 | 默认host解相对CPU误差 |','|---|---:|---:|---:|---:|']
    for s in probe['systems']:
        host=next(v for v in s['cpu_solution_checks'] if v['phase']==1 and v['mode']=='host');graph=next(v for v in s['cpu_solution_checks'] if v['phase']==1 and v['mode']=='graph')
        lines.append(f'| {s["frame"]} | {s["cpu_reference"]["true_relative_residual"]:.3e} | {host["cpu_true_relative_residual"]:.6f} | {graph["cpu_true_relative_residual"]:.6f} | {host["relative_solution_error_to_cpu"]:.6f} |')
    lines+=['', 'CPU参考为FP64稀疏LU，真相对残差均远低于1e-8。默认rho=1e-4并不意味着欧氏真残差1e-4；本例默认方向真残差约4.2%/22.8%，解相对CPU参考误差约0.70%/0.97%。这是线性方向近似程度，不能直接解释为位置/速度误差，也不自动构成相对原版的质量失败。host/Graph默认真残差近乎一致，不支持Graph本身把线性精度大幅恶化的解释。',
        '', '已有诊断工具另重放rho=1e-16作准确度对照：迭代从6/22增加至约72/197–198，真残差约2.45e-7–4.82e-7。更严rho也不是对应真残差保证；本轮没有将它设为生产阈值，更没有通过全局收紧PCG“修复”轨迹。',
        '', '## 5. 本轮判定与唯一下一步','',
        '| 结论 | 状态 |','|---|---|',
        '| 原版关闭速度导出仍存在长窗分岔 | 已验证 |',
        '| 新速度导出直接改写物理状态或修补缺失同步 | 已排除（限本批处理代码路径） |',
        '| 构建身份/速度路径是分岔主因 | 在本预算内未隔离 |',
        '| 同已冻结数据的legacy MAS作用存在重复微差 | 已验证（f2/f25） |',
        '| MAS的限制/延拓是具体主因、解释全部非线性分岔 | 待验证 |',
        '| 新/旧候选同材料质量验收与2× | 未通过；不推广默认 |','',
        '下一步限一个受保护系统中的阶段归因探针：先复用无接触f2冻结系统，分别记录legacy MAS限制、局部作用、延拓和整个作用的同输入重复差；各阶段固定上游scratch后记录，核对清零/覆盖和FP32多写者。每阶段1次预热＋3次观测，总计至多16次，不重跑长轨迹、不放宽阈值。只有定位到可复现阶段后才考虑一个默认关闭的有界数值候选；不能把整套慢的稳定MAS直接替换进IPC，也不能先写新kernel再找原因。',
        '', '质量协议本轮不重建。若未来重新验收，必须用与正式条件一致的全新仅基线标定和独立复核，冻结后另跑新候选；不能取本批通过的raw三次重新画范围，也不能追认旧失败。当前材料协议无法认证的结论仍在。',
        '', '## 6. 可复现命令与交付','', '```text',
        'E:/Anaconda/envs/DL/python.exe tools/active/batch.py --plan configs/active/ipc_observer_causal_20261005.json',
        'E:/Anaconda/envs/DL/python.exe tools/active/observer_causal.py analyze',
        'E:/Anaconda/envs/DL/python.exe tools/active/batch.py --plan configs/active/ipc_observer_fixed_probe_20261005.json',
        'E:/Anaconda/envs/DL/python.exe tools/active/observer_fixed_probe.py',
        'E:/Anaconda/envs/DL/python.exe tools/active/observer_evidence_verify.py',
        '```','',
        '三个臂各5条完整轨迹均完成，初态/拓扑/边界/质量/body_ids哈希全部一致，实际速度文件数量0/0/101，位置及已导出速度有限、无生产PCG触顶或breakdown。两固定系统输入/方向恢复检查与独立CPU参考通过。GPU锁已释放；不称17次诊断为性能验收。','',
        '最终证据核验通过：17次运行配置/完成状态检查、75项活动源码/对象/程序身份检查，以及旧质量协议和预声明协议哈希一致性检查。冻结原版/观测程序不输出resolved_config，此部分依据请求环境、程序身份、原生日志与实际导出文件数核对；两个活动程序探针有resolved_config且与请求一致。','',
        '- [最终证据核验](ipc_observer_causal_20261005_verification.json)',
        '- [预声明协议](ipc_observer_causal_20261005_protocol.json)',
        '- [完整15运行分析](ipc_observer_causal_20261005_analysis.json)',
        '- [固定系统＋CPU参考](ipc_observer_fixed_probe_20261005_analysis.json)',
        '- [应用代码审查](OBSERVER_NEUTRALITY_CODE_AUDIT_20261005.md)',
        '- [历史前缀审计](BASELINE_DIVERGENCE_AUDIT_20261005.md)',
        '- [质量协议复核](QUALITY_PROTOCOL_REVIEW_20261005.md)',
        '- [数值路径审查](BASELINE_NUMERICAL_PATH_AUDIT_20261005.md)',
        '- [原实施结果及旧失败](IPC_EXECUTION_RESULTS_20261005.md)','']
    (folder/'IPC_OBSERVER_CAUSAL_RESULTS_20261005.md').write_text('\n'.join(lines),encoding='utf-8')
    print(str(folder/'IPC_OBSERVER_CAUSAL_RESULTS_20261005.md'))

if __name__=='__main__':main()
