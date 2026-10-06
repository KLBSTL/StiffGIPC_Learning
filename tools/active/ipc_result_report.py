"""Chinese implementation report and scientific figures from saved evidence."""
import collections
import json
import statistics
from pathlib import Path
import numpy as np
from config import ROOT, read, sha
from ipc_benchmark import TAG, SCENES

def figure(perf,residual):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    names=['Hanging cloth','Fixed bunny cloth'];x=np.arange(2);width=.19
    colors=['#b7b7b7','#4776b4','#e1aa47','#419773']
    for j,(v,label) in enumerate([('base','Stiff'),('reference','Frozen Graph'),('combined_host','Execution host'),('combined','Execution Graph')]):
        y=[perf['scenes'][k][v]['paired_base_speedup'] for k in SCENES]
        bars=axes[0].bar(x+(j-1.5)*width,y,width,label=label,color=colors[j])
        axes[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    axes[0].set_xticks(x,names);axes[0].set_ylim(0,2.18);axes[0].set_ylabel('Paired median speedup vs Stiff')
    axes[0].axhline(2,color='#777777',ls='--',lw=1);axes[0].legend(ncol=2,loc='upper left',fontsize=8)
    for j,(v,label) in enumerate([('gated','Gate 0.01 / 0.03'),('gate001','Gate 0.001 / 0.03')]):
        y=[residual['summary'][k][v]['pcg']/residual['summary'][k]['shadow']['pcg'] for k in SCENES]
        bars=axes[1].bar(x+(j-.5)*.3,y,.3,label=label,color=['#df9863','#bb5b62'][j]);axes[1].bar_label(bars,fmt='%.3f',padding=3)
    axes[1].axhline(1,color='#777777',ls='--',lw=1);axes[1].set_ylim(0,2.25)
    axes[1].set_xticks(x,names);axes[1].set_ylabel('PCG work / legacy shadow');axes[1].legend(fontsize=8)
    fig.suptitle('Local IPC execution and stopping experiments (100 frames, dt=0.01)')
    fig.text(.5,.01,'RTX 3070 Laptop, 3 paired desktop repeats. Performance and quality are not certified.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]);folder=ROOT/'reports/active/figures';folder.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(folder/f'{TAG}_performance.{suffix}',dpi=170)
    plt.close(fig)

def main():
    p=ROOT/'reports/active';perf=read(p/f'{TAG}_performance.json');res=read(p/f'{TAG}_residual_analysis.json')
    comp=read(p/f'{TAG}_components_analysis.json');cost=read(p/f'{TAG}_selected_cost_corrected.json')
    evidence=read(p/f'{TAG}_residual_evidence.json');ccd=read(p/f'{TAG}_accepted_ccd.json');traj=read(p/f'{TAG}_trajectory_comparison.json')
    material=read(p/f'{TAG}_final_material_checks.json')['records'];audit=read(p/f'{TAG}_audit_material_checks.json')['records']
    mixed=next(r for r in read(p/f'{TAG}_audit_analysis.json')['runs'] if r['scene_key']=='mixed')
    cpu=read(p/f'{TAG}_historical_cpu_systems.json');fixture=read(ROOT/f'runs/active/{TAG}_fixtures/result.json')
    manifest=read(ROOT/'builds/active/manifest.json');figure(perf,res)
    lines=['# IPC＋CUDA Graph 执行优化与残差门控实施结果（2026-10-05）','',
        '本轮按批准计划完成本机实施和有界实验。IPC 停止配置、完整帧成本追踪、CPU/GPU 残差核对、冻结构建身份、已有组件消融、两种门控、条件补偿对照和独立接受路径 CCD 均已落地。',
        '',f'**结论：局部诊断有增量收益，但阶段目标及质量认证未通过。** 相对冻结 Graph 的两场景配对中位加速比几何平均为 {perf["combined_vs_frozen_graph_geometric_mean"]:.6f}×，低于预声明 1.10×。固定兔子的基线及候选存在冻结材料范围越界；没有修改范围来追认通过。',
        '', 'IPC 默认仍为 `legacy`，`.01` 累计阈值、最少 6 次更新、`pcg_rho_tol=1e-4` 不变。新门控仅显式启用；本轮执行组合也未推广默认。按用户决定先做本机；本轮没有连接 AutoDL。',
        '', '## 1. 实施内容与真实身份','',
        '| 项目 | 已完成的实现 |','|---|---|',
        '| 独立 IPC 配置 | `solver/ipc_options.h`；legacy / movement_only / gated / compensated；累计阈值与原 movement 阈值分开，旧配置仍可读 |',
        '| 残差观测 | `solver/ipc_residual.inl`；GPU 最大范数，排除固定 FEM 自由度，分别输出 ABD/FEM，非有限传播，独立 CPU 审计开关 |',
        '| 接受步预算 | `solver/ipc_budget.h`；固定第六次有效更新前参考，零/全步和非法步处理、预算恒等式、历史否决 |',
        '| 求解观测 | `core/GIPC.cu` 输出退出原因、有效更新、alpha、beta、Kappa、接触量、系统编号；门控显式替换旧累计出口，保留原 movement 时序 |',
        '| 成本追踪 | 完整物理帧、退出轮装配、普通/FullCCD BVH、查询、CCD、能量与线搜索；修复嵌套追踪过早 flush 和退出轮 CUDA 事件释放 |',
        '| 仅观测原版 | `sources/stiff_base_observed` 只替换应用导出文件，导出真实速度，冻结材料及求解代码保持原版 |',
        '| 公共工具 | `ipc_benchmark.py`、`ipc_delivery.py`、公共 runner/config/validator；每运行 requested/resolved 配置及程序身份落盘 |',
        '', '| 程序 | SHA-256 |','|---|---|']
    for label,path in [('原版 Stiff','builds/local-base/Release/gipc.exe'),('冻结 Graph','builds/ipc-reference/Release/gipc.exe'),('观测原版','builds/base-observed/Release/gipc.exe'),('当前活动程序','builds/active/Release/gipc.exe')]:
        lines.append(f'| {label} | `{sha(ROOT/path)}` |')
    lines+=['',f'活动构建记录实际源文件/对象/链接及依赖身份。观测原版补全了 35 个实际编译单元和链接输入；保留首次不完整的 manifest 及旧运行快照，未重解释旧数据。当前活动 build manifest 编译单元 {len(manifest["compiler_inputs"])} 个。',
        '', '观测中性边界：3 帧原版/观测原版方向、PCG 与退出原因一致，位置最大分量差约 1.8e-11/3.1e-11 米；这不能代替 100 帧等价。后续原版复测超出观测基线范围，完整观测中性与质量认证仍待确认。',
        '', '## 2. 本机完整性能矩阵','',
        'RTX 3070 Laptop，共享桌面；dt=.01、从零连续 100 帧、相同材料和停止参数。3 轮交错配对，重型诊断关闭。下表速度是**每轮耗时比的中位数**，不是中位耗时的比。加载与导出在求解计时外，Graph 构建、矩阵转换、MAS 准备在求解计时内。','',
        '| 场景 | 配置 | 时间中位 / 秒 | 相对同轮原版 | 相对冻结 Graph |','|---|---|---:|---:|---:|']
    labels={'base':'原版 Stiff','reference':'冻结 IPC＋Graph','combined_host':'执行组合 host','combined':'执行组合 Graph'}
    for k,label in zip(SCENES,['悬挂','固定兔子']):
        for v,name in labels.items():
            r=perf['scenes'][k][v];lines.append(f'| {label} | {name} | {r["median_seconds"]:.4f} | {r["paired_base_speedup"]:.3f}× | {r["paired_frozen_graph_speedup"]:.3f}× |')
    lines+=['','执行组合为 FullCCD swept refit＋batched energy＋energy reuse；普通离散 BVH refit没有实现。组合内部 Graph 相对 host：悬挂 '+f'{perf["scenes"]["hang"]["graph_in_new_combination"]["paired_speedup"]:.3f}×，固定兔子 {perf["scenes"]["fixed_bunny"]["graph_in_new_combination"]["paired_speedup"]:.3f}×。',
        '',f'相对冻结 Graph 几何平均 {perf["combined_vs_frozen_graph_geometric_mean"]:.6f}×，3 对样本的诊断性单侧 95% 配对自助法下界 {perf["diagnostic_one_sided_95_bootstrap_lower"]:.6f}×。即使将 1.099 四舍五入为 1.10，也不满足预先声明门槛。共享负载、只有三对、质量未通过，均不允许认证阶段目标或 2×。',
        '', '![本机性能与门控工作量]('+(p/'figures'/f'{TAG}_performance.png').as_posix()+')','',
        '## 3. 已有组件消融与成本归因','', '| 场景 | 当前 Graph 秒 | +refit 秒 | +batch 秒 | +reuse 秒 | 全组合秒 |','|---|---:|---:|---:|---:|---:|']
    for k,label in zip(SCENES,['悬挂','固定兔子']):
        d=comp['summary'][k];lines.append('| '+label+' | '+' | '.join(f'{d[v]["seconds"]:.4f}' for v in ['graph','refit','batch','reuse','combined'])+' |')
    lines+=['','36 组消融均完成。单项收益不稳定；本轮同活动程序消融，全组合相对 Graph 的中位耗时比约 1.063/1.062×。此表与上节独立最终矩阵是不同轮次，不能拼接最快样本。',
        '', '完整窗口 trace 包含固定算子探针，已删除探针子树成本再分析；没有把探针成本或 CPU 等待再次计入正式性能。','',
        '| 场景 / 窗口 | 扣探针后帧区间 ms | 扣探针后线性 ms | 删除探针 ms |','|---|---:|---:|---:|']
    for k,r in cost.items():lines.append(f'| {k} / '+','.join(map(str,r['frames']))+f' | {r["frame_excluding_probe_ms"]:.3f} | {r["linear_excluding_probe_ms"]:.3f} | {r["probe_ms_removed"]:.3f} |')
    lines+=['','这些 CUDA 事件区间仍包含诊断和提交间隙；嵌套 inclusive 范围不能相加。Nsight 另取无算子探针的一帧节点追踪，确认以下启动决策：','',
        '| 新候选 | 本轮证据 | 决策 |','|---|---|---|',
        '| MAS 输出与 r.z 融合 | 全部 PCG 点积 kernel 仅占 PCG 投影时间约 4.01%/4.31%；可消除 r.z 子集更小，低于 10% | 不启动；未排除其他场景可能有收益 |',
        '| Hessian 静态/动态符号分离 | 整个矩阵转换约占帧 3.32%/3.53%；符号部分只是子集 | 不启动；不足整场 10% 门槛 |',
        '| 普通 BVH refit | 建树＋查询约 13.68%/13.19%，但建树全部成本上界仅 5.53%/3.34%；较长选定区间为 2.88%/2.16%，refit还要更新 bounds | 无支持整场至少 5% 的预测，不启动 |',
        '', '没有为了用完两个候选名额而编写 kernel。主要剩余成本是不可删去的碰撞查询、线搜索，以及线性准备/SpMV；目前证据不足以证明新的有限融合能越过局部门槛。',
        '', '## 4. 残差门控有界评估','',
        'legacy 保留原累计出口；gated/compensated替换旧累计出口，保留原 movement 判定。第六次有效接受更新前冻结参考，不随后重置。原始混合坐标范数仅是算法诊断，不是米、米/秒误差保证。','',
        '| 场景 | 配置 | 秒中位（含影子观测） | 方向中位 | PCG中位 | 峰值拉伸重复范围 |','|---|---|---:|---:|---:|---|']
    for k,label in zip(SCENES,['悬挂','固定兔子']):
        for v,name in [('shadow','legacy＋shadow'),('gated','gated .01/.03'),('gate001','gated .001/.03')]:
            r=res['summary'][k][v];a,b=r['max_stretch_range'];lines.append(f'| {label} | {name} | {r["seconds"]:.4f} | {r["directions"]} | {r["pcg"]} | {a:.9f}–{b:.9f} |')
    lines+=['','新增门控相对 legacy＋shadow：悬挂 PCG约 +26%、方向 +25%、耗时 +23%；固定兔子 PCG约 +90%、方向约 +95%、耗时约 +95%。表中含残差读回，不能当正式速度组；工作量增加来自门控延迟退出，而不是 Graph 失败。',
        '', '悬挂峰值拉伸从约1.128876到1.126931，降幅约0.17%；固定兔子gated两次通过一次越界，兼容阈值一次通过两次越界，最差峰值达1.041638/1.040568。残差更严格没有保证本项目材料指标更好，因此不推广。',
        '', '所有观测预算有效，恒等式最大误差为 '+f'{max(r["budget_identity_max_abs"] for r in evidence["runs"]):.3e}。movement 出口仍可在相对残差>.03时退出，报告没有把它标成门控证明。',
        '', '独立终端装配审计：悬挂f43和固定兔子f59均以累计出口退出，最后alpha=1、beta=0，但终端相对残差约0.123601/0.159523，均高于.03。额外装配约2.10/2.55ms，残差观测另计；不加入正式速度组。',
        '', '历史补偿：legacy影子观测未出现独立否决；延迟求解的gated轨迹在固定兔子f30出现两次历史否决，兼容轨迹f30/f63各一次。按条件仅运行28–30和57–59两个窗口，均从零推进，保留旧CCD。','',
        '| 窗口 | 配置 | 窗口方向 | 窗口PCG | 窗口最大拉伸 |','|---|---|---:|---:|---:|']
    for r in evidence['compensation_windows']:lines.append(f'| {r["window"]} | {r["variant"]} | {r["directions"]} | {r["pcg"]} | {r["window_max_stretch"]:.9f} |')
    lines+=['','两个独立运行存在数值分岔：28–30没有稳定的方向减少，57–59的小幅拉伸/工作差异没有配对重复证明，整体从零耗时也无可靠改善。全步会清除本递推的历史影响，CPU预算单元检查覆盖这一恒等式。本分支停止，不扩大参数网格、不进入长测。',
        '', '## 5. 质量、碰撞与兼容边界','',
        '每主场景观测原版3次标定＋2次独立复核，复核均通过。候选测试前冻结整段最大/高分位拉伸及固定漂移范围。固定兔子后续正式原版3次却均越界：有的是最大拉伸，有的是p99；不能在看过候选后扩大范围。','',
        '| 场景 | 原版正式材料通过 | 冻结Graph通过 | 组合host通过 | 组合Graph通过 |','|---|---:|---:|---:|---:|']
    for k,label in zip(SCENES,['悬挂','固定兔子']):
        values=[sum(r['material_passed'] for r in material if r['scene_key']==k and r['variant']==v) for v in labels]
        lines.append('| '+label+' | '+' | '.join(str(v)+'/3' for v in values)+' |')
    lines+=['','固定兔子组合Graph最差峰值1.038670877，冻结上界1.036172766；未通过。当前协议在原版本身也失败，质量认证不可用。四条独立接受路径诊断的材料指标恰好通过，不能用它们覆盖正式组失败。','',
        '| 独立CCD轨迹 | 接受段 | 实际检查路径（含帧间闭合桥） | 碰撞标记 |','|---|---:|---:|---:|']
    for r in ccd['runs']:lines.append(f'| {r["scene_key"]} / {r["variant"]} | {r["accepted_segments"]} | {r["ccd"]["paths_checked"]} | {r["ccd"]["conservative_collision_flags"]} |')
    lines+=['','全部100帧子步编号连续、首尾与逐帧端点逐字节闭合，独立CPU BVH＋Tight-Inclusion通过。'+f'总共{sum(r["ccd"]["paths_checked"] for r in ccd["runs"])}条路径零标记。只覆盖这些独立诊断轨迹，未宣称覆盖全部计时轨迹。',
        '', '真实速度单独导出并比较：','', '| 场景 / 体 | 量 | 基线重复最大帧RMS范围 | 组合相对基线范围 |','|---|---|---|---|']
    for k,r in traj.items():
        for group in ['cloth','FEM','ABD']:
            for kind,unit in [('position','m'),('velocity','m/s')]:
                b=[v[kind][group]['max_frame_rms'] for v in r['baseline_pairwise'] if group in v[kind]]
                c=[v[kind][group]['max_frame_rms'] for v in r['candidate_vs_baseline'] if group in v[kind]]
                if b:lines.append(f'| {k}/{group} | {kind} | {min(b):.6g}–{max(b):.6g} {unit} | {min(c):.6g}–{max(c):.6g} {unit} |')
    mf=mixed['frames'];lines+=['',f'混合兔子兼容100帧完成，耗时{mixed["seconds"]:.3f}s，方向{mixed["directions"]}、PCG{mixed["pcg"]}；PCG失败{mixed["pcg_failures"]}，FEM最小J={min(r["fem_min_J"] for r in mf):.9f}，最大非正单元数={max(r["fem_nonpositive"] for r in mf)}，ABD最小J={min(r["abd_min_J"] for r in mf):.9f}。这是兼容诊断，没有同轮混合材料范围或独立CCD，不能称混合质量验收通过。',
        '', '## 6. 验证、预算与关闭的分支','',
        f'- 配置契约：9项、36子检查通过；IPC预算C++检查通过（全步、零步、重复接受、非法输入、历史否决）。',
        f'- GPU组件/守卫/六份历史系统：8项顶层fixture全部通过。覆盖Graph重放、地址/容量增长、零RHS、异常pivot、非有限逆、关闭开关恢复等已有边界。',
        f'- 三场景3帧CPU/GPU非线性残差审计通过；这是梯度归约一致性，未把PCG rho阈值解释为真残差。',
        f'- 六历史CPU线性系统：{len(cpu["systems"])}份完成；整体passed={cpu["passed"]}，准确参考最大真相对残差={max(s["cpu_reference"]["true_relative_residual"] for s in cpu["systems"]):.3e}（要求1e-8）。',
        '- 最终92个活动运行的requested/resolved及实际执行模式逐项通过；源文件/对象身份通过。无改动增量构建0对象变化，程序SHA保持不变。',
        '- 未启动新执行kernel：三项候选均缺少所需整场收益证据；不新增低精度、cuSPARSE、Tensor Core或整个Newton Graph。',
        '- 无PCG触顶、breakdown、非有限导出状态的声明按已完成运行逐项核验；不以此代替材料质量。',
        '- 300帧计划已生成但未运行：阶段收益未过、固定兔子质量失败；不越过长测门槛。AutoDL七组配对遵照用户决定暂缓。',
        '- 保留最初嵌套trace flush失败、构建/工具修复日志与所有失败材料记录；没有全量清理。',
        '', '## 7. 可复现入口与后续决策','',
        '解释器：`E:/Anaconda/envs/DL/python.exe`。公共入口采用独立运行名称，旧运行目录不可覆盖。已执行命令及配置：','',
        '```text',
        'python tools/active/build.py build --label ipc_frame_trace_20261005 --jobs 2',
        'python -m pytest tools/active/test_contracts.py -q',
        'builds/active/Release/ipc_budget_test.exe',
        'python tools/active/fixtures.py --name ipc_revision_20261005_fixtures',
        'python tools/active/batch.py --plan configs/active/ipc_revision_20261005_components.json',
        'python tools/active/batch.py --plan configs/active/ipc_revision_20261005_residual.json',
        'python tools/active/batch.py --plan configs/active/ipc_revision_20261005_final.json',
        'python tools/active/batch.py --plan configs/active/ipc_revision_20261005_compensation.json',
        'python tools/active/batch.py --plan configs/active/ipc_revision_20261005_audit.json',
        'python tools/active/ipc_delivery.py audit',
        'python tools/active/verify_systems.py --historical-fixtures runs/active/ipc_revision_20261005_fixtures --output reports/active/ipc_revision_20261005_historical_cpu_systems.json',
        '```','',
        '后续不继续调整AL-TOI。先在独立阶段查清固定兔子原版的长窗重复范围为何不能覆盖正式组，验证导出频率/观测同步与原子装配数值变化的作用；不能把这个范围直接当成候选可额外形变预算。只有重新预声明并独立复核的协议能认证时，才决定是否将现有执行组合送AutoDL。',
        '', '若重新进入执行优化，优先需要证明碰撞查询或线搜索的可消除成本；当前三项候选的低占比证据不支持继续凭kernel想法迭代。残差门控保留诊断接口，停止作为加速主线。',
        '', '## 8. 证据文件与结论状态','',
        '| 结论 | 状态 |','|---|---|',
        '| 配置、预算、残差归约和已有边界fixture正确性 | 已验证（限测试覆盖） |',
        '| 执行组合较冻结Graph有约10%诊断增量 | 支持但未证明（未过严格1.10和质量认证） |',
        '| 残差门控能加速当前IPC | 已排除（限两主场景及两配置） |',
        '| 历史补偿有独立可重复质量收益 | 待验证；本轮有限分支关闭 |',
        '| 新MAS融合/符号分离/普通BVHrefit达推广门槛 | 待验证；启动条件不满足，本轮未实施 |',
        '| 两场景同材料质量阶段验收、300帧、4090验收、2× | 未完成；不宣称通过 |','']
    for name in ['quality_protocol','performance','components_analysis','final_material_checks','cost_decisions','selected_cost_corrected','residual_evidence','residual_material_checks','accepted_ccd','trajectory_comparison','observer_build_provenance','historical_cpu_systems','final_verification']:
        lines.append(f'- [{name}]({TAG}_{name}.json)')
    lines+=['','运行/失败证据统一登记到 [EXPERIMENT_INDEX.json](EXPERIMENT_INDEX.json)。实施计划见 [IPC_EXECUTION_IMPLEMENTATION_PLAN_20261005.md](IPC_EXECUTION_IMPLEMENTATION_PLAN_20261005.md)。']
    target=p/'IPC_EXECUTION_RESULTS_20261005.md';target.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(str(target))

if __name__=='__main__':main()
