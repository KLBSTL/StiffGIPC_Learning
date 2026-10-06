# 固定兔子基线长窗分歧审计（2026-10-05）

## 结论与边界

**长窗分岔在原版自身重复中已存在，速度同步导出不是它的必要条件。** 当前证据仍不能证明同步导出完全中性，也不能将组合候选的材料越界自动豁免。旧冻结协议不变，旧失败保留。

本审计只读取 `ipc_revision_20261005` 的 baseline、final、audit 数据，覆盖 19 条固定兔子 100 帧轨迹及原版 3 帧观测对照；没有 GPU 实验，没有修改程序、公共工具、协议或索引。全部 19 条完整轨迹的初态、拓扑、边界类型、质量文件分别具有唯一且相同的 SHA-256，初始输入身份一致。布料 RMS 使用 5776 个布料顶点；下文的 RMS 阈值用于定位放大时刻，不是新的验收容差。

主要发现：

1. 所有类别在物理帧 2 的接受步长已经有约 `1e-8` 的数值差；帧 25–27 首次出现 PCG 迭代整数差，之后方向数及接触数量分岔，末段达到厘米级位置差。原版自身重复也经历这条链。
2. 材料越界晚于数值分岔：最大拉伸在帧 44 首次越过冻结整段上界；p99 拉伸在帧 49 越过上界。不能只诊断越界帧而跳过帧 24–32 的首次分支。
3. 退出种类几乎完全一致。observed、raw base、frozen Graph、combined Graph 及两条独立 audit 均为逐帧相同的 22 次 movement、78 次累计出口；combined host r2 仅在帧 73 改为 movement。主要分岔不是新停止模式造成的。
4. observed 的 3 次标定加 2 次复核通过，并不足以覆盖后续原版重复；这是既有协议覆盖不足的证据，不能成为看过候选后扩大范围的理由。

## 1. 材料失败发生在哪里

冻结整段上界仍为：最大边长比 `1.0361727662908526`，最大逐帧 p99 边长比 `1.0135636480371302`，固定漂移 `1.0000005721958499e-10 m`。

| 完整轨迹 | 失败指标 | 首次越界帧 | 整段峰值 | 峰值帧 | 超过冻结上界 |
|---|---|---:|---:|---:|---:|
| raw base final r1 | p99 拉伸 | 49 | 1.0135841334 | 49 | 2.04854e-5 |
| raw base final r2 | 最大拉伸 | 44 | 1.0372743445 | 45 | 1.10158e-3 |
| raw base final r3 | p99 拉伸 | 49 | 1.0135764312 | 49 | 1.27832e-5 |
| frozen Graph final r3 | 最大拉伸 | 44 | 1.0375805650 | 45 | 1.40780e-3 |
| combined Graph final r3 | 最大拉伸 | 44 | 1.0386708768 | 47 | 2.49811e-3 |

observed r1–r5、两条独立 accepted audit，以及 combined host 三次都没有超过这些整段材料上界。observed r4/r5 在帧 31/32 已超过三次标定的逐帧最大拉伸包络加 `1e-6`，但其整段指标仍通过；这说明逐帧包络越界和预声明整段材料失败是两个不同判断。本审计只把逐帧包络用于定位，不重解释旧验收。

原版 final r1/r3 的失败来自 p99，r2 来自最大拉伸，不能把“原版三次失败”合并描述为所有样本都发生相同的额外峰值形变。候选最差峰值仍高于本轮 raw base 最差峰值，因此现有重复性证据也不足以排除候选材料退化。

## 2. 同一程序重复已经形成显著轨迹分岔

| 重复组 | 样本数 | 两两对照最大帧布料位置 RMS 范围 / m | 最大帧真实速度 RMS 范围 / m/s | 首次 PCG 整数序列差帧 |
|---|---:|---|---|---|
| observed base | 5 | 0.011662–0.025501 | 0.146620–0.248285 | 25–26 |
| raw base final | 3 | 0.027255–0.037656 | 未导出 | 25–26 |
| frozen Graph final | 3 | 0.018614–0.028112 | 未导出 | 25–26 |
| combined Graph final | 3 | 0.022891–0.032256 | 未导出 | 25–27 |

raw base 三次运行均没有速度同步导出，其自身位置差仍达到 `2.73–3.77 cm`，甚至大于 observed 五次中的最大值。由此已排除“没有新增速度导出，就不会出现这种长窗分岔”的解释。不同组的运行时间、GPU调度、构建身份及输出工作不同，以上范围不能单独用于推断哪种导出方式降低或提高了分岔概率。

前一次四配置测试已经报告原版自身位置 RMS 约 `25.416 mm`；本轮现象并非第一次出现在新增 IPC 门控接口以后。旧报告的逐帧严格质量判定保持历史含义，本审计没有追溯改判。

真实速度只有 observed 五次以及两条 accepted_audit 可读。raw base、frozen Graph、combined Graph/host 的 final 计时组没有 `velocity_*.bin`；本审计不将位置有限差分称为实际速度，也不据缺失速度断言这些组速度等价。

## 3. 首次分支的前缀证据

| 时刻 | 直接观测 | 可以支持的判断 |
|---|---|---|
| 帧 1–3 | 原版 3 帧 observer_check 与 observed r1 PCG/方向数/退出完全相同；布料位置 RMS 最大 `6.81e-12 m`，最大顶点差 `3.09e-11 m` | 短前缀差仅为微小数值量；不能证明后续 100 帧中性 |
| 帧 2 | 19 条轨迹首个接受 alpha 范围 `0.631308862397191–0.6313088755614841`，跨度 `1.31643e-8` | 接受路径已出现微差，早于 PCG 整数分支和末段材料失败 |
| 帧 22 | 所有轨迹首次有 native narrow self pair；距离诊断出现微差。例 observed r1/r2 为 `0.002211213123/0.002211212943 m` | 距离数值变化；不是接触数量或分类已经不同 |
| 帧 25–27 | 首次 PCG 迭代整数序列不同，原版自身亦如此；与 observed r1 的位置 RMS 跨过 `1e-6 m` 多发生于帧 25–27 | 数值微差进入离散求解分支，随后有放大条件 |
| 帧 29–32 | 相对 observed r1，native pair 数量/分类首次不同 | 活动接触离散差晚于首次 PCG 整数差，不应把帧22距离差误称活动集差 |
| 帧 31–42 | 相对 observed r1 的原版/Graph/组合代表轨迹开始方向数量差；observed 内部两两首次方向差分布更宽，最晚到帧49 | 同一种退出理由也可能接受不同数量的方向和不同 alpha |
| 帧 44/49 | 最大/p99 拉伸首次超过整段冻结上界 | 材料指标失败已是前缀分岔的下游现象；尚不能确定具体算子因果 |

一个最小例子是 observed r1 对 raw base final r1：帧 25 的 PCG 序列分别为

```text
observed: 22, 44, 47, 53, 55, 56, 58, 57, 55
raw base: 22, 44, 47, 53, 54, 56, 58, 57, 55
```

双方均接受 9 个方向并以累计出口退出，初次整数差仅 1 次 PCG 迭代。该帧第 4 个 alpha 已分别为 `0.527531610286/0.527459179287`；布料位置 RMS 为 `2.843e-7 m`。到帧 33 超过 `1e-4 m`，帧 39 超过 `1e-3 m`，帧 58 超过 `1e-2 m`。这是可复核的放大时序，不是某个 kernel 已被证明错误的结论。

实际速度对照同样放大。observed r1 对 observed accepted_audit：帧 25 的速度 RMS 超过 `1e-6 m/s`，帧 32 超过 `0.001 m/s`，帧 34 超过 `0.01 m/s`，帧 50 超过 `0.1 m/s`；峰值 `0.22110 m/s` 在帧 71。r1 对 combined accepted_audit 的对应帧为 25、32、33、44，峰值 `0.28022 m/s` 在帧 81。它们属于不同独立诊断轨迹，不继承为所有计时样本的速度结果。

## 4. 观测同步的因果结论

| 假设 | 当前证据状态 |
|---|---|
| 新增真实速度导出是出现长窗分岔的必要条件 | 已排除：raw base 自身重复已分岔 |
| 同步导出完全不影响数值分布/首次离散分支 | 待验证：缺少同程序、同其余配置的交错导出开关对照 |
| 新 IPC 停止模式是本轮 legacy 长窗差异的原因 | 不支持：配置 legacy，几乎全部逐帧退出类型一致；已有原版自身差异 |
| 原子装配/并行归约微差与接触分支放大有关 | 支持但未证明：前缀时序一致，尚未锁定算子、运行顺序或第一处输入差 |
| 当前材料协议能够认证同质量 | 未完成：原版独立复测本身越界，不允许自动扩大范围 |

下一项最小因果对照可以直接复用同一 `base_observed.exe`：应用已有 `GIPC_TRACE_VELOCITY` 条件（`gl_main.cu:1847`），公共配置的 `trace_velocity=false/true` 即可控制，无需改求解器或新构建。父任务已经独立预声明 raw_off、observed_off、observed_on 三臂各5轮、从零连续100帧，共15次。raw_off 对 observed_off 隔离应用观测构建差异，observed_off 对 observed_on 隔离实际速度导出；其新结果由父任务统一分析，本报告不提前宣布结论。设置间差异需与同设置重复差比较，重点保留帧25的PCG整数分支、帧29–32的接触数量，以及帧31–36的方向差。

若这项对照不支持同步导出存在超过自身重复的前缀影响，就停止该因果分支，保留诊断限制。若出现可重复差异，先定位其第一处输入/系统差，不改变停止容差。本报告不启动重新标定，不制定更宽的材料预算，也不据现有候选反向选择基线范围；100帧材料认证仍标为未完成。

## 5. 复现与证据

只读 CPU 分析命令：

```text
E:/Anaconda/envs/DL/python.exe E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/baseline_divergence_audit_20261005.py
```

脚本只重新生成其自身审计 JSON/日志，没有写入原始运行目录。JSON 保存每个材料指标的最早越界、峰值帧、重复组范围、两两求解时序差、代表位置/真实速度前缀曲线及初始输入哈希。

- [审计数值 JSON](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/baseline_divergence_audit_20261005.json)
- [冻结协议](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_revision_20261005_quality_protocol.json)
- [基线逐帧分析](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_revision_20261005_baseline_analysis.json)
- [正式矩阵逐帧分析](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_revision_20261005_final_analysis.json)
- [独立审计逐帧分析](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_revision_20261005_audit_analysis.json)
- [历史四配置结果](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/FOURWAY_CLOTH_RESULTS_20261005.md)

所有帧号为从零推进后物理帧的端点编号；位置/速度初态是帧0。所有阈值跨越仅描述实测差异，均不代表对真值误差的保证。
