# TOI 限制步长接触诊断（v30，2026-10-01）

## 结论

本轮完成只读诊断实现、本机 7 次运行和三条完整接受路径的独立 CCD 检查。当前外层求解多的直接证据是：**限制安全步长的接触经常没有参加当轮 trial 求解；已经参加的接触也经常仍违反线性约束。** 抽样的乘子、slack、gamma 更新符合实现公式，未发现更新符号或数值计算的明显错误。

跨帧保留 `C/lambda/gamma` 的现有组合值得进一步拆分：本次外层次数 642 → 519（减少 19.2%），前五次外层的安全 alpha 中位数 0.00983 → 0.29339，活动集总加入次数大幅减少，开始能够淘汰旧接触。不过额外内层方向 203 → 233，质量仍未达标，不能直接启用为默认配置。

本轮开启了 CPU 拷贝、接触重建探针和子步导出，**计时均为诊断数据，正式质量匹配加速比为 N/A**。各分支仅运行一次，且已有同后端长轨迹波动，分支差异不能全部归因于单一开关。

## 1. 修改和实验身份

- 工作目录：`E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929`。修改仅发生在本任务目录。
- 源码版本：`v30-safe-alpha-blocker-audit`；553 个源文件/资源逐项复核一致。
- `source_digest`：`bb22624213574627b7a916bffb4b75055bd0a2dabb61a36a21d4ef414da84c60`。
- v30 二进制 SHA256：`b1c359d1da5a85dbde0d1eeb38f1522f0896091dee7f50e2e177a38003783786`。
- v29 原二进制、求解器源码、runner 和 manifest 已保存于 `builds/local-fused-v29`、`research/v29_before_blocker_audit`；归档身份复核通过。
- 新诊断默认关闭，仅指定帧窗口时启用。读取 CCD 的真实最小 alpha 及限制接触，不修改 GPU 接触、乘子或 trial/safe 状态。缺失接触的线性平面只在 CPU 副本上构造，日志明确标为诊断探针，未用于实际求解。
- 捕获 trial 系统接触的时机是内层完成、乘子更新之前；另记录活动集更新后的存在性，避免把“求解后才加入”误认作“已经参加求解”。

主实现：`sources/stiff_fused/StiffGIPC/solver/toi_solver.cu`；CLI：`tools/run_local.py`；串行运行协议：`configs/local_blocker_v30_20261001.json`。

## 2. 配置和完成性

本机 RTX 3070 Laptop 8 GiB、CUDA 13、Windows Release / SM86。场景 `cloth_sphere7_l`，共 3,084 顶点，其中可变形布料 2,601 顶点、质量 0.512 kg，固定 ABD 球 483 顶点、质量约 305.386 kg。

共同参数：`dt=0.01 s`、累计 TOI epsilon/Newton tol `0.01`、`pcg_tol=1e-4`、`mu_mode=diagonal`、`suite=0`（host，无 Graph）。60 帧对应 0.60 s；诊断窗口为第 22–60 帧，outer 从 0 编号。

| 阶段 | 运行 | 结果 |
|---|---|---|
| 组件 | 接触 6 项、体积 17 项 | 全通过 |
| 开关检查 | 默认 TOI，2 帧，诊断 off/on | 两次完成，状态有限 |
| 接触试跑 | 默认 TOI，30 帧，诊断 off/on | 两次完成，状态有限 |
| 主诊断 | 默认 TOI，60 帧 | 完成，无 PCG/外层上限 |
| 主诊断 | TOI + velocity_tol=1.0 m/s，60 帧 | 完成，无 PCG/外层上限 |
| 主诊断 | TOI + 跨帧 C/lambda/gamma，60 帧 | 完成，无 PCG/外层上限 |

七次运行初态、拓扑、质量、边界和体编号一致，实际二进制、runner 与协议身份一致。GPU 仿真串行执行。构建曾因沙箱访问 Windows SDK 被拒而重试；首次版本目录启动缺少 freeglut/glew32 DLL，补齐后组件验证通过。这些启动失败没有计入实验结果。

### 诊断开关的限制

2 帧 off/on 最大顶点差为 `1.05e-12 m`。30 帧最大顶点差为 `0.11265 m`，布料最大 RMS/初始布料尺度为 1.281%；首次超过 `1e-8 m` 的状态差在第 11 帧，而接触采样从第 22 帧才启用。与已有同后端波动一致，本轮不能将长轨迹差归因于采样，也不能宣称长轨迹等价已经通过。当前结论基于每次运行自身的 trial、实际接触和 CCD 最小值。

## 3. 工作量

“方向求解”指一次全局线性系统/PCG 求解；PCG 内部迭代另列。方向数 = 外层数 + 额外内层方向数。

| 指标，全部 60 帧 | 默认 TOI | velocity_tol=1.0 | 跨帧保留组合 |
|---|---:|---:|---:|
| 全局方向求解 | 845 | 663 | 752 |
| 外层次数 | 642 | 652 | 519 |
| 额外内层方向 | 203 | 11 | 233 |
| PCG 内部迭代 | 38,342 | 26,614 | 30,013 |
| 第 22–60 帧外层数 | 589 | 599 | 466 |
| 单帧最大外层数 | 22 | 20 | 19 |
| 活动接触加入次数，非唯一接触计数 | 132,619 | 133,987 | 6,886 |
| 活动接触删除次数 | 0 | 0 | 6,324 |
| 最大已发布活动接触数 | 5,296 | 5,719 | 1,373 |
| 首五次 outer 的安全 alpha 中位数，第 22–60 帧 | 0.00983 | 0.01086 | 0.29339 |
| 诊断求解时间，秒 | 20.278 | 17.200 | 13.737 |

三组前 21 帧外层均为 53 次。跨帧组合的改善集中在接触阶段：第 22–60 帧外层下降约 20.9%。`velocity_tol=1.0` 主要减少额外内层方向，外层未减少，再次支持前轮判断。

![工作量与阻挡接触分类](figures/TOI_BLOCKER_V30_WORK.png)

## 4. 限制 alpha 的接触在哪里

每个 `alpha<1` 的 outer 取一个规范化的全局 CCD 最小接触作为分类单位；遇到并列最小值时只代表其中一个。另记录最小的前四个唯一接触用于一致性检查，不能把这些抽样当作全部接触统计。

| 最小接触分类，第 22–60 帧 | 默认 TOI | velocity_tol=1.0 | 跨帧保留组合 |
|---|---:|---:|---:|
| 当轮 trial 未包含，更新后仍缺失 | 154 | 159 | 149 |
| 当轮 trial 未包含，求解后才加入 | 314 | 313 | 253 |
| trial 已包含，线性约束仍违反 | 88 | 89 | 28 |
| trial 已包含，线性约束满足但 CCD 仍限步 | 30 | 30 | 12 |
| alpha<1 的 outer 总数 | 586 | 591 | 442 |

默认组中，468/586 = **79.9%** 的限制接触没有进入当轮 trial 系统。前五次 outer 中为 175/195 = 89.7%；第五次之后仍为 293/391 = 74.9%。因此问题并非只有每帧第一次冷启动。

默认组已进入 trial 的 118 个最小接触中，88/118 = **74.6%** 仍违反实际线性约束。这些违反接触中 48 个使用的 lambda 为零，82 个 gamma 为 1；因此不能统一解释为 gamma 衰减造成约束“失效”。违反量/`delta` 的中位数约 −1.54。独立 full CCD 仍承担了截断 trial 的工作。

跨帧组合的缺失比例为 402/442 = 91.0%，高于默认；但限制 outer 的绝对数、前五步的限步强度和活动集加入工作均下降。它缓解了一部分启动/重建成本，**没有解决动态阻挡接触进入求解不及时的问题**。本次第 23–60 帧继承活动接触数中位数为 752.5，默认和 velocity 分支均为 0。

### 具体证据

1. 默认组第 22 帧、outer 0，PT `[1,1073,1117,2217]`：`alpha=0.134812`，trial 和更新后集合都没有该接触。安全端无符号距离约 3.633 mm，trial 端无符号距离约 17.858 mm，但 safe 平面上的 trial 约束约 **−18.109 mm**。这说明 trial 可能沿路径越过球面后在另一侧结束；端点 unsigned distance 大不能代替完整路径检查。缺失平面是诊断副本。
2. 第 22 帧、outer 1，PT `[1,1117,1119,2217]`：`alpha=0.0315234`，接触在 trial 求解后加入，其 lambda 为 0；它无法纠正已经算出的这一轮方向。
3. 第 22 帧、outer 4，PT `[1,1116,1117,1125]`：实际系统包含此接触，`c_trial=-0.312837 mm`，`lambda=0.0096424 → 0.0191960`，`gamma=1`，`alpha=0.207644`。乘子在按违反量加强，但尚需后续 outer 才能影响新方向。
4. 第 23 帧、outer 6，EE `[28,59,2228,2255]`：实际线性约束 `c_trial=+0.004211 mm`，仍得到 `alpha=0.613289`。线性平面满足不足以保证整条非线性几何路径安全，独立 CCD 必须保留。

## 5. 对照实现和论文后的解释

用户提供论文的本地冻结内容位于 `research/paper_extracted.txt`、`paper_page_06.png`、`paper_page_07.png`。

- 论文算法 1 接收上一时间步的 `C_t`；当前默认每帧清空 C/lambda/gamma，跨帧状态是可选开关。此差异有直接源码证据，本次也观察到跨帧组合改善早期 alpha 和活动集工作量。
- 论文算法 1 先用 `C[k]` 解出新 trial，再用 **上一 trial** 更新活动集，随后对新 trial 做安全 CCD。当前时序与这一步一致。因此“求解后才加入”本身不能直接判成移植时序错误。
- 当前每顶点最早候选筛选使用上一 trial 的扫掠路径；新 trial 的实际全局最小阻挡接触可能尚未被选择。当前日志证实缺失，尚未逐个重放算法 3 的候选选择原因，不能把所有缺失归为同一 bug。
- 算法 2 的 `r=1` 是能量线搜索全步，不代表接触约束已无违反。当前内层停止后，AL 在 outer 之间更新 lambda；有限罚参数及局部线性化仍可能需要多轮。不能把 `r=1` 与安全 CCD 的 `alpha=1` 混同。
- 默认组抽样窗口 mu 维持约 30.5386；现有升罚规则只针对连续 `alpha<1e-4` 的停滞。本次最小接受 alpha 约 `1.63e-4`，未触发升罚。它不响应一般性的负线性残差；是否需要新的残差驱动策略须用受控案例验证，不能仅凭本轮数据提高 mu。

这条证据链解释了当前开销：缺失/不足的接触约束产生跨越接触面的 trial → 完整 CCD 给出小 alpha → safe 只推进一点 → 更新活动集、重建系统和再次求解。当前尚未形成论文目标中的较大安全步长。

### 数值更新核对

共核对 1,135 个已进入系统的前四最小接触副本：

| 检查 | 三组最大偏差 |
|---|---:|
| CPU 重建探针 vs 实际 GPU 线性平面系数，混合系数单位 | `1.66e-10` |
| lambda 更新相对误差，分母至少为 1 | `1.33e-16` |
| slack 更新绝对误差 | `3.47e-18 m` |
| gamma 更新绝对误差 | 0 |

CCD 最小值重建在所有已审计 outer 内通过 `1e-12` 一致性检查。这里验证更新算式和诊断探针，不能替代全局 KKT 残差或 PCG 真残差审计。

## 6. 安全和物理质量

独立 CPU BVH + Tight-Inclusion 检查所有导出的已接受路径；每个帧初态及每个 outer 的 safe 状态均已覆盖。路径数量包含相邻帧之间的重复初态段。

| 组别 | 路径数 | 保守碰撞标记 | 四面体翻转 | 结果 |
|---|---:|---:|---:|---|
| 默认 TOI | 701 | 0 | 0 | 通过 |
| velocity_tol=1.0 | 711 | 0 | 0 | 通过 |
| 跨帧组合 | 578 | 0 | 0 | 通过 |

这证明本轮接受路径的检查通过；物理质量另行比较。固定 ABD 球没有由 `boundary_types` 标为固定，必须依赖场景 fixed_mode 和拓扑排除，避免固定球的大质量稀释布料误差。

| 布料单独指标，0–0.60 s | 默认 TOI | velocity_tol=1.0 | 跨帧组合 |
|---|---:|---:|---:|
| 对严格 base dt/4 的最大质量 RMS / 布料初始尺度 | 4.757% | 5.199% | 3.163% |
| 末态速度 RMS 差 | 0.609 m/s | 0.648 m/s | 0.460 m/s |
| 全程最大局部边伸长比 | 1.376 | 1.403 | 1.143 |
| 每帧 95 分位边伸长比的全程最大值 | 1.056 | 1.060 | 1.010 |

严格参考本身在共同 0.60 s 窗口内，dt/2 → dt/4 的布料最大位置差为 **2.333%**，末态速度差为 **0.576 m/s**，尚未收敛为精确真值。本轮三组均未通过建议的 1% 位置质量门槛；跨帧组合的较低偏差和拉伸仅是本次诊断结果。

![共同时间、视角和尺度下的实际导出几何](figures/TOI_BLOCKER_V30_GEOMETRY.png)

## 7. 下一步执行顺序

1. **先拆分跨帧状态。** 保留现有默认和 v30 归档；新增只继承 C、重置 lambda/gamma 的独立选项，再分开继承乘子、gamma 和摩擦状态。优先用固定输入重放和第 21–30 帧短窗验证，随后 60 帧。每个对照只改变一个因素，继续记录阻挡接触和布料质量。
2. **核对缺失阻挡接触的选择原因。** 固定一个有代表性的 safe/previous_trial/trial，重放算法 3，区分“路径尚未包含”“每顶点最小候选被其他接触抢占”“canonical key 对应错误”“被删除”。只有发现确定的实现错误才修复。可进一步试验把完整 CCD 已确认的阻挡接触保护到下一轮求解中，作为单独算法消融，防止不受控扩大 C。
3. **处理已在系统但违反的约束。** 固定输入重放 lambda、AL 残差和真线性残差，再比较局部约束残差驱动的罚参数/内层策略；保持完整 CCD。当前证据不支持只继续放宽 velocity_tol 或只减少 Kmin。
4. **最后再测速度。** 先控制同后端重复差并建立更充分收敛的共同参考；通过完整路径与物理门槛后，关闭诊断/子步导出，交错配对重复，再组合 Graph。

本轮完成的是阻挡接触诊断；上述求解策略仍待实施和验收。当前默认未改为 persist，未声明已经修复 TOI 或获得相对 base 的加速。

## 8. 复现和交付文件

在任务目录内，以 `E:/Anaconda/envs/DL/python.exe` 运行：

```powershell
& 'D:/computer/cmake/bin/cmake.exe' --build builds/local-fused --target gipc --config Release --parallel 2
$env:GIPC_VALIDATE_COMPONENTS = (Join-Path (Get-Location).Path 'builds/v30_toi_components.json')
& 'builds/local-fused-v30/Release/gipc.exe'
Remove-Item Env:GIPC_VALIDATE_COMPONENTS
& 'E:/Anaconda/envs/DL/python.exe' tools/run_blocker_audit.py --stage smoke
& 'E:/Anaconda/envs/DL/python.exe' tools/run_blocker_audit.py --stage pilot
& 'E:/Anaconda/envs/DL/python.exe' tools/run_blocker_audit.py --stage diagnose
```

构建后版本目录须包含相同构建得到的 gipc.exe、freeglut.dll、glew32.dll。已有完整运行会按身份复用；新测量需使用新协议/目录，禁止覆盖旧结果。精确命令记录在 `runs/local/local_blocker_v30_20261001/*_status.json`。

三组独立验证命令，依次替换 `full` 为 `vel100`、`persist`：

```powershell
& 'builds/validator/Release/validate_path.exe' 'runs/local/local_v30_blocker_full_audit60/trace' 'reports/v30_full_accepted_ccd.json' substeps --stable-nh1
& 'E:/Anaconda/envs/DL/python.exe' tools/report_blocker_audit.py
& 'E:/Anaconda/envs/DL/python.exe' tools/verify_blocker_audit.py
& 'E:/Anaconda/envs/DL/python.exe' tools/plot_blocker_audit.py
```

以上构建、组件、七次运行、三次验证、分析和最终身份/覆盖复核均返回成功。最终复核摘要：`passed=true`，553 源文件、7 次完成运行，接受路径 `[701,711,578]`。

- 机器可读汇总：`reports/TOI_BLOCKER_AUDIT_V30_20261001.json`。
- 每个采样接触：`reports/TOI_BLOCKER_AUDIT_V30_20261001.csv`。
- 身份及导出覆盖复核：`reports/v30_blocker_verification.json`。
- 独立安全结果：`reports/v30_{full,vel100,persist}_accepted_ccd.json`。
- 图件：`reports/figures/TOI_BLOCKER_V30_{WORK,GEOMETRY}.{png,pdf}`，均已打开检查。
- 构建/组件/运行/CCD/分析日志：`builds/v30_*.log`。
