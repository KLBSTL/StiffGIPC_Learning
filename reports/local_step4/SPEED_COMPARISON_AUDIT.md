# 加速比比较口径与源码整理回归核查

2026-10-06。只读核查已有 requested/resolved、result、stats、构建封存与历史报告；没有运行 GPU、修改协议或改写旧结果。包含 Step4 悬挂首对已完成结果，不预写后续反序结果。

**现有证据不支持“源码整理使速度大幅回退”。主要混淆是把接触池的增量比当成整套实现相对 Stiff 的总比，并比较了不同帧窗、硬件和算法。共享桌面环境确实使绝对时间及增量比显著变化；新一对相同程序、配置、近乎相同工作量的复测直接支持这一点，但还不是隔离负载因素的因果实验。**

## 1. 这些数字分别是什么

统一记号：总比为 `T(Stiff)/T(当前)`；池增量比为 `T(当前池关)/T(当前池开)`。二者分母、分子不同。必须用同场景、同帧窗、同轮实测值；不能跨轮或跨机器乘比率。

| 证据组 | 硬件／窗口 | 真正比较对象 | 实测比率 | 可比性边界 |
|---|---|---|---:|---|
| Step3 固定兔子三对 | 3070 Laptop／59 帧 | Stiff / 当前池关 | 1.56842、1.46298、1.47482；中位 **1.47482** | 当前含 Graph、FullCCD refit、能量批量/复用、普通 BVH 缓存；不是 Graph 单项 |
| Step3 固定兔子第三轮 | 同机／59 帧 | 当前池关 / 池开 | **1.07134** | 仅池增量，净省 6.659%；同轮 Stiff / 池开仍是 **1.58004** |
| Step3 悬挂首对 | 同机／51 帧 | 当前池关 / 池开 | **1.13711** | 仅池增量，净省 12.057%；该轮没有悬挂 Stiff 分母 |
| Step2 固定兔子两对 | 同机／59 帧 | Stiff / 当前池关 | **1.33626、1.64953** | 与 Step3 当前请求配置相同；两个样本本身跨度很大 |
| 历史四配置，悬挂 | 同机／100 帧 | Stiff / IPC Graph-only | **1.46899** | refit/batch/reuse 均关，未用当前普通 BVH 缓存；与最新组合不同 |
| 历史四配置，固定兔子 | 同机／100 帧 | Stiff / IPC Graph-only | **1.31720** | 同上；100 帧与 59 帧不能直接归因为实现变化 |
| 4090 旧短窗，悬挂 | RTX 4090／21 帧 | Stiff / AL TOI Graph 新因子作用 | **1.88724** | backend=toi_al、Cholesky/factor_inverse/warp；非当前 IPC，且窗口更短 |
| Step4 新悬挂首对 | 3070 Laptop／51 帧 | 当前池关 / 池开 | **1.03632** | 与 Step3 相同程序和配置；资源释放后仅一对，净省 3.504%，不可预设后两轮结果 |

历史四配置的三对比率直接由各 `result.json.solver_seconds` 重算，分别为悬挂 1.48587/1.41804/1.46899，固定 1.35523/1.31720/1.27791。4090 三对重算为 1.86712/1.90959/1.88724。以上中位不是强制用两个时间中位相除。

因此，若把“1.887× 降到 1.071×”作为退速依据，比较口径已不成立。最新固定池关总比 1.475× 也没有低于历史固定 Graph-only 的 1.317×；反过来也不能因其更高就宣称源码整理提速。

## 2. 同配置也出现多大变化：给出数值，不只称为噪声

Step2 固定池关两次为 **10.3818867、8.3167238 s**，慢者比快者多 **24.83%**；相应 Stiff 为 **13.8729119、13.7186635 s**。Step3 池关为 **9.5384428、10.1310887、10.1472227 s**，Stiff 为 **14.9602536、14.8215340、14.9653516 s**。

两批池关时间中位比较，Step3 高 **8.36%**；同一个冻结 Stiff 程序的时间中位也高 **8.44%**。这支持运行环境或共同工作量因素参与变化，反对仅把当前程序的变化归因于代码删除；不证明两批时间可以精确用一个负载系数校正。Step3 总比中位 1.47482 处于 Step2 的 1.33626–1.64953 范围内，不能把 Step2 最有利单次 1.64953 当作已稳定达到的旧水平。

### 资源实际波动及失败窗口

- Step2 四次完整固定窗口起始 free **3735–3850 MiB（3.65–3.76 GiB）**，起步 GPU util **35–41%**。随后 fixed-on 起始 **3092 MiB**、预算 **1556 MiB**，最后 free **920 MiB**，变化 **2172 MiB**，于 22/59 帧止损；hang-off 起始 **3061 MiB**、预算 **1525 MiB**，最后 **1065 MiB**，变化 **1996 MiB**，于 40/51 帧止损。
- Step3 固定完成组起始 free **3725–3749 MiB**；悬挂首对依次 **3524、3449 MiB**。悬挂第二轮起始 **3081 MiB**、预算 **1545 MiB**，末次 **1059 MiB**、变化 **2022 MiB**，同样在 40/51 帧止损。GPU 起步 util 实际在 **31–49%** 之间；WDDM 进程显存大量为 N/A，无法将全卡变化全部归属仿真。
- 根任务后续状态观测曾为 **2897 MiB（2.83 GiB，约2.9 GiB）**，用户释放应用后为 **5674 MiB**；这是状态快照，不冒充某次运行的起始值。新 Step4 真正两臂 requested 起始为 **5886、5887 MiB（约5.75 GiB）**，util 分别 **22%、4%**。因此此前“约2.9–3.7 GiB”的可用资源波动具有实际记录，不是抽象噪声描述。

资源中止只能解释为什么完整配对缺失，不能用失败前缀补整场时间，也不能仅凭 free 下降断言发生显存换页或其造成多少减速。显存、GPU占用、功耗、时钟与调度没有独立控制，本报告不把它们混成单一已证原因。

### Step4 首对：同程序、同配置的新反证

| 51 帧悬挂 | Step3 秒 | Step4 秒 | 绝对耗时下降 | Newton / PCG |
|---|---:|---:|---:|---|
| 当前池关 | 3.1687754 | 1.9057210 | **39.859%** | 两次均 297 / 9103 |
| 当前池开 | 2.7867039 | 1.8389386 | **34.010%** | 297 / 9104 → 297 / 9103 |

直接核对两批 requested：对应 arm 的 `expanded_config` 相等，程序 SHA 均为 `b2ad19884fc01b81827d560f9f1984346b5488851995c2fd495f7427eac0f812`。没有通过改源码、停止条件或大幅减少线性迭代取得上述绝对时间变化。池增量比反而 **1.13711→1.03632**：原来一对的 13.7% 比率增益不能当稳定的纯池执行收益。新首对旧材料界限通过，仍缺重复、独立接受路径 CCD 和完整同质量认证；不据一次 3.5% 净省直接封存或推广候选。

## 3. 源码整理是否改变默认公式或工作量

**已证明到源码身份的范围：**本次独立比较 Step2/Step3 实际运行目录内 `build_manifest.json` 的 322 项 source/input 记录，只有 **3 项**改变：`solver/toi_options.h`、`solver/toi_solver.cu`、`core/retired_components.h`。其余 319 项哈希相等，应用 DLL 文件名/哈希也相等。具体 IPC 核心与线性文件如下：

| Step2→Step3 未变文件 | SHA-256 |
|---|---|
| core/GIPC.cu | `5e4ffa38535ccff16095c9c5d59f005f5b0de9bf4616f8b277c333314e346539` |
| linear_system/solver/pcg_solver.cu | `1deea2593200a2a32751a0610c0762f5fcdd546b43f04f2533a59cee9219061f` |
| linear_system/solver/pcg_graph_impl.inl | `899984d1055f4fa92d8f01d93ecffc085f275f2a08723cfdb0ab48c0b2a8d3ac` |
| solver/MASPreconditioner.cu | `233f74da8422f283d2ea43f14d45cfc5295f872c3342b5afab75e15b931a103b` |
| solver/ipc_budget.h | `0b8fd0acf790ccde3412c62b42a3d807f1088bb981ca2f9bd3327fe0b8858b0d` |
| solver/ipc_residual_controller.h | `00007663a499bc8296959f1c8df276cfa25e9f96499e52e24d180f04e37d0a41` |

Step3 删除的是关闭的 `velocity_only` 与 full-step probe 两个 **AL 入口**。既有 `CODE_CLEANUP_STEP3.md` 记录关闭分支常量化后的 token 一致和 256 种退出布尔组合核对；这与本次实际构建封存哈希对比一致。不能把这次删除解释为更改了 IPC 累计退出公式。

Step2 更早删除 MAS final-dot、SpMV+pAp、ordered restriction 失败分支的 37 项 CPU 静态核对见 `CODE_CLEANUP_STEP2_STATIC_CHECKS.json`：保留 host/Graph 原关闭路径、previous-rho 次序、MAS三条作用和普通SpMV。该记录支持关闭路径源码语义保持；**没有整理前后同一固定系统、同一负载、同一构建选项的受控配对计时，不能据此声称整理完全无性能影响，也不能反向声称整理造成退速。**

**实际工作量没有证明完全相等。**固定 Step2 池关两次 Newton 为 327/322、PCG 为 19059/18930；Step3 为 327/330/328、19324/19700/19227。固定 Stiff 自身跨次也从 324/336、19171/20030 变为 327/331/322、19416/19520/18985。默认公式保持不意味着浮点并行顺序、轨迹及停止次数逐位保持；这些结果不支持“工作量大幅增加导致速度塌陷”，但也不足以把所有差异归为计时噪声。

## 4. 三种停止/求解路线不能混称 TOI

1. **当前 Step2/3/4：**raw requested 和 resolved 均为 `contact_backend=ipc`、`ipc_stopping.termination=legacy`、累计容差 `.01`、minimum 6、movement 出口保留、PCG rho `1e-4`。MAS 实际 `cholesky=false/wide=false/inverse64=false`；请求中的 serial/triangular 对 legacy 稳定因子作用不生效，resolved 明确为 inactive。`cost_observation.active=false/gpu_events_effective=false`，没有把默认 requested cost_events=true 错当成本追踪开启；实际速度导出开启，性能仅共享桌面诊断。
2. **IPC compensated：**仍是 IPC 的双容差补偿/残差门控路线。本批 `termination=legacy`，没有开启 compensated。不能用日志中存在 residual 配置字段就声称该出口实际触发，更不能解释为它减少/增加了这批迭代。
3. **4090 历史 AL TOI：**raw `backend=toi_al`，Cholesky + factor_inverse、warp restriction、初值选择与 restart guard 开启；子步/physics/velocity 导出开启，21/45/35 帧短窗。悬挂第一轮 Stiff 和 TOI 的 21 帧 native self/ground pair 峰值均为 **0**，该 1.887× 不能作为当前接触阶段优化的已达水平。其固定兔子同轮总比只有 **0.948×**，混合兔子 **0.374×**，不能只选悬挂最有利数字代表所有场景。

## 5. 可交付结论与剩余验证

**已证明：**分母混用确实存在风险；最新固定总比并非 1.071×。Step2/3 配置完全相同、IPC主要源文件未变；Step4 同程序同配置且近乎同工作量仍出现 34–40% 绝对耗时变化。资源预算实际中断了后续配对，失败数据没有计入完整加速比。

**支持但未证：**桌面共享 GPU 环境显著参与绝对速度与增量比变化；池可减少部分线搜索成本，但其可重复净收益尚未确定。新空闲显存恢复后的首对更适合否定“原 1.137× 已稳定成立”，不构成硬件独占或负载单因素证明。

**待验证：**源码整理前后严格匹配的执行时间差、两布料稳定池净收益、同质量 2×、接受路径独立CCD及100/300帧结果。下一轮应继续已声明的反序配对并用同状态算子成本解释差值；不改停止门槛，不将不同硬件/帧窗的最佳历史比率作为回归基准。

## 精确证据路径

- [Step3逐轮报告](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_step3/ROUND_REVIEW.json)；原始根 `runs/local_step3_20261006/{fixed_r1,fixed_r2,fixed_r3,hang_r1,hang_r2}/<run>/`，本核查读取各 `requested.json`、`resolved_config.json`、`result.json`、`output/stats.json`。
- [Step2逐轮报告](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_step2/ROUND_REVIEW.md)；原始根 `runs/local_step2_20261006/`。源码身份比较直接读取两批 `fixed_r1/fixed_off_r1/build_manifest.json`。
- [Step4首对池关结果](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/runs/local_step4_pool_20261006/hang_r1/hang_off_r1/result.json)、[池开结果](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/runs/local_step4_pool_20261006/hang_r1/hang_on_r1/result.json)，以及同目录 requested/stats；本报告不包含随后轮次。
- [历史四配置报告](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/FOURWAY_CLOTH_RESULTS_20261005.md)；原始根 `runs/active/fourway_cloth_20261005_{hang,fixed_bunny}_{base,graph}_r{1,2,3}/` 的 requested/result 已重读。旧大清理保留这些元数据；不宣称已删轨迹仍可完整重放。
- [4090历史报告](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/AUTODL_FACTOR_RESULTS_20261005.md)；实读紧凑回传根 `downloads/autodl_factor_20261004_4090/runs/active/autodl_window_hang_{stiff,toi_graph}_r{1,2,3}/` 的 requested/result，以及r1 output/stats。
- [Step2静态核对](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/docs/CODE_CLEANUP_STEP2_STATIC_CHECKS.json)、[Step3关闭分支核对](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/docs/CODE_CLEANUP_STEP3.md)。静态语义检查、运行轨迹检查和性能因果证明各有范围，不能互相替代。
