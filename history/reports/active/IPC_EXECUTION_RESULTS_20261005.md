# IPC＋CUDA Graph 执行优化与残差门控实施结果（2026-10-05）

本轮按批准计划完成本机实施和有界实验。IPC 停止配置、完整帧成本追踪、CPU/GPU 残差核对、冻结构建身份、已有组件消融、两种门控、条件补偿对照和独立接受路径 CCD 均已落地。

**结论：局部诊断有增量收益，但阶段目标及质量认证未通过。** 相对冻结 Graph 的两场景配对中位加速比几何平均为 1.099423×，低于预声明 1.10×。固定兔子的基线及候选存在冻结材料范围越界；没有修改范围来追认通过。

IPC 默认仍为 `legacy`，`.01` 累计阈值、最少 6 次更新、`pcg_rho_tol=1e-4` 不变。新门控仅显式启用；本轮执行组合也未推广默认。按用户决定先做本机；本轮没有连接 AutoDL。

## 1. 实施内容与真实身份

| 项目 | 已完成的实现 |
|---|---|
| 独立 IPC 配置 | `solver/ipc_options.h`；legacy / movement_only / gated / compensated；累计阈值与原 movement 阈值分开，旧配置仍可读 |
| 残差观测 | `solver/ipc_residual.inl`；GPU 最大范数，排除固定 FEM 自由度，分别输出 ABD/FEM，非有限传播，独立 CPU 审计开关 |
| 接受步预算 | `solver/ipc_budget.h`；固定第六次有效更新前参考，零/全步和非法步处理、预算恒等式、历史否决 |
| 求解观测 | `core/GIPC.cu` 输出退出原因、有效更新、alpha、beta、Kappa、接触量、系统编号；门控显式替换旧累计出口，保留原 movement 时序 |
| 成本追踪 | 完整物理帧、退出轮装配、普通/FullCCD BVH、查询、CCD、能量与线搜索；修复嵌套追踪过早 flush 和退出轮 CUDA 事件释放 |
| 仅观测原版 | `sources/stiff_base_observed` 只替换应用导出文件，导出真实速度，冻结材料及求解代码保持原版 |
| 公共工具 | `ipc_benchmark.py`、`ipc_delivery.py`、公共 runner/config/validator；每运行 requested/resolved 配置及程序身份落盘 |

| 程序 | SHA-256 |
|---|---|
| 原版 Stiff | `1c175da5a664bbc98657ddb6690a9902d821732c19e9b18f1f3ce0e868e05205` |
| 冻结 Graph | `57782cfeabdc35eb3151187a0f466911ef7b2ecf831b13ecc60ef221385b75f6` |
| 观测原版 | `39aa0710b640c5a88cfa2b462c61ce471936078d92ca7c544dc753a063e651fe` |
| 当前活动程序 | `fd76e02b94f6438640e6b4483abd1d094d675a50810f148ed96861d915eec8c4` |

活动构建记录实际源文件/对象/链接及依赖身份。观测原版补全了 35 个实际编译单元和链接输入；保留首次不完整的 manifest 及旧运行快照，未重解释旧数据。当前活动 build manifest 编译单元 37 个。

观测中性边界：3 帧原版/观测原版方向、PCG 与退出原因一致，位置最大分量差约 1.8e-11/3.1e-11 米；这不能代替 100 帧等价。后续原版复测超出观测基线范围，完整观测中性与质量认证仍待确认。

## 2. 本机完整性能矩阵

RTX 3070 Laptop，共享桌面；dt=.01、从零连续 100 帧、相同材料和停止参数。3 轮交错配对，重型诊断关闭。下表速度是**每轮耗时比的中位数**，不是中位耗时的比。加载与导出在求解计时外，Graph 构建、矩阵转换、MAS 准备在求解计时内。

| 场景 | 配置 | 时间中位 / 秒 | 相对同轮原版 | 相对冻结 Graph |
|---|---|---:|---:|---:|
| 悬挂 | 原版 Stiff | 4.8028 | 1.000× | 0.686× |
| 悬挂 | 冻结 IPC＋Graph | 3.2951 | 1.458× | 1.000× |
| 悬挂 | 执行组合 host | 4.7665 | 1.040× | 0.691× |
| 悬挂 | 执行组合 Graph | 3.0371 | 1.586× | 1.085× |
| 固定兔子 | 原版 Stiff | 14.8598 | 1.000× | 0.755× |
| 固定兔子 | 冻结 IPC＋Graph | 11.0813 | 1.324× | 1.000× |
| 固定兔子 | 执行组合 host | 14.1917 | 1.036× | 0.781× |
| 固定兔子 | 执行组合 Graph | 10.0279 | 1.482× | 1.114× |

执行组合为 FullCCD swept refit＋batched energy＋energy reuse；普通离散 BVH refit没有实现。组合内部 Graph 相对 host：悬挂 1.526×，固定兔子 1.420×。

相对冻结 Graph 几何平均 1.099423×，3 对样本的诊断性单侧 95% 配对自助法下界 1.030187×。即使将 1.099 四舍五入为 1.10，也不满足预先声明门槛。共享负载、只有三对、质量未通过，均不允许认证阶段目标或 2×。

![本机性能与门控工作量](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/figures/ipc_revision_20261005_performance.png)

## 3. 已有组件消融与成本归因

| 场景 | 当前 Graph 秒 | +refit 秒 | +batch 秒 | +reuse 秒 | 全组合秒 |
|---|---:|---:|---:|---:|---:|
| 悬挂 | 3.3186 | 3.2481 | 3.2561 | 3.4042 | 3.1218 |
| 固定兔子 | 11.1162 | 10.7113 | 11.0457 | 10.8714 | 10.4627 |

36 组消融均完成。单项收益不稳定；本轮同活动程序消融，全组合相对 Graph 的中位耗时比约 1.063/1.062×。此表与上节独立最终矩阵是不同轮次，不能拼接最快样本。

完整窗口 trace 包含固定算子探针，已删除探针子树成本再分析；没有把探针成本或 CPU 等待再次计入正式性能。

| 场景 / 窗口 | 扣探针后帧区间 ms | 扣探针后线性 ms | 删除探针 ms |
|---|---:|---:|---:|
| hang / 1,2,3,41,42,43 | 765.747 | 238.735 | 710.005 |
| fixed_bunny / 1,2,3,39,40,41,57,58,59 | 1267.730 | 526.375 | 1565.515 |
| mixed / 1,2,3,24,25,26,33,34,35 | 2008.403 | 695.047 | 4907.700 |

这些 CUDA 事件区间仍包含诊断和提交间隙；嵌套 inclusive 范围不能相加。Nsight 另取无算子探针的一帧节点追踪，确认以下启动决策：

| 新候选 | 本轮证据 | 决策 |
|---|---|---|
| MAS 输出与 r.z 融合 | 全部 PCG 点积 kernel 仅占 PCG 投影时间约 4.01%/4.31%；可消除 r.z 子集更小，低于 10% | 不启动；未排除其他场景可能有收益 |
| Hessian 静态/动态符号分离 | 整个矩阵转换约占帧 3.32%/3.53%；符号部分只是子集 | 不启动；不足整场 10% 门槛 |
| 普通 BVH refit | 建树＋查询约 13.68%/13.19%，但建树全部成本上界仅 5.53%/3.34%；较长选定区间为 2.88%/2.16%，refit还要更新 bounds | 无支持整场至少 5% 的预测，不启动 |

没有为了用完两个候选名额而编写 kernel。主要剩余成本是不可删去的碰撞查询、线搜索，以及线性准备/SpMV；目前证据不足以证明新的有限融合能越过局部门槛。

## 4. 残差门控有界评估

legacy 保留原累计出口；gated/compensated替换旧累计出口，保留原 movement 判定。第六次有效接受更新前冻结参考，不随后重置。原始混合坐标范数仅是算法诊断，不是米、米/秒误差保证。

| 场景 | 配置 | 秒中位（含影子观测） | 方向中位 | PCG中位 | 峰值拉伸重复范围 |
|---|---|---:|---:|---:|---|
| 悬挂 | legacy＋shadow | 3.4363 | 442 | 14021 | 1.128876196–1.128876226 |
| 悬挂 | gated .01/.03 | 4.2385 | 554 | 17637 | 1.126931360–1.126931373 |
| 悬挂 | gated .001/.03 | 4.2978 | 554 | 17662 | 1.126931361–1.126931377 |
| 固定兔子 | legacy＋shadow | 11.2962 | 587 | 37751 | 1.035229691–1.035533308 |
| 固定兔子 | gated .01/.03 | 22.0430 | 1143 | 71901 | 1.034700164–1.041638345 |
| 固定兔子 | gated .001/.03 | 22.5221 | 1136 | 71094 | 1.035459194–1.040568425 |

新增门控相对 legacy＋shadow：悬挂 PCG约 +26%、方向 +25%、耗时 +23%；固定兔子 PCG约 +90%、方向约 +95%、耗时约 +95%。表中含残差读回，不能当正式速度组；工作量增加来自门控延迟退出，而不是 Graph 失败。

悬挂峰值拉伸从约1.128876到1.126931，降幅约0.17%；固定兔子gated两次通过一次越界，兼容阈值一次通过两次越界，最差峰值达1.041638/1.040568。残差更严格没有保证本项目材料指标更好，因此不推广。

所有观测预算有效，恒等式最大误差为 6.661e-16。movement 出口仍可在相对残差>.03时退出，报告没有把它标成门控证明。

独立终端装配审计：悬挂f43和固定兔子f59均以累计出口退出，最后alpha=1、beta=0，但终端相对残差约0.123601/0.159523，均高于.03。额外装配约2.10/2.55ms，残差观测另计；不加入正式速度组。

历史补偿：legacy影子观测未出现独立否决；延迟求解的gated轨迹在固定兔子f30出现两次历史否决，兼容轨迹f30/f63各一次。按条件仅运行28–30和57–59两个窗口，均从零推进，保留旧CCD。

| 窗口 | 配置 | 窗口方向 | 窗口PCG | 窗口最大拉伸 |
|---|---|---:|---:|---:|
| [28, 30] | gated | 32 | 1825 | 1.027468295 |
| [28, 30] | compensated | 32 | 1825 | 1.027482840 |
| [57, 59] | gated | 42 | 3319 | 1.024758767 |
| [57, 59] | compensated | 39 | 3134 | 1.022851054 |

两个独立运行存在数值分岔：28–30没有稳定的方向减少，57–59的小幅拉伸/工作差异没有配对重复证明，整体从零耗时也无可靠改善。全步会清除本递推的历史影响，CPU预算单元检查覆盖这一恒等式。本分支停止，不扩大参数网格、不进入长测。

## 5. 质量、碰撞与兼容边界

每主场景观测原版3次标定＋2次独立复核，复核均通过。候选测试前冻结整段最大/高分位拉伸及固定漂移范围。固定兔子后续正式原版3次却均越界：有的是最大拉伸，有的是p99；不能在看过候选后扩大范围。

| 场景 | 原版正式材料通过 | 冻结Graph通过 | 组合host通过 | 组合Graph通过 |
|---|---:|---:|---:|---:|
| 悬挂 | 3/3 | 3/3 | 3/3 | 3/3 |
| 固定兔子 | 0/3 | 2/3 | 3/3 | 2/3 |

固定兔子组合Graph最差峰值1.038670877，冻结上界1.036172766；未通过。当前协议在原版本身也失败，质量认证不可用。四条独立接受路径诊断的材料指标恰好通过，不能用它们覆盖正式组失败。

| 独立CCD轨迹 | 接受段 | 实际检查路径（含帧间闭合桥） | 碰撞标记 |
|---|---:|---:|---:|
| hang / observed | 442 | 541 | 0 |
| hang / combined | 442 | 541 | 0 |
| fixed_bunny / observed | 571 | 670 | 0 |
| fixed_bunny / combined | 584 | 683 | 0 |

全部100帧子步编号连续、首尾与逐帧端点逐字节闭合，独立CPU BVH＋Tight-Inclusion通过。总共2435条路径零标记。只覆盖这些独立诊断轨迹，未宣称覆盖全部计时轨迹。

真实速度单独导出并比较：

| 场景 / 体 | 量 | 基线重复最大帧RMS范围 | 组合相对基线范围 |
|---|---|---|---|
| hang/cloth | position | 1.83914e-06–2.40488e-06 m | 1.10268e-06–2.21551e-06 m |
| hang/cloth | velocity | 6.526e-05–0.000111476 m/s | 4.59569e-05–0.000102669 m/s |
| fixed_bunny/cloth | position | 0.0134829–0.0217103 m | 0.0294302–0.0347966 m |
| fixed_bunny/cloth | velocity | 0.18939–0.212821 m/s | 0.23171–0.280219 m/s |
| fixed_bunny/ABD | position | 1.39582e-17–1.75388e-17 m | 3.0924e-18–1.71894e-17 m |
| fixed_bunny/ABD | velocity | 1.39582e-15–1.75388e-15 m/s | 3.0924e-16–1.71894e-15 m/s |

混合兔子兼容100帧完成，耗时25.048s，方向606、PCG19815；PCG失败0，FEM最小J=-0.689471665，最大非正单元数=4，ABD最小J=0.999982609。这是兼容诊断，没有同轮混合材料范围或独立CCD，不能称混合质量验收通过。

## 6. 验证、预算与关闭的分支

- 配置契约：9项、36子检查通过；IPC预算C++检查通过（全步、零步、重复接受、非法输入、历史否决）。
- GPU组件/守卫/六份历史系统：8项顶层fixture全部通过。覆盖Graph重放、地址/容量增长、零RHS、异常pivot、非有限逆、关闭开关恢复等已有边界。
- 三场景3帧CPU/GPU非线性残差审计通过；这是梯度归约一致性，未把PCG rho阈值解释为真残差。
- 六历史CPU线性系统：6份完成；整体passed=True，准确参考最大真相对残差=5.124e-11（要求1e-8）。
- 最终92个活动运行的requested/resolved及实际执行模式逐项通过；源文件/对象身份通过。无改动增量构建0对象变化，程序SHA保持不变。
- 未启动新执行kernel：三项候选均缺少所需整场收益证据；不新增低精度、cuSPARSE、Tensor Core或整个Newton Graph。
- 无PCG触顶、breakdown、非有限导出状态的声明按已完成运行逐项核验；不以此代替材料质量。
- 300帧计划已生成但未运行：阶段收益未过、固定兔子质量失败；不越过长测门槛。AutoDL七组配对遵照用户决定暂缓。
- 保留最初嵌套trace flush失败、构建/工具修复日志与所有失败材料记录；没有全量清理。

## 7. 可复现入口与后续决策

解释器：`E:/Anaconda/envs/DL/python.exe`。公共入口采用独立运行名称，旧运行目录不可覆盖。已执行命令及配置：

```text
python tools/active/build.py build --label ipc_frame_trace_20261005 --jobs 2
python -m pytest tools/active/test_contracts.py -q
builds/active/Release/ipc_budget_test.exe
python tools/active/fixtures.py --name ipc_revision_20261005_fixtures
python tools/active/batch.py --plan configs/active/ipc_revision_20261005_components.json
python tools/active/batch.py --plan configs/active/ipc_revision_20261005_residual.json
python tools/active/batch.py --plan configs/active/ipc_revision_20261005_final.json
python tools/active/batch.py --plan configs/active/ipc_revision_20261005_compensation.json
python tools/active/batch.py --plan configs/active/ipc_revision_20261005_audit.json
python tools/active/ipc_delivery.py audit
python tools/active/verify_systems.py --historical-fixtures runs/active/ipc_revision_20261005_fixtures --output reports/active/ipc_revision_20261005_historical_cpu_systems.json
```

后续不继续调整AL-TOI。先在独立阶段查清固定兔子原版的长窗重复范围为何不能覆盖正式组，验证导出频率/观测同步与原子装配数值变化的作用；不能把这个范围直接当成候选可额外形变预算。只有重新预声明并独立复核的协议能认证时，才决定是否将现有执行组合送AutoDL。

若重新进入执行优化，优先需要证明碰撞查询或线搜索的可消除成本；当前三项候选的低占比证据不支持继续凭kernel想法迭代。残差门控保留诊断接口，停止作为加速主线。

## 8. 证据文件与结论状态

| 结论 | 状态 |
|---|---|
| 配置、预算、残差归约和已有边界fixture正确性 | 已验证（限测试覆盖） |
| 执行组合较冻结Graph有约10%诊断增量 | 支持但未证明（未过严格1.10和质量认证） |
| 残差门控能加速当前IPC | 已排除（限两主场景及两配置） |
| 历史补偿有独立可重复质量收益 | 待验证；本轮有限分支关闭 |
| 新MAS融合/符号分离/普通BVHrefit达推广门槛 | 待验证；启动条件不满足，本轮未实施 |
| 两场景同材料质量阶段验收、300帧、4090验收、2× | 未完成；不宣称通过 |

- [quality_protocol](ipc_revision_20261005_quality_protocol.json)
- [performance](ipc_revision_20261005_performance.json)
- [components_analysis](ipc_revision_20261005_components_analysis.json)
- [final_material_checks](ipc_revision_20261005_final_material_checks.json)
- [cost_decisions](ipc_revision_20261005_cost_decisions.json)
- [selected_cost_corrected](ipc_revision_20261005_selected_cost_corrected.json)
- [residual_evidence](ipc_revision_20261005_residual_evidence.json)
- [residual_material_checks](ipc_revision_20261005_residual_material_checks.json)
- [accepted_ccd](ipc_revision_20261005_accepted_ccd.json)
- [trajectory_comparison](ipc_revision_20261005_trajectory_comparison.json)
- [observer_build_provenance](ipc_revision_20261005_observer_build_provenance.json)
- [historical_cpu_systems](ipc_revision_20261005_historical_cpu_systems.json)
- [final_verification](ipc_revision_20261005_final_verification.json)

运行/失败证据统一登记到 [EXPERIMENT_INDEX.json](EXPERIMENT_INDEX.json)。实施计划见 [IPC_EXECUTION_IMPLEMENTATION_PLAN_20261005.md](IPC_EXECUTION_IMPLEMENTATION_PLAN_20261005.md)。
