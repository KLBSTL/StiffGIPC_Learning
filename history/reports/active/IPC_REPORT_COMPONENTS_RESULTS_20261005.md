# 两份报告组件实现与本机测试结果（2026-10-05）

已实现 **MAS 最终输出与 rᵀz 融合**、**普通离散 BVH refit** 两个独立组件，并完成本机局部守卫。完整 100 帧速度和质量验收因既有显存保留线停止，原版 Stiff 和旧组合也出现同类资源失败。因此不宣布新加速比、不推广默认开关；两组件均默认关闭。

## 1. 实现范围与程序身份

用户本轮明确要求先实现几个报告组件再测试，覆盖此前成本证据不足而暂缓编写候选的决定。原报告是设计依据，其历史计时不视作本轮实测。第二份报告的 gated/compensated 残差规则此前已实现并完成有界评估，本轮没有重新调参；Hessian 符号拆分没有同时展开。

| 组件 | 已实现内容 | 保留边界 |
| --- | --- | --- |
| MAS final collect + rho | FEM 最终 z 写入时产生 FP64 partial；最终非 FEM 坐标另行贡献；统一设备归约；host/conditional Graph 初始与后续 rho 接入 | 原 MAS 精度/限制/局部作用/延拓不变；最终 writer 与区间检查；一次 PCG 内不切换；独立 scratch 与 Graph 地址/尺寸签名 |
| 普通离散 BVH refit | invalid/discrete/swept 状态；全部当前 bounds 更新；首次、存储/映射变化、swept、固定周期时重建；同状态重建查询核对 | 不复用旧接触，不裁掉安全 CCD；原 query/helper 保留；原地拓扑改变显式 invalidate；周期按 Construct 调用计数 |
| 公共配置与观测 | 严格独立开关；requested/resolved/实际 PCG 和逐帧计数核对；固定系统 study 和查询验证独立 | combined 不自动开启新组件；旧请求可读；冻结原版拒绝活动开关；诊断和性能分组 |

代码职责、旧路径、生命周期、检错顺序及命令见 [调试指南](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/IPC_COMPONENT_DEBUG_GUIDE_20261005.md)。实现计划见 [本轮计划](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/IPC_REPORT_COMPONENTS_PLAN_20261005.md)。

Release 构建完成：37 个编译单元，18 个对象改变，0 编译错误；日志有 425 条既有模板/编译器警告，本轮没有声称无警告。当前程序 SHA256：

`9ae1c09394e9f36aed90499dae578064f28921bd96467f6c5a45206cea2a8df0`

完整源码—对象—包含依赖—链接输入记录位于 [构建 manifest](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/builds/active/provenance/20261005T143313_739171Z_ipc_report_components_20261005/manifest.json)。native 源码在该构建后未变；后续只修改 Python 分析、索引和说明文档。

IPC legacy、累计阈值 .01、min_updates=6、PCG rho1e-4、legacy restriction atomic、材料和碰撞安全参数未变。旧质量协议 SHA256 仍为 `1cfb9045d6988fa606b8bb76afdd71e20661ef602c84296c49cdda9a0d07d78f`。历史质量失败和基线范围不改判。

## 2. 实际运行与资源停止

GPU 为 RTX 3070 Laptop 8 GiB。启动空闲显存约 3377–3453 MiB，保留 1536 MiB，运行可用预算约 1841–1917 MiB。达到既有保留线后公共 runner 停止；没有放宽预算、终止其他应用或清理用户数据。资源状态为 `memory_budget`，不是数值 breakdown，也不能判作成功。

| 阶段 | 计划 | 实际尝试 | 完成 | 资源失败 | 未执行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 12 | 12 | 10 | 2（混合） | 0 |
| guards v2 | 6 | 5 | 4 | 1（固定兔子接触窗） | 1（混合） |
| 100 帧性能 | 30 | 10 | 0 | 10 | 20（同配置后续重复） |
| 独立质量 | 12 | 0 | 0 | 0 | 12（先前场景资源失败） |
| 合计场景运行 | 60 | 27 | 14 | 13 | 33 |

另有一组 8 项既有 GPU fixture，8/8 通过。配置契约为 13 项测试、68 个子检查，通过。

两布料 old/dot/discrete/both Graph、both host 的 3 帧启动均完成，配置与实际执行一致；混合 old/both 在 3/2 个记录帧后触发资源线。性能第一轮中，悬挂 Stiff 完成前 40 帧，其余四臂 41 帧；固定兔子 Stiff 24 帧，其余四臂 22 帧。十次请求均为 100 帧且均失败，分析没有将前缀改写成完整测试。

性能 batch 原始 JSON 在执行结束时漏写最后追加的 skipped 列表，原始证据保留；另生成 [执行覆盖清单](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_report_components_20261005_execution_coverage.json)，逐项对照已冻结计划和实际尝试推导 20 项未执行配置及其前序同配置失败原因。runner 已修正最后一次写出，质量阶段 12 项跳过均有原始记录。这是记录完整性修复，不是补跑或改判结果。

## 3. 局部正确性证据

### MAS 融合

悬挂和固定兔子 f2/direction1 两个受保护系统均通过：实际 RHS、零 RHS、signed-tail 三输入，各 host、组件 Graph 首次、重放，共每系统 9 次 collect 比较，完整 z 逐位一致。实际 RHS 的最大 rho 差分别约 8.47e-22、4.49e-20，均在记录的 FP64 舍入界限内。

完整 fused PCG 的 host/conditional Graph 零 RHS 两种模式均 0 次迭代，x/z/r 有限且全零。主工作缓冲、A/b/M、RHS、MAS scratch、生产 Graph handle/key/地址恢复检查全部通过。dot smoke 实际非零初始 rho 路径生效：悬挂 9 个方向、固定兔子 7 个方向，无 PCG failure。

独立 FP64 sparse LU 准确参考真相对残差分别 2.78e-15、1.64e-14；这是线性参考结果，不是默认 PCG 或 rho 的误差保证。C++ long double 点积对照不被自动称为高于 FP64 的精度。

此检查冻结旧局部作用，隔离 final collect 正确性，不要求旧原子 MAS 反复作用逐位重复。没有取得非零固定系统完整准备＋PCG 的重复性能证据；活动 ABD 混合体、实际 Graph 容量增长/失效、非 legacy 实验路径仍待验证。

### 普通 BVH

两个 3 帧初始窗分别通过 face/edge 各 9/7 次同状态检查，但比较接触条目为零，不能用它们声称非空接触覆盖。预先声明的固定兔子 59 帧接触诊断在第 23 个记录帧后被资源线停止：

- face/edge 各 65 次、合计 130 次同状态 refit/rebuild 查询检查通过，失败 0。
- 累计比较 face29＋edge39＝68 条 pair 多重集合条目，含重复，不是 68 个唯一接触。
- 诊断从零容量起算，scratch 增长重试 face16＋edge12＝28 次；不是生产缓冲溢出覆盖。
- 树拓扑、原 ID、栈深度、bounds、typed DCD/CCD 多重集合及 MatIndex 类型内双射检查通过。

这是保存前缀上的局部检查；59 帧请求仍失败。混合活动体、原地拓扑编辑和完整接受路径 CPU CCD 本轮没有新增覆盖。详见 [局部守卫证据](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_report_components_20261005_guard_checks.json)。

## 4. 速度：只能报告不完整前缀诊断

完整 100 帧配对数组为空，中位加速比为 null。下表仅比较第一轮失败运行共同保存的悬挂 40 帧、固定兔子 22 帧；它们是事后取共同前缀、每臂一次观测，未预声明为计时窗，也没有置信区间，不能作为正式加速比。

old 为本轮程序关闭两个新组件、保留已有 FullCCD refit＋批量能量＋能量复用的 combined Graph；dot/discrete/both 只增加相应新组件。Stiff 为原版。

| 场景/共同前缀 | 配置 | 求解秒数 | old/该配置 | Stiff/该配置 | PCG |
| --- | --- | ---: | ---: | ---: | ---: |
| 悬挂40帧 | Stiff | 3.327493 | 0.58932 | 1.00000 | 6011 |
| 悬挂40帧 | old | 1.960960 | 1.00000 | 1.69687 | 6011 |
| 悬挂40帧 | dot | 1.942957 | 1.00927 | 1.71259 | 6011 |
| 悬挂40帧 | discrete | 1.990831 | 0.98500 | 1.67141 | 6011 |
| 悬挂40帧 | both | 1.947144 | 1.00710 | 1.70891 | 6011 |
| 固定兔子22帧 | Stiff | 1.565146 | 0.86362 | 1.00000 | 906 |
| 固定兔子22帧 | old | 1.351697 | 1.00000 | 1.15791 | 906 |
| 固定兔子22帧 | dot | 1.354393 | 0.99801 | 1.15561 | 906 |
| 固定兔子22帧 | discrete | 1.332908 | 1.01410 | 1.17423 | 906 |
| 固定兔子22帧 | both | 1.351861 | 0.99988 | 1.15777 | 906 |

在这些有限前缀中，新增组合相对 old 约 +0.71% / -0.01%；没有稳定增量收益证据。相对 Stiff 的差距主要来自旧组合，不能全部算成新增组件收益。

**已确认的实现限制：**discrete/both 在保存的悬挂 40 帧中每棵树 207 次、固定兔子 22 帧每棵树 59 次普通 Construct，全部因 swept 状态而重建，生产 refit=0。当前普通和 FullCCD 共用节点，每次 FullCCD 标记 swept；下一次普通构建正确失效。因此仅实现 refit 入口不能减少这条调用序列的建树工作。诊断主动执行 refit 不等于生产命中。

这一判断限于已记录窗口。原版 Stiff 不含新计数，分析 v2 显式标记缺失，填零不得解释为原版不建树。分析 v1 保留，v2 仅补计数可用性和 disabled rebuilds，不改变成功/失败判定。详见 [分析 v2](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_report_components_20261005_analysis_v2.json)。

## 5. 质量：局部没有数值失败，整段仍待验

共同前缀全部导出状态有限，无 PCG cap/breakdown，五臂方向数一致：悬挂207、固定兔子59。悬挂最大拉伸跨五臂范围 1.12887618675–1.12887623098，p99 约 1.032409331–1.032409342；最大拉伸跨度约 4.42e-8。固定兔子最大拉伸 1.000016863697–1.000016863996，跨度约 2.98e-10；固定漂移约 3.10e-17–5.72e-17 m，悬挂固定漂移为零。

这些指标支持保存前缀内没有明显材料差异，但不能外推到接触发展后的 100 帧。独立实际速度/位置分体比较、完整材料协议、接受路径 CPU CCD 都未启动，因为质量阶段 12 项全部由既有资源失败跳过。没有继承历史 CCD 零标记，没有新增 300 帧/AutoDL，也没有改变此前质量无法认证的结论。

## 6. 本轮结论与有限后续

| 结论 | 状态 |
| --- | --- |
| 两个组件、独立配置、旧路径、实际执行观测已实现并编入当前程序 | 已验证 |
| 两布料固定 final collect 与受保护零 RHS、保存接触前缀 BVH 查询等价 | 已验证（局部范围） |
| 新 rho 归约不改变长期质量 | 待验证 |
| MAS fusion 提供稳定整场增量收益 | 待验证，现有单次前缀不支持宣布 |
| 当前共享树结构下普通 refit 在保存前缀提供收益 | 已排除（生产命中为零） |
| 两组件完整100帧质量/速度、活动混合体兼容 | 待验证，资源失败 |
| 新组件可加入最快默认组合 | 未成立，默认关闭 |

下一轮先解决测试资源可用性，在相同预算下重新取得完整矩阵。MAS 先补非零冻结系统 host/Graph 的准备＋求解对照和增长/失效检查，再决定是否保留。普通 BVH 如继续，只允许围绕独立 ordinary/swept 缓存作一次单独设计及显存/收益核算，不能删除现有失效规则；本轮未实现缓存拆分。不扩大周期/容差网格，不启动 TOI 再调参。

本轮命令及不可覆盖复现约定见调试指南。CPU 交付核验见 [verification](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_report_components_20261005_verification.json)：保留396项历史索引和全部旧决定，追加本轮实际运行、跳过配置及两个待推广候选；质量/性能认证和 default_promoted 均 false。
