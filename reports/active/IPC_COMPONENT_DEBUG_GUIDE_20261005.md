# IPC 两个执行组件：代码与调试指南

本文说明本轮新增代码的职责、开关、证据和检错入口。两个组件均默认关闭，当前只完成局部正确性检查，完整速度和质量验收未完成。活动代码沿用一个 overlay，v50/v54 和原版程序保留。

## 1. 文件职责

| 模块 | 文件（相对项目根目录） | 职责 |
| --- | --- | --- |
| MAS 配置 | `sources/stiff_active/StiffGIPC/solver/mas_fused_dot_options.h` | 严格解析独立开关，不把环境解析塞入每次 GPU 作用 |
| MAS 作用 | `solver/MASPreconditioner.cu/.cuh`、`solver/mas_fused_dot_kernels.cuh`（均位于活动 StiffGIPC） | 保留原限制/局部作用/延拓，只在最终 FEM z 写入时产生 FP64 点积 partial |
| 最终写者检查 | `linear_system/linear_system/global_linear_system.cu/.h`、`i_linear_system_solver.cu/.h` | 确认唯一 MAS、区间不重叠、写入顺序正确；非 FEM 的最终贡献单独计入 |
| PCG 接入 | `linear_system/solver/pcg_solver.cu/.h`、`pcg_graph_impl.inl` | 初始和后续 rho 统一选择；PCGSolver 独立拥有 partial/CUB scratch；Graph 签名包含地址和尺寸 |
| 固定系统检错 | `linear_system/solver/pcg_mas_dot_study.inl`、`pcg_fixed_study.inl` | 保存/恢复主状态，固定局部 MAS 结果，核对旧/新输出、rho 和零 RHS |
| 普通 BVH 策略 | `collision/discrete_bvh.h/.inl` | 状态、存储签名、固定重建周期、诊断计数和同状态查询核对 |
| BVH 调用接入 | `collision/mlbvh.cu/.cuh` | 原 Construct 按策略选择；FullCCD 标记 swept；原 GPU 建树/查询 helper 保留 |
| 框架观测 | `core/GIPC.cu`、`solver/toi_options.h` | 配置展开及逐帧计数；重置计数不重置树状态 |
| 公共入口 | `tools/active/config.py`、`run.py`、`validate_run.py` | 请求/解析/实际执行三层核对，冻结程序拒绝新增活动开关 |
| 有限实验 | `tools/active/report_components.py` | 准备配置、串行运行、同场景同配置失败后停止重复、保留不可覆盖分析 |

更细的实现契约见 [MAS_FUSED_DOT.md](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/sources/stiff_active/StiffGIPC/solver/MAS_FUSED_DOT.md) 和 [DISCRETE_BVH_REFIT.md](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/sources/stiff_active/StiffGIPC/collision/DISCRETE_BVH_REFIT.md)。其中 implementation-time/handoff 段记录子代理当时的验证范围，最新本机结果以本轮结果书为准。

## 2. 独立配置与旧路径

| runner 参数 | 默认值 | 环境变量 | 检错边界 |
| --- | --- | --- | --- |
| `mas_fused_dot` | false | `GIPC_MAS_FUSED_DOT` | 本轮公共实验限 IPC legacy MAS；最终 owner 无效时记录原因，不能在一次 PCG 内静默换预条件器 |
| `fixed_mas_dot_study` | false | `GIPC_FIXED_MAS_DOT_STUDY` | 只允许末帧 direction1、单固定系统、仅 fixed 诊断，不与其他 fixed study 混用 |
| `discrete_bvh_refit` | false | `GIPC_DISCRETE_BVH_REFIT` | 无有效普通树、swept 或存储签名变化时重建 |
| `discrete_bvh_rebuild_interval` | 8 | `GIPC_DISCRETE_BVH_REBUILD_INTERVAL` | 1–1024 整数；按普通 Construct 调用计数，不按物理帧计数；1 是全重建对照 |
| `discrete_bvh_validate` | false | `GIPC_DISCRETE_BVH_VALIDATE` | 必须同时启用 refit；额外查询/传输/同步属于诊断成本 |

布尔参数只接受布尔值；原生环境只接受 0/1。两个性能开关不由 combined preset 自动开启。设回 false 即使用旧 GPU 执行路径，无需移除代码。IPC 停止模式 legacy、累计阈值 .01、min_updates=6、PCG rho=1e-4、材料和完整安全 CCD 均未调整。

## 3. MAS 检错顺序

1. 看 `requested.json` 的 expanded_config 与 `resolved_config.json` 的 `report_components`，确认请求和解析一致。
2. 看 `output/stats.json` 每个 `newton[].pcg` 的 `mas_fused_dot_requested/effective/fallback_reason/partial_count`。只看请求 true 不能证明实际接入；本轮两布料 dot smoke 全部实际生效。
3. 若 effective=false，先查 MAS 个数、全局偏移、FEM 总节点数、最终 writer 顺序和重叠区间。不能为得到 true 删除这些检查。
4. 若 rho 不一致，先在同进程冻结 local action，检查完整 z 是否逐位一致，再检查 rho 的记录界限。旧 legacy 限制含 FP32 原子聚合，反复执行完整 MAS 的微差不能直接归因到新 collect。
5. 查看 `fixed/f2_n1_mas_dot_study.json`：实际 RHS/零 RHS/signed-tail 三输入，各 host、组件 Graph 首次与重放；全部 work、A/b/M、RHS、MAS scratch、生产 Graph 恢复布尔值必须 true。
6. 初始 rho 与后续 rho 必须用同一 chosen path。零 RHS 的完整 fused PCG 应 0 次迭代且 x/z/r 有限为零；这不能代替非零全 PCG 或混合活动 ABD 验收。

partial 与 CUB 临时内存属于 PCGSolver，捕获前准备。地址、长度、临时字节数和有效模式改变时 Graph 失效；捕获内部不得分配、重新解析环境或读回。实际容量增长/失效、活动 ABD 混合系统本轮尚未覆盖。

## 4. 普通 BVH 检错顺序

树状态明确为 invalid / discrete / swept。完整建树建立 discrete；FullCCD 的 Construct/Refit 标记 swept。普通 refit 更新全部当前叶和内部 AABB，只复用拓扑，不复用接触集。借用拓扑在 init 与 invalidate 之间不可原地改编号；原地编辑必须调用 `invalidate_discrete_topology()`。

每帧 `discrete_bvh.face/edge` 中：

- `construct_calls` 是普通构建请求次数；关闭组件时看 `disabled_rebuilds`。
- `production_rebuilds/refits` 是实际生产选择；`swept_rebuilds` 解释共享 FullCCD 状态导致的重建。
- `diagnostic_refits/validation_calls/validation_passed` 是主动同状态检查，不能当生产命中率。
- `diagnostic_pairs_compared` 是累计多重集合比较条目，含重复，不是唯一接触数。
- `diagnostic_overflow_retries` 是诊断从零容量起算后的 scratch 增长重试，不证明生产缓冲溢出被覆盖。

原版 Stiff 没有新计数，分析中的 `discrete_bvh_counters_available=false` 表示缺失；不能把填零解释为没有 BVH 成本。

validate 会核对树根/链接/ID/深度≤64、当前 bounds、各类型 DCD/CCD pair 多重集合和 MatIndex 类型内双射。出现不一致会失败，不静默接受。诊断最后保留全重建树 age1，因此诊断轨迹和耗时不能用来认证生产 refit 收益。

本轮观测的普通构建都紧随 swept 操作，生产 refit 为零。若后续要提高命中率，应单独设计普通/swept 缓存并计算新增常驻显存；不能直接忽略 swept 失效。本轮没有实施该改动。

## 5. 可复现入口

在 `E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929`、PowerShell 7 下：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/active/build.py build --label ipc_report_components_20261005 --jobs 2
& 'E:/Anaconda/envs/DL/python.exe' tools/active/test_contracts.py
& 'E:/Anaconda/envs/DL/python.exe' tools/active/fixtures.py --name ipc_report_components_20261005_fixtures
& 'E:/Anaconda/envs/DL/python.exe' tools/active/report_components.py run --stage smoke
& 'E:/Anaconda/envs/DL/python.exe' tools/active/report_components.py run --stage guards
& 'E:/Anaconda/envs/DL/python.exe' tools/active/check_report_component_guards.py
& 'E:/Anaconda/envs/DL/python.exe' tools/active/report_components.py run --stage performance
& 'E:/Anaconda/envs/DL/python.exe' tools/active/report_components.py run --stage quality
& 'E:/Anaconda/envs/DL/python.exe' tools/active/report_components.py analyze --revision 2
& 'E:/Anaconda/envs/DL/python.exe' tools/active/verify_report_components_delivery.py
```

这些是本轮执行命令，输出名不可覆盖，不能直接在已有证据上重跑。新轮先修改 TAG/实验身份并冻结新的有限预算，不删除原文件来绕过检查。smoke/guards/performance/quality 的精确配置分别保存在 `configs/active/ipc_report_components_20261005_{stage}.json`；guards 实际使用 `_guards_v2.json`，原文件保留。

CPU 分析不启动 GPU。`report_component_ccd.py` 只在 both 两条完整质量轨迹存在时使用；本轮质量阶段因资源失败全部跳过，未执行 CCD 审计。资源失败仍是失败，已导出的有限前缀可以作诊断但不能替代完整窗口。
