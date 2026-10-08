# 报告 5 / 6 的线性系统审查

审查日期：2026-10-08。范围：PCG、MAS、SpMV、向量归约、精度/停止规则、库替换，以及完整线性准备到解分发的数据流。两份输入报告均已全文读取；它们是待核查材料，未将其中的示例补丁当作执行指令。

## 1. 结论与证据边界

1. 活动工程已经具有默认 MAS、连接性层次构造、CUB 点积归约、设备端条件 PCG Graph、持久 workspace、MAS 的多种稳定参考路径。报告 5 的“先加入 MAS”和报告 6 的“把 CPU 串行点积替换为库”不符合当前源码。报告中 `src/solver/mas_precond.cu`、`SolverGPU.cu`、`Solver.cu` 等是示意路径，不能作为本仓库补丁定位。
2. 全局矩阵、PCG 向量与标量是 FP64；冻结的 legacy MAS 内部原本已有 FP32 residual、inverse 和 local-action 数据。保持当前数值路径意味着不再降精度，也不能把当前执行模式误称为“所有 MAS 中间计算均为 FP64”。切换 wide/inverse64/Cholesky 会改变预条件算子，不可自动归入同路径的执行优化。
3. 已确认一个活动 SpMV 的 CUB 接口合约缺陷：不足 32 项的尾 warp 提前退出后仍调用 32-lane `HeadSegmentedReduce`，并连续复用临时存储缺少 `__syncwarp()`。它已在主任务授权下做最小正确性修复；本分工未运行 GPU，现有质量差距的因果归属仍未知。
4. 七场景 120 帧现有单次 `base/all` 为 1.0753–1.4139×，几何平均 1.2629×，时间总和比 1.2266×，不支持报告 5 的“当前 1.6–1.95×”作为这轮起点，也不能从通用 CUDA 收益例子推出本工程已具备同质量 2×。
5. 线性阶段仍有较大成本，但它包含矩阵转换、预条件准备、PCG 和解分发，不等同于 SpMV。最多提出两个候选：精确符号结构复用的完整准备流程、完整算子视图到 row-gather SpMV 的数据流。第二项必须先补足热点证据；不重开退休融合、K4 性能分支或接触池。

活动源码审查时 HEAD：`9192ad1b8637757290440ff5ae16e01d301465e2`。七场景既有构建来源为 `3d8c15e`。本轮读取 `BUILD_IDENTITY.json` 中实际 source SHA256，与本地相关文件比对：`pcg_solver.cu`、`pcg_graph_impl.inl`、`MASPreconditioner.cu` 完全一致。SpMV 在该计时版本中尚未包含这次正确性修复；不能把旧计时作为修复版程序的实测。

| 证据文件 | 用途与限制 |
|---|---|
| `D:/C_files/deep-research-report (5).md`，184 行 | 报告 5 逐项建议与示例代码；外部材料，不是执行授权 |
| `D:/C_files/deep-research-report (6).md`，401 行 | 报告 6 逐项建议与示例代码；相同边界 |
| `AGENTS.md:3–15` | 活动源码/冻结 baseline；材料、完整 CCD、legacy `.01`/min6、rho `1e-4` 固定；不从局部或 profiler 数据认证速度 |
| `reports/autodl_seven_20261008/TIMINGS.csv` | 21 条完成记录、硬检查全部通过，每场景每臂一次；没有 active-host 臂、基线重复制定或完整质量认证 |
| `reports/autodl_seven_20261008/REPORT.md:3–24` | arm 定义、计时口径、配置与质量缺项 |
| `reports/autodl_seven_20261008/BUILD_IDENTITY.json` | 实际 Linux 编译 source 身份；本审查对照相关文件 hash |
| `reports/report_execution_20261007/COST_sphere.json`、`COST_fixed.json` | 从零 100 帧诊断 CPU inclusive 成本；嵌套 scope 不能相加为 GPU wall，Graph 内核的捕获期 CPU 提交不能充当每迭代 GPU 成本 |
| `reports/report_execution_20261007/MAIN_ANALYSIS.md:73–96` | 两个受限 Nsight 捕获及 SpMV/能量归约启动门槛；不是七场景整场热点证明 |

## 2. 生产数据流与固定数值语义

`GIPC::calculateMovingDirection` → `GlobalLinearSystem::solve_linear_system` → `build_linear_system` → PCG → `distribute_solution`。

| 阶段 | 具体位置 | 当前工作 |
|---|---|---|
| RHS / 子系统 | `linear_system/linear_system/global_linear_system.cu:61–71` | resize RHS/x，所有子系统写入当前 RHS |
| ABD 准备 | 同文件 `74–83` | 独立 ABD 局部预条件准备；不可把 FEM MAS 替换当作整个全局预条件器替换 |
| 矩阵转换 | 同文件 `85–90,378–396`；`linear_system/utils/converter.cu:138–181,186–210,247–296` | 每次构建 raw triplet keys，radix sort、值重排、分段/scan、unique 数读回、FP64 块归约 |
| FEM 子系统索引 | `linear_system/linear_system/i_preconditioner.cu:54–87` | 从 compact BCOO 索引筛选 FEM 块，DeviceSelect 并读回 count；scratch 与 converter 共用，不能直接把该数组当持久缓存 |
| MAS 层次 | `solver/MASPreconditioner.cu:2439–2466,1876–1923` | 无接触静态复用已有 opt-in；否则根据当前完整碰撞连接重建层次，含每层 metadata 读回 |
| MAS 当前数值 | 同文件 `2478–2517,2217–2297` | 当前 Hessian 的数值填充/聚合、legacy inverse 或稳定 factor，每个新系统重新准备 |
| PCG 循环 | `linear_system/solver/pcg_solver.cu:466–571`；`pcg_graph_impl.inl:135–312` | host 或条件 Graph；每轮 `Ap=A p`，`p·Ap`，x/r 更新，M 应用，rho 点积，p 更新 |
| 解分发 | `linear_system/linear_system/global_linear_system.cu:113–120,268–272` | 给各子系统取回本次解，等待设备 |

固定语义：`gipc/type_define.h:11–21` 定义 `Float=double`；`app/gl_main.cu:672,730` 给出/读取 `pcg_threshold=1e-4`，`core/gipc_system.cu:61` 写入配置。host 在 x/r 更新后检查上一轮 `rho = rᵀz` 的 `abs(rho) <= tol*rho_initial`（`pcg_solver.cu:533–540`）；Graph 用 `s[8]` 保留同一上一轮 rho（`pcg_graph_impl.inl:18,30–52`）。这不是 `||r||/||b|| <= 1e-4` 的真残差阈值；放宽 rho 或换 M 都可能改变当前停止工作量和求解结果。

Newton/IPC `.01`、min6 是另一层停止规则，见 `solver/ipc_options.h:9–11,14–39`；它不是 PCG 最少 6 次迭代。既有 breakdown、零 rho 非零 residual 和迭代上限保护保留，不把错误返回当正常收敛。

## 3. 逐点对照

状态仅使用：已实现、部分、未实现、不适用、错误。同一建议有实现与示例错误时拆开记录。

| 报告与行号 / 建议 | 当前实现位置 | 状态 | 源码/计时证据与纠正 |
|---|---|---|---|
| 5:29、137–147：假设对角/无预条件，加入 connectivity-enhanced MAS | `solver/PCG_SOLVER.cuh:25`；`core/gipc_system.cu:76–88`；`MASPreconditioner.cu:1848–1923` | 错误 | `P_type=1` 默认进入 FEM MAS；碰撞连接与多层 cluster 构造已存在。“新接入 MAS”的性能不应算到未来候选 |
| 5:138：按 Morton/网格重分组以减少迭代 | `app/gl_main.cu:1072–1099,1148`；`MASPreconditioner.cu:1636–1923` | 不适用 | 活动路径已有预分区映射与运行时连接增强。重做分区会改变 M 和同 rho 下的停止行为，不属于保证旧算子语义的执行优化；没有本轮证据证明重分区更快 |
| 6:141–144：调优 MAS / 其它预条件器 | `MASPreconditioner.cuh:47–62`；`.cu:2487–2517,2525–2619` | 部分 | legacy、wide、inverse64、Cholesky 均已存在；serial/warp 限制及 triangular/factor_inverse 也有独立入口。不能把保留的研究模式说成未实现，也不能用减少迭代数代替含 prepare+apply 的净计时 |
| 5:29：放宽 PCG 终止容差；5:164：收紧或放松验证容差 | `pcg_solver.cu:533–551`；`solver/ipc_options.h:9–11`；`tools/seven_eval/run.py:31–33` | 不适用 | 当前实验固定 rho `1e-4`、legacy `.01`/min6；事后放宽验收范围不能提供同质量速度证据 |
| 5:31、149–161；6:27、189–207、382–396：double→float/half | `gipc/type_define.h:14`；`math/eigen_data.h:13–17,125–127`；`MASPreconditioner.cuh:49–54` | 不适用 | A/b/x/r/p/Ap/rho 固定 FP64。legacy MAS 原有 float 内部数据须如实披露并保留，不再降精度；将其强制提升/切换稳定路径也需要单独数值研究 |
| 5:126–135：向量更新与点积融合 | `pcg_solver.cu:59–76`；`pcg_graph_impl.inl:24–54,244–256`；`core/retired_components.h:28–30` | 部分 | x/r 双更新已共用 kernel；只保留 opt-in diagonal update fusion，MAS 不适用。MAS final-dot、SpMV+pAp 已退休，不能再次按“新融合”恢复 |
| 5:39、6:86–115：persistent / 合并全部 PCG 小核 | `pcg_graph_impl.inl:239–273`；`pcg_graph_options.h` | 不适用 | 同一 PCG 迭代有跨整个网格的归约依赖；普通线程块持久循环没有全网格屏障保证。条件 Graph 是既有实现；K4 已有失败性能证据，本计划不重开 |
| 5:41：一个流 SpMV，另一个流预计算下一轮预条件/向量 | `pcg_solver.cu:502–564` | 错误 | `M(r_{k+1})` 依赖 `pᵀAp→alpha→r_{k+1}`，`p_{k+1}` 又依赖新 rho。并非独立工作，直接并发会读旧数据；pipelined CG 是改变算法的另行研究 |
| 5:15、45；6:25：warp 归约取代所有 atomic | `spmv.cu:61–85`；`MASPreconditioner.cu:857–890,1080–1104`；`pcg_solver.cu:43–54,98–105` | 部分 | SpMV 上行 head segment、MAS local action 已 warp 归约；非对角转置 scatter 和跨 cluster 粗层写入仍需全局累加，warp 归约不能直接替代不同 warp/地址的全局冲突 |
| 5:13、43；6:117–139、323–342：SoA / shared-memory 重构 | `spmv.cu:58–64`；`MASPreconditioner.cu:1057–1073`；`MASPreconditioner.cuh:47–54` | 部分 | MAS 已 shared-cache residual，小矩阵及三分量向量已有针对性布局。3×3块操作同时读xyz，单凭 AoS→SoA 名称不能推断净收益；全局节点数据改型成本需算入 |
| 5:11、43：统一强制提高 occupancy / maxrregcount / volatile | `spmv.cu:110–122`；`MASPreconditioner.cu:2358–2370`；`CMakeLists.txt:115–132` | 未实现 | 可测热点寄存器、spill、带宽及活跃 warps，再调 launch。未采集本轮 Nsight Compute，不能断言低 occupancy 是根因；盲目限制寄存器或加 volatile 可能增加 spill，不能默认列为收益 |
| 5:17、19；6:35–65：把 PCG 的标量同步移至设备，Graph 重用 | `pcg_graph_impl.inl:151–179,192–211,237–293` | 已实现 | Graph 内 alpha/beta/stop、CUB dot 均在设备；缓存签名含矩阵地址/尺寸、M地址/模式、向量和workspace。仍有每solve初始/最终读回，不能称为整帧Graph完全覆盖；base/graph 没有 active-host 对照，不能隔离纯Graph因果增益 |
| 6:163–187、344–364：循环内 malloc/free 改预分配 | `cuda_tools/cuda_cub_wrappers.h:22–72`；`pcg_solver.cu:123–131`；`pcg_graph_impl.inl:151–156` | 已实现 | CUB workspace 是 thread_local 持久 buffer，仅容量增长时重分配；PCG 向量/MAS buffers 是容量复用。禁止把 resize 调用数量直接等同于 cudaMalloc 数量。Graph/事件等 lifecycle 成本仍可单独核查 |
| 6:366–380：CPU串行 dot 换 cuBLAS Sdot | `pcg_solver.cu:43–54,85–107`；`pcg_graph_impl.inl:157–163` | 错误 | 当前已是 GPU BlockReduce+DeviceReduce。Sdot 接受float，与double向量不符；示例host result仍会同步。正确对照需Ddot、device scalar、正确stream并包含handle/workspace/graph成本，收益未知 |
| 6:147–152：`float alpha = cublasSdot(...)` | NVIDIA cuBLAS API；当前 `pcg_solver.cu:514–516` | 错误 | cuBLAS 返回 `cublasStatus_t`，结果由末尾指针输出。返回status不能当步长alpha，且 PCG 的alpha是rho/curvature，不是单个 r·r |
| 6:153–160：`cusolverSpScsrlsvqr` 是 cuSPARSE 迭代 PCG 替换，输出iter/res | 当前 BCOO：`global_linear_system.cu:378–435`；官方cuSOLVER API | 错误 | 这是FP32 CSR稀疏QR直接求解器；接口末尾只有singularity，无iter/res两个参数。tol是奇异性判据，不是rho停止阈值。需展开本算子和转换CSR；更换直接求解器不属于保留当前PCG停止语义的执行优化 |
| 6:141–161、367：用 cuSPARSE 加速 mat-vec | `global_linear_system.cu:421–434`；`utils/spmv.cu:25–123` | 未实现 | 当前生产SpMV仍为自写对称BCOO scatter。库链接/handle封装已存在（`CMakeLists.txt:95`；`cuda_linear_system.h:144–181`），不代表活动PCG已调用库SpMV。可作为候选2的对照后端，先核算视图构建与数值更新成本 |
| 5:47：缓存不变求逆/库 batched inverse | `MASPreconditioner.cu:2217–2297` | 不适用 | 同拓扑不等于同Hessian；当前每系统数值填充/逆准备是必要工作。不能跨Newton直接复用旧inverse/factor；仅符号准备可以在精确不变条件下复用 |
| 5:51、53、68；6:161、380：只比位移1e-5 / 迭代数 / residual通过 | `pcg_solver.cu:341–374`；`tools/local/cpu_fixed_reference.py`；`reports/autodl_seven_20261008/REPORT.md` | 部分 | 同系统审计、CPU真残差参考等已有入口，但数值阈值应由冻结协议和基线重复标定；rho不是物理误差证书。独立CCD、材料、真实速度、完整轨迹及统计配对不可由局部残差替代 |
| 6:207：`cuda-memcheck --precision` 检查舍入误差 | 官方 Compute Sanitizer 文档 | 错误 | memory/sync sanitizer发现越界、数据竞争、同步合约错误；不提供该浮点舍入认证选项。舍入/解误差要与独立CPU FP64参考、真残差、能量/轨迹检查比较 |
| 5:93、134、145、159、170–179；6:3–5：按通用经验叠加收益达到2× | 七场景 `TIMINGS.csv`；`MAIN_ANALYSIS.md:79–96` | 错误 | 百分比是猜测或外部特定微基准，不能累计为本工程实测；现有all=1.0753–1.4139×，且本轮缺质量/统计认证 |

## 4. 可以确认的 bug 与修复状态

### L-B1：生产 SpMV 违反 CUB 32-lane collective 前置条件

计时版本 `spmv.cu:35–37` 在 `global_thread_id>=triplet_count` 时直接返回。末尾 warp 若只有 1–31 项仍在 `:73–78` 调用 `cub::WarpReduce<Float,32>::HeadSegmentedReduce`；参与线程数不是 LogicalWarpThreads 的整数倍。同一个 `temp_storage_float[warp_id]` 对 xyz 连续复用且没有同步。CUB 官方分别规定整 logical warp 参与和临时存储复用前同步：[WarpReduce API](https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpReduce.html)、[warp-level temporary storage](https://nvidia.github.io/cccl/unstable/cub/developer/warp_level.html)。这些是可确认的接口合约错误，不等于已证明当前硬件一定出现错误结果。

同工程 `gipc/utils/parallel_algorithm/details/fast_segmental_reduce.inl:35–66` 已使用安全模式：无效lane零值、独立head、满warp归约、复用前 `__syncwarp()`。MAS legacy restriction则在 `MASPreconditioner.cu:807–810` 先 ballot 并在后续使用相应有效mask；不能机械地把所有return都判成同一缺陷。MAS local block是由bank16/padded cluster生成256的整块工作量，不能仅凭其return判为尾warp缺陷。

主任务授权的修复仅修改活动 `spmv.cu`：保留每个存储块的原 `Aij*xj` 与非对角 `Aijᵀ*xi` 作用、FP64、分段与atomic公式；无效lane不读任何矩阵/索引/x，不写y，只参与零值独立head归约；xyz之间加两次 `__syncwarp()`。最终源码相关位置：`spmv.cu:35–36,45–59,68–85`。

初次补丁的 `if(valid)` 位置被交叉复核发现放错，已在GPU运行前纠正；最终完整diff重新核查。所有native/test/CMake文件随后按主任务要求冻结。最终 `spmv.cu` SHA256：`e6ba57b385a42afd6a4f6828b38928dc3d0a80349923eb2d2cac02f98d928f10`；计时版本：`76fa87eaf92a836b086d195a8f2647e3f66eb1c2b7b07167fb0a64884432fa23`。

新增 `tests/sym_spmv_tests.cu` 直接链接生产 `spmv.cu`，不复制测试版CUDA核。覆盖0/1/31/32/33/255/256/257/4097条目、整长行跨warp/block、mixed/空行、非对称3×3块转置、双存储方向、diagonal只作用一次、a=0/.75/-1.25和b=0/.375/-.5、零维向量、输入/输出尾guard。b=0时live初值为NaN，CPU参考显式置0，检查生产清零不依赖旧y。共171case，每case主机提交和3次captured Graph重放，共513replay。CPU参考用独立逐块FP64公式与dyadic输入，检查误差≤1e-12；这是算子级检查，未把该阈值移到整场物理认证。

已做：`git diff --check` 当前通过、源码diff/guard静态复核。未由本分工做：build、GPU、sanitizer、完整PCG/轨迹及修复后速度；主任务串行负责。不得从“合约修好”推出“更快”或“历史质量差距已解释”。

验证命令（占位build目录须由主任务替换为已封存活动构建目录）：

```text
cmake --build <active-build-dir> --target sym_spmv_tests --config Release
ctest --test-dir <active-build-dir> -C Release -R ^sym_spmv_equivalence$ --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 <sym_spmv_tests executable>
compute-sanitizer --tool synccheck --error-exitcode 1 <sym_spmv_tests executable>
compute-sanitizer --tool racecheck --error-exitcode 1 <sym_spmv_tests executable>
```

报告6中的Sdot/QR参数错误属于待审报告的错误代码，不是活动工程已经执行这些错误调用；二者不混为一类。

## 5. 七场景的2×成本预算

使用 `solver_seconds`，不使用带不对称导出/加载成本的 `wall_seconds`。all为conditional Graph+FullCCD refit+batched energy+energy reuse+ordinary BVH refit，legacy MAS、rho=1e-4、dt=.01、legacy `.01`/min6、完整CCD。`linear_seconds` 是 `calculateMovingDirection` 期间的原生阶段事件包络，包含build/precondition/PCG/distribution；GPU事件区间也包含host同步造成的设备间隙。阶段覆盖与总CPU包络不完全相同，下表是工作量与其它阶段固定时的预算估算，不是严格的Amdahl证明或可保证收益。

`需省秒 = T_all - T_base/2`；`需削线性比例 = 需省秒/L_all`。

| 场景 | base 秒 | all 秒 | base/all | all线性秒 | 线性/all | 达2×需省秒 | 若仅削线性，需要削掉 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cloth_hang_l | 2.2876 | 1.6180 | 1.4139× | .7240 | 44.75% | .4742 | 65.49% |
| cloth_hang_m | 4.7361 | 3.5797 | 1.3230× | 1.9344 | 54.04% | 1.2117 | 62.64% |
| cloth_sphere7_l | 5.2871 | 4.3459 | 1.2166× | 1.5497 | 35.66% | 1.7023 | 109.85% |
| cloth_sphere7_m | 8.3087 | 6.6191 | 1.2553× | 2.9646 | 44.79% | 2.4647 | 83.14% |
| cloth_fixed_bunny_l | 6.5606 | 4.8759 | 1.3455× | 2.4188 | 49.61% | 1.5956 | 65.97% |
| cloth_fixed_bunny_m | 12.0072 | 9.6854 | 1.2397× | 5.8557 | 60.46% | 3.6818 | 62.88% |
| bunny_cloth_bunny_l | 10.6731 | 9.9258 | 1.0753× | 3.4616 | 34.87% | 4.5893 | 132.58% |

即使能将线性阶段减半，现有预算也不支持多数场景直接达2×；落球L与混合兔子还必须降低非线性/碰撞/线搜索成本。不能以一项局部SpMV达到2×就宣布原版Stiff整场2×。

既有准备成本支持把范围扩到完整linear workflow：

| 2026-10-07诊断场景 | matrix convert CPU inclusive | MAS hierarchy CPU inclusive | 两个不重叠scope之和 / 物理CPU范围 |
|---|---:|---:|---:|
| sphere | 520.189ms，5.475% | 427.932ms，4.504% | 948.121ms，9.979% |
| fixed bunny | 918.645ms，5.457% | 656.117ms，3.897% | 1574.762ms，9.354% |

它们可能含等待前序GPU工作的时间，不能将全scope都称作可消除的符号计算。`graph.initial_readback`为462.576/1142.589ms、`graph.final_readback`为1213.031/4229.898ms；最终读回多数是等待整次PCG完成，不是80-byte传输本身的成本，删读回并不自动省掉这些数值。

SpMV证据较弱：sphere第49帧捕获SpMV+clear/scale=10.549ms，占GPU union约4.04%、诊断帧CPU wall约2.26%；fixed第65帧是Graph整体跟踪，内部SpMV未知。Graph捕获期间记录的`mas.apply`或`pcg.reduce_device` CPU提交成本不能乘上迭代数充当真实GPU成本。所有候选须补充有代表性的各阶段成本，不能从全线性占比推导某个小核热点。

## 6. 最多两个有价值的后续候选

### C1：精确符号结构复用，覆盖 conversion → 子系统索引 → MAS hierarchy

目标是去掉同一结构被逐系统重复准备的工作；当前无接触MAS复用是已有功能，先单独核查其收益，不能作为新发明。新候选把符号与数值流程显式分开，并扩展到新鲜完整接触检测之后的精确拓扑相等命中。

实现边界：

- `GlobalLinearSystem::build_linear_system/convert_new`、`Converter`：缓存原始有序 `(row,col)` 输入长度/索引身份、radix稳定排列、分段partition、compact rows/cols与子系统块索引。命中时仍 gather 当前FP64数值并执行相同数值归约。raw输入数组会被compact输出覆写，缓存必须使用私有持久storage，不能别名到当次共享scratch。
- `MASPreconditioner::ReorderRealtime/setPreconditioner_bcoo`：在每次完整碰撞检测已得到当前pairs后，比较完整有序pairs与owner/topology/partition/尺寸/层数身份；相等才复用层次。hash只用于快速拒绝，不能只依赖可能碰撞的hash通过等价门控。先采用严格有序相等；无序规范化必须另证不会改变hierarchy或数值归约顺序。
- 每次继续重算当前 RHS、Hessian numeric fill、numeric aggregation、inverse/factor 与PCG。禁止复用旧Hessian/M数值；禁止基于“小位移”“接触数未变”或“近似连通相同”判命中。
- 任何索引/owner/尺寸/输入拓扑变化都回退现路径；版本、cache地址/容量、实际活跃维度进入Graph失效判断。不改材质、PCG/newton停止、CCD；不保存接触对以替代下次检测，因此不属于接触池。

性能启动门槛：先做有界from-zero成本记录，量化输入结构命中率、比较/维护/内存成本及可省的critical-path时间。需要证明 `实测可省准备成本 - 结构比较/维护/新增数值搬运成本 >= 整场5%`；或对应热点≥10%并有可信的减半路线。现有9.35–9.98% CPU范围只支持候选价值调查，不等于满足净门槛。

正确性门槛：固定状态上新旧compact结构、作用`Av`、MAS持久算子身份、`M⁻¹v`、完整PCG停序和x、真残差、breakdown与恢复检查；同时覆盖无接触/新接触/接触顺序变化/碰撞类型变化/容量增长/ABD+FEM offsets。随后从零完整轨迹，冻结协议验材料、真实速度、独立接受路径CCD与hard numeric checks。

性能推广门槛：至少3个交错完整配对供有限筛选，再按主任务冻结的统计方案重复验证；必须算全部prepare+PCG+distribution与CPU/GPU包络，不只cache hit kernel。组件中位须≥1.05×、跨目标场景无显著回退；低命中或净收益不足即停止这一分支，不展开参数网格。最终2×继续用原版Stiff、修复后的active-host、Graph、all及candidate-only消融核对。

### C2：完整对称算子视图准备 → FP64 row-gather SpMV

这项尚未实现；2026-10-07曾因启动证据不足跳过，不恢复已退休的SpMV+pAp融合。先在大M场景、混合ABD/FEM以及接触/无接触窗口补足operator成本；若仍仅约4%总工作，不实施该重构。

数据流：对每个当前存储块生成row-adjacency引用，diagonal一条，off-diagonal两条并显式transpose标记。数值保持来自原FP64 block storage，不重新物理组装、不把下三角存储条目过滤掉。现算子对每一个stored offdiagonal都加其转置；若两种方向都stored，也都必须贡献。新路径每行gather、归约后直接写`aAx+by`，包含空行和b=0，无全局scatter atomic。保留legacy作为参考；完整视图生命周期及Graph地址/后端/尺寸签名都要管理。

可在同一算子后端实验中用FP64 cuSPARSE作对照，但必须先构建该算子的完整CSR/BSR、正确处理所有存储方向/重复块并计入转换、值更新、descriptor及workspace成本；不替换为QR求解器。单独cuBLAS Ddot替换CUB点积缺少热点证据，不另列第三候选。

性能门槛：目标整体SpMV+清零/scale在代表性完整运行中≥10%关键路径，且预计`保存迭代成本 - 每系统视图准备/更新/失效成本`≥5%整场。固定系统测`T_prepare + N_iterations*T_spmv`，不能仅重复旧图内已准备SpMV。operator-only中位≥1.10×供初筛；整场最终≥1.05×才考虑保留。若视图/CSR准备抵消收益或真实热点不足，按有限决定停止。

正确性门槛：本次生产SpMV fixture是第一道；另覆盖block排序/多个存储方向、空行、跨warp長行、一般a/b、矩阵尺寸/容量变化、完整host/Graph PCG停序与真残差，再沿用C1相同完整物理协议。减少atomic会改变FP64相加顺序，局部误差可用冻结容差比较，不能据此放宽整场质量范围。

## 7. 库建议的精确纠正

cuBLAS Ddot接受double并把标量写入result指针；其返回是status。host result pointer模式会等GPU结果读回，device pointer模式才允许异步标量流水：[cuBLAS dot](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-dot)、[scalar pointer modes](https://docs.nvidia.com/cuda/cublas/index.html#scalar-parameters)。因此“换cuBLAS就消除同步”是不完整的。当前Graph已经device dot，库替换要比较实际kernel、初始化/缓存及Graph兼容性。

`cusolverSp<t>csrlsvqr`是稀疏QR直接求解，必须提供完整CSR；输出只有singularity，tol判奇异性，且当前官方已将此API标为deprecated：[cuSOLVER csrlsvqr](https://docs.nvidia.com/cuda/cusolver/index.html#cusolversp-t-csrlsvqr-deprecated)。本轮不以它替换PCG，也不把更换求解算法的结果认证为同停止规则执行优化。

内存/同步检查用[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)的memcheck/synccheck/racecheck；浮点质量另用独立FP64参考与物理协议。不存在报告6所暗示的通用 `cuda-memcheck --precision` 舍入认证。

## 8. 本分工交付与未完成验证

交付：本审查；授权的SpMV正确性源码补丁；生产SpMV测试；独立CMake/CTest target。未修改baseline、材料、rho、Newton停止、完整CCD，未恢复任何退休候选；未提交Git；没有GPU执行或新性能结果。

源码已冻结给主任务全量构建与串行GPU验证。以上候选均为条件化计划，所有收益待实测；达到或超过2×仍必须同时通过完整同质量和受控统计性能验收。
