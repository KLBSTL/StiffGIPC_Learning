# K1 / K4 同进程固定系统重放设计

日期：2026-10-06。性质：只读源码审查与实施建议；本文件没有对应的新编译、GPU 运行或提速结论。采用 benchmark-regression 的固定身份、有限预算和诊断与正式计时分离规则。

## 1. 最小入口与冻结范围

建议在 `PCGSolver::fixed_system_study` 的帧/方向筛选及完整快照之后，增加一个独立、默认关闭、与其他 study 互斥的 **K replay 早返回分支**。复用 `pcg_graph`，不要递归调用 `solve()`，也不要执行旧 study 后面的算子微基准、严格容差或因子候选循环。

具体复用点：

| 源码位置 | 已有能力 / 边界 |
|---|---|
| `StiffGIPC/linear_system/solver/pcg_solver.cu:363` | 主 PCG 已求出当前 direction 后调用 study；调用者尚未分发这个 direction，可在此冻结本次系统。 |
| `.../solver/pcg_fixed_study.inl:5–28` | `total_Frames+1` 为物理帧；`newton.size()` 为一基 direction；已有帧/方向选择、主解保存、系统导出、solve context。 |
| `.../solver/pcg_fixed_study.inl:30–61` | MAS stage 早分支可作为结构范例，但不是完整 PCG 隔离实现。 |
| `.../solver/pcg_fixed_study.inl:84–110` | 旧路径有每算子 128 次重复，必须由新分支提前返回避开。 |
| `.../solver/pcg_graph_impl.inl:63–198` | 真正完整的 conditional Graph solve，包含初始化 M、rho、首次读回、捕获或重放、末次读回。 |
| `.../linear_system/i_linear_system_solver.h:12–22,76` | solver 不能复制，系统绑定为 GlobalLinearSystem 私有 friend 接口；不能仅在 study 构造两个未绑定 PCGSolver 就调用。 |

选择 `cloth_fixed_bunny_l`，物理帧 **2、57，各 direction 1**，从零连续推进 59 帧；每个被选系统在本进程内直接复用当前 A/b/M，不加载 checkpoint，不重装配或重新准备 MAS。生产路径保持 K1，study 的 K4 只作用于私有求解状态。也可将两帧分别置于两个有限进程；后者仍须分别从零推进，不能把 f2 快照当作 f57。

f2 是已有短系统入口，旧 legacy MAS 样本为 17,340 DOF、默认 6 次 PCG；这是历史事实而非新程序保证。f57 是接触窗入口，记录实际迭代数、接触数量和系统哈希；若本次系统并不长，标记“未覆盖长系统”，不临时换方向或扩大搜索。

冻结条件：IPC、dt=.01、legacy cumulative=.01/minimum updates=6、rho=1e-4、相同 max_iter，MAS legacy（Cholesky/wide/inverse64 均关闭，原 atomic restriction），`diagnostic_fixed_iterations=0`、fused-diagonal 关闭；仅 K=1/4 有差别。不得在这项执行实验中提高 rho 精度或强制固定迭代数。

## 2. A/b/M 是完整系统，M 的 scratch 仍须恢复

`global_linear_system.cu:160–219` 导出最终 global triplet、整个 `m_b`、全局/ABD/MAS 预条件数据。必须显式 `GIPC_MAS_SNAPSHOT=1`，检查 `preconditioner_export_complete=true`；仅有地址 signature 不足以声称 M 完整导出。

矩阵 ABI：int32 block row/col、逐块 column-major FP64 3×3，**每个实际存储非对角块均增加转置，对角原样**，不能因 meta 写“upper”而筛除 lower 项。最终 A 已含 ABD；不能拿 ABD 逆块补造 A。已有 `reports/local_step2/CPU_REFERENCE.md` 对该布局和完整 RHS 做过独立 CPU 复核。

`MASPreconditioner.cuh:75–99` 的 `diagnostic_buffers` 已列出所有 owned buffers；`d_multiLevelR/Z/R64/Z64` 是会被应用改写的 scratch。新 study 必须备份并逐字节恢复它们，检查全部 A/b/M before/after 哈希一致；不能沿旧通用 study 的 `snapshot_operator_identity` 将四份 scratch 排除后便称“完全恢复”。现有 visitor 是只读导出接口，恢复需极小、明确的专用写回入口，不能依赖未经检查的 const_cast 或对象内存布局。

将 source/compiler/exe/build seal、requested/resolved、frame/direction/linear-system context、A/b/M 字节数及哈希写入独立 JSON。保存 14 条正式 repeat 的解文件，关联 pair/order/K/iterations；已有 CPU loader 的矩阵部分可复用，解索引另适配。f2 旧默认 PCG 真残差约 **4.1984e-2**，并非 1e-8；CPU 参考满足 1e-8 不能作为生产 rho 停止解也满足 1e-8 的证据。

## 3. 私有工作区与两个 Graph cache

当前 `pcg_graph_impl.inl:113–178` 只有一份 cache；K 加入 key 后，直接在同一 cache 上 K1/K4 交替会每次失效重捕获。最低要求是 study 持有 **两个私有 arm cache**，工作向量地址在本系统内不再改变；production graph/exec 必须保持存活、暂不使用，并在 study 结束原样交还。

建议在同一已绑定 solver 上用一个有界 RAII 诊断状态槽：保存生产 owner，安装私有 x/r/z/p/Ap、reduction_result、graph_scalars 和 CUB storage；两臂可以共享地址稳定的私有工作区，仅切换独立 GraphState（graph、exec、key、max_iter、tol、capture/hit/invalidation counters、K/ABI）。不要释放生产 owner 后重新分配同尺寸内存来冒充恢复。若采用 move/swap，先确认 DeviceBuffer 的所有权移动行为；DenseVector 本身没有公开交换生产缓存的 API，需实施时明确封装。

成功或异常退出均必须：

1. 等待私有 GPU 操作结束，销毁两臂临时 graph/exec。
2. 恢复原工作 buffer owner/地址/容量/内容、全部 scalar（包括新增 active）和 CUB storage；恢复 MAS scratch。
3. 恢复生产 graph/exec/key/max_iter/tol/counters、config、diagnostic flags、原 `pcg` info 与任何临时 cost/solve context。
4. 原 direction `x`、只读 A/b/M、MAS scratch、生产地址签名与 counters 均通过逐位/逐字段核验，只向 info 追加 study 文件链接。

`pcg_fixed_study.inl:225–236` 目前只强制只读 operator 与主 x 不变，**没有**上述完整恢复合同；现成旧 study 不能直接宣称满足本实验。`pcg_graph` 会写当前 `Statistics...pcg`；每次先清临时 info，采样后复制到独立结果，最终恢复原 info，避免 `zero_residual`、cache 字段和累计统计串臂。

私有 GPU 缓冲的预分配、M scratch 备份与导出计入诊断资源预算；若显存不足立即退出，不扩大原预算。A/M 本体共享只读，不能为方便再造两份大矩阵或因子。

## 4. 有限顺序与计时定义

每系统固定 **2 对 warmup + 7 对 measurement**，即每臂 warmup 2 次、正式 7 次，共 18 次完整 solve。warmup 次序 K1,K4,K4,K1；正式 pair 奇数 K1→K4、偶数 K4→K1。每次 xwork=0，且恢复到声明的同一 MAS scratch 初始快照后再开始；恢复和输出导出均在计时外，单独记开销。不要增加 host/strict/强制迭代网格。

首个 warmup 各臂允许捕获；第二 warmup 与所有正式 solve 必须 cache hit。若真实指针/容量变化导致捕获，保留该记录并判稳态性能比较不可用，不排除慢样本再重试。短/长系统各建自己的臂 cache，不跨不同 A/M 地址复用。

| 字段建议 | 范围与解释 |
|---|---|
| `solve_host_ms` | 进入 `pcg_graph` 至同步末次读回返回，含初始化、签名、捕获（若有）、重放等待和读回；不含 x0/scratch 复位及质量导出。 |
| `solve_event_ms` | 同一 cudaStreamPerThread 上，从 PCG 初始化前至完整 solve 后的一对事件；包含 GPU 流空档，**不是 kernel active 时间之和**。只在 warm cache 上作为正式结果。 |
| `initialization_event_ms` | r=b、M 初次作用、rho/p 初始化到 initial readback 前；明确是 PCG 初始化，不是 Hessian/矩阵转换/M 因子准备。 |
| `replay_event_ms` | CUDA Graph launch 前后同流事件；end 在同步读回前排入，含完整 conditional loop。 |
| `capture_instantiate_host_ms` | 复用已有 `graph_capture_instantiate_host_ms`；只在 cold warmup 记录，不摊到稳态 7 对也不隐去生产真实 capture。 |
| `initial_readback_host_ms` / `final_readback_host_ms` | CPU 阻塞区间；final 包含等待 Graph，不可与 replay_event 相加当作总成本。 |
| `private_setup/reset/export_restore_host_ms` | 独立诊断成本，不能冒充生产求解开销或忽略资源消耗。 |

事件只用于私有 study，不能在 conditional body 内植入新的事件节点。已有 `CostScope("graph.replay")` 是 CPU launch 包络，不是 Graph GPU 完成时间。原 A 转换与 M 准备已在进入 study 前发生一次，报告为“冻结共同前置成本，未由 replay 重新测量”，不能写成零成本，也不能以 solve 微收益直接宣称整场收益。

按各 pair 的 K1/K4 比值报告中位数与全部 7 对；保存实际迭代数、rho_initial/rho_stop、breakdown/iteration_limit、相对残差、解差异、cache hit 与新增 active 最终值。残差 SpMV、CPU 拷贝、哈希和比较都在 timed solve 之后。若迭代工作量出现系统性差异，先处理停止语义/数值问题，不将减少迭代计为调度收益。legacy 原子路径自身可能非逐位重复；原解差异、K1 自身重复范围及独立 CPU 残差须同时呈现，不事后扩大门槛。

## 5. 10 scalar → active 的具体检查点

当前 `pcg_graph_impl.inl`：s0 rho、s1 pAp、s2 new-rho、s3 alpha、s4 beta、s5 converged、s6 iterations、s7 initial-rho、s8 previous-rho、s9 error。固定长度触点包括 `:80 resize(10)`、`:100 initial[10]`、`:184 report[10]`；`pcg_guard_fixture.inl` 也分配 10 并用 host s[10]。增加 active 后必须统一常量/ABI、读回字节数、快照长度与 fixture 初始化，Graph key 加 K、scalar ABI、额外 buffer 地址/尺寸。只加 K 而遗漏 scalar 容量/新地址不能保证缓存有效。

每次 solve 的 graph_init 都必须重置 active，包括复用上一轮已经停止的 cache；零 RHS 早返回不得保留上次 active。K4 组内第 1/2/3 步停机后，后续子步不得继续更新 x/r、计数、rho/previous-rho 或错误码。当前 `graph_p_continue:40–55` 会无条件改 p，并由线程 0 增计数；不能仅把 conditional handle 更新挪到第 4 步而执行四次真实迭代。

`graph_alpha:16` 先保存 previous-rho；`graph_beta:30` 在检查新 rho 之前，先处理 previous-rho 已收敛的情况。新 active 不能把这一轮原本正常结束变成下一次 M 的负 rho 错误；`max_iter-1` 和迭代返回值也须完全保留。被屏蔽阶段若依然执行 SpMV/M，它们可能只写 scratch，但不能污染有效 scalar/错误状态；这类尾部额外 GPU 工作必须体现在完整 solve 计时中，不按“有效迭代”删掉。

原 generic guard 只是 scalar kernel/host 分类器，不是完整 A/b/M K4 重放。进入上述两系统比较前，应由主实现的有限 fixture 覆盖完整零 RHS、在组内 1/2/3/4 位置停止、上限余数、坏 rho/pAp 和“previous-rho 已停止而 new-rho 异常”；历史六 M fixture 仅验证 M，不能代替完整 PCG 停序验证。缓冲增长与 cache invalidation 若本批没有实际触发，要列为未覆盖，不能由地址 key 设计推断已测试通过。

## 6. 执行入口建议与停止条件

沿 `tools/local/fixture_windows.py:85–96` 的 fixed-cloth legacy 环境构造新独立诊断命令，保留所有输入与 resolved 校验，并选 `GIPC_FIXED_STUDY_FRAMES=2,57`、`GIPC_FIXED_STUDY_DIRECTIONS=1`、`GIPC_MAS_SNAPSHOT=1`、新的单一 K replay selector。旧 `GIPC_FIXED_STUDY_COMPACT=1` 本身不会跳过算子微基准，不能作为 bounded replay 开关。此文不假造尚不存在的 CLI/ENV 名称。

外部绝对 120 秒、原显存/磁盘/外来 GPU 与串行 lock 保护不变；到期、分配失败或任一恢复失败立即终止，保留部分记录，不续跑补齐。完成两个固定系统只支持调度机制判断；随后仍需无诊断、从零、相同配置的有限整场配对与质量检查。K4 是否更快和是否适合默认开启目前均待验证，固定系统收益不代表 2× 整场目标已达成。

本次核验：只读上述源码、已有 fixture 配置与 CPU_REFERENCE；未运行编译、CPU 求解或 GPU，未修改 native/tool/protocol。
