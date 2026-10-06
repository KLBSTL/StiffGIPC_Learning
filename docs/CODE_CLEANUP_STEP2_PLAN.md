# 第二批清理计划：封存的线性执行候选

日期：2026-10-06。审查基线：原生 `925aff1`，公共测试工具 `edaf238`。
**这是只读审查后的实施清单，尚未实施；当前 AutoDL 编译及候选池实验期间不得修改原生源码。**
执行须等待当前实验完成、证据分析归档及主任务的实施信号。本文不改变实验预算，也不提出新的性能候选。

## 1. 决定与证据

第二批删除三个已经封存的活动实现，以及它们专属的选项、设备工作区、Graph 签名字段和诊断研究入口。保留历史报告和清理前 Git 版本：`archive/pre-cleanup-20261006`，备份提交 `eaa1b2131bca06de41531c66ed64e74dd4316da0`。

| 组件 | 已有证据 | 本轮决定 |
|---|---|---|
| MAS final-dot | `history/reports/active/IPC_COMPONENT_TUNING_RESULTS_20261006.md` §3：两布料相对旧执行组合 1.002× / 1.021×，几何平均 1.011×，未到预声明门槛 | 删除活动候选，不重开调参/微基准 |
| SRBK SpMV+pAp | 同报告 §3：0.983× / 0.974×，几何平均 0.978×；42 组短窗工作量一致 | 删除活动候选，保留普通 SpMV 与普通点积 |
| legacy ordered restriction | `history/reports/active/IPC_LEGACY_ORDERED_RESULTS_20261005.md` §1、§4：两个完成系统数值检查通过，计入准备后约慢 14%–19%，其余两系统受资源门槛阻止 | 删除 legacy 有序分支及专属研究，恢复当前 atomic 默认直通 |

这些是有限窗口未获得收益的结论，不是数学方法失效或所有硬件上的永久否定。用户要求压缩活动实现，因此从可关闭候选升级为明确不可用；旧代码由 Git 归档承担。专属文件总量约 55 KB，主要价值是减少复杂度，不把此项描述为大容量磁盘回收。

必须保留：legacy 默认 MAS、稳定 Cholesky MAS、`mas_restrict=serial|warp`、`mas_factor_action=triangular|factor_inverse`、已选择的 `factor_inverse/world_block` 默认、conditional PCG、PCG 守卫、普通 SpMV、完整 CCD、普通/swept 独立 BVH 缓存、contact pool、成本追踪、通用固定系统快照及六份历史输入。`mas_restrict=warp` 与本次删除的 `legacy_restrict=ordered` 是不同实现，不能混删。材料、停止条件及 AL/compensated 控制器不在本次清理范围。

## 2. 删除的专属文件

以下路径均相对 `StiffGIPC/`。删除采用精确文本补丁；不匹配目录通配符，不改 `baseline/` 或原实验工作区。

| 分组 | 文件 |
|---|---|
| MAS final-dot | `solver/mas_fused_dot_options.h`；`solver/mas_fused_dot_kernels.cuh`；`linear_system/solver/pcg_mas_dot_study.inl` |
| SpMV+pAp | `solver/spmv_quadratic_options.h`；`linear_system/utils/spmv_quadratic_kernel.cuh`；`linear_system/utils/spmv_quadratic_reference.h`；`linear_system/utils/spmv_quadratic_cpu_check.py`；`linear_system/solver/pcg_spmv_quadratic.inl`；`linear_system/solver/pcg_spmv_quadratic_study.inl` |
| legacy ordered | `solver/legacy_restrict_options.h`；`solver/legacy_ordered_restrict.cuh`；`linear_system/solver/pcg_legacy_restrict_study.inl` |

保留专属 `.md` 说明，在开头加“活动实现已删除，复现实验请使用归档版本”的标识；不重写报告中的旧实测结论。无独立 `.cu` 编译单元被删除，预计仍是 37 个 TU，但受影响头文件依赖必须重编，不能只重编表面修改的 TU。

## 3. 共享源码最小补丁

行号来自 `edaf238`；实施时用函数名定位并检查与当前未提交修改是否重叠。

### 3.1 MAS final-dot

1. `linear_system/solver/pcg_solver.h:47–59`：删除 requested/effective/supported、reason、partial count、两份设备缓冲、CUB 字节数和四个候选方法。保留相邻 `release_graph()`。
2. `pcg_solver.cu:118–168`：删除 `prepare_mas_dot` / `apply_mas_dot` / `apply_mas_dot_host`。`:190` 不再准备候选；在公共 PCG info 初始化处写下文的退休兼容元数据。
3. host PCG 的初始 rho 和迭代 rho 都恢复为当前候选关闭分支：先完整 `apply_preconditioner(z,r)`，再原 `My_PCG_General_v_v_Reduction_Algorithm`。不改其 scratch 参数、返回值、`rho0`、previous-rho 停止顺序或 breakdown 判断。初始零 RHS 路径同样保持。
4. `linear_system/linear_system/global_linear_system.cu:25–106`：删除匿名 `mas_dot_complement` 和四个 fusion/diagnostic 转发方法；同步删除 `.h:99–103` 声明。保留普通 `apply_preconditioner` 的全局/ABD/局部写入顺序；不将 mixed ABD 贡献改成仅 FEM 点积。
5. `i_linear_system_solver.h:62–66` / `.cu:31–39`：删除专属转发声明和定义，保留普通 apply、SpMV、fused diagonal update、snapshot 和 Graph 接口。
6. `preconditioner/fem_mas_preconditioner.h:27–31` / `.cu:8–29`：删除四个 fusion/partial scratch 包装。保留 `apply`、`diagnostic_snapshot`、`diagnostic_buffers` 和普通 Graph 签名。
7. `solver/MASPreconditioner.cu:2408–2425,2562–2577`：删除 collect 支持查询和 `collect_fused_dot`。`CollectFinalZ` 恢复单个 `Z` 参数，`preconditioning` 恢复 `(R,Z)`；删除 Cholesky/wide/legacy 的 dot 分支，保持关闭分支的原 collect kernels、参数和发射形状。
8. `MASPreconditioner.cuh:76,170–176` 同步精简声明。仅删除 stage probe 中“不能与 fused rho 同用”的死检查；通用受保护 stage probe 保留。

### 3.2 SpMV+pAp

1. `pcg_solver.h:61–73` 和专属 `.inl` 删除候选状态、partial/CUB 缓冲及方法。`pcg_solver.cu:191` 不再准备候选。
2. host PCG 的每次曲率计算恢复当前关闭分支：恰好一次普通 `spmv(p,Ap)`，随后原 `dot(p,Ap)`。避免在删除条件后留下双重 SpMV 或漏掉清零。
3. `global_linear_system.cu:522–547` / `.h:116–120` 及 `i_linear_system_solver` 对应声明/定义删除 quadratic 方法；普通 `spmv` 原样保留。
4. `linear_system/utils/spmv.cu:125–140` / `.h:12–16` 删除候选方法及专属 include。普通 kernel、`a/b` 缩放、输出清零、lower/upper 每个已存块的对称扩展规则和原子累加不变。
5. 删除 `pcg_solver.cu` 尾部三个专属 MAS/SpMV inl includes。普通 BlockReduce、CUB 和 Graph reduction scratch 仍使用，不能按“候选也用了它”删掉公共依赖。

### 3.3 conditional Graph

`linear_system/solver/pcg_graph_impl.inl` 只做常量关闭分支化简：

- `:79` 的 `fused_update` 去掉 `&& !mas_dot_effective`，保留原 diagonal update 条件及 diagnostic override。这等价于当前两个候选关闭时的行为。
- `:95–101` 初始预条件与点积保持普通顺序；Graph init、零 rho 判断和首次 host readback 不变。
- `:128–137` 从 `captured_key` 删除两候选共 10 个值：各自 mode、partial count、CUB 字节数、partial 地址、reduce 地址。
- `:170–175` 恢复普通 SpMV 后点积；`:176–186` 保留 alpha、dx/r 与可用 diagonal update 分支，再普通预条件与 rho 点积。
- 完整保留 `graph_beta`、previous-rho continue、max_iter、tol、异常标记、最终 readback、capture/cache counters 和设备号。
- 保留 A/映射/预条件器签名、x/r/z/p/Ap 地址、公共 reduction storage/scalar 地址及大小、fused_diag_update 与 fixed_iterations 键。删除对象不能导致保留缓冲的签名漏项。

缓存不跨进程/可执行文件序列化，新程序不应接管旧 Graph handle。记录新的源/对象/链接/二进制身份。

### 3.4 legacy ordered 与共享稳定映射

1. `MASPreconditioner.cu:2305–2308` 删除仅 legacy ordered/study 触发的映射准备与互斥检查。
2. `BuildMultiLevelR:2333–2342` 删除 ordered 两 kernel 分支，原 `#ifdef GROUP` / 非 GROUP atomic 路径原样保留。
3. `preconditioning:2657` 将 `if(!legacy_ordered_restrict()) memset` 恢复无条件的原 coarse-R 清零，保留长度、类型和 stream。删除所有 `LegacyRestrictionArm`/thread-local override 专属实现。
4. `MASPreconditioner.cuh:205` 只删最后一个 ordered mode 签名值；**保留** `d_restriction_starts/nodes` 的所有权、分配/释放、快照及 Graph 地址。它们由 `prepare_cholesky:2186` 使用，并由 `:2593–2596` 稳定 serial/warp 限制读取。
5. `prepare_restriction_map(bool measure):2189–2222` 保留 CSR 构造、排序、读回/上传及现有 CostScope。去掉只供 ordered study 的 `measure` 参数、显式同步、wall 起止时间、`restriction_map_prepare_ms` 成员及 getter，改为无参；稳定路径此前传入默认 false，正常 kernel/数据顺序不变。保留稳定映射计时 `mas.restriction_map_*`。
6. `fem_mas_preconditioner.cu:37` 不再输出专属 `restriction_map_prepare_ms` 测量字段；新 schema 中缺失表示未测，不写假的 0 ms。历史 JSON 中该字段仍可读取。
7. `global_linear_system.cu:329–338` 删除 `measure_legacy` 同步计时包裹，只保留原 `build_linear_system()` 及前后 linear_stage。确认 `<chrono>` / `<cub/block/block_reduce.cuh>` 无其他使用后再移除该 TU 的 include。

### 3.5 专属 study 路由

`pcg_fixed_study.inl:22–25,34–38` 删除 MAS-dot、SpMV-quadratic、legacy-ordered 三个 early dispatch。保留通用 host/Graph frozen-system study、restriction/factor 研究、stage-only 探针、A/b/M 身份与状态恢复检查。旧研究开关必须在 main 入口明确拒绝，不能删 dispatch 后静默落入通用研究。

## 4. 原生拒绝与历史配置兼容

在既有 `core/retired_components.h::reject_retired_components()` 增加如下检查。该函数已经是 `main` 的第一个操作，早于 fixture dispatch、场景加载和 GPU 初始化，不依赖 Python runner：

| 环境变量 | 当前程序允许 | 拒绝规则 |
|---|---|---|
| `GIPC_MAS_FUSED_DOT`、`GIPC_FIXED_MAS_DOT_STUDY` | 未设置或严格 `0` | `1` 报组件已退休；其他字符串报严格布尔错误 |
| `GIPC_SPMV_FUSED_QUADRATIC`、`GIPC_FIXED_SPMV_QUADRATIC_STUDY` | 未设置或严格 `0` | 同上 |
| `GIPC_FIXED_LEGACY_RESTRICT_STUDY` | 未设置或严格 `0` | 同上 |
| `GIPC_LEGACY_RESTRICT` | 未设置或严格 `atomic` | `ordered` 报已退休；包括空串在内其他值报非法枚举 |

不新增不存在的旧 fixture 变量别名。`GIPC_MAS_REPLAY_FIXTURE`、`GIPC_MAS_CHOLESKY_FIXTURE`、`GIPC_PCG_GUARD_FIXTURE` 和 contact-pool fixture 保留。

`toi_options.h` 去掉三个 options include。继续输出 `legacy_restrict="atomic"`、三个 fixed-study=false、两 requested=false，并新增对应 `*_available=false`。每个实际 PCG info 继续提供两组 `requested=false/effective=false/fallback_reason="retired_component"/partial_count=0`；available=false 可同时提供。这里 0 是该次实际没有候选 partial，不伪造研究/耗时记录。这兼容现有 `tools/bench/validate_run.py:77–83` 的关闭状态契约。

公共工具处理与 native 变更分开，必须在实施后共同冻结：

1. `tools/bench/config.py::expand`、`matches_requested` 保留历史键、枚举及旧省略字段兼容，不能在解析阶段拒绝历史 true/ordered 记录，不能重算覆盖旧请求哈希。
2. 新增单独执行能力检查，在 `linux_runner.py::execute` 创建运行目录/读取 GPU 之前调用；拒绝五个 true 布尔及 ordered。也可覆盖第一批已退休组件。**当前 edaf238 没有这个 Python 能力检查**；当前固定 plans 均关闭，原生入口已可阻止第一批请求，因此不是正在执行场景的误用证据。
3. 所有未来执行入口都调用同一能力检查；仅浏览、分析或读取 JSON 不调用它。不要把解析、导出 env 与“当前二进制可执行”混为一个功能。
4. CPU 合同检查：历史缺省/显式 true/ordered 的展开与 `matches_requested` 仍正确；同一 true/ordered 配置执行前失败；当前默认和三个历史 study=false 可执行；坏 bool/enum 仍失败。历史原始结果文件只读。

## 5. 固定系统保留与验证分层

### 5.1 六份历史输入的身份与现有入口边界

当前新仓库保留 native replay 实现，但未找到六份二进制输入或 portable `fixtures.py`。不得在全量清理时按旧下载缓存删除这些数据。先把以下前缀登记为明确保留的只读测试资产，列所有文件相对路径、大小和 SHA256；可独立打包/迁移必要输入，不迁回历史源码树。

以下路径相对原工作区 `stiff_toi_cudagraph_20260929/`：

| 名称 | 输入前缀 |
|---|---|
| smoke | `downloads/autodl_perf_v37_20261003/runs/autodl/autodl_perf_v37_mas_smoke/mas_audit_initial` |
| default | `downloads/autodl_perf_v39_20261003/runs/autodl/autodl_perf_v39_wide_bunny/mas_audit_failure` |
| strict | `downloads/autodl_perf_v39_20261003/runs/autodl/autodl_perf_v39_wide_bunny_strict/mas_audit_failure` |
| v41_graph | `downloads/autodl_perf_v41_20261003/runs/autodl/autodl_perf_v41_double_bunny/mas_audit_failure` |
| v41_host | `downloads/autodl_perf_v41_20261003/runs/autodl/autodl_perf_v41_double_bunny_host/mas_audit_failure` |
| v41_wide | `downloads/autodl_perf_v41_20261003/runs/autodl/autodl_perf_v41_wide_guard/mas_audit_failure` |

**不能扩大现有测试的表述：**`mas_cholesky_fixture.inl` 读取 MAS 因子/映射和 10 个 probe，测试 M 作用、host/Graph 重放、地址增长、zero RHS 与坏 pivot。它没有装入历史全局 A/b 执行生产 PCGSolver。`pcg_fixed_study.inl` 能对当前进程的 A/b/M 做完整 host/Graph 对照，但没有历史文件载入入口。旧 Python replay 也不能自动充当当前生产 PCGSolver 的验证。

### 5.2 哪些检查足够，哪些必须回放

| 修改部分 | 最少检查 | 是否要求六系统 |
|---|---|---|
| 仅删专属、不再被引用的文件/文档；退休错误入口；解析与 metadata | include/符号闭包、CPU 契约、clean build、原有组件/PCG guard 与默认短 smoke | 不需为此创造新算子测试 |
| 普通 SpMV 原方法原样保留，仅删 unused quadratic 方法 | 规范化函数文本/字节对照、原 SpMV/PCG 回归 | 单独这项不需新 SpMV 微基准；与下项共同提交仍走其门槛 |
| MAS preconditioning/collect 参数缩减、ordered 分支去除、稳定 CSR 共享代码及 MAS Graph key | 当前 disabled 路径静态等价；六份历史 M fixture；稳定 serial/warp、默认 factor_inverse 与保留 triangular 各相关模式；增长/zero/pivot checks | 六份历史 M 数据全部重放，缺一项记未完成 |
| PCG host/conditional Graph 初始与循环 rho/pAp 分支化简、删 partial/scratch/key | generic guard；当前同进程 frozen A/b/M host/Graph（含零 RHS、迭代上限、真实残差、生产状态恢复）；mixed ABD+FEM 覆盖 | 这是必须覆盖完整 A/b/M+生产 PCG 的部分，不能只拿六个 M fixture 代替 |

若验收要求字面上的“六份历史 A/b/M 完整生产 PCG 重放”，须先确认六份资产确实含全局矩阵/右端项/所有 ABD 与 FEM 预条件器数据，并补一个有限的、只读快照载入入口或找到已有完整生产入口。该入口只重建冻结系统，不推进场景、不替换算法，沿旧守卫，不增加 rho 扫描或性能研究；此工作必须明确排期，未完成时写“六 M 已验、历史全 PCG 未验”，不能改称通过。当前计划不实现该入口。

完整验收建议采用两层：六历史 M＋当前连续推进所得 frozen A/b/M 的 host/Graph 对照。若主任务坚持历史完整 PCG，则补齐前一段缺口再宣布第二批清理完成。CPU 准确参考、默认 rho 与真残差继续分开，不把 1e-4 rho 当欧氏残差门槛。只读 A/b/M 身份与主状态恢复要求逐位；atomic 路径输出数值比较使用预先已有重复范围，不能在失败后放宽。

## 6. 实施与停止顺序

1. 等待当前 AutoDL contact-pool 轮结束，冻结其结果和运行身份。本计划文件不属于当前 native 编译输入；不改已封存 manifest。
2. 保存实施前 commit、有效配置和六系统资产清单；确认执行能力检查与 native 两处均拒绝旧候选。
3. 一次性按第 2–4 节修改；不顺带整理其它求解器或优化普通 kernel。静态对照各条 disabled 路径、公共签名/缓冲和前后 include 闭包。
4. CPU 配置/拒绝合同、`git diff --check`；干净重建并记录所有 TU、对象和链接输入身份。不存在“只是删代码所以不用重编”例外。
5. 串行完成第 5 节回归；一旦共享默认/稳定路径不一致立即停止，定位清理错误，不能通过切算法、加迭代上限或放宽停止规则处理。
6. 用既有有限接触窗口确认默认 IPC/Graph 与 mixed 安全回归；需要主性能确认时按同一轮/同预算重新配对，不拿清理前后不同机器或诊断开关相混的时间宣称加速。没有新算子，不重复退役候选的完整性能网格。
7. 更新清理实绩文档和保留能力表，标明 GPU/六系统/完整 PCG 的实际覆盖与缺口，再提交；历史报告及失败证据保留。

本计划仅做路径、依赖、历史证据和入口检查；没有修改 native、没有编译、没有启动 GPU、没有宣称清理后数值或性能通过。
