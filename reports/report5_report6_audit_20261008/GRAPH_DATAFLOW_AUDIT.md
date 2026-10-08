# 报告 5 / 6：CUDA Graph 与数据流审计

审计日期：2026-10-08。范围：活动源码 `StiffGIPC/` 的 conditional PCG Graph、缓存失效、容量和指针生命周期、scratch 归属、主机读回、事件/流依赖，以及两报告的 Graph、异步传输、allocator 和布局建议。两报告已完整阅读；本审计没有启动 GPU，没有修改 native 或 `baseline/`，没有提交 Git。下列行号是本轮审计快照；并行修复会移动行号，因此同时给出函数名。

## 1. 性能事实与证据边界

报告 5 第 3 行“当前已有 1.6–1.95×”不能作为这次计划的起点。`reports/autodl_seven_20261008/ANALYSIS.md`、`REPORT.md` 和 21 行 `TIMINGS.csv` 的当前结果为：

- Graph-only / 冻结 Stiff：1.049–1.269×，七场景描述性几何平均 1.187×。
- all / 冻结 Stiff：1.075–1.414×，描述性几何平均 1.263×。
- all = K1 conditional Graph + FullCCD refit + 批量能量 + 能量复用 + 普通 BVH refit；不是 K4、接触池、AL-TOI 或退休融合。
- 每臂每场景仅一次；没有重复置信区间、活动 host 对照、分量独立消融或同质量认证。Graph/base 的所有差异不能严格归因于 CUDA Graph。

`BUILD_IDENTITY.json` 显示本轮是独立 clean build，活动 37 个、base 35 个实际编译单元，CUDA 12.8、compute/sm_89，程序/源码/输入身份有保存。当前以下文件 SHA256 与该身份中的源码逐项相等：`pcg_graph_impl.inl`、`global_linear_system.cu`、`MASPreconditioner.cuh`、`cuda_buffer_view.h`、`cuda_cub_wrappers.h`。因此本审计的 Graph 指针/缓存判断对应已测实现；当前 SpMV 和 GIPC 正在并行纠错的工作树不能直接套用旧性能结论。

七场景 all 的线性阶段占求解时间 34.9–60.5%，CCD＋线搜索占 24.7–47.5%。这些是组合阶段，不能称为单个 kernel 成本。达到 >2×仍需节省当前 all 约 29.3–46.2%；报告 5 的 Graph“再加 5–10%”、报告 6 的 Graph“极高收益”都没有当前证据支持。

## 2. 两报告建议逐点对照

状态含义：已实现＝存在活动实现；部分＝只覆盖一段或缺少收益证据；缺失＝没有该具体机制；错误＝示例/前提会破坏依赖、精度或生命周期；不适用＝当前工作约束和已有结构不支持直接采用。

| 报告建议位置 | 状态 | 活动代码映射 | 审计结论 / 纠正 |
|---|---|---|---|
| 5:9、19、77–94；6:35–65、246–277：引入/重用 Graph | **PCG 已实现，整帧部分；示例错误** | `linear_system/solver/pcg_solver.cu::solve` 116–150；`pcg_graph_impl.inl::pcg_graph` 135–312 | 当前已减少逐 PCG 迭代主机决策；不能把它当作尚未实施的第一收益。报告中的 `src/cuda/graph.cu`、`CaptureSimulationGraph`、`SimulationRunner.cpp` 均不是当前实际路径。 |
| 5:19：初始化构图，后续复用 | **有条件已实现** | `pcg_graph_impl.inl` 192–281 | 图按矩阵/预条件器/scratch 地址及形状、停止参数缓存。接触变化导致失效时需要重建；“仅初始化一次”不成立。 |
| 5:78：所有 CUDA 操作纳入一图；6:37：碰撞/TOI/装配/线搜索/PCG 一次捕获 | **缺失，但不可直接照抄** | `core/GIPC.cu::buildCP` 9070–9139、`buildFullCP` 9142–9180、`solve_subIP` 11399–11525 | 主机根据 live count、overflow、可行步长、能量、Kappa、停止规则做动态决策。需要先明确设备决策和容量协议；不是扩张现有捕获括号就能实现。 |
| 5:81–91：`cudaGraphLaunch(graph,stream)` | **示例错误** | 当前 `pcg_graph_impl.inl` 277、286 | launch 接收 `cudaGraphExec_t`，而注释说构建的是 `cudaGraph_t`；必须先 instantiate 并管理 exec 生命周期。 |
| 6:48–62、258–274：函数级 static 图一次捕获所有迭代 | **错误 / 不适用** | 当前 `PCGSolver` 逐实例拥有 graph/exec，`pcg_solver.h` 41–49 | 示例没有设备、参数地址、节点尺寸、接触、停止条件和对象生命周期失效机制；CPU 条件也不会在 replay 时重新执行。不能复制到当前仿真。 |
| 5:33：多帧批处理/持久核 | **缺失，当前不启动** | `GIPC.cu::IPC_Solver` 11669–11886；`app/gl_main.cu` 1896–1917 | 帧间物理/边界目标、CCD、Kappa和接受状态有依赖；现有 all 没有该机制。没有时间线证据支持多帧批处理，更不能固定停止次数。 |
| 5:17、41；6:67–84、279–298：多流＋异步传输 | **生产多流缺失，局部异步已实现；示例有依赖错误** | `global_linear_system.cu::apply_preconditioner` 365–366；`MASPreconditioner.cu::preconditioning` 2568–2571、2597–2601 | identity preconditioner 的 D2D copy 和 MAS workspace memset 已异步。`cudaStreamWaitEvent(0,event,0)`只建立 GPU 流依赖，不保证 CPU 已可读下载结果；报告注释“如果后续需要 CPU 处理”不正确。 |
| 5:41：一流 SpMV，另一流预计算下一次迭代向量/预条件 | **错误的独立性假设** | `pcg_graph_impl.inl` 241–256；host `pcg_solver.cu` 501–565 | `Ap_k→alpha_k→r_{k+1}→M^{-1}r_{k+1}→rho_{k+1}→beta_k→p_{k+1}`是关键串行依赖；下一迭代预条件器需要刚更新的 r。必须证明真正独立子任务后才能分流。 |
| 6:82、294–296：每次创建 event，record/wait | **错误的生命周期示例** | 当前生产没有这类多流 event fork/join | 示例未 destroy、未给 stream1 上游输入就绪边和 host 输出完成边；一段同流 copy→kernel→主流等待本身并没有可重叠工作。应长期复用事件，明确输入/输出和缓冲区所有者。 |
| 5:13、17；6:118–139、324–342：AoS→SoA / pinned / 预取 | **字段布局部分已实现，坐标 SoA 缺失；收益未证实** | `fem/device_fem_data.cuh` 21–29：位置/速度/力已独立字段数组；位置仍 `double3`；`global_matrix.h` 与 `global_linear_system.cu` 分开 values/rows/cols | 当前不是示例那种把位置速度全塞一个 `Particle`。三坐标 SoA 会影响 Eigen、MAS、ABD/碰撞和导出 ABI；不能以 `float*` 示例变相降精度。没有 memory transaction/带宽热点实测，当前不列高收益候选。 |
| 6:164–187、345–364：循环 malloc/free 全移初始化 | **容量复用已实现；static 裸指针示例错误** | `cuda_buffer_view.h::DeviceBuffer` 314–559；`cuda_cub_wrappers.h::TempBuffer` 22–72；`solver/PCG_SOLVER.cuh` 19–31 | 重用容量已存在，增长时仍需实际重分配。`static ptr`只第一次按 size 分配，后续增长会越界；没有 owner/device/stream/析构，不能代替当前 RAII。 |
| 6:164–187、345–364：移除所有同步 | **部分可研究，不可直接采用** | `GIPC.cu` 9111–12、9162、11449/61、11525、11832；`pcg_graph_impl.inl` 179、293 | 有的读回决定容量重跑/步长/错误退出，有的是计时包络。必须逐点分类。把 D2H 改成 Async 然后直接读 stack 标量不是正确优化。 |
| 5:39、125–135；6:86–115、300–321：融合位置/速度 | **示例会改变结果；当前不重开旧融合** | 当前 Newton 接受位置后 `IPC_Solver` 11828 更新速度 | 6的before先位置后速度，after先速度后位置，使用不同速度；不等价。持久线程也必须保持跨block全局阶段依赖，不能用普通block barrier替代kernel边界。 |
| 5:61–71；6:详细瓶颈分析、8剖面工具 | **已有观测框架，算子证据部分缺失** | `gipc/cost_trace.h` 136–260；`pcg_solver.cu` 160起 operator probe | cost trace可提供scope/NVTX/事件；捕获内事件被 guard 关闭。事件区间含host提交空隙且嵌套inclusive，不能直接相加当全wall。优先Nsight时间线＋独立算子probe；不要用微型Graph toy收益替代端到端收益。 |

## 3. Conditional Graph 实现与失效链

### 3.1 捕获/重放边界

`PCGSolver::solve`先清 x、准备 z/p/r/Ap；`pcg_graph`在捕获前 resize 标量和专用 CUB workspace，拷贝 b→r、应用已组装预条件器并求初始 rho。捕获只包含 PCG while body，不包含动态装配、资源分配或主机读回。

`pcg_graph_impl.inl` 220–238 创建拥有 conditional handle 的 root graph，把 `cudaStreamPerThread` 捕获到 WHILE body。241–256 为 K1：SpMV、p·Ap、alpha、x/r 更新、预条件器、r·z、beta、零rho检查、p更新及继续条件。275结束 capture，277 instantiate，286 replay，293只在图外读回结果。

当前 K1 保留“更新 x/r 后检查 previous rho”的官方 host 时序（host `pcg_solver.cu` 533–535，Graph `graph_beta` 30–38、`graph_p_continue` 49–52），不是新 residual stopping。Graph在停止轮仍会计算一次不会再用于解更新的预条件器/归约，这是额外执行成本，不能宣称所有节点都严格对应host工作量；它不是当前确认的物理错误。

`CostCaptureGuard`（`cost_trace.h` 104–108、143–147）让 capture期间的 CostScope不创建/record计时事件。MAS apply的生产路径没有分配、D2H和Timer同步；SpMV当前也是kernel-only路径。K4只是显式实验选项：`pcg_graph_options.h` 8–20，当前评估使用K1，不应顺手切换。

CUDA官方允许捕获 `cudaStreamPerThread`，禁止capture期间对该流/事件或覆盖它的device/context做同步/查询；conditional body支持设备条件循环。当前`--default-stream=per-thread`（根`CMakeLists.txt` 127）与显式per-thread CUB/memset匹配，不存在“默认流一律不能捕获”的问题。[CUDA Graphs Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

### 3.2 签名是否覆盖 captured 参数

| captured 对象/参数 | 失效键来源 | 结论 |
|---|---|---|
| 全局 BCOO values/row/col、live unique block count | `GlobalLinearSystem::graph_signature` 400–408 | 覆盖地址和kernel grid/count。矩阵内容在相同地址更新可以复用，不需要每次内容hash。 |
| 子系统维度和offset | `global_linear_system.cu` 409–417；MAS本地offset 54；ABDoffset 41 | 覆盖captured view尺寸/局部段。 |
| diag3x3地址与数目 | `diag_preconditioner.h` 10–12 | 覆盖。当前legacy MAS+ABD仍有各自签名。 |
| ABD inverse12x12地址、body count、offset | `abd_preconditioner.cu` 37–41 | 覆盖。 |
| MAS totalNodes/totalMapNodes/clusters/levels | `MASPreconditioner.cuh::graph_signature` 174–179 | 覆盖所有生产apply尺寸、memset长度及kernel launch参数。 |
| MAS restrict/local/prolong所用persistent和scratch指针 | `MASPreconditioner.cuh` 180–199 | 当前legacy和wide/64/cholesky路径所读指针均在键中；MAS模式与factor/restrict动作也入键。组装专用、apply不使用的buffer不必入键。 |
| x/r/z/p/Ap、CUB temp、s地址，n，CUB bytes | `pcg_graph_impl.inl` 196–204 | 覆盖。p/z/Ap兼作partials但阶段在同一流串行，未发现跨阶段并发覆盖。 |
| fused/chunk/scalars size/fixed iteration/current device | 同文件205–208 | 覆盖；当前生产不启用退休融合/K4/fixed。 |
| max_iter、rho tolerance | 同文件209–211，278–279 | 覆盖，变化后destroy旧exec和graph并重新capture。 |

结论：**本轮未确认当前单设备、单host线程、串行solve的captured pointer/shape漏键bug。** graphmiss本身不代表缓存错误：live block count和MAS cluster数真实变化时，现有captured launch尺寸确实已变；不能为了提高命中率去掉这些键。七场景本地汇总没有graph cache-hit/capture/invalidations逐solve分布，不能根据当前代码推测recapture已是主导成本。

### 3.3 容量增长与所有权

- `DeviceBuffer`是move-only owner，resize在容量内只改逻辑size；discard/preserve增长有50%余量及size溢出检查（`cuda_buffer_view.h` 336–403、446–489）。pointer变化随后进入Graph签名。
- PCG专用`graph_reduce_storage`避免graph引用通用thread-local CUB temp；每次capture前query/reserve bytes（`pcg_graph_impl.inl` 151–162）。通用CUB workspace归calling host thread（`cuda_cub_wrappers.h` 54–72），不适合未经改造的同线程多设备或多流并发；当前活动执行未使用这些模式，属于扩展前置条件，不是本轮生产bug。
- MAS assembly在capture前动态准备cluster/matrix/R/Z容量（`MASPreconditioner.cu` 2469–2505），apply只重复写自有scratch。不能在独立流里同时apply同一MAS owner。
- 每次PCG最终D2H返回后，旧graph不再in-flight；下次resize即使先释放老storage，再做签名失效，也不会重放旧地址。若今后移除最后读回/改成异步，就必须在增长、销毁和下一solve前建立明确完成事件。
- `release_graph`先销毁exec再销毁graph；`PCGSolver`析构调用它，然后成员buffer析构（`pcg_graph_impl.inl` 127–133）。正常所有权顺序正确。
- `buildCP`/`buildFullCP`先有限容量写入并完整计数，overflow后增长并重新查询，成功后收缩逻辑live range（`GIPC.cu` 9084–9138、9153–9179）。这些路径不允许用未经校验的固定容量来换Graph静态形状。
- `_vertexes`/`_moveDir`等是借用指针（`GIPC.cuh` 36–49），不能独立free；新增layout、allocator或graph owner必须维护alias更新。

## 4. 主机读回/同步的实际作用

| 位置 | 数据量/目的 | 能否直接删或异步替换 |
|---|---|---|
| `pcg_graph_impl.inl`179 | 初始10个double，80B；判断rho错误/真零并记录初始值 | **不可直接删**；若设备handle先决定是否进loop，可研究合并到最后读回，见候选A。 |
| 同文件293 | 最终10个double，80B；迭代数、停止、breakdown及下一Newton所需解就绪 | **必须有host完成边界**；Async后仍需事件/流同步再读host。 |
| host `pcg_solver.cu`104–105 | 每次dot 8B；host计算alpha/beta/判断 | Graph已消除逐迭代读回，不是尚未实现的机会。 |
| `converter.cu::_make_unique_indices`231–234 | unique block count 4B；后续kernel grid、压缩矩阵live size | 改设备count传参+容量grid才可能移除，不能机械换Async。 |
| `GIPC.cu::buildCP`9111–12；`buildFullCP`9162 | 20B＋4B/4B；容量overflow重跑、CCD/narrow count | 保留计数完整性、增长和失败语义；批量合并可能省API调用，当前没有独立收益证据。 |
| `GIPC.cu`11449/61、11525、11832 | terminal assembly/每Newton/每frame事件计时和完成包络 | 可检查是否只同步最后事件，但必须保留elapsed可用条件；见候选B。 |
| `gipc/utils/timer.cpp`54、63、77 | 默认启用Timer的构造/elapsed/析构全设备同步 | 普通嵌套计时本身引入同步；Nsight/NVTX才能区分真实计算与仪器开销。不能把关闭计时后的时间直接与旧计时口径比较。 |
| `app/gl_main.cu`1900/1919 | frames.csv CPU同步包络、最终导出前完成 | 基准口径的一部分；初期保持，使计时不包含异步漏算。 |
| `cost_trace.h`213–214 | 仅events diagnostic模式外层flush同步一次 | 关闭GIPC_COST_EVENTS可用NVTX/CPU-only模式；不能以instrumented run认证性能。 |

D2H `cudaMemcpy`返回时host数据已经完成；pageable `cudaMemcpyAsync`也可能host阻塞，Async这个名字本身不证明重叠。报告中的异步例子应先补齐pinned owner、数据生命周期、输入event和结果完成event。[CUDA Runtime API Synchronization Behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)

## 5. 已确认 native 缺陷与最小修复建议

**D-GRAPH-01：物理帧计时事件确定泄漏。** `GIPC::IPC_Solver`在11681–11683每帧创建start/end0；11831–11834记录和elapsed，函数11886结束前没有destroy。全文件现有destroy只覆盖`solve_subIP`六事件正常/terminal路径。120帧至少遗留240个event handle；错误/异常退出也没有owner收尾。这是代码控制流确认的资源缺陷，尚未用GPU验证其长期资源耗尽表现，不能宣称其修复带来某个加速百分比。

最小修复：在局部RAII owner中将每个句柄初始化为nullptr，逐个checked create，析构对非空句柄无抛出destroy；保留现有record/sync/elapsed位置及计时字段。构造阶段如果采用可抛出错误检查，需要清理已创建的前一个event。`solve_subIP`的六event在PCG/lineSearch抛异常时也漏掉手动destroy，可同一生命周期修复覆盖，不需要改变算法或同步。

已先发给root，root负责native补丁；本文件不与collision/solver agent同时改`GIPC.cu`。审计时也发现工作中的SpMV尾部guard暂态不完整，已提醒root核对并发编辑，**不把该临时文本当作Graph历史缓存缺陷**。

## 6. 本范围最多两个性能研究候选

两项都只是有现有工作量/阶段证据支撑的候选，没有新的kernel或API微测，因此不承诺速度，不独立承担>2×主目标。K4、接触池、旧融合不重开。

### A. 合并 Graph 初始读回到最终读回

证据：当前Graph每个非零solve有两次80B D2H；七场景Graph臂分别有483/584/679/654/708/687/736个linear direction，至多966–1472次Graph scalar读回。相应初始读回数据总量只有约38.6–58.9KB/120帧，因此优化目标是API延迟/停顿，不能声称PCIe带宽瓶颈。

目标：`pcg_graph_impl.inl`166–190、220–238、289–312。初始化后由设备kernel写conditional handle决定初始rho合法且非零才进入WHILE；初始error/zero直接跳过body，最终一次读回同时处理rho_initial、zero、error、iterations。保留原FP64/CUB、K1、rho=1e-4、previous-rho停止位置，不能固定循环次数。

进入条件：非instrumented配对任务前先用已有NVTX/CPU-only trace取得initial-readback关键路径占比；只有可移除停顿在目标场景达到预先登记的工程门槛才进入实现。零RHS、零rho但非零residual、NaN/负rho、curvature breakdown、迭代触顶、pointer/shape增长失效必须回归；新增Graph初始化节点地址和语义入缓存协议。CUDA官方允许device条件/嵌套条件节点，具体实现需按实际CUDA12.8支持集验证。[CUDA 12.8 Programming Guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html)

停止条件：配对总solver时间无稳定下降，或错误/zero/iteration统计变化，或质量合同未通过，即关闭候选。不要为了省初始读回删掉错误guard。

### B. 把生产计时等待限定到实际完成事件

证据：七场景每120帧有483–737个linear direction；当前每个接受Newton在`GIPC.cu`11525全device同步，帧尾11832又同步，batch计时1900还保留同步。生产主要工作使用calling thread默认流，Graph/CUB/memset明确为`cudaStreamPerThread`。当前计时调用次数具备数量级依据，但没有各API耗时依据。

目标：在证明所有被计时生产工作都已归该流或显式join后，研究11525等待end4、11832等待end0的等价替换；复用/RAII计时事件可减少管理开销，但资源泄漏修复应先独立落地。batch的frames.csv包络、必要D2H依赖和质量导出时机保持。

进入条件：Nsight确认没有未join的library/其他stream工作；CUDA error观测不被延后到下一frame；同一输入/输出/停止配置配对，事件计时字段完整。不要将StreamWaitEvent误当host等待，不直接删除同步。若全部设备工作本就只有这一条stream，device→event替换可能没有实质收益，须接受这一结果。

停止条件：收益小于重复噪声、event字段失真、host计时漏算、并发输入依赖不完整，关闭候选。当前没有多流大收益或allocator大收益的直接证据，不新增第三项泛化候选。

## 7. 后续必须保留的验证合同

1. 先修确定生命周期缺陷并保留数值算法；root汇总native补丁后再做干净构建与身份保存。
2. 对Graph缓存至少覆盖：首次capture→同key replay、matrix live count变化、容量指针增长、MAS cluster/level变化、ABD body/offset变化、rho/max_iter/chunk/device变化；记录captures/hits/invalidations，不能只确认进程退出0。
3. 保存FP64外部类型及legacy MAS内部既有实现、材料、rho=1e-4、IPC dt=.01、legacy累计阈值.01/min6、完整CCD。现有`gipc::Float=F64`，legacy MAS已有float scratch/预条件矩阵（`math/eigen_data.h`13–15、`MASPreconditioner.cuh`49–51）；“保留FP64”不授权改变legacy MAS精度模式。
4. 固定system host vs K1 Graph对照解决输出/guard/迭代时序，cache增长场景用内存检查；benchmark再用冻结Stiff分母并报告活动host/Graph/all，避免把native纠错收益都归给Graph。
5. 每次GPU任务串行，先窄场景，再七场景完整120帧配对和重复；记录源/编译/二进制/输入/resolved config身份，保存正确性/材料/轨迹及接受路径CCD证据。质量/性能未认证必须继续明确。

只读验证命令包括：PowerShell7 `rg -n`逐symbol/调用映射、`Get-Content`完整报告与源段、`Import-Csv reports/autodl_seven_20261008/TIMINGS.csv`统计、`ConvertFrom-Json`读取身份的小字段、`Get-FileHash -Algorithm SHA256`对照上述五文件。没有在本审计新增GPU、仿真、profiling或性能认证。
