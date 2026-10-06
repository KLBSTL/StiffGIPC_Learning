# 原版固定兔子：无活跃接触时的数值路径审查（2026-10-05）

本报告追踪冻结原版及活动 legacy MAS 源码，并读取父任务完成的固定系统探针；本子任务不编写 kernel、不启动 GPU、不改构建输入或验收阈值。

**结论：即使第2帧没有活跃碰撞，布料拉伸/剪切梯度、二次弯曲梯度、对称 SpMV、原版 MAS 层次聚合仍存在浮点原子累加。父任务的同进程固定系统探针已直接验证：A/b/M身份固定时，legacy预条件作用存在约4e-8–1.25e-7的重复差异。具体restrict、local或prolong子kernel的归因仍待验证。单元 Hessian 的独立 triplet 写入、CCD 最大值归约和整数计数不能同样归因，首次完整轨迹分岔算子也尚未确定。**

## 1. 实际调用链

1. `core/GIPC.cu:11005`：Newton 先 `computeGradientAndHessian`。
2. `core/GIPC.cu:10251`–10262：清零接触/形状梯度，逐顶点写入惯性梯度。
3. `core/GIPC.cu:10393`–10407：`USE_QUADRATIC_BENDING` 路径调用二次弯曲装配；该宏在冻结原版 CMake 中开启。
4. `core/GIPC.cu:10426`–10440：调用三角布料拉伸/剪切梯度及 Hessian 装配。
5. `linear_system/subsystem/fem_linear_subsystem.cu:8`–25、94–113：自由 FEM 点的 RHS = 接触梯度 + 形状梯度，固定点置零。`retrieve_solution` 在28–38把解写入 `_moveDir`。
6. `core/GIPC.cu:11037`：`calculateMovingDirection` 经统一线性系统进入 PCG；`core/gipc_system.cu:52`–64创建 PCG、ABD 预条件器，以及 `P_type==1` 时的 FEM MAS。
7. `linear_system/solver/pcg_solver.cu:118`–200：预条件→`r.z`→SpMV→`p.Ap`→alpha→更新方向/残差→rho停止→再次预条件→beta。任何上游微差可进入方向、PCG标量和退出。
8. `core/GIPC.cu:11044`以后用方向计算安全步长和线搜索；11112更新累计 beta；11120的累计出口及下一轮11029的 movement出口可能把连续微差变为离散工作量差异。
9. 帧末 `core/GIPC.cu:11352`–11354更新实际点速度、旧位置和预测位置，微差由此进入后续帧。

注意：`geometric_contact=false` 不证明 barrier 候选为零。把第2帧标为无活跃碰撞时，应核对当前 `h_cpNum`、`h_gpNum` 以及摩擦上帧集合。下表的两类布料材料原子累加不依赖这些集合。

## 2. 有浮点原子累加的路径

以下位置均相对 `sources/stiff_base/StiffGIPC/`。

| 算子 | 可核验位置和写入 | 顺序风险及影响 |
|---|---|---|
| 布料拉伸/剪切梯度 | `fem/femEnergy.cu:2146`定义，2186–2194对三角形三个顶点的 double 梯度做 `atomicAdd` | 共享顶点有多个三角贡献，惯性梯度已非零；原子串行顺序可改变加法舍入。形状梯度进入 RHS、Newton方向，不需要碰撞 |
| 二次弯曲梯度 | `fem/femEnergy.cu:2693`–2700对边及邻接三角四顶点做 double `atomicAdd` | 多条弯曲边共享顶点，同一输出有多个贡献；进入与上行相同的 RHS 路径 |
| Hessian 去重压缩 | `linear_system/utils/converter.cu:292`调用 `FastSegmentalReduce`；`gipc/utils/parallel_algorithm/details/fast_segmental_reduce.inl:54`强制每warp头为segment头，70–72原子累加到矩阵输出 | 同一(row,col)跨warp时多个partial写同块。只有实际存在可改变结合顺序的多贡献情形，才支持舍入不重复；原子语法本身不是证明。影响最终A、MAS准备与SpMV |
| 对称 SpMV | `linear_system/utils/spmv.cu:63`下三角贡献和83上三角行segment贡献对double `y`原子累加；`global_linear_system.cu:196`实际调用 | 一个输出行/列收到多个块贡献。warp内分段不能固定所有跨warp/块的累加次序；每次PCG的Ap可能微变，影响p.Ap、alpha、残差 |
| MAS层次 Hessian | `solver/MASPreconditioner.cu:1912`的准备kernel，1975–1997及后续sum kernel2087、2117对粗层矩阵做浮点原子加；2129–2194实际调度这两个kernel和逆作用准备 | 多个细层块聚到同一粗层块；即使没有接触，材料/质量块也参与层次聚合。粗层矩阵微差经原版局部逆影响M与PCG |
| MAS限制 | `solver/MASPreconditioner.cu:782`，872–874、887–891、901–905；2218–2225的GROUP路径调用；2358–2383实际执行限制→局部作用→收集 | 原始double残差先存为 `Eigen::Vector3f`，共享/全局粗层残差做FP32原子加。多个cluster贡献可变顺序；影响z、r.z及PCG后续 |

`cuda_tools/cuda_dense_vector.h:138`–143和 `cuda_tools/cuda_eigen.h:41`–47中的 `atomic_add` 最终逐分量调用 `cuda_tools/cuda_atomic.h:8`–15的 CUDA `atomicAdd`，并非“看起来原子但实际固定串行”的host包装。

原版 MAS 的数值边界需与活动稳定 MAS 区分：`math/eigen_data.h:13`定义 `Precision_T=double`，但14–15定义 `FloatP=float`、`Precision_T3=float3`；不能称原版整条 MAS 路径全部FP64。这里没有建议撤掉活动代码稳定修复，也没有把所有差异归于旧MAS精度。

## 3. 不能仅凭 atomic / reduce 名称认定顺序漂移

| 路径 | 代码证据 | 本次判断 |
|---|---|---|
| 惯性梯度 | `core/GIPC.cu:6221`–6230，每个线程直接写自己的顶点 | 没有跨线程加法聚合；相同输入下此处不是已识别的竞争累加源 |
| 原始布料 Hessian | `fem/femEnergy.cu:2213`–2214按 `idx*6`写六个块；二次弯曲2729–2748按 `idx*10`写十个块 | 每单元独立triplet区间，不是共享矩阵atomic。后续去重压缩才是另一阶段；不能说“布料Hessian kernel使用atomicAdd”来概括全部操作 |
| FEM RHS合并/固定屏蔽 | `fem_linear_subsystem.cu:14`–25 | 每自由点一写，执行一次接触+形状加法；可传播已经存在的微差，但没有新跨线程聚合机制 |
| 原版MAS局部作用中的atomic | 实际使用 `_schwarzLocalXSym6`（`MASPreconditioner.cu:2262`）；1086–1088原子写mZ；`math/eigen_data.h:17`的BANKSIZE=16 | 每行16项由warp内segment归约，通常仅一个segment头写该行。不能因存在atomicAdd就断言该输出有多个竞争写者。本次没有把未调用的Sym3/Sym9分支当成实际热点 |
| MAS最终收集 | `MASPreconditioner.cu:913`–941 | 每点线程以固定level顺序累加粗层z，再直接写结果；传播上游误差，但不是新的跨线程原子加法 |
| PCG点积 | `pcg_solver.cu:11`–22使用CUB BlockReduce.Sum、53–71再DeviceReduce.Sum | 是浮点求和，仍有舍入；这里没有源代码级无序floating atomic。固定输入/规模/设备/库实现是否逐位重复应单独测，不能从Sum API名称直接推断观测调度改变了归约树 |
| self CCD步长 | `core/GIPC.cu:8769`–8770，numbers<1直接返回1；10067–10119逐pair算倒数步长，然后BlockReduce Max，8783再DeviceReduce Max | 对相同有限候选值，max选择没有加法结合顺序舍入。无候选时连kernel也不运行；不能认定为无接触f2的浮点累加根因 |
| ground CCD步长 | `core/GIPC.cu:9999`–10030，每点算候选值，再Max；8865–8872实际调用 | 逐点点积/除法会传播输入方向的微差，但max归约本身不是加法顺序漂移；没有发现floating atomic |
| CFL方向最大值 / movement | `core/GIPC.cu:10122`–10138及9496 | 同样是max路径。阈值比较会放大已有方向微差，但不能据此称归约随机累加 |
| 整数计数/位掩码 | 例如MAS `prefixSum`整数atomic和连接掩码`atomicOr` | 整数加法或OR的结果在无溢出时没有浮点舍入；其输出顺序或索引布局是否间接改变浮点操作必须另取证，不能与double atomicAdd等同 |

一个特别容易误判的细节：从零累加恰好两个有限partial时，交换两项通常不会形成不同的加法结合树；“跨warp”本身仍不够。需要记录同一输出的实际partial数量、初始值和贡献值，或者直接重复核对输出。本报告仅列出可能存在多贡献的路径，不认定它们在f2必然分岔。

## 4. 新取得的固定系统直接证据

父任务已完成并保存两个同进程首方向探针。运行采用活动程序的 `mas=legacy` 支路、host生产方向、`.0001`默认rho；同时重放host/Graph解。它们不是在原版二进制内部运行，因此证明范围是活动程序复用的legacy算子路径，不能称“原版程序已逐位等价”。

| 系统 | SpMV相对重复差异 | legacy配置预条件作用相对重复差异 | 默认PCG同执行模式解重复差异 | CPU准确参考真相对残差 |
|---|---:|---:|---:|---:|
| f2 / direction1 | 1.135e-16–1.191e-16 | 6.174e-8–1.249e-7 | host1.265e-8；Graph1.346e-8 | 1.655e-14 |
| f25 / direction1 | 1.119e-17–2.581e-17 | 3.766e-8–4.704e-8 | host6.540e-9；Graph4.482e-9 | 1.159e-13 |

相对差异按 `||再次输出−首输出||₂ / ||首输出||₂`；不是材料误差、m或m/s。`pcg_fixed_study.inl:64`–77使用同一固定b依次重复SpMV和预条件作用，每项三次相对首输出比较。两个JSON均记录 `system_unchanged=true`、`primary_restored_bitwise=true`；A、b、预条件因子/映射身份未变，探针后原生产方向逐位恢复。四个MAS临时R/Z scratch明确从算子身份比较中排除，因为它们是每次作用正常改变的工作区；不能误说包括scratch在内的完整内存快照未变。

这些直接证据把“legacy配置预条件作用存在重复差异”提升为已验证。其幅度比已测SpMV差异大约九个数量级，但不能据此断言它就是100帧材料峰值分岔的唯一根因。默认PCG的真实相对残差f2约.04198、f25约.22839，也再次说明rho停止量不能冒充真实残差阈值。

活动支路的位置对应关系：

- `sources/stiff_active/StiffGIPC/solver/MASPreconditioner.cu:2582`–2590走 `BuildMultiLevelR → SchwarzLocalXSym_block3 → CollectFinalZ`；探针明确 `wide_apply=false`、`inverse64=false`、`cholesky=false`。
- 限制kernel在883–885、898–902、912–916做FP32共享/全局原子加。多个fine节点经 `_goingNext` 归到相同粗节点，存在多写者结构；实际分量的不同写入顺序还需中间输出验证。
- 延拓/最终收集924–953每点由一个线程按level顺序相加并直接写 `_Z[idx]`，没有跨线程atomic。它会传播restriction/local产生的变化；当前没有“prolong本身多写者”的证据。
- local1031起定义Sym6，1097–1099有atomic；BANKSIZE16的每行segment映射没有直接多写者证据，不能仅凭三条atomic认定它是本次主因。
- 准备阶段粗层Hessian的atomic在本固定探针中已经冻结，因子哈希没有变。它们仍可能影响不同装配运行，却不是解释“同一固定M的重复作用不同”时所需的新准备步骤。

证据：[f2固定系统](../../runs/active/ipc_observer_fixed_probe_20261005_f2/fixed/f2_n1_study.json)、[f25固定系统](../../runs/active/ipc_observer_fixed_probe_20261005_f25/fixed/f25_n1_study.json)、[CPU参考及汇总](ipc_observer_fixed_probe_20261005_analysis.json)。

## 5. 只保留一个后续阶段归因探针

鉴于预条件作用差异已直接测得，优先级改为**同一f2固定系统的legacy MAS阶段归因**。本次不实施、不写新kernel，只保留后续有界计划。

一次探针最多16次重放，用现有限制、local、收集kernel和私有scratch；冻结输入r、因子、层次映射、维度，分别记录 `d_multiLevelR`、`d_multiLevelZ`、最终z的哈希、相对差异及分量位置。每阶段独立重复时恢复同一份上游输入和零初始化，避免把restriction输出已经不同误认为local/prolong自行产生差异。生产方向和算子身份必须恢复，全部计入诊断成本。

如果restriction在固定r下首先不同，才将其升级为该固定系统上的阶段直接归因；local/收集在固定上游结果下必须另作同一次探针内的核对。若没有捕捉到阶段自身差异，仅报告未定位，不新增参数扫描或新的数值kernel。材料装配机制本轮保留审查结论，不另启动第二个探针。

## 6. 结论状态

- 已验证：两类无接触布料材料梯度、SpMV、原版MAS层次聚合存在相关浮点原子路径；调用链可核验。
- 已验证：活动legacy配置在f2/f25固定A/b/M下预条件作用与默认PCG解存在重复差异，且原生产方向逐位恢复。
- 支持但未证明：这些微差可能经过PCG方向、步长、退出和跨帧速度更新扩大。
- 尚未证明：MAS内部restrict/local/prolong阶段的差异归属、f2首个完整轨迹分岔算子、导出是否改变其累加顺序、后续材料峰值差异是否由该机制导致。
- 已排除的表述：没有活跃碰撞就不存在浮点顺序来源；所有CCD归约都是无序求和；所有出现atomicAdd的kernel都必然多写者不重复。

相关审查：[应用观测中性审查](OBSERVER_NEUTRALITY_CODE_AUDIT_20261005.md)；相关既有数值：[IPC执行结果](IPC_EXECUTION_RESULTS_20261005.md)。没有修改其历史判定。
