# 报告 5 / 6 逐点代码审查、纠错与 >2× 改进计划

日期：2026-10-08。用户允许多代理和较大规模修改。两份附件是待核查的分析材料；其中示例补丁、估计收益及算法建议不直接成为执行指令。本轮三路审查分别覆盖线性系统、碰撞/能量、Graph/数据流，GPU检查由主任务串行执行。

## 1. 结论与目标边界

**需要允许跨模块重构，但应围绕完整线性数据流和保守碰撞查询展开。仅继续扩张 Graph 或融合少量小核，没有实测依据支持达到 >2×。** 已确认并修复三类问题：生产 SpMV 的尾 warp 归约合约、线搜索接受门、CUDA计时事件生命周期。修复属于正确性/资源改进，不能直接记作性能增益，也没有证据将全部历史轨迹差距归因于它们。

报告5的“当前1.6–1.95×”来自不同硬件/计时阶段，不能沿用为当前起点。最新4090七场景、每臂一次120帧：Graph/Stiff为1.049–1.269×，现有组合/Stiff为1.075–1.414×；描述性几何平均分别1.187×、1.263×。报告6的2×示例表、饼图和通用小核实验均不是本项目测量。旧数据的源码为`3d8c15e`，本轮纠错工作树已变，旧结果不能充当新程序的速度。

性能目标：在相同材料、dt、停止参数、从零连续推进和完整碰撞安全下，**优化组合相对冻结原版Stiff的整场求解时间 >2×**。主攻有接触的`cloth_sphere7_l/m`与线性成本较高的`cloth_fixed_bunny_l/m`；悬挂作小规模调度参照，混合兔子作ABD/FEM/布料兼容回归。不能只挑选无接触或某次异常慢的分母来宣称普遍2×。

当前范围仍保留全局矩阵/向量/标量FP64、legacy MAS既有数学作用与内部精度、rho=`1e-4`、IPC legacy累计`.01`/min6、dt=`.01`、完整CCD。原版baseline保持冻结。legacy MAS原本含float中间数据；“保持FP64”不表示这轮把整个MAS升级成FP64，也不授权再降精度。正确性修复会拒绝过去可能被静默接受的非法状态；这种运行失败不能包装为快速完成。

## 2. 逐点对照索引

报告5共184行、报告6共401行。下表统一合并重复建议；每个报告的编号步骤和补丁都在右列映射。详细行号、源码依据和反例分别在[线性审查](LINEAR_AUDIT.md)、[碰撞/能量审查](COLLISION_ENERGY_AUDIT.md)、[Graph/数据流审查](GRAPH_DATAFLOW_AUDIT.md)。源码行号以审查快照为准，函数名用于修复后的定位。

状态：“已有”表示活动实现存在，不等于通过质量/收益认证；“候选”表示待门槛验证；“错误”指报告前提或示例错误；“不采用”表示本轮边界外或缺少证据。

| ID | 报告5建议 | 报告6对应建议 | 实际代码/实现点 | 判断与处理 |
|---|---|---|---|---|
| P01 | §1启动开销、§6-1 Graph | 候选1、补丁1 | `linear_system/solver/pcg_graph_impl.inl::pcg_graph`，`PCGSolver::solve` | conditional PCG已有，K1与设备标量复用；不重新移植并重复计算收益 |
| P02 | 所有仿真操作纳入Graph | 一次捕获全部模拟步骤 | `core/GIPC.cu::solve_subIP/lineSearch/buildCP/buildFullCP` | 动态计数、overflow、Kappa、能量和接受条件在host；整帧尚未设备化，不能扩大capture括号代替设计 |
| P03 | 初始化一次、参数化图 | static graph一次初始化 | `GlobalLinearSystem::graph_signature`，`MASPreconditioner::graph_signature` | 当前地址、尺寸、live count、模式、停止参数等键已覆盖；static示例遗漏失效和生命周期；`cudaGraphLaunch`需要exec而非原graph |
| P04 | 多帧批处理 | 主循环所有step捕获再replay | `GIPC::IPC_Solver`，`app/gl_main.cu` | 帧间依赖、边界目标、CCD和退出决策必须保留；本轮不固定迭代或批量跳过帧间检查 |
| P05 | occupancy、launch bounds、寄存器限制 | SM/带宽指标 | `MASPreconditioner.cu`局部作用、`mlbvh.cu`查询 | 用Nsight Compute确认register spill、stall、eligible warps；提高occupancy不是线性提速保证，不全局设置maxrregcount |
| P06 | 共享/常量/纹理缓存、预取 | 候选4、补丁4 SoA | `device_fem_data.cuh`，`gipc/type_define.h`，矩阵values/rows/cols | 字段已分开、位置仍double3；不存在示例Particle布局。仅在热点视图中改布局，维护ABD/Eigen/导出别名；禁止换float |
| P07 | atomic、warp归约、分支 | 原子竞争/融合 | `linear_system/utils/spmv.cu`，`fast_segmental_reduce.inl` | warp分段归约已有但尾部合约有缺陷，已修；行邻接SpMV仅有条件候选，不能恢复退休pAp融合 |
| P08 | pinned、CPU工作搬GPU | 候选2、补丁2 | `pcg_graph_impl.inl`读回，`converter.cu`计数，`buildCP/buildFullCP` | 每次读回需区分必要host决策、API停顿和实际GPU等待；少量字节不证明PCIe带宽瓶颈 |
| P09 | stream并行PCG下一轮 | 多流异步例子 | `pcg_graph_impl.inl` K1依赖链，MAS自有scratch | 下一轮r/z/p依赖当前alpha/beta；不能并行预计算。先独立scratch、event输入边/输出完成边；StreamWaitEvent不保证CPU可读结果 |
| P10 | O(n²)改网格/BVH | 新CCD扫掠/区间算法概述 | `collision/mlbvh.cu` Morton LBVH与query，`ACCD.cu` | 当前已有LBVH和ACCD，错误前提；均匀哈希中心入单桶不足以覆盖长primitive及swept轨迹 |
| P11 | BVH refit与历史一致性 | 减少重复工作 | `collision/discrete_bvh.inl`，`GIPC::buildBVH_FULLCCD` | 普通/FullCCD refit已有；每次更新完整bounds并查询当前状态，旧接触不能代替安全检测；构建收益不等于查询收益 |
| P12 | 小相对速度跳CCD | 新CCD/TOI作为加速来源 | `GIPC::self_largestFeasibleStepSize`，`ACCD.cu` | 不采用；near-gap小位移仍可能穿越。只允许有保守证明的几何排除，不能调速度阈值漏碰 |
| P13 | 二分/迭代上限后接受近似TOI | 最新CCD算法通用引述 | 当前ACCD及FullCCD保守候选 | 不把未匹配几何/保守界的算法替换为执行优化；完整CCD保留，失败不能接受不安全步长 |
| P14 | 球/平面解析、极小位移跳过 | narrow-phase剖析 | ground/self CCD入口与三角形/边对 | 解析公式只适用证明匹配的primitive，不能把布料triangle CCD换成球体CCD；收益待真实pair类型成本 |
| P15 | 新增connectivity-enhanced MAS | 候选5、补丁6 | `solver/PCG_SOLVER.cuh`默认P_type，METIS映射、`MASPreconditioner::ReorderRealtime` | 已有连接增强MAS、分区与padding；不得再计为新功能。精确符号复用可研究，数值算子仍fresh |
| P16 | 新预条件器/块Jacobi/重分组 | 调参/其它预条件器 | legacy/wide/inverse64/Cholesky模式 | 改M会改变迭代和rho意义，本轮不作为执行提速；准备+应用+PCG净成本比单迭代数更重要 |
| P17 | 放松rho | 精度/稳定性分析 | host/Graph previous-rho停止、`ipc_options.h` | 不采用；rho是rᵀz预算，不是真残差阈值；不增加分母工作、不改收敛条件 |
| P18 | FP32/半精度 | 候选7、补丁7 | `Float=double`，legacy MAS既有float | 不再降精度，不凭Tensor Core概述替换稀疏块作用；精度研究单列未来范围 |
| P19 | 近似材料/阻尼/strain limiting | 物理精度可接受 | FEM/布料材料与ABD耦合入口 | 改材料/积分方法不属于执行优化；薄壳模型已在当前Stiff内，不能重记创新收益 |
| P20 | Hessian局部归约/快速组装 | 原子/访存优化 | `Converter`排序/分段压缩，`GlobalLinearSystem::build_linear_system` | 已有并行归约；新机会是符号/数值流程分离和精确有序索引复用，不缓存旧Hessian数值 |
| P21 | kernel fusion向量更新/点积 | 候选3、补丁3 | PCG Graph K1、diag fused update、MAS/ABD作用 | 已有部分融合；必须在最终z完成后归约；旧MAS/pAp失败路径不复活 |
| P22 | position/velocity融合例子 | before先pos，after先vel | `IPC_Solver::updateVelocities`在接受位置后 | 报告6示例改变积分顺序，非等价融合；不能用显式粒子示例改隐式IPC |
| P23 | persistent工作队列 | 融合/打包 | `mlbvh.cu`各query kernel | 可调查查询负载均衡，但普通block barrier不是全grid屏障；先保持遍历顺序与pair身份，不强制整个Newton持久化 |
| P24 | volatile降寄存器 | 寄存器/线程效率 | 重型查询/MAS局部作用 | volatile不是通用优化开关；spill/寄存器/带宽依据不足前不推广 |
| P25 | 预计算相似矩阵逆/批处理 | cuBLAS/cuSPARSE/cuSOLVER | MAS每系统数值prepare与自写CUB点积 | 不可沿用旧数值因子；库是否更快需含格式转换、handle、workspace与完整solve |
| P26 | 库替换向量/线性求解 | Sdot、csrlsvqr补丁 | 当前FP64设备归约、对称半存储BCOO | Sdot是FP32、host结果可重新引入同步；csrlsvqr是直接QR，非PCG，参数末尾singularity而非iter/res；示例不能编译且算法不等价 |
| P27 | 延迟计算/复用 | 候选6、补丁5 allocator | `DeviceBuffer`，专用Graph CUB scratch，`PCG_SOLVER.cuh` | 容量复用已有；static裸ptr不随尺寸增长且无device/owner。确认事件泄漏已修，不能猜循环malloc占比 |
| P28 | 移除同步 | 候选6、补丁5 | `Timer`，Newton/帧事件、batch solver包络 | 某些同步决定计数/安全/输出就绪，另一些是计时成本；先证明stream join再限到event。CPU等待不可与GPU执行相加 |
| P29 | 1e-5差异即通过 | 任意可视/物理误差 | material/state质量分析、真实速度导出 | 不采用无单位/无基线重复依据的万能阈值；材料、轨迹、真实速度分开，数值容差不是额外形变预算 |
| P30 | 接触数比较判断无漏碰 | 相同seed/近似输出回归 | BVH候选完整pair集合与独立CCD | 数量相同不足；需pair身份、类型、转置/owner、保守覆盖与独立接受路径CCD |
| P31 | nvprof/cuda-memcheck | --trace=cuda,cpu、--precision | 受保护profiler与新算子检查 | 使用Nsight Systems/Compute与Compute Sanitizer；cuda-memcheck --precision不存在，sanitizer不检查舍入误差；trace=cuda,nvtx配合CPU采样配置 |
| P32 | synthetic CCD/小系统/玩具图 | 微型核官方收益 | 实际冻结A/b/M、目标cloth窗口 | toy用于边界，不代替ABD/FEM固定系统和整场计时；代表窗口从零推进，Graph内部测量单列 |
| P33 | 10次/t检验、示例场景 | 3090/A100表和示意饼图 | `tools/bench`，七场景结果 | 按预声明交错配对和有限预算；示例数据不能出现在实测图；固定/接触/混合规模都保留 |
| P34 | 任意必要时改容差、>2就停止 | 简单“同精度/可接受误差”总结 | 冻结参考/质量协议/硬失败门 | 停止规则固定；超过2但质量未过仍不能推广。严格纯Graph贡献需同一活动身份host/Graph对照 |

官方核查：occupancy提高不必带来同比性能提高，寄存器限制可能导致spill，应依热点测量。[CUDA 12.8最佳实践](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-best-practices-guide/index.html)
cuSOLVER QR的接口、直接分解含义及对称矩阵需补全要求见[CUDA 12.8 cuSOLVER](https://docs.nvidia.com/cuda/archive/12.8.1/cusolver/index.html#cusolversp-t-csrlsvqr)。Graph/异步复制的依赖需按实际API合约处理，不能删除必要host完成边。[运行时同步行为](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)

## 3. 本轮已经落地的纠错

### F1：生产对称SpMV的warp归约

旧路径不足32条目的尾warp提前return，再调用`WarpReduce<Float,32>::HeadSegmentedReduce`，不满足32-lane参与合约；xyz连续复用TempStorage无中间同步。修复只让无效lane携零值和独立head参加归约，在每次复用前`__syncwarp()`，有效lane才读矩阵/索引/x或写y。非对角转置贡献、对角一次贡献、FP64和原atomic公式保留。

`tests/sym_spmv_tests.cu`链接真实`spmv.cu`：171个case、每个host提交及三次Graph重放，覆盖尾长0/1/31/32/33/255/256/257/4097、跨warp/block长行、空行、非对称块、双方向存储、一般a/b、b=0时NaN旧y、x/y尾guard。独立CPU FP64参考。CUB的参与约束见[官方源码](https://github.com/NVIDIA/cccl/blob/v2.7.0/cub/cub/warp/warp_reduce.cuh)。不能由此推断它必然解释现有厘米级轨迹分岔。

### F2：线搜索必须接受有限且满足原能量条件的当前状态

旧循环允许9次能量回退后只打印警告，仍进入postLineSearch；NaN比较也会绕过循环。后置相交回退改变状态后，reuse关闭时没有强制重算最终能量。修复将判断抽到`solver/line_search_acceptance.h`，保留原armijoParam=0和总计9次能量回退预算。非有限值、阈值溢出、步长无进展及预算耗尽拒绝运行，记录原因/能量/alpha/工作量后走现有controlled failure。后置安全回退改变状态必须重评最终能量，并只能消耗剩余原预算。正常一次通过路径不增加能量评估。

这会改变原本非法路径的行为，不能声称逐位保持全部旧运行。CPU helper覆盖2304个边界检查；还需要真实场景验证修复是否触发，以及与冻结原版合法路径的质量对照。

### F3：事件所有权和异常释放

`IPC_Solver`每帧创建两个事件，没有destroy；120帧至少遗留240个句柄。`solve_subIP`六事件虽正常/terminal手动destroy，异常路径仍会漏。新增`cuda_tools/scoped_cuda_events.h::ScopedCudaEvents<N>`，创建失败清理部分句柄、析构无抛出释放；两处改用局部owner，移除重复手动destroy。保留所有record/sync/elapsed边界，不顺手改变计时口径或删除同步。

具体构建、GPU检查和未做项目统一见[实施与验证记录](IMPLEMENTATION_AND_VALIDATION.md)；代理审查文件中的“未运行GPU”指其分工范围。

## 4. 达到2×还要降低多少成本

来自七场景一次完整120帧，单位秒。公式`目标=base/2`，`仍需省=all-base/2`，均由[cost_budget.py](cost_budget.py)读原CSV重算，输入SHA保存于[COST_BUDGET.json](COST_BUDGET.json)。阶段字段是计时包络；下面只是保持其它成本/工作量不变的预算估算，不是独占kernel归因或可保证的严格Amdahl证明。

| 场景 | base | 当前all | 当前总倍数 | 2×目标秒 | all仍需降低 | all线性占比 | CCD+线搜索占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 悬挂L | 2.288 | 1.618 | 1.414× | 1.144 | 29.3% | 44.7% | 39.3% |
| 悬挂M | 4.736 | 3.580 | 1.323× | 2.368 | 33.8% | 54.0% | 30.7% |
| 落球L | 5.287 | 4.346 | 1.217× | 2.644 | 39.2% | 35.7% | 47.5% |
| 落球M | 8.309 | 6.619 | 1.255× | 4.154 | 37.2% | 44.8% | 41.7% |
| 固定兔子L | 6.561 | 4.876 | 1.346× | 3.280 | 32.7% | 49.6% | 32.5% |
| 固定兔子M | 12.007 | 9.685 | 1.240× | 6.004 | 38.0% | 60.5% | 24.7% |
| 混合兔子L | 10.673 | 9.926 | 1.075× | 5.337 | 46.2% | 34.9% | 40.9% |

线性阶段减半依然不足；落球L和混合兔子甚至清零全部线性成本也不足2×。这明确要求两个主要成本方向协同。固定兔子M适合证明线性流水线收益，落球适合证明查询收益；混合兔子还需要关注装配约21.1%的成本，不能预先保证全部七场景都过2×。

旧本机剖析只提供方向：matrix_convert+mas.hierarchy的CPU inclusive范围约9.35–9.98%；落球峰值帧碰撞查询约54.5%的GPU union，而SpMV约4.04%、能量归约约0.106%。这些不能移用为4090整场独占比例。Graph初始读回80B可研究API停顿，最终读回多半等待整个PCG完成，删除它不能省掉整个等待区间。分段能量归约和普通小核融合目前缺少承担2×缺口的依据。

## 5. 执行计划：纠错基准 → 成本归因 → 两条重构主线 → 验收

### 阶段0：先建立修复后的有效参考

1. 完成本轮全量独立构建、两项新测试和现有CTest；SpMV做memcheck/racecheck/synccheck。源/对象/编译/链接/程序哈希保存；不覆盖旧21次证据。
2. 正确性组至少包含无接触、带接触布料、固定ABD、混合ABD/FEM；检查修复后的line_search_failure、PCGbreakdown/触顶、有限状态、当前alpha和退出原因。旧版本若触发非法接受，不把其速度继续当“合法等价候选”的参考。
3. 冻结原版Stiff、纠错后的活动host、活动Graph、Graph+四已有组件四臂；活动host/Graph同源码和执行组件，只改变Graph开关。由此解决“Graph/base包含其它活动源码差异”的归因缺口。
4. 原版真实速度仍有观测中性缺口；保留独立observer，未证明中性前不替换正式速度分母或声称速度等价。原版3次标定+2次独立复核冻结材料范围；候选不能扩大范围。

通过门：算子/guard/配置与程序身份一致，合法线搜索接受闭合；基线复核失败则质量协议不可认证，单列原因。阶段0的修复不得被计入执行组件性能。

### 阶段1：有界剖析，确定可消除的关键路径

- 4090可用后优先从零运行固定兔子M、落球L各120帧的诊断组；本机用于结构/算子筛查，WDDM共享负载只报告诊断速度。每个新native身份最多两次Nsight Systems捕获：一个Graph整体、一个节点级。代表窗口含1–3、落球22–24及实测最贵连续三帧；混合24–26/33–35作兼容。
- 沿用`gipc/cost_trace.h`/NVTX，区分matrix conversion、FEM索引、MAS hierarchy/numeric填充/factor、限制/local/prolong/ABD、SpMV、CUB归约、Graphinit/readback；碰撞拆普通/FullCCD build/refit、query、isIntersected、ACCD；能量拆九项生产、ABD生产、归约/读回。
- 新增仅诊断的symbolic有序key/pair相等命中率、比较成本、Graph invalidation原因与次数；BVH记录每query访问节点数/leaf tests/最大路径和warp长尾。计数缓冲独立拥有，关闭诊断不运行对应探针。不能把同count当同结构。
- Nsight Systems只分析不重叠GPU活动及关键等待，CPUinclusive不与GPU相加。Nsight Compute只查明确热点的spill、DRAM transactions、cache命中和执行长尾。官方工具命令按安装版本help核查，不用报告不存在的参数。[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

交付两张成本表：完整solver包络及剩余未归因部分；互斥GPUkernel/memcpy时间和关键依赖。某候选的“可消除成本减新增准备/维护”不能达到整场5%，立即跳过，不写kernel后寻找理由。

### 主线A：精确符号结构与数值作用分离（允许跨模块重构）

对应`GlobalLinearSystem::build_linear_system/convert_new` → `Converter` → `LocalPreconditioner`索引 → `MASPreconditioner::ReorderRealtime/setPreconditioner_bcoo`。

1. 定义唯一有owner的`LinearStructure`：ordered raw keys、稳定排列、segment partitions、compact row/col、FEM块索引及精确碰撞连接身份；与`LinearNumericState`的当前rhs/Hessian/M数据分离。初版仅严格有序输入相等命中，不引入规范化顺序差异。
2. 每次先完整碰撞检测取得当前pairs。hash仅快速拒绝；通过命中须验证完整有序key/pairs、owner/topology/partition/维度/offset/level相等。缓存不能借用会被compact覆写的共享scratch，也不能以小位移/同count/近似连通判相等。
3. 命中复用排序/分段/索引/层次；仍gather当前FP64块值并按原顺序归约，更新全部rhs、Hessian数值、MAS聚合与inverse/factor。每次PCG内部M固定。无接触静态MAS复用是已有组件，先核查，不重计创新。
4. 不命中使用原完整准备路径，并更新自己拥有的缓存；变更容量/地址/活跃维度/模式进入Graph失效协议。此处“未命中”是显式算法路径，异常不是静默切回。
5. 若已确认SpMV原子/访存仍是主要热点，再在同一结构owner内建立row-gather视图：非对角两引用、对角一引用、转置标记、每Newton准备一次、不复制两份数值。直接形成aAx+by并覆盖空行；保留生产legacy开关和host/Graph同入口。该子候选未满足热点≥10%之前不实施，不恢复pAp融合。

验证：相同状态上索引/排序/segments逐项相等，A作用、MAS作用、对称性/正二次型、真残差和PCG停序；覆盖新/删接触、pair顺序、类型、缓冲增长、ABD/FEM offsets、零rhs/异常pivot/非正曲率/预算。固定系统两次预热+七对交错完整求解。准备复用要求净整场≥5%；row-gather要求包含邻接准备的完整线性阶段≥15%且整场≥5%。另一主场景不得回退>3%。每个子候选最多一次有据修订。

该路线可较大改动源码组织，但同状态数值作用必须验证；不能承诺省掉60%的线性成本。若精确相等命中率低或比较/搬运抵消收益，封存，不扩展近似cache条件。

### 主线B：保守BVH查询数据流重构

对应`collision/mlbvh.cu`的离散/swept查询及edge-triangle安全查询。先确认每线程`stack[65]`是否产生显著local-memory spill及长尾；否则不能只凭数组存在假定热点原因。

1. 候选为有界stackless/escape-index遍历视图。建树后生成owner拥有的escape index，topology变化必须更新，refit仅更新完整当前bounds；不替换接触集、不减少安全查询。
2. 初版保持原DFS子节点顺序、primitive原ID、leaf eligibility及输出pair类型/转置语义。减少栈与访存才是本轮假设；不同时改SAH、treelet、query排序、bucket或工作队列，否则不能隔离收益。
3. 生产与诊断scratch独立。新旧同状态查询逐项对照规范化后的完整pair身份和类型；还核对未规范化顺序以确定浮点装配顺序是否变化。仅统计pair数量相等不通过。
4. 覆盖长edge/triangle、跨cell swept轨迹、near-gap低速、高速擦边、退化primitive、零候选、深树、容量overflow/growth、固定物体和ABD/FEM；FullCCD/ACCD全部保留，独立接受路径CPU CCD最终审计。
5. 只有完整build/refit+query+narrow+view维护净省≥15%、整场≥5%且质量通过才推进120帧配对。若主要成本是isIntersected窄相而非栈，则停止该实现，提交确切瓶颈；不改成报告的速度阈值跳CCD。

辅助机会是能量readback合并：当前批量九段CUB仍另读ABD kinetic/shape。只有关键停顿门槛成立才把两个ABD标量接到设备11标量输出、一次D2H、保留原CPU求和顺序。保留九类生产kernel和逐项CUB，不同时上segmented/多流/能量Graph；一轮未形成≥5%整场增量就停。它属于该非线性数据流的次级机会，不能代替查询主线承担2×预算。

### 次级Graph机会（不作为第三条大重构主线）

设备初始化guard先决定是否进入conditional loop，可研究将初始80B读回合并到最终一次；必须保留initial rho、真零/假零/非有限/负rho、previous-rho停序和错误统计。生产计时可研究同步end4/end0替代device-wide等待，但必须证明其它stream已join并保留batch包络。当前没有关键停顿占比证据，先不实施；不能修改cache签名提高表面命中率。

## 6. 每轮分析、性能与质量验收

每个候选先固定系统/同状态算子，再三对交错120帧筛选；保留一项新假设、一个明确成本预算，完成后立刻记录“省了哪段、新增哪段、Newton/PCG/energy/query工作量是否变化、质量是否退化、下轮是否继续”。绝不相加报告估计百分比，也不相乘重叠组件消融倍数。

正式最终矩阵：冻结Stiff；纠错活动host；纠错活动Graph；优化host；优化Graph。已有四组件明确列配置与移除消融；报告Graph同身份收益、新组件总增量及相对Stiff总收益。每场景从零120帧、dt=.01、相同输入和停止规则，关闭重型诊断。图构建/prepare/转换全部计入solver，加载/导出另列。七对交错、受控AutoDL负载；配对中位>2且单侧95%配对bootstrap置信下界>2，质量协议同时通过才认证“>2×”。否则只报告实测值和不确定性，不用最好单次值替代。

材料指标采用预先冻结的整段最大/高分位布料拉伸，混合FEM最小J/非正单元数/负体积，ABD翻转/固定漂移；位置与真实速度按体类型分别列。非有限、PCGbreakdown/触顶、line_search_failure、独立CCD标记为硬失败。原版允许出现的FEM翻转不新增禁止规则，但不得用非正数量下降掩盖更差minJ。基线复核失败标记无法认证，不看候选后放宽范围。最终主场景各一次完整120帧接受路径审计，两个接触布料各300帧，混合120帧兼容。

预算：沿用已授权120帧单次180秒、300帧600秒；GPU串行，保留磁盘/显存/外来负载保护。超时/资源/数值失败保留证据，不自动重试或临时延长。初版→一次修订仍未过净门槛即封存；两条主线没有足够共同成本潜力时提交未达2×原因，不扩展到低精度、改变M/停止、OptiX、多卡或AL-TOI参数网格。

## 7. 交付与推进顺序

1. 本文件：全局判断、34项合并对照、主线边界、详细实现/验证/停止计划。
2. 三份模块审查：精确报告行号→文件/函数/源码行号和候选证据。
3. `COST_BUDGET.json`及脚本：七场景预算，源CSV身份，可复算。
4. `IMPLEMENTATION_AND_VALIDATION.md`及验证收据：本轮实际修复、构建/检查结果、当前缺口。

顺序固定：**纠错后的有效参考 → 非重叠成本与cache/query诊断 → 精确符号数据流 → 重新分析 → 保守查询视图 → 重新分析 → host/Graph组件配对及独立质量 → 受控4090最终验收。** 多代理可以并行审查/实现有独立文件所有权的模块，GPU任务串行；集成后必须交叉复核，不能只接受代理的“已完成”。

目前“已验证”的是旧七场景数值和本轮检查收据范围内的合约；“支持但未证明”的是跨模块两条主线价值；“已排除”的是报告错误示例可直接套用、已有MAS可重复计收益及低速跳CCD；“待验证”的是新程序的质量、整场净收益和>2×。大规模改动已获授权，>2×仍必须由实际测试证明。
