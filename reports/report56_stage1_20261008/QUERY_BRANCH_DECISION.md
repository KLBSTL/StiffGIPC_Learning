# Query 分支决定：暂缓 stackless 生产视图

现有证据尚未建立“扣除视图生成／维护后，完整120帧净省至少5%”的启动门槛。决定暂缓生产 stackless 分支，保留碰撞优化方向。新动态计数支持把 DCD EE 列为下一项因果检查的首选，但未证明净潜力低于或达到5%；缺口是遍历的独占时间与替代视图的净收益，而不是查询正确性烟测。

原计划要求先满足整场净潜力门槛，再实现保守视图；局部完整 query（含准备）至少15%改进，三对从零120帧整场至少5%、另一主场景退化不超过3%、完整质量门通过后才保留。当前没有 stackless A/B 或视图维护计时。

## 数据与采样范围

四次运行已关闭：同一新程序 observer-off/probe 各两场景，全部120帧、exit=0、有限值、配置及既有原生守卫通过。诊断 flag 不构成独立物理质量或性能认证。每场480个 VF/EE launch summary 来自240个私有样本（DCD120、FullCCD120），保留 typed multiset、duplicates／orientation／MatIndex，未将原子输出数组顺序作正确性标准。

DCD 第1个样本来自初始化、contact_model_id=0、linear_system_id=0、outer=-1、inner=-1；其余119个 DCD 和全部120个 FullCCD 都是 outer=-1、inner=0 的首次成功 production pass。FullCCD 与 DCD 各自选择本帧最先遇到的状态，并非同一状态；没有观察后来 Newton inner、线搜索 backtrack、容量重试的全部工作。第1个 DCD 的 JSON frame 被显式设为1，但不在选中 cost frame 的事件上下文内，故每场 BVH diagnostic cost 只有239条（DCD119 + FullCCD120）。

| 场景 | cost DCD调用 / 对应首样本 | cost FullCCD调用 / 首样本 | VF / EE query数每样本 | FullCCD alpha范围 |
|---|---:|---:|---:|---:|
| cloth_sphere7_l | 724 / 119 | 676 / 120 | 3083 / 9040 | 0.348178–1.000000 |
| cloth_fixed_bunny_l | 698 / 119 | 691 / 120 | 16194 / 48273 | 0.226098–1.000000 |

首样本占上述已记录 query 调用约16–18%，这些比例是调用覆盖率，不能据此将其节点分布倍乘为全部调用成本。私有 replay 使用当前树和 query launch ID 顺序，warp_max 是每32条实际 launch lane（含最后部分 warp）的节点计数最大值，并非 Morton ID 排序后的统计。

## 实际工作量

下表均来自真实首样本：均值按 sampled query／warp 数加权；max 是这些样本的精确最大值。节点 p95 为120个“各 launch 内 query p95”的中位数，p99为各 launch p99 的最高值；它们都不是混合全部帧的 pooled 分位数。默认 raw records 关闭，仅摘要和 top16，不能恢复 pooled p95/p99 或逐 query 节点／窄相联合分布。

| 场景 / query | AABB均值 | launch p95中位 | launch p99最高 | AABB max | 叶均值 | narrow均值 / max | pending栈max | warp_max均值 | warp/query均值比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cloth_sphere7_l / DCD_VF | 52.61 | 84 | 112 | 194 | 7.00 | 1.171 / 34 | 7 | 86.45 | 1.643 |
| cloth_sphere7_l / DCD_EE | 85.41 | 146 | 226 | 356 | 14.04 | 1.418 / 38 | 9 | 149.50 | 1.750 |
| cloth_sphere7_l / FullCCD_VF | 56.92 | 84 | 110 | 194 | 7.18 | 0.000 / 0 | 10 | 86.22 | 1.515 |
| cloth_sphere7_l / FullCCD_EE | 88.79 | 139 | 214 | 284 | 14.46 | 0.000 / 0 | 10 | 131.00 | 1.475 |
| cloth_fixed_bunny_l / DCD_VF | 58.56 | 86 | 110 | 210 | 6.58 | 0.384 / 10 | 7 | 91.82 | 1.568 |
| cloth_fixed_bunny_l / DCD_EE | 87.10 | 124 | 162 | 280 | 12.92 | 0.499 / 23 | 10 | 124.56 | 1.430 |
| cloth_fixed_bunny_l / FullCCD_VF | 67.08 | 98 | 138 | 188 | 6.81 | 0.000 / 0 | 9 | 99.31 | 1.481 |
| cloth_fixed_bunny_l / FullCCD_EE | 95.95 | 138 | 210 | 326 | 13.40 | 0.000 / 0 | 10 | 130.35 | 1.358 |

每 query 平均约53–96次 AABB、6.6–14.5次重叠叶，最大356次 AABB；pending 栈最高7–10，未接近65个源码槽。warp 的最大节点次数均值是 query 均值的1.36–1.75倍，支持存在访问工作不均衡；它不是实际 stall／divergence 时长，也不能推出节省同样百分比。stackless 若仅改变存储／DFS控制，不会自动减少这些 AABB、过滤、叶或输出原子工作。

DCD narrow_calls 计数的是进入 eligible leaf 的距离分类入口，均值0.384–1.418、最高38；其内部 FP64 距离运算没有独立计时，不能用次数比例把完整 DCD kernel 当成纯遍历。leaf_overlaps 在共享 primitive／body／固定边界过滤前统计，故叶数与 narrow数差距也不是可跳过碰撞的授权。FullCCD narrow_calls=0，因为这四个 swept query 仅产生候选；真正 point-triangle／edge-edge ACCD 在 `_cub_reduct_self_step`，保持完整安全计算。

## 单帧 Nsight 与整场事件包络

已有 PROFILE_COST_ANALYSIS 的 sphere49/fixed40 是冻结旧程序、Graph K1、refit／batched energy／energy reuse／discrete refit 均关闭的单帧；本次 probe/observer-off 是新程序且四组件开启。只能把该单帧作热点拆分证据，不能移用为当前四组件整場比例或首样本的直接计时配对。

| GPU kernel union，仅 capture单帧 | sphere49 ms | fixed40 ms |
|---|---:|---:|
| collision.discrete_query | 50.908665 | 19.515101 |
| collision.swept_query | 7.240881 | 7.362674 |
| collision.safety_edge_triangle | 20.939380 | 8.450385 |
| collision.accd_self_narrow | 4.194935 | 2.165564 |

DCD kernel含遍历、过滤、距离分类与输出；FullCCD候选 kernel含遍历、过滤与输出；ACCD另列。安全 `_edgeTriIntersectionQuery` 在两帧分别20.939/8.450ms，但本次工作量探针未覆盖它，不能将该安全耗时预支给四 query 的 stackless 视图。Graph 的 CPU范围、内部node时间、GPU活跃并集不相加。

| 120帧观察量 | sphere ms | fixed ms |
|---|---:|---:|
| GPU-event inclusive sum: ipc.physical_frame | 10524.152 | 17059.295 |
| GPU-event inclusive sum: collision.discrete_query | 2247.248 | 3149.180 |
| GPU-event inclusive sum: collision.swept_query | 1672.425 | 4186.502 |
| GPU-event inclusive sum: collision.self_ccd | 162.886 | 224.666 |
| GPU-event inclusive sum: diagnostic.bvh_query_workload | 1941.482 | 4888.388 |
| BVH私有 replay / CPU比较 diagnostic_host（240样本） | 1827.941 | 4783.325 |

| production query 的实际父／私有子event分解 | 父event ms | 私有BQ子event ms | 父减子event余量 ms | 父CPU ms | 私有子CPU ms | 父减子CPU余量 ms |
|---|---:|---:|---:|---:|---:|---:|
| cloth_sphere7_l / collision.discrete_query | 2247.248 | 649.066 | 1598.182 | 2238.885 | 642.306 | 1596.579 |
| cloth_sphere7_l / collision.swept_query | 1672.425 | 1292.416 | 380.009 | 1675.068 | 1285.410 | 389.657 |
| cloth_fixed_bunny_l / collision.discrete_query | 3149.180 | 1440.705 | 1708.475 | 3139.805 | 1432.119 | 1707.686 |
| cloth_fixed_bunny_l / collision.swept_query | 4186.502 | 3447.683 | 738.819 | 4247.480 | 3438.852 | 808.628 |

这里按真实 parent_scope_id 对119个 DCD 与120个 FullCCD 私有子范围进行分解，未将子范围再加到父范围。余量约 DCD1598/1708ms、FullCCD380/739ms：它包含生产遍历、距离分类／候选过滤、Ground query（DCD）、清零／计数读回、事件／调度空隙，以及诊断边界影响，绝非纯遍历核函数时间。仅删掉子event没有恢复未诊断的同工作量执行。它说明剩余查询值得继续调查，不能单独满足整场净5%。

这些 GPU event 是 stream elapsed（含提交空隙和嵌套范围），不是互斥 kernel union。DCD／FullCCD的 production父包络包含上面的私有 probe；physical frame还包含事件、线性结构探针、读回、CPU比较与文件写入。diagnostic_host从私有比较开始到返回，不含JSON文件写入及完整CostScope收尾。239条诊断event与240条host测量覆盖不同，不能相减或相加得出未观测的纯生产遍历成本。

| 完整臂观测 | sphere | fixed |
|---|---:|---:|
| observer_off solver s | 7.635986 | 11.549531 |
| probe solver s | 11.111230 | 17.622110 |
| off → probe directions | 669 → 676 | 701 → 691 |
| off → probe pcg_iterations | 29139 → 29192 | 44572 → 44238 |
| off → probe energy_evaluations | 846 → 844 | 828 → 818 |
| off → probe energy_backtracks | 57 → 48 | 7 → 7 |

两臂完整执行工作量不同，且 probe同时启用CostScope GPU事件和线性结构观察；solver秒数差只表示两次独立诊断观测，不能归为纯BVH probe开销或未来收益。observer-off只有一次／场景，本机WDDM桌面负载未受控；原生守卫通过也不补齐独立质量协议。

## 静态资源与动态性能证据

冻结旧exe的 CUDA13 `cuobjdump --dump-resource-usage` 提供以下实际 cubin 信息，覆盖生产kernel（排除 pool／Ordered诊断版本）。它与本次新诊断程序的资源身份不能混同。

| 生产 kernel | REG / thread | STACK bytes | LOCAL bytes |
|---|---:|---:|---:|
| `_selfQuery_ee_ccd` | 40 | 264 | 0 |
| `_selfQuery_ee` | 138 | 360 | 0 |
| `_selfQuery_vf_ccd` | 40 | 264 | 0 |
| `_selfQuery_vf` | 162 | 704 | 0 |
| `_edgeTriIntersectionQuery` | 58 | 800 | 0 |

这确认编译后的静态栈帧非零，比仅看到源码 stack[65] 更强；它未区分数组、函数调用与临时对象的贡献，也不证明运行中的动态 spill、访问延迟或瓶颈。Nsight Systems各 query/ACCD 的 localMemoryPerThread=0 与 LOCAL=0 同样不能消除实际 STACK。样本栈高水位低也不能直接缩减当前槽数或保证所有状态安全。

一次有界 Nsight Compute 捕获已完成：新 observer/probe 相同程序、四组件开启、从零120帧，只捕获 sphere 第49帧首个 production `_selfQuery_ee`，一个kernel，raw计数1pass，child exit=0。原collector因期待long CSV而产生 `counter_check_failed / KeyError: Metric Name`，原回执保留；本CPU分析独立读取wide CSV，跳过单位行，未重跑GPU。

| 动态指标：一个真实 DCD EE launch | 实际值 |
|---|---:|
| duration | 1.189248 ms |
| local load sectors | 1,213,799 |
| local store sectors | 615,043 |
| achieved active-warp occupancy | 12.57% |
| long-scoreboard / active warp | 21.98% |
| launch blocks / SMs / waves per SM | 36 / 40 / 0.90 |
| registers requested / allocated per thread | 138 / 144 |

动态 local流量现在有真实证据，但属于完整DCD EE，尚未定位到DFS栈、距离分类临时量或spill指令。long-scoreboard百分比不能转换为等量wall-time收益；36 blocks少于40 SM也造成launch不足，低occupancy不是stackless可以自动解决的证据。该单launch不是所有调用的平均，也不是与probe首样本同一state的计数对照。

最小CPU来源检查已完成：[新程序单函数SASS分析](NEW_EE_SASS_ANALYSIS.md)。新d4da6程序只提取production `_selfQuery_ee`，共4968条静态指令，82处local站点；地址和控制流把4处32-bit站点映射为节点栈（0x120 init、0x360 pop、0x9c30左push、0x13500右push）。其余36处STL.64／42处LDL.64位于R1+0x00..0x58，和从0x60开始的uint32节点栈槽区分开，出现在FP64／CALL邻域。

能区分静态节点栈站点与其他64-bit frame值，却不能严格将所有其他值命名为某个窄相临时对象、caller保存或spill。mlbvh实际编译命令有-lineinfo；当前cuobjdump SASS无源码PC注解，不能写成binary没有line information。NCU没有逐PC执行／local事务／stall计数，4/82的静态站点比例绝不能当动态流量／时间份额。这次有界来源检查到此结束，不再增加GPU计数或扩张追踪。

## 门槛判断与最小下一步

| 原门槛证据 | 当前状态 |
|---|---|
| 完整 query 的整场独占成本≥10% | 未建立：单帧配置不同；整场事件含诊断；DCD未拆出距离分类 |
| stack 控制／访存或长尾可被 escape视图因果削减 | 工作不均衡、静态栈、动态local流量已观察；识别栈指令PC，但动态栈份额与候选收益未测 |
| 减少的关键路径成本 − 生成／维护／失效成本≥whole 5% | 未建立：没有escape视图成本或完整调用覆盖 |
| 局部完整query≥15%、三对整场≥5%、另一主场景≤3%退化 | 没有候选对照，不具备验收证据 |
| 完整碰撞安全／质量认证 | replay oracle与原生守卫通过；独立质量协议仍未认证 |

已完成原提出的最小CPU来源检查。现在更有依据的结论是：DCD EE值得优先调查，但观测到的local成本同时含节点栈和其他frame值，缺少它们的动态份额；DCD父包络余量还混合VF／EE／ground／距离／读回。尚无可信的“可消除时间减维护成本≥whole5%”估计，依原门槛暂缓生产stackless分支，本次有界分析结束。不能把“未达到证明门槛”写成“已证明整个碰撞优化方向无效”。

未来若完整未诊断调用成本建立净≥5%的可信潜力，则依原计划进入单query候选：私有同树／同state的原DFS与escape对照，计数关闭，保持完整距离／tuple oracle，分列视图生成／拓扑失效／refit复用成本。可用实际长尾坐标是 sphere DCD EE 首样本 frame46（node max356，inner0，contact_model226，linear205）／frame50（warp倍率峰值），fixed frame53（node max280，contact_model298，linear276）；它们是数据坐标，不承诺已保存可重放state。后续Newton／backtrack仍需覆盖。严禁把遍历-only减掉窄相当作生产提速，也不能删除树过滤或碰撞候选。

达到门槛后进入候选实施，再完成原约定的三对从零120帧关闭诊断A/B和完整质量验收。当前决定仅限stackless启动门，不排除更好的树布局、保守空间查询或窄相实现；这些方向需各自成本与安全证据。

## 可复现文件

- [CPU分析脚本](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/report56_stage1_20261008/analyze_query_branch.py)
- [聚合数据与完整选择上下文](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/report56_stage1_20261008/QUERY_BRANCH_DECISION.json)
- [单帧成本分析](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/report56_stage1_20261008/PROFILE_COST_ANALYSIS.md)
- [冻结静态资源原始日志](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/report56_stage1_20261008/frozen_resource_usage.log)

- [单次NCU原始wide CSV](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/runs/report56_stage1_20261008/query_counters_f49/counter.csv)

- [新程序SASS来源检查](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/report56_stage1_20261008/NEW_EE_SASS_ANALYSIS.md)

复现：`E:/Anaconda/envs/DL/python.exe reports/report56_stage1_20261008/analyze_query_branch.py`。脚本要求completed ledger／run／evidence身份，读取输入前后核对SHA256，只生成这份Markdown和JSON，不修改native/tools、不调用GPU或构建。
