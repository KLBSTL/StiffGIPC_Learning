# 装配与现有执行组合：独立选择审查

2026-10-06。只读审查；没有运行 GPU、构建、修改原生源码或冻结运行器，没有追加 Nsight 捕获。

**结论：现在不启动新的 Hessian/排序 kernel。唯一建议是用已有开关做普通 BVH refit × FullCCD refit 的四格有限消融。** 两个开关会改变扫掠树拓扑的实际刷新频率，不能把各自收益简单相加。它是选组合的诊断实验，不是已证明的优化，更不是 2× 承诺。batched energy、energy reuse 与 conditional PCG 保持当前设置；不恢复任何退休组件。

## 1. 本轮证据和成本上界

读取 `reports/local_step4/PROFILE_OFF.json`、`PROFILE_ON.json`、`PROFILE_REVIEW.md`，以及两个 `runs/local_step4_profile_fixed_{off,on}_node_20261006` 的 requested/resolved/stats。它们使用同一程序 `b2ad19884fc01b81827d560f9f1984346b5488851995c2fd495f7427eac0f812`、同一 source digest `9e57e82e14d58703348e951be92c0652b958428ddbdc2295e6fecabbff62b738`。均从零完成固定兔子 59 帧，仅捕获 f57。OFF/ON 指 contact_pool，其他配置一致，但轨迹、接触数和线性系统不同，不能当冻结 A/b 对照。

下表取 pool OFF。CPU scope 为 inclusive 包络；GPU 值为实际 kernel activity sum。两类不能相加；同类父子也不能重复相加。

| 项目 | f57 ms | 占 CPU 物理帧 161.738546 ms 的比例 | 判断 |
|---|---:|---:|---|
| gradient/Hessian assembly，CPU | 13.470796 | 8.329% | 包含接触、ABD 转换、材料、同步；不是单 kernel |
| 最终 global matrix conversion，CPU | 7.636593 | 4.722% | 全部删除都不到本帧 5%；实际不可全删 |
| triangle gradient/Hessian，GPU | 6.063095 | 3.749% | 最大单个材料装配核；没有实测支持可省绝大部分 |
| energy evaluation，CPU，7次 | 3.149944 | 1.948% | 当前 batch/reuse 已生效，不优先重写 |
| MAS hierarchy，CPU | 4.105707 | 2.538% | 当前有接触，不能使用无接触拓扑复用 |
| 普通 VF/EE，GPU，6+6次 | 21.635034 | 13.377% | 树遍历是真实工作；已拒绝的 eligibility 不复活 |
| swept VF/EE，GPU，6+6次 | 8.669873 | 5.361% | 完整安全 CCD 必须保留 |
| 全部 BVH kernel，GPU | 1.613785 | 0.998% | 不能只看 build 下降而忽略 query 增长 |
| edge–triangle 安全检查，GPU | 9.784111 | 6.049% | 四格均保持完整调用与谓词 |

普通+swept查询+全部BVH合计31.918692ms，约帧包络19.735%。这只是有关工作集合的设备活动预算：要减少5%帧耗时，理想串行情形也需净省8.086927ms，即这部分约25.34%，还须抵扣更多重建。没有证据已经能达到这个幅度。edge–triangle也可能随普通树拓扑改变成本，但不把它额外承诺为可省收益。

若 triangle kernel 真能提速2×，理想帧收益也只有约1.87%；最终 conversion 即使提速2×，理想包络相减约2.36%。这些局部数值均不能填补整场2×缺口。WDDM与node trace开销、工作量分岔进一步限制外推。

## 2. 装配、转换和排序的源码依赖

- `StiffGIPC/core/GIPC.cu:10313`：`computeGradientAndHessian` 清空梯度，计算当前接触、摩擦/地面贡献；`:10393–10395` 更新接触 hash 并调用 `partitionContactHessian`。接触集合和摩擦历史会变，不能假定整个稀疏结构不变。
- `core/GIPC.cu:9880`：partition先 radix sort 接触类别，再 reorder，并在`:9938`起读回四个分区起点，供后续不同 ABD/FEM 写入区间使用。这是**类别/归属分区**，不等价于最终全局矩阵的 (row,col) 去重。四次标量读回存在，但目前没有将其可消除等待与前序GPU工作拆开的证据，不能把API阻塞时长全算为可省成本。
- `abd_system/abd_system_function/setup_abd_system_gradient_and_hessian.cu:584`：先压缩顶点级 ABD–ABD 接触块；`:602`起据压缩数量预留容量，`:650`附近通过 J 映射扩展到 ABD 自由度，并写入刚体自身 Hessian；`:476`外层再次合并转换后的 ABD 块。两次操作处于**映射前/映射后**，不是相同输入上的重复排序。ABD–FEM块也在同一函数中单独转换后加入。
- `core/GIPC.cu:10429`起追加 FEM/布料/软约束/质量对角，保留2F不重叠 staging容量；`:10463`弯曲、`:10495`三角膜材料根据当前位置计算。
- `fem/femEnergy.cu:2146`：三角核从当前位置构造Ds/F、应力和投影Hessian，再写六个3×3块。预计算 rest inverse、面积不意味着当前投影材料 Hessian 是常量。
- `linear_system/linear_system/global_linear_system.cu:369`：最终 `convert_new` 对完整矩阵排序/合并并暴露 live unique block 前缀。`:391` Graph signature绑定块值、行列地址和unique数量。
- `linear_system/utils/converter.cu:138,186,247`：验证不重叠 staging→hash/index→radix sort→按排序索引搬值→分区scan→读回unique数量→清零prefix→`FastSegmentalReduce`。缓存排序映射需要证明输入顺序/键不变，不能只证明材料拓扑不变；改变分段规约顺序还涉及浮点输出验证。

Step4实际共有18次 `set_dst_val_kernel`、18次分段规约，即六个方向各三次转换。它们跨ABD准备与最终matrix_convert；不能把18次全部归到六个最终转换scope。搬值GPU合计2.509788ms、分段规约0.722591ms。全帧radix onesweep 1.939869ms还包含其他排序来源，不能无相关性分析就全归Hessian。现有证据不足以删除上述阶段或支持一个小范围的新装配kernel达到5%净收益。

## 3. 哪些保留组件实际开启

| 组件 | 当前实际配置/执行 | 边界 |
|---|---|---|
| conditional PCG | 开启 | 不改变rho/previous-rho停机时序；不再研究图外初始化 |
| FullCCD refit | requested/resolved=true；f57每棵树refit6、完整重建0 | D=1时长期保留扫掠拓扑，无周期重建 |
| 普通BVH refit | enabled=true，interval=8；f57每棵树refit5、重建1 | 保留普通/扫掠独立storage |
| batched energy | true | `core/GIPC.cu:10729`九类partial/CUB后一次读回；ABD能量仍单独计算；未更改材料 |
| accepted energy reuse | true | `core/GIPC.cu:11336`复用且`:11347`要求Kappa不变；`:10941`缓存缺失时重算；当前7次能量与6次更新吻合 |
| 无接触MAS拓扑reuse | **false** | 不能说combined已开启该项 |
| contact pool | 下一选择固定false | Step4净省4.783%未达组件门槛；不继续池维护优化 |

无接触MAS的证据链：`tools/bench/config.py:216`清掉全部环境GIPC项，`:246`设置`GIPC_ACCEL_SUITE=0`，`:257–258`只显式映射refit/batch/reuse；未设置`GIPC_MAS_STATIC_TOPOLOGY`。`core/accel_features.h:4`因此解析false，两个实际resolved也都明确为false。`solver/toi_options.h:202`仅报告该项，不会开启它。

即使另行开启，`solver/MASPreconditioner.cu:2439`要求cpNum==0、owner相同且已有有效无接触层次；`:2465`一旦有接触就将缓存标为失效。`linear_system/preconditioner/fem_mas_preconditioner.cu:62–74`传入当前接触计数；f57六个方向active_pairs均非零（OFF1089/1252/1224/1161/1136/1104）。初始化`:2642`也失效缓存，`:2516`的数值矩阵准备仍每方向执行。该项只能研究无接触前缀，不能作为接触布料优化支线，本轮不新增开关或启用。

## 4. 唯一推荐：BVH现有组合四格消融

假设限定为：**省去Morton重建可能使后续查询变贵；独立双树缓存改变了FullCCD refit的拓扑寿命。** 不是假定refit错误，也不是预先宣布关掉更快。

源边：

1. `core/GIPC.cu:9050`仅按`GIPC_CCD_BVH_REFIT`选择 `RefitFullCCD` 或 `ConstructFullCCD`。
2. `collision/mlbvh.cu:2467` / `:2584`：D=1时选择独立swept storage，仅签名/容量/有效性失败才重建；否则更新叶AABB及内部AABB，保留Morton顺序。没有周期阈值。
3. `collision/discrete_bvh.inl:66–124`：storage交换和signature；`:172–196`普通树按interval8重建。D=0则普通query前 `ConstructRebuild` 刷新**同一棵树**，随后S=1基于该普通拓扑作swept refit。D=1/S=1则扫掠拓扑可跨很长时间保留。因此需四格而非把两个单独组件收益相加。

| 臂 | discrete_bvh_refit (D) | refit / FullCCD (S) | 其他差量 |
|---|---:|---:|---|
| A，当前参考 | true | true | 无 |
| B | true | false | 无 |
| C | false | true | 无 |
| D | false | false | 无 |

所有臂固定IPC/conditional Graph/legacy MAS、batch=true、reuse=true、contact_pool=false、材料/dt/停止规则/资源预算不变；interval保持8，不扫描参数。这里只给选择方案，不修改或扩展旧seal/协议。旧seal两次profile槽已耗尽，本方案不含新capture。

有限执行建议，供主计划统一裁定：两布料各从零跑已声明接触前缀（悬挂51、固定兔子59），先正序A/B/C/D，再反序D/C/B/A，共最多16次；每次沿原120秒及显存/磁盘/外来GPU保护。先分析第一遍再决定是否执行第二遍。任一资源失败保存为失败、不用前缀算比值、不自动重跑。每场景最多一个非A胜者；确有同方向净省后，仅追加该胜者与A的一次反序配对，两场景最多4次，总上限20次。父计划更小预算优先；这不是自动运行授权或新增性能网格。

必须比较整段和接触段 wall、Newton/PCG/accepted alpha、普通/完整CCD候选与线搜索工作量、BVH实际统计、峰值内存；不能只用BVH构建毫秒选胜者。用相同轮次参考A配对；当前A也不能因是参考而默认质量通过。

## 5. 验证和有限决定

无需新算子或六份MAS输入重放：上述消融没有改A/M数学、PCG代码或新scratch，目标是现有树构建策略的选择。仍需当前同程序组件夹具、BVH/refit守卫、零/一leaf、普通/扫掠切换及容量增长已有测试结果；新组合不能借其他程序身份的通过替代。

正确性先于计时：

- 独立诊断组沿现有`component_audit`核对refit与完整重建候选，沿`discrete_bvh_validate`核对普通 typed multiset及叶bounds；关闭这些重型诊断再计时。
- **已有FullCCD audit限制必须保留：**`core/GIPC.cu:9060–9067`排序后`unique`，只核set，且每帧只审一次。它不能单独证明每次query的typed多重性、原始顺序、最终alpha等价。若计划需要更强同状态保证，应先声明缺口并单独设计小诊断；本审查不改原生补探针，也不拿该set审计冒充全覆盖。
- 完整safe CCD、`isIntersected`、ground、fixed/body过滤均保持；不删碰撞对。BVH遍历顺序可以改变原子累加和后续轨迹，应分别报告实际alpha/工作量与材料质量，不能要求整个轨迹逐位相等，也不能放宽旧质量界限。
- 固定兔子旧质量门槛未全通过的事实继续保留；候选若未过仍只能给诊断速度，不能晋级质量/2×认证。无非有限、PCG预算触顶、固定漂移等硬失败；布料/ABD/FEM分项按原协议。

晋级只考虑三次完整配对、同方向稳定且预声明整段净省至少5%、现行质量要求通过的组合；否则保留A/报告无收益，结束本分支，不再扫描重建间隔、不重开退休kernel。即使某场景胜出，它也只确定当前执行组合，后续相对Stiff、100/300帧及2×仍独立待验证。现有单帧成本不能预言结果。

本报告只新增自身文件。结论分类：组件实际配置与成本已证明；双树改变拓扑寿命由源码支持；重建可能改善遍历是待验证假设；新Hessian/排序实现的5%启动门槛目前未满足。
