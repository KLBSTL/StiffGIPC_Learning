# SRBK 融合与双容差补偿 TOI：实现及本机验证

已完成用户选定的两项：**SRBK SpMV＋pᵀAp 融合**，以及**报告中的双容差补偿 TOI 独立入口与状态模块**。本机20次有界运行全部完成，8项既有GPU回归通过；真实23帧窗口覆盖补偿激活、延后验收及一次compensated退出。未取得完整场景速度/质量验收证据，融合默认关闭，legacy仍是默认停止规则。

任务在10月5日冻结身份 `ipc_spmv_compensated_20261005`，跨日继续后保留所有配置、程序和运行身份，结果书日期为10月6日。

## 1. 代码完成了什么

| 项目 | 实现与实际生效 | 边界 |
| --- | --- | --- |
| SpMV 二次型融合 | 原SRBK输出保留；每存储块直接形成二次型partial，host/conditional Graph接入pAp；独立partial/CUB及地址/尺寸图签名 | 不读未完成的Ap原子结果；按既有每块对称扩展支持lower/upper，不筛i<j；原PCG停止和守卫不改 |
| compensated独立入口 | `preset=toi_compensated` 展开IPC接触后端、Graph、compensated/.001/.03/min6；可显式execution=host | 复用已有公式，不重复申报新算法；与AL/Robust的`toi`/`toi_al`分开；不自动开启其他执行组件 |
| 独立控制模块 | `IpcResidualController`拥有固定reference、beta/u/z、pending审计和出口选择；日志离开主循环 | 物理装配、材料、CCD、线搜索不重写；legacy无观测时不创建控制器、不多测残差 |
| 修正的边界 | 动画完成及最少有效更新门控；激活前非法输入sticky-invalid；严格原生整数/布尔解析；normal/terminal记录分开；真实预算与旧beta分列 | 对gated/compensated补齐报告语义，不改变legacy默认行为；movement出口继续优先 |

代码按职责拆分，包含旧路径、配置、生效原因、算子、独立CPU参考、受保护study及状态生命周期。详见 [调试指南](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/IPC_SPMV_COMPENSATED_DEBUG_20261006.md)、[SpMV模块说明](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/sources/stiff_active/StiffGIPC/linear_system/utils/SPMV_FUSED_QUADRATIC.md)、[补偿TOI模块说明](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/sources/stiff_active/StiffGIPC/solver/IPC_COMPENSATED_TOI.md)。

新的SpMV、先前MAS点积融合、普通BVH refit默认均false；旧legacy累计阈值.01、movement .01、min6、PCG rho1e-4、atomic MAS、材料和完整安全CCD不变。命名compensated入口的.001/.03是独立停止实验，不成为全局默认。旧质量协议与历史失败决定保留。

## 2. 构建和程序身份

初始Release构建37单元、11对象改变，SHA `290356b4f0d35bc7185baac374d407a8ddb3c9d6ef934dc0b2341650257854be`，未用于任何GPU实验。GPU前静态复核发现尾部无效线程跳过warp collective但仍参加block collective的问题；修正为全部lane参与，零贡献尾部不写输出，并在组件Graph检查前poison输出/标量/partial，防止旧结果掩盖失效重放。

第二次构建仅更新3个依赖对象，最终程序SHA256：

`5269bb3d4668f016f845a3f48af38fef5ed1538caba5f40057e9d15d79075af4`

所有GPU数据均使用此身份。两次GPU程序构建均0编译错误；没有声称消除了既有编译警告。完整源/对象/包含依赖/链接证据见 [最终manifest](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/builds/active/provenance/20261005T160145_988186Z_ipc_spmv_compensated_20261005_tail_guard/manifest.json)。后续native未再修改。

首个纯CPU test build在默认沙箱因读取已安装Windows SDK元数据被拒失败，自动批准读取后重试成功；两个日志保留。这不是自动审批拒绝或数值失败。

## 3. 验证结果

| 检查 | 实际结果 |
| --- | --- |
| 公共配置/历史兼容 | 15项测试、88子检查通过 |
| 纯CPU补偿控制测试 | 2923个性质/控制/解析检查通过，其中800步预算审计（w=1、w=1/30） |
| SpMV CPU独立检查 | 9个block-count产权边界、Decimal展开矩阵恒等、旧kernel保留、图签名、tail collective及poison检查通过 |
| 既有GPU回归 | 8/8通过 |
| 两布料f2固定系统 | 2/2完成，三种输入各3次，共18次组件比较＋6次私有空矩阵host/Graph检查通过 |
| 两布料短窗 | legacy/spmv/compensated/both × host/Graph × 两场景，共16次3帧，全完成 |
| 控制专项23帧 | compensated+CPU残差核对、legacy+shadow独立terminal两次均完成 |
| 运行合计 | 20/20完成、配置/状态/PCG检查通过，本轮无GPU资源失败或未执行配置 |

GPU为共享桌面RTX3070 Laptop。所有GPU实验串行，显存保留线和120秒上限未改变，无清理或终止用户应用。本轮成功短窗不改判上一轮100帧资源失败。

### SpMV固定系统

悬挂固定矩阵13081个存储块、5817标量；固定兔子39536块、17340标量，均真实覆盖最后一个不满256的block。三输入为实际已解Newton方向、零向量、signed-tail；各host、组件Graph首次、重放，与旧SpMV+dot及独立逐系数CPU展开比较。

实际方向中，新Ap最大CPU差约1.16e-21 / 3.57e-21；二次型CPU差约3.18e-21 / 3.75e-19，均在GPU前声明的FP64贡献累加界内。完整Ap并非逐位一致，符合既有原子顺序可变的边界；没有把数值界限改成材料质量容差。CPU `long double`实际53位mantissa，不称高于FP64的准确参考。

每系统另有私有空矩阵host/Graph首次/replay，Ap和二次型均有限为零。主PCG向量、旧新scratch、RHS、A/M身份、MAS scratch、生产Graph handle/key/计数/参数恢复检查全部通过。16次短窗共128个方向，其中开启spmv/both的8次运行共64个方向，实际requested/effective全部true，无PCG failure。

lower/upper混合扩展有CPU算术覆盖；这两个真实系统lower_blocks=0，不宣称已跑GPU lower fixture。新组件没有再次跑完整零RHS PCG；没有实际容量增长/缓存失效或活动混合ABD非零系统GPU覆盖。这些与组件零向量/空矩阵覆盖分开报告。

### compensated真实控制时序

固定兔子23帧compensated诊断共有89次观察；第23帧3次active观察、2次延后审计、1次compensated退出：

| 状态 | 正接受次数 | reference | 相对残差 | beta | 补偿budget | 决定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 第六更新前 | 5 | 2.2134976741e-5 | 1.000000 | 1 | 1 | 冻结，不退出 |
| 第六步后正常装配 | 6 | 保持 | 0.17287706 | 0 | 0.00576257 | beta已足够，残差/预算未通过，继续 |
| 第七步后正常装配 | 7 | 保持 | 0.02325969 | 0 | 0.00077532 | compensated通过，PCG前退出 |

这两步最终alpha均1，animation_rate=1；全步清除了递推的旧历史影响。本次没有独立历史否决，不能据此宣称历史补偿提供额外质量收益。第七次退出行无PCG，并记录正常装配成本；GPU残差与CPU自由分量核对通过。独立Decimal日志重放核对预算递推、reference、最终alpha、有效接受计数和允许退出条件。

独立legacy+shadow终端诊断共有88次观察；第23帧按旧累计规则在第六次更新后退出，额外terminal装配耗时3.2322ms，normal/terminal两条记录同时存在，审计计数1。终端归一化残差约.1728746，补偿入口的进一步工作是实际发生的，不能包装成减少迭代。compensated与legacy终端两份23帧分别为66/65个方向、1069/1039次PCG；累计阈值及CPU审计开关不同，不构成性能配对。

20次运行共265条残差记录全部逐条复核，normal/terminal命名字段与数组顺序一致，接受计数和门控布尔重算一致。收尾加强了CPU审计器：缺失数组、漏行、重复和乱序不再静默跳过；只重审原JSON，不修改GPU数据或冻结协议。

动画拒绝、min边界、非法输入、历史独立否决等由生产控制器CPU测试覆盖；自然GPU样本中动画均已完成，没有故意修改物理场景来制造门控失败。

## 4. 质量与计时只限三帧诊断

三个startup帧尚未激活六次接受更新后的补偿预算。两布料所有四配置host/Graph的PCG与方向数一致：悬挂129次PCG/9方向，固定兔子75次PCG/7方向。全部导出位置/实际速度有限，无PCG触顶/失稳。相对同execution的legacy逐体比较：

| 场景 | 八臂最大拉伸范围 | 最大位置差 | 最大实际速度差 |
| --- | --- | ---: | ---: |
| 悬挂 | 1.004295202003–1.004295202140 | 6.76e-11 m | 6.49e-9 m/s |
| 固定兔子 | 1.000002360639–1.000002360642 | 2.84e-11 m | 2.76e-9 m/s |

位置/速度分cloth/FEM/ABD保存在分析JSON，表中为分体最大值。局部差异小不等于接触发展后的100帧材料协议通过；没有新增独立接受路径CCD审计。

下表是每臂一次、完整3帧、含初始化/捕获/必要残差观测的 **legacy耗时/该配置耗时**，仅诊断，不作为最终加速比或推广条件：

| 场景/执行 | SpMV | compensated | 两者组合 |
| --- | ---: | ---: | ---: |
| 悬挂host | 1.056 | 1.046 | 0.998 |
| 悬挂Graph | 1.083 | 0.873 | 1.037 |
| 固定兔子host | 1.044 | 0.973 | 1.013 |
| 固定兔子Graph | 0.969 | 0.998 | 1.108 |

没有交错重复和置信区间，停止规则也尚未在这三帧发挥作用，不能把波动算成TOI加速。真实23帧两条重诊断轨迹观测方式不同，也不能直接相除当性能对照。

## 5. 结论与后续条件

| 判断 | 状态 |
| --- | --- |
| 独立SpMV融合已实现并实际接入host/Graph | 已验证 |
| 两个固定系统及零/空矩阵组件结果、恢复正确 | 已验证（局部范围） |
| compensated独立入口、状态时序和真实预算退出 | 已验证 |
| compensated减少当前IPC迭代 | 不支持；真实样本多一次更新 |
| 历史补偿独立质量收益 | 未证明，本轮自然GPU样本无独立历史否决 |
| SpMV稳定整场收益、100帧材料/CCD质量 | 待验证 |
| 新默认组合 | 不推广，legacy及组件默认保持 |

本轮只执行预声明有限检查，无100/300帧、AutoDL或参数网格。后续先补非零固定PCG整体作用/真残差与实际Graph增长失效；若局部准备＋求解没有净收益就停止融合推广。补偿模式保留清楚入口及诊断能力，速度和质量仍须与相同执行组件的legacy分别验收，不以更严格退出的更小残差代替材料质量收益。

精确配置、命令、scope和检错见调试指南；原始数据见 [分析JSON](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_spmv_compensated_20261005_analysis.json)、[CPU检查](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_spmv_compensated_20261005_cpu_checks.json)、[协议](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_spmv_compensated_20261005_protocol.json)。交付核验见 [verification](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_spmv_compensated_20261005_verification.json)，原460项索引和历史决定不改，仅追加本轮25项证据。
