# 组件优化、改错及本机配对结果

已修复普通 BVH refit 因与 FullCCD 共用树而始终不能复用的限制。现在两套缓存独立，生产普通 refit 命中约87.5%，完整碰撞检测及原停止规则保留。本机70次场景运行全部完成，8项GPU回归通过。100帧新组合相对同轮原版 Stiff 的配对中位加速为 **悬挂1.737×、固定兔子1.419×**；仅本轮BVH修订相对旧执行组合为 **1.089×、1.005×**。

没有取得2×、同质量受控负载或新默认推广证据。悬挂获得可见增量，固定兔子完整窗口收益很小；MAS和SpMV融合没有纳入本轮最快组合。新开关默认仍关闭。本轮不继续扩大参数或修订次数。

## 1. 修改与检错

| 工作 | 结果 | 证据状态 |
| --- | --- | --- |
| 普通/swept缓存分离 | 8个拥有缓冲及scene完整交换；各自保存签名、有效状态；首次或失效FullCCD refit先完整构建 | 已验证：代码身份、自然调用与局部守卫 |
| 缓存生命周期 | init/MALLOC/FREE/显式拓扑失效处理两套状态；默认off不分配备用缓存；公开节点对应最近操作 | 已静态复核；动态改映射地址/数量等分支GPU覆盖待验证 |
| 查询正确性 | 全部当前bounds刷新，不复用接触集；原GPU query及CCD helper不改；原ID筛选保留 | 非空同状态查询通过，完整接受路径独立审计待验证 |
| 线性融合审查 | 没有发现空complement kernel或重复partial清零；SpMV少一次向量遍历，但增加SRBK计算和块归约 | 不做无证据小改动，实测筛选决定保留 |
| 状态比较工具 | 新分析入口显式报告冻结原版没有真实速度导出；有数据时逐体比较，缺失不估算、不当零 | 已验证：12个原版配对明确标记速度不可用 |

原版和当前 IPC 的 legacy累计阈值.01、movement .01、min6、PCG rho1e-4、atomic MAS、材料及完整CCD均不改。本轮快配置是IPC执行优化，不启用AL-TOI或双容差补偿。补偿只做了一次23帧组合兼容，仍真实触发一次compensated出口、多一次接受更新；未重开停止条件调参。

负责代码位于 [mlbvh.cuh](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/sources/stiff_active/StiffGIPC/collision/mlbvh.cuh)、[mlbvh.cu](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/sources/stiff_active/StiffGIPC/collision/mlbvh.cu)、[discrete_bvh.inl](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/sources/stiff_active/StiffGIPC/collision/discrete_bvh.inl)。所有权、失效、计数及调试合同见 [模块说明](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/sources/stiff_active/StiffGIPC/collision/DISCRETE_BVH_REFIT.md)。

## 2. 程序、预算与覆盖

本轮只有一次BVH执行修订，没有新的线性kernel或收敛调参。构建37单元、11对象重编，程序SHA256为：

`fb3b46246a710febecd8badb9e5b1fcf82125fc5deebefa0b9b2c19cd6017d1d`

源/对象/包含依赖及链接记录见 [manifest](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/builds/active/provenance/20261005T164849_797592Z_ipc_component_tuning_20261006/manifest.json)。37个编译单元均记录身份，布局改变的依赖对象重编；GPU运行期间native冻结，0编译错误，不声称消除了既有警告。此前`5269bb3d…075af4`源码/程序archive、485项历史实验及旧决定保留。

| 阶段 | 完成/请求 | 范围 |
| --- | ---: | --- |
| CPU配置契约 | 15项/88子检查 | 默认、命名入口、旧配置及严格参数 |
| 既有GPU回归 | 8/8 | 组件、PCG守卫、六历史系统 |
| 局部guard | 4/4 | 悬挂3帧查询、固定23帧非空查询、混合3帧、固定23帧compensated＋三个组件 |
| 筛选 | 42/42 | 两布料×7臂×3轮，从零23帧 |
| 主对照 | 24/24 | 两布料×4臂×3轮，从零100帧，dt=.01 |

本轮无资源失败、未执行配置或PCG触顶/breakdown。RTX3070 Laptop共享桌面，既有显存保留、120秒及串行锁不变；没有清理数据、终止用户应用、连接AutoDL或运行300帧。短窗以两场景几何平均≥1.05且各场景不回退3%为预声明长测条件，选择及完整计划在100帧前冻结，没有按长测结果选最快臂。详见 [计划](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/IPC_COMPONENT_TUNING_PLAN_20261006.md)。

## 3. 筛选：不是所有组件同时开更快

旧组合为Graph＋FullCCD swept refit＋批量能量＋能量复用。表为旧组合耗时/候选耗时，各场景三轮配对中位，仅23帧筛选：

| 增加组件 | 悬挂 | 固定兔子 | 两场景几何平均 | 本轮决定 |
| --- | ---: | ---: | ---: | --- |
| MAS final-dot | 1.002× | 1.021× | 1.011× | 未达到筛选收益，停止推广 |
| SpMV+pAp | 0.983× | 0.974× | 0.978× | 两场景略慢，停止推广 |
| 普通BVH独立缓存 | 1.103× | 1.041× | 1.072× | 唯一进入100帧 |
| 三组件全开 | 1.027× | 1.033× | 1.030× | 弱于BVH单项，停止推广 |

全部42组短窗工作量一致：悬挂每组2911次PCG/107方向，固定每组1039次PCG/65方向。由此短窗差异来自执行与共享负载，不是减少求解任务。有限融合有额外计算/归约成本，不能用减少kernel数推断必然加速；此处实测也不足以把慢化全部归因于某个算子。

## 4. 100帧性能与成本

速度列为**同轮比值的中位**，不是时间中位相除。四臂均100帧、相同材料/停止条件/预算、无重型诊断；加载和导出单列于raw result，求解计时包括图管理和线性准备。

| 场景 | 新BVH组合秒中位 | 相对Stiff | 相对Graph-only | 相对旧组合 |
| --- | ---: | ---: | ---: | ---: |
| 悬挂 | 5.3975 | 1.7366× | 1.2369× | 1.0894× |
| 固定兔子 | 18.8258 | 1.4187× | 1.0514× | 1.0051× |

相对旧组合的三个配对值：悬挂`1.1202/1.0894/1.0853`，固定`1.0199/1.0051/.9982`。两场景增量几何平均1.0464×，小幅改善；固定兔子没有足够证据满足5%整场推广门槛。

| 参照 | 两场景几何平均 | 单侧95%诊断自助法下界 |
| --- | ---: | ---: |
| Stiff | 1.5696× | 1.4744× |
| Graph-only | 1.1404× | 0.9621× |
| 旧执行组合 | 1.0464× | 1.0408× |

只有三个配对轮次，自助法采用三整轮重采样、两场景按轮聚类，穷举27种；不把小样本下界当受控GPU认证。Graph-only比较中悬挂第三对为.8805×，共享负载波动明显，Graph-only速度要求的置信下界没有超过1；加上质量协议不可认证，本轮阶段目标仍未通过。2×挑战目标未达到。

正常100帧每树：悬挂443次Construct中388次refit、55次周期重建；固定590–602次中517–527次refit、73–75次周期重建。普通`swept_rebuilds=0`，FullCCD首次fallback构建一次，之后refit持续工作。40个运行/树组合逐帧满足周期8恒等式，未把诊断主动refit计为生产收益。

备用swept容量实测：悬挂1.876MiB、固定16.093MiB、混合21.898MiB（最后仅3帧guard）。此前按所有buffer统一1.5倍得到的306N估算过大；实际单树初次为`204N-68+12 floor(N/2)`字节，约210N。只有排序scratch采用1.5倍增长，文档已纠正；native计数一直正确。

阶段中位ms支持悬挂主要改善在线搜索段：旧组合1669.7→1253.9ms，CCD570.9→572.1ms，线性2706.0→2618.8ms。固定线搜索5563.7→5223.0ms，但线性8531.1→8690.5ms、CCD1652.5→1736.1ms。线搜索包含普通碰撞构建/查询；没有细化kernel计时，不把整个段下降都归给建树。长窗Newton/PCG已有差异及负载变化，同段计时不是同工作量微基准。

## 5. 质量与轨迹差异

冻结质量协议SHA仍为`1cfb9045d6988fa606b8bb76afdd71e20661ef602c84296c49cdda9a0d07d78f`，范围不扩张。候选6/6通过原整段材料阈值，但固定原版自身0/3通过，因此不能认证同质量加速。速度数字保留为诊断，不用通过的候选轨迹覆盖基线失败。

| 材料通过次数/3 | Stiff | Graph-only | 旧组合 | 新BVH组合 |
| --- | ---: | ---: | ---: | ---: |
| 悬挂 | 3 | 3 | 3 | 3 |
| 固定兔子 | 0 | 3 | 2 | 3 |

固定原版三次均p99越界，一次同时最大拉伸越界；旧组合一次最大拉伸越界。新组合最大拉伸范围：悬挂1.1288762016–1.1288762136，固定1.0354432135–1.0356971866；固定p99为1.0132665666–1.0133898344。全部70次导出位置有限，PCG无cap/breakdown，混合guard无ABD翻转；这不等于所有时刻的独立接受路径CCD审计通过。

每轮新BVH与旧组合的逐体最大位置/实际速度差：

| 连续范围 | 悬挂位置 | 悬挂速度 | 固定兔子位置 | 固定兔子速度 |
| --- | ---: | ---: | ---: | ---: |
| 23帧 | 1.51e-6m | 9.82e-5m/s | 2.73e-9m | 2.45e-7m/s |
| 100帧 | 6.40e-4m | .05131m/s | .17415m | 3.09355m/s |

这些是所有配对与帧的单点最大值，不是总体RMS。布料/FEM/ABD分项保存在分析JSON。固定兔子后期分岔明显，不能据材料通过便称轨迹一致。原版自身三次最大位置差.21068m；旧组合自身.17168m/3.11625m/s，新组合自身.14450m/2.02124m/s。这些只说明本项目存在显著重复分岔，不作为看过候选后放宽的质量容差，也没有证明差异必然由BVH或任何单个模块造成。

冻结原版没有真实速度文件，本轮12个与旧组合的原版配对及原版重复比较明确标记不可用；没有用位置差分估算速度。活动程序的100帧实际速度完整、有限。原版真实速度同轮基线仍需独立观测构建取得，不能继承其他身份旧运行的数据。

固定兔子非空query guard两树各65次全部通过，共29face＋39edge类型化多重集条目；16＋12次增长为诊断scratch重试，非生产接触缓冲扩容。自然周期、FullCCD首次fallback及缓存切换有运行证据；动态地址/数量/原地映射变化和新100帧独立CPU接受路径CCD未覆盖，不继承历史零标记。

## 6. 保留配置和下一步边界

值得保留的**诊断配置**为`preset=combined`、`discrete_bvh_refit=true`、周期8，MAS/SpMV融合false，IPC legacy。悬挂有可重复的本轮增量，后续受控硬件复测有价值；固定兔子当前瓶颈仍主要在线性与Newton/线搜索工作，继续调整BVH周期或把SpMV强行开启不会自动跨越收益门槛。

本轮停止MAS/SpMV/全开组合的推广，停止固定兔子BVH速度分支扩参。源码作为明确可关闭的组件保留，默认配置不推广。后续必须以完整成本和质量缺口为依据，而不是再次枚举组合、加上限或放宽rho；本轮不启动新的算法重写或无界确定性任务。

精确命令（项目根目录，PowerShell7）：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/active/build.py build --label ipc_component_tuning_20261006 --jobs 2
& 'E:/Anaconda/envs/DL/python.exe' tools/active/component_tuning.py prepare
& 'E:/Anaconda/envs/DL/python.exe' tools/active/fixtures.py --name ipc_component_tuning_20261006_fixtures
& 'E:/Anaconda/envs/DL/python.exe' tools/active/component_tuning.py run --stage guards
& 'E:/Anaconda/envs/DL/python.exe' tools/active/component_tuning.py run --stage screen
& 'E:/Anaconda/envs/DL/python.exe' tools/active/component_tuning.py select
& 'E:/Anaconda/envs/DL/python.exe' tools/active/component_tuning.py run --stage full
& 'E:/Anaconda/envs/DL/python.exe' tools/active/component_tuning.py analyze
& 'E:/Anaconda/envs/DL/python.exe' tools/active/component_tuning_evidence.py
& 'E:/Anaconda/envs/DL/python.exe' tools/active/verify_component_tuning.py
```

输出名不可覆盖，新实验须另冻结身份与预算。首次离线分析因冻结原版缺速度文件失败，修正新比较入口后只重审既有数据，没有重跑GPU或修改结果。数据见 [selection](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_component_tuning_20261006_selection.json)、[analysis](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_component_tuning_20261006_analysis.json)、[详细证据](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_component_tuning_20261006_evidence.json)、[交付核验](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_component_tuning_20261006_verification.json)。485项历史索引与旧决定保留，新增本轮实际运行、局部结论及推广否决，未改判历史质量或资源失败。
