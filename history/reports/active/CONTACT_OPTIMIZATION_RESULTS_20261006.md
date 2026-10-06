# 接触布料优化：本轮成本、实现与决定

本轮从零完成两种有barrier接触的布料100帧基线、一个独立CCD执行候选、同状态正确性检查、三轮交错短窗及两个接触帧的Nsight节点追踪。**当前组合相对官方Stiff为1.730×／1.486×，尚未达到2×。新CCD候选没有形成稳定净收益，默认关闭并封存，不进入新的100／300帧候选测试。**

全部速度是RTX3070 Laptop共享桌面的诊断值。100帧基线每场景只有一对，不是受控GPU性能认证。物理质量、轨迹差异和局部算子等价分别评价。

## 1. 身份、配置与真实接触窗口

| 项目 | 实际身份／配置 |
|---|---|
| 官方Stiff | SHA256 `1c175da5a664bbc98657ddb6690a9902d821732c19e9b18f1f3ce0e868e05205` |
| 本轮100帧当前组合 | SHA256 `a9465968046f18472f2fd5085c5849b5575df012f0d343ed62167471d87b370b` |
| 新组件fixture、guard、screen、profile | SHA256 `b1a1ac4506adcf999b073100a6853c625d37dd78594d91df35d33d08146b6497` |
| 新构建 | Release，37编译单元，5个对象变化；实际命令、include依赖、对象与链接输入已保存 |
| 当前组合 | IPC＋conditional PCG Graph＋FullCCD refit＋批量能量＋能量复用＋普通BVH refit（周期8） |
| 停止／模型 | legacy，累计阈值.01，min6，PCG rho阈值1e-4，dt=.01；材料不变 |
| 新／旧实验开关 | bounded_ccd默认false；eligibility、MAS点积融合、SpMV二次型融合保持false；没有切换AL TOI |

悬挂 `cloth_hang_l` 在本次f41–50有活动接触，f51脱离，短测从零51帧。固定兔子 `cloth_fixed_bunny_l` 在f22–100持续活动接触，重点f45–59，短测从零59帧。窗口在候选运行前冻结。`geometric_contact=false`来自更严格的距离分类，不能据此否定IPC barrier接触；native narrow pairs及Newton active_pairs另列。

基线程序与新程序分别保存身份，不能将新程序的短窗结果冒充其100帧结果。所有运行保留requested、resolved（活动程序）、程序及输入哈希、逐帧工作量和GPU采样。GPU串行，既有显存保留线和120秒单次上限没有放宽。

## 2. 完整100帧成本

| 场景 | Stiff / 当前组合 | 相对Stiff | 到2×还需减少当前时间 |
|---|---:|---:|---:|
| 悬挂 | 9.1698s / 5.3006s | 1.72995× | 0.7157s，13.50% |
| 固定兔子 | 29.5393s / 19.8840s | 1.48558× | 5.1144s，25.72% |

当前组合的阶段分解如下。事件区间包含提交间隙，是求解阶段耗时，不能直接当成纯kernel执行时间。线性阶段包含转换、MAS准备、PCG和解分发。

| 阶段 | 悬挂：秒 / 总时间占比 | 固定兔子：秒 / 总时间占比 |
|---|---:|---:|
| 已计装配 | 0.5261 / 9.93% | 2.4394 / 12.27% |
| 线性准备＋PCG | 2.4809 / 46.80% | 9.3711 / 47.13% |
| CCD | 0.6233 / 11.76% | 1.7979 / 9.04% |
| 线搜索 | 1.2460 / 23.51% | 5.5767 / 28.05% |
| 状态更新 | 0.1177 / 2.22% | 0.1235 / 0.62% |
| 退出轮装配（另计） | 0.0472 / 0.89% | 0.0449 / 0.23% |
| 扣除上项后的未归类余量 | 0.2595 / 4.90% | 0.5306 / 2.67% |
| 总计 | 5.3006 / 100% | 19.8840 / 100% |

退出轮装配没有再次算进原assembly；未归类余量包括帧级初始化及观测范围之外工作，不能自动解释为可删除CPU开销。所有时间恒等式已核验。

接触段相对Stiff为悬挂f41–50 **1.54407×**，固定f22–100 **1.50706×**；固定f45–59仅 **1.42446×**。悬挂接触段线性35.97%、线搜索27.46%、装配16.97%、CCD14.54%；固定持续接触段线性47.78%、线搜索27.69%、装配12.34%、CCD8.93%。不能用悬挂无接触后期掩盖接触段成本。

按不改变当前工作结构的粗预算，只靠线性阶段需要分别净减其约29%／55%才能到2×；即使把整个CCD阶段清零，也仍不足两场景2×。因此新CCD是一次有界冗余验证，不是2×的完整方案。

## 3. 本轮实现的组件

新增 `collision/ipc_bounded_ccd.h/.inl/.md`，配置、CCD算式与诊断入口分开；`ipc_bounded_ccd_observer.h`单独记录host计数；GIPC只负责第二次swept调用和最终CFL检查。公共runner显式记录 `bounded_ccd` / `bounded_ccd_validate`，冻结程序禁止请求这个组件，旧配置比较不会改写历史哈希。

旧路径按temp_alpha构造swept候选，第二次ACCD仍可能推进到更远，最后再取min(temp_alpha, step)。新路径仅在0<temp_alpha<1、ccd_size=1时生效，保留全部候选、距离分类、原算式、原gap退出、原toc累加及原toc>1返回顺序。在有限、非负、单调前缀越过请求上界且实际倒数往返值也达到上界后，才结束不再影响该请求的推进。第一遍、边界移动和AL TOI原样。

不能无条件声称未执行后缀的浮点行为等价。验证以原导出设备函数为真值，分别核对逐pair有效步长、完整倒数／Max／倒数归约、实际生产输出及CFL后的最终alpha。计数副本须与原函数／生产版本交叉核对；诊断使用独立scratch，不覆盖生产输入及队列。无效请求走旧函数；非有限几何在诊断中明确拒绝，不将旧函数的偶然有限返回视作几何合法。

另补充 `mas.numeric_fill`、`mas.numeric_aggregate`、`mas.legacy_factor`、`mas.inverse64_factor` 四个scope，分开原来的数值填充、层级聚合和求逆。正式速度组关闭成本观测，Nsight使用cost_events=false，不新增观测event／flush同步。Graph内按实际节点符号分类，未识别符号及通用CUB操作保留为未知。

## 4. 正确性与工作量

17项公共配置测试通过，分析器CPU自检及旧SQLite核对通过；新15组GPU CCD fixture（859对）与8项既有GPU回归全部通过。覆盖PT／EE、零运动、接近／分离、近接触、擦碰、近平行／薄三角形、0／1／255／256／257数量及0／1／负／NaN／Inf请求，两个非有限几何样本按预期拒绝。输入复制回读逐位不变；更广泛退化几何、实际缓存增长及300帧仍未覆盖。

| 从零guard | 第二遍查询 / 有效新查询 | 验证pairs | 截断pairs | 旧 / 新诊断ACCD迭代 |
|---|---:|---:|---:|---:|
| 悬挂51帧 | 152 / 36 | 420,733 | 119,892（28.50%） | 798,699 / 442,581（减44.59%） |
| 固定兔子59帧 | 200 / 31 | 1,578,152 | 50,647（3.21%） | 1,666,942 / 1,603,984（减3.78%） |
| 混合体3帧 | 0 / 0 | 未触发 | 未触发 | 未触发 |

两布料67次同状态核对共1,998,885对，逐对有效步长、完整归约、生产步长及最终alpha均逐位一致，零失败。混合3帧只是兼容启动，不证明新组件覆盖活动ABD／FEM接触。

固定兔子的原查询本来平均每对仅约1.056次诊断迭代，能省的工作很少。悬挂的请求上界中位.1161，固定为.5454；悬挂较短请求有更多可截断后续工作。上述仅覆盖生效查询，计数副本迭代不是整场时间，也不是Newton／PCG减少。

## 5. 三轮交错速度与质量

| 场景 | r1 off/on | r2 off/on | r3 off/on | 配对中位 | 固定接触窗配对中位 |
|---|---:|---:|---:|---:|---:|
| 悬挂51帧 | .99863× | .99148× | 1.04662× | **.99863×** | 1.01495× |
| 固定兔子59帧 | .98730× | .96120× | 1.05356× | **.98730×** | .98542× |

全部12次完成，无PCG触顶／breakdown／非有限位置。悬挂方向数均297，PCG9,101–9,105；固定方向316–329，PCG18,516–19,365。固定PCG工作量在配对中变化，速度差不能全部归给CCD kernel。第三轮多个阶段一起变快，也不能取这一轮的最佳值推广。

悬挂CCD阶段配对中位约1.080×，但仅这一阶段的改善被其他阶段及负载变化抵消。固定CCD阶段中位约.966×。**局部44.6%迭代减少未转化为稳定整场收益，固定兔子削减空间本来就小。**不追加block／阈值网格，不修改停止规则。

本次四组100帧基线材料指标均在原冻结界限内；固定Stiff本次通过不撤销上轮0/3失败。短窗新候选两场景各3/3通过原材料数值检查；固定off仅2/3，r1最大伸长1.037315868超过冻结1.036172766。既有重复稳定性与协议认证缺口仍存在，不放宽阈值。

100帧当前／Stiff布料位置最大逐帧RMS差：悬挂1.395微米，固定27.448毫米（f45–59为12.470毫米）；Stiff真实速度缺失，不能从位移差分补造。短窗off/on最大逐帧RMS差：悬挂位置8.780微米／实际速度0.000235m/s；固定位置15.685毫米／实际速度0.299833m/s。固定最大单点差为0.103316m／2.42530m/s，ABD漂移仅浮点量级。轨迹差异不是材料失败的同义词，也不能仅凭两种算法达同一标量阈值就宣布同质量。

本轮没有新的独立接受路径CPU CCD、完整候选100／300帧、AutoDL或默认推广。GPU安全检测保留、同状态CCD相同不等于独立接受路径审计完成。

## 6. 新追踪与下一轮选择

两次实际节点捕获覆盖hang41／fixed49，分别从零42／50帧；新程序b1a1，bounded_ccd=false。全部GPU活动union为103.845／157.164ms，CPU物理帧区间312.150／259.640ms。这些是profile诊断轨迹，fixed49为7方向／540PCG，不与100帧基线该帧8方向／659PCG假装同状态。

MAS数值填充＋聚合＋求逆总GPU10.439／11.822ms，仅占诊断帧3.34%／4.55%，连全部清零都不够5%。不再先写一个MAS小kernel。Graph WHILE迭代边界发现16.393／33.398ms无活动包络（5.25%／12.86%），但长尾明显，含WDDM、共享GPU及profiler因素，不能当成已证明可消除调度时间。CPU final_readback包含等待Graph执行，绝不与GPU时间相加。

**下一唯一优先实现方向：复用每个Newton方向的swept barrier候选池，线搜索仅重新做离散分类／压缩，替代普通VF／EE query。**本次普通query实际GPU22.855／32.644ms，占诊断帧7.32%／12.57%；这是明确执行工作。仅按这两帧粗算，须省约68%／40%的原query成本，再扣维护、包含检查、分类和压缩，才可能净减帧时间5%。尚无整场收益保证；2×缺口仍需组合收益或更多有证据的工作削减。

源码证明和边界：FullCCD查询已使用sqrt(dHat)，swept bounds包含起点及请求端点。实数域同一线性路径内普通barrier重叠属于该集合。但CCD int4缺少近平行EE编码所需原edge ID，候选池必须另保留原始ID、方向、重复及类型信息。ABD的J(q-alpha*dq)与x-alpha*(Jdq)浮点不严格相同；有限double反例已确认，必须每trial实际bounds包含检查，不成立明确回退旧query。`isIntersected`仍使用BVH，初版只替普通query，不顺手删buildBVH、安全CCD或地面检测。

下一轮有限方案：独立default-off入口，先同trial核对完整typed DCD／CCD多重集、近平行编码、类型内置换、分项能量和接受alpha，再三轮交错51／59帧；维护＋查询＋窄相净收益不达5%即停止。显式验证零pool、尾块、扩容、跨方向／拓扑／dHat／mapping失效、包含失败回退及固定ABD。通过短窗及材料门槛后才安排三轮100帧；质量协议不更改。

WHILE分块是备用研究：先用同一冻结A/b/M、未profiled事件对照，证实边界成本可重复，再考虑小幅有限展开。必须锁存首次收敛、previous-rho时序、错误和预算，禁止停止后修改x/r／计数；需计入额外尾轮作用，不能天真重复capture代码。没有证据不与候选池同时实施。

## 7. 复现与证据

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/active/batch.py --plan configs/active/ipc_contact_20261006_baseline.json
& 'E:/Anaconda/envs/DL/python.exe' tools/active/build.py build --label ipc_bounded_20261006 --jobs 2
& 'E:/Anaconda/envs/DL/python.exe' tools/active/fixtures.py --name ipc_bounded_20261006_fixtures --bounded-ccd
& 'E:/Anaconda/envs/DL/python.exe' tools/active/batch.py --plan configs/active/ipc_bounded_20261006_guards.json
& 'E:/Anaconda/envs/DL/python.exe' tools/active/batch.py --plan configs/active/ipc_bounded_20261006_screen.json
& 'E:/Anaconda/envs/DL/python.exe' tools/active/batch.py --plan configs/active/ipc_bounded_20261006_profile.json
& 'E:/Anaconda/envs/DL/python.exe' tools/active/export_cost_plan.py --plan configs/active/ipc_bounded_20261006_profile.json
```

命令按本轮记录列出；基线先于新构建执行，复现应恢复对应归档程序和清单。现有输出不可覆盖，重跑须使用新名称与新计划。100帧性能包括图构建、转换和预条件器；加载／trace导出另列。

成本证据：`contact_cost_rank_20261006_agent.json`、`contact_linear_cost_20261006_agent.json`及两份SQLite／linear_profile。质量：`contact_baseline_quality_20261006_agent.json`；机制：`bounded_ccd_mechanism_20261006_agent.json`；运行：三个ipc_bounded阶段batch／analysis和初始baseline_batch；身份／索引核验：`ipc_bounded_20261006_verification.json`；下一轮详细边界：`NEXT_CONTACT_POOL_PLAN_20261006.md`。594项历史索引及旧决定保留，本轮不清理旧数据。

结论分级：**已验证**本轮完成、身份、局部CCD等价和未过速度门槛；**支持但未证明**候选池复用的整场潜力；**已排除本轮推广**bounded CCD、再追加MAS小融合；**待验证**候选池机器包含／正确性／净收益、WHILE调度归因、完整独立质量认证和2×。
